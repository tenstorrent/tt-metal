# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from PIL import Image, ImageOps

import ttnn

from ...encoders.qwen3vl.loader_flux3_klein import build_flux3_klein_text_encoder
from ...models.transformers.flux3_klein.transformer_block_flux3_klein import PRECISION_BF16, Flux3KleinPrecision
from ...models.transformers.flux3_klein.transformer_flux3_klein import (
    EARLY_DEPTH,
    IN_CHANNELS,
    SINGLE_DEPTH,
    Flux3KleinPlan,
    Flux3KleinPlanParts,
    Flux3KleinSequenceLayout,
    Flux3KleinTransformer,
    ids_digest,
    padded_length,
    timestep_inputs,
)
from ...parallel.config import DiTGParallelConfigNoCFG, EncoderParallelConfig, ParallelFactor
from ...parallel.manager import CCLManager
from ...utils import cache, tensor
from ...utils.dbcache import DBCacheConfig
from .canvas_flux3_klein import reference_canvases
from .dbcache_flux3_klein import Flux3KleinDBCache
from .packing import (
    DOWNSAMPLING_RATIO,
    MAX_TEXT_LENGTH,
    TEXT_BUCKETS,
    TEXT_BUCKETS_BY_LAYOUT,
    image_ids,
    latent_shape,
    output_shape,
    pad_rows,
    reference_ids,
    text_bucket,
)
from .prompt_encoder import Flux3KleinPromptEncoder
from .scheduler import GUIDANCE, STEPS, Flux3KleinScheduler
from .vae_flux3_klein import Flux3KleinVae
from .weights_flux3_klein import checkpoint_id, resolve_weights_dir

# Klein's q / k norms split their work by (row, head): bit-identical, faster. The program factory reads it at compile time.
os.environ.setdefault("TT_DIT_RMSNORM_HEAD_SPLIT", "1")

# The first release serves fixed shapes (width, height), all warmed up at startup, and at most one reference image,
# prepared as BFL's single reference: fitted (aspect kept, never cropped) to at most 4 MP and a 4096 side, rounded up to
# 16 px and reflect-padded. Its tokens fill one of two buckets; the bucket's padding is masked.
SIZES = (1024, 2048, 4096)
SHAPES = (*((size, size) for size in SIZES), (4096, 2304), (2304, 4096))
DEFAULT_SIZE = 2048
MAX_REFERENCES = 1
MATCH_MAX_PIXELS = 4 * 1024 * 1024
MIN_SIDE, MAX_SIDE = 64, 4096
REFERENCE_TOKEN_BUCKETS = (4096, MATCH_MAX_PIXELS // DOWNSAMPLING_RATIO**2)
PLAN_CACHE_SIZE = 64
# the request parts of the most recent (prompt length, reference size and ids) per plan, so a repeat uploads nothing
# (4096^2 with a 4 MP reference: ~3.3 MB per device each)
PLAN_PARTS_CACHE_SIZE = 64
# The media server's 1 / 4 MP reference choice (reference_size) predates BFL's fit and is ignored: every shape takes
# any reference; kept so its callers keep working.
REFERENCE_SIZES = {shape: (1024, 2048) for shape in SHAPES}


@dataclass(frozen=True)
class Flux3KleinRequest:
    prompt: str
    model: str = "distilled"
    width: int = DEFAULT_SIZE
    height: int = DEFAULT_SIZE
    seed: int | None = None
    steps: int | None = None
    guidance: float | None = None
    images: tuple = ()
    reference_size: int | None = None  # ignored (REFERENCE_SIZES)
    cache: DBCacheConfig | None = None
    return_final_tokens: bool = False


@dataclass
class _PlanSample:
    """One denoise branch's plan: its cache keys, layout and ids, and its parts as host tensors unless cached."""

    key: tuple
    parts_key: tuple
    layout: Flux3KleinSequenceLayout
    ids: dict[str, torch.Tensor]
    parts: Flux3KleinPlanParts | None


@dataclass
class Flux3KleinResult:
    image: Image.Image | None  # None on the ranks other than 0 of a multi-host run
    prompt: str
    seed: int
    timings: dict[str, float]
    cached_steps: int = 0
    final_tokens: torch.Tensor | None = None


def layout_buckets(buckets: tuple[int, ...], width: int, height: int, references: int) -> tuple[int, ...]:
    """The text buckets a request of this layout picks from: TEXT_BUCKETS narrowed per layout."""
    return TEXT_BUCKETS_BY_LAYOUT.get(((width, height), references), buckets) if buckets is TEXT_BUCKETS else buckets


def make_noise(seed: int, h: int, w: int) -> torch.Tensor:
    """BFL's draw: [1, C, h, w] in fp32, rounded to bf16, one row per latent pixel."""
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn((1, IN_CHANNELS["image"], h, w), generator=generator)
    return noise.to(torch.bfloat16).flatten(2).transpose(1, 2)[0].float().contiguous()


def caption_seed(caption: str) -> int:
    return int(hashlib.sha256(caption.encode()).hexdigest(), 16) % 2**32


def reference_canvas(width: int, height: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """(fitted size, canvas) of a single reference as BFL's pipeline.reference_canvas: scaled down to fit 4 MP and a
    4096 side, the canvas the fitted size rounded up to 16 (sides 64-4096) and at most 4 MP."""
    scale = min(1.0, math.sqrt(MATCH_MAX_PIXELS / (width * height)), MAX_SIDE / max(width, height))
    while True:
        fit = (max(1, round(width * scale)), max(1, round(height * scale)))
        canvas = tuple(min(MAX_SIDE, max(MIN_SIDE, ttnn.core.roundup(v, DOWNSAMPLING_RATIO))) for v in fit)
        if canvas[0] * canvas[1] <= MATCH_MAX_PIXELS:
            return fit, canvas
        scale *= 0.995


def prepare_reference(image: Image.Image) -> Image.Image:
    """BFL's single reference (pipeline.prepare_reference): RGB, fitted by reference_canvas, reflect-padded to its
    canvas, centred. The output size is the request's own, so the padding box is not needed."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    fit, (w, h) = reference_canvas(image.width, image.height)
    if fit != image.size:
        image = image.resize(fit, Image.Resampling.LANCZOS)
    if (w, h) == image.size:
        return image
    left, top = (w - image.width) // 2, (h - image.height) // 2
    padding = ((top, h - image.height - top), (left, w - image.width - left), (0, 0))
    mode = "reflect" if min(image.size) > DOWNSAMPLING_RATIO else "symmetric"
    return Image.fromarray(np.pad(np.asarray(image), padding, mode=mode))


def reference_bucket(tokens: int) -> int:
    if tokens > REFERENCE_TOKEN_BUCKETS[-1]:
        raise ValueError(f"{tokens} reference tokens exceed {REFERENCE_TOKEN_BUCKETS[-1]}")
    return next(b for b in REFERENCE_TOKEN_BUCKETS if tokens <= b)


# The guidance and Euler arithmetic round to bf16 where the reference does; every op runs in fp32 with explicit
# typecasts so the update matches the reference bit for bit given the same velocity.
def combine_cfg(conditional: ttnn.Tensor, unconditional: ttnn.Tensor, guidance: ttnn.Tensor) -> ttnn.Tensor:
    c32 = ttnn.typecast(conditional, ttnn.float32)
    u32 = ttnn.typecast(unconditional, ttnn.float32)
    d = ttnn.typecast(ttnn.typecast(ttnn.subtract(c32, u32), ttnn.bfloat16), ttnn.float32)
    ttnn.deallocate(c32)
    m = ttnn.typecast(ttnn.typecast(ttnn.multiply(d, guidance), ttnn.bfloat16), ttnn.float32)
    ttnn.deallocate(d)
    v = ttnn.typecast(ttnn.add(u32, m), ttnn.bfloat16)
    ttnn.deallocate(u32)
    ttnn.deallocate(m)
    return v


def euler_update(x: ttnn.Tensor, velocity: ttnn.Tensor, dt: ttnn.Tensor, mask: ttnn.Tensor) -> ttnn.Tensor:
    v32 = ttnn.typecast(velocity, ttnn.float32)
    p = ttnn.typecast(ttnn.typecast(ttnn.multiply(v32, dt), ttnn.bfloat16), ttnn.float32)
    ttnn.deallocate(v32)
    # the mask keeps the alignment rows at zero
    delta = ttnn.multiply(p, mask)
    ttnn.deallocate(p)
    x_new = ttnn.add(x, delta)
    ttnn.deallocate(delta)
    return x_new


def image_rows(image: ttnn.Tensor) -> ttnn.Tensor:
    """The [1, H, W, 3] image as one [1, H, W * 3] row per image row."""
    n, h, w, c = (int(d) for d in image.shape)
    return ttnn.reshape(ttnn.to_layout(image, ttnn.ROW_MAJOR_LAYOUT), (n, h, w * c))


def image_to_uint8(image: ttnn.Tensor) -> ttnn.Tensor:
    """The reference's ((x + 1) * 127.5).clamp().to(uint8) on device, as one [1, H, W * 3] row per image row."""
    return rows_to_uint8(image_rows(image))


def rows_to_uint8(x: ttnn.Tensor) -> ttnn.Tensor:
    x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
    x = ttnn.typecast(x, ttnn.float32)
    x = ttnn.add(x, 1.0)
    x = ttnn.multiply(x, 127.5)
    x = ttnn.clamp(x, min=0.0, max=255.0)
    x = ttnn.floor(x)
    return ttnn.typecast(x, ttnn.uint8)


class Flux3KleinPipeline:
    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        *,
        snapshots: dict[str, Path],
        text_length: int = MAX_TEXT_LENGTH,
        text_buckets: tuple[int, ...] | None = TEXT_BUCKETS,
        sp_axis: int = 1,
        tp_axis: int = 0,
        topology: ttnn.Topology = ttnn.Topology.Ring,
        num_links: int = 2,
        precision: Flux3KleinPrecision = PRECISION_BF16,
    ) -> None:
        self._device = mesh_device
        self._text_length = text_length
        self._text_buckets = text_buckets
        self._sp_axis = sp_axis
        self._tp_axis = tp_axis
        mesh_shape = tuple(mesh_device.shape)
        shared = next(iter(snapshots.values()))
        if ttnn.using_distributed_env():
            # the TP collectives stay on one host: each host holds whole TP columns, the SP axis spans the hosts
            local = tensor.local_mesh_coords(mesh_device)
            if len(local) != mesh_shape[tp_axis] * len({c[1 - tp_axis] for c in local}):
                raise ValueError(f"this host does not hold whole TP columns (axis {tp_axis}) of the {mesh_shape} mesh")

        encoder_parallel = EncoderParallelConfig(
            tensor_parallel=ParallelFactor(factor=mesh_shape[tp_axis], mesh_axis=tp_axis)
        )
        encoder, text_config = build_flux3_klein_text_encoder(
            shared,
            mesh_device=mesh_device,
            parallel_config=encoder_parallel,
            ccl_manager=CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology),
            # the TP shard is ~1.8 GB per chip: keep it resident instead of all-gathering every weight per prompt
            is_fsdp=False,
        )
        self._text_encoder = encoder
        self._prompt_encoder = Flux3KleinPromptEncoder(encoder, shared, mesh_device, text_config=text_config)

        self._ccl = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)
        parallel = DiTGParallelConfigNoCFG(
            tensor_parallel=ParallelFactor(factor=mesh_shape[tp_axis], mesh_axis=tp_axis),
            sequence_parallel=ParallelFactor(factor=mesh_shape[sp_axis], mesh_axis=sp_axis),
        )
        self._transformers: dict[str, Flux3KleinTransformer] = {}
        for model, snapshot in snapshots.items():
            transformer = Flux3KleinTransformer(
                flow_map=model == "distilled",
                device=mesh_device,
                ccl_manager=self._ccl,
                parallel_config=parallel,
                precision=precision,
            )
            cache.load_model(
                transformer,
                model_name=f"flux3_klein_{model}_{checkpoint_id(Path(snapshot) / 'model.safetensors')}",
                subfolder=f"transformer_D{EARLY_DEPTH}_S{SINGLE_DEPTH}{'_flowmap' if transformer.flow_map else ''}",
                parallel_config=parallel,
                mesh_shape=mesh_shape,
                mesh_device=mesh_device,
                dtype=precision.name,
                get_torch_state_dict=lambda snapshot=snapshot: _load_dit_state(snapshot),
            )
            self._transformers[model] = transformer

        self._vae = Flux3KleinVae(
            shared,
            device=mesh_device,
            ccl_manager=CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology),
            dit_ccl_manager=self._ccl,
            dit_sp_axis=sp_axis,
        )
        # the base model's unconditional sample keeps the reference's own text length
        self._null_text = self._prompt_encoder.encode("") if "base" in snapshots else None
        # Device constants that depend only on the request's layout (RoPE tables, modulation and row-gather indices,
        # the latent row mask), kept across requests instead of rebuilt and uploaded each time.
        self._plan_cache: dict[tuple, Flux3KleinPlan] = {}
        self._plan_parts_cache: dict[tuple, Flux3KleinPlanParts] = {}
        self._mask_cache: dict[tuple[int, int], ttnn.Tensor] = {}

    @classmethod
    def create(cls, mesh_device: ttnn.MeshDevice, *, models: tuple[str, ...] = ("distilled",), **kwargs):
        return cls(mesh_device, snapshots={model: resolve_weights_dir(model) for model in models}, **kwargs)

    def warmup(
        self, sizes: tuple[int | tuple[int, int], ...] = SHAPES, reference_buckets: tuple[int, ...] | None = None
    ) -> None:
        """Compile every program of ``sizes`` (square sides or (width, height); every served shape by default): the
        VAE encoder on every reference canvas, then each size without a reference and with a square reference that
        fills each token bucket (or those of ``reference_buckets``), at every text bucket that layout picks from (a
        long prompt reaches the larger buckets). A reference of any other shape runs its canvas's and its bucket's
        programs."""
        for canvas in reference_canvases():
            self._vae.encode_canvas(Image.new("RGB", canvas))
        buckets = set(self._text_buckets or (self._text_length,))
        if self._text_buckets is TEXT_BUCKETS:
            buckets.update(b for layout in TEXT_BUCKETS_BY_LAYOUT.values() for b in layout)
        for bucket in sorted(buckets):
            self._prompt_encoder.warmup(bucket)
        saved = self._text_buckets
        try:
            for model in self._transformers:
                for width, height in map(output_shape, sizes):
                    served = (b for b in REFERENCE_TOKEN_BUCKETS if reference_buckets is None or b in reference_buckets)
                    for side in (0, *(math.isqrt(b) * DOWNSAMPLING_RATIO for b in served)):
                        for bucket in layout_buckets(saved, width, height, int(side > 0)) if saved else (None,):
                            self._text_buckets = (bucket,) if bucket else None
                            self.generate(
                                Flux3KleinRequest(
                                    "warmup",
                                    model=model,
                                    width=width,
                                    height=height,
                                    seed=0,
                                    images=(Image.new("RGB", (side, side)),) if side else (),
                                )
                            )
        finally:
            self._text_buckets = saved

    def generate(self, request: Flux3KleinRequest) -> Flux3KleinResult:
        if request.model not in self._transformers:
            raise ValueError(f"model {request.model!r} is not loaded ({tuple(self._transformers)})")
        width, height = request.width, request.height
        if (width, height) not in SHAPES:
            raise ValueError(f"{width}x{height} is not one of the served shapes {SHAPES} (width, height)")
        if len(request.images) > MAX_REFERENCES:
            raise ValueError(f"at most {MAX_REFERENCES} reference image")
        if request.reference_size is not None:
            logger.info(f"reference_size {request.reference_size} ignored: the reference is fitted as BFL's (<= 4 MP)")
        steps = request.steps or STEPS[request.model]
        guidance = GUIDANCE[request.model] if request.guidance is None else request.guidance
        if request.model == "distilled" and guidance not in (None, 1.0):
            raise ValueError("the distilled model takes no guidance")
        guidance = None if request.model == "distilled" else guidance
        reference = None
        if request.images:
            reference = request.images[0]
            if not isinstance(reference, Image.Image):
                reference = Image.open(reference)
            reference = prepare_reference(reference)
        timings = {}

        prompt = request.prompt
        seed = caption_seed(prompt) if request.seed is None else request.seed
        h, w = latent_shape(height, width)
        # the noise and its host tensor are built on a helper thread while this thread dispatches the text encoder
        sp = self._transformers[request.model].sp_factor
        latent_job = self._prestep_pool().submit(self._prepare_latent_host, seed, h, w, sp)

        t0 = time.time()
        text_rows = self._text_length
        if self._text_buckets:
            buckets = layout_buckets(self._text_buckets, width, height, len(request.images))
            text_rows = text_bucket(self._prompt_encoder.tokenize(prompt, self._text_length)[2], buckets)
        text = self._prompt_encoder.encode(prompt, text_rows)
        # the reference's own latent shape: h, w stay the generated image's (the pre-step helper's noise uses them)
        cond_ids = None if reference is None else reference_ids(*latent_shape(reference.height, reference.width), 0)
        # a new prompt length or reference size: its plan parts are built on the helper thread while the encoders run
        plan_job = self._prestep_pool().submit(
            self._plan_samples, request.model, h, w, text, cond_ids, guidance is not None
        )
        timings["encode"] = time.time() - t0

        t0 = time.time()
        cond_tokens = None if reference is None else self._vae.encode_canvas(reference)
        timings["encode_reference"] = time.time() - t0

        t0 = time.time()
        noise, x_host = latent_job.result()
        state, cached_steps = self._denoise(
            request.model, noise, plan_job.result(), text, cond_tokens, steps, guidance, request.cache, x_host=x_host
        )
        ttnn.deallocate(text[0])
        timings["denoise"] = time.time() - t0

        t0 = time.time()
        decoded = self._vae.decode(state, size=(width, height))
        ttnn.synchronize_device(self._device)
        timings["decode"] = time.time() - t0

        t0 = time.time()
        rows = self._read_image(decoded)
        image = None if rows is None else Image.fromarray(rows.reshape(height, width, 3).numpy())
        ttnn.deallocate(decoded)
        final_tokens = None
        if request.return_final_tokens:
            if ttnn.using_distributed_env():
                raise ValueError("return_final_tokens is single-host only")
            final_tokens = tensor.to_torch(state, mesh_axes=[None, self._sp_axis, None])[0][: h * w].clone()
        ttnn.deallocate(state)
        timings["to_host"] = time.time() - t0
        logger.info(f"FLUX 3 Klein {request.model} {width}x{height} refs {len(request.images)}: {timings}")
        return Flux3KleinResult(
            image=image,
            prompt=prompt,
            seed=seed,
            timings=timings,
            cached_steps=cached_steps,
            final_tokens=final_tokens,
        )

    def _prestep_pool(self) -> ThreadPoolExecutor:
        pool = getattr(self, "_prestep_executor", None)
        if pool is None:
            pool = self._prestep_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="klein-prestep")
        return pool

    def _prepare_latent_host(self, seed: int, h: int, w: int, sp: int) -> tuple[torch.Tensor, ttnn.Tensor]:
        """Host half of the latent upload: the noise and its padded, SP-sharded fp32 host tensor (no device access)."""
        noise = make_noise(seed, h, w)
        rows_padded = padded_length(noise.shape[0], sp)
        x_host = tensor.from_torch(
            pad_rows(noise, rows_padded)[None],
            device=self._device,
            dtype=ttnn.float32,
            mesh_axes=[None, self._sp_axis, None],
            on_host=True,
        )
        return noise, x_host

    def _read_image(self, decoded: ttnn.Tensor) -> torch.Tensor | None:
        """The uint8 [H, W * 3] rows of the decoded image, H-sharded over the VAE's h axis. Across hosts the TP
        replicas are split over the columns first (every chip then reads unique rows), the rows are gathered on device
        over the host axis and read by rank 0 alone; the other ranks get None."""
        if not ttnn.using_distributed_env():
            u8 = image_to_uint8(decoded)
            rows = tensor.to_torch(u8, mesh_axes=[None, self._vae.h_axis, None])[0]
            ttnn.deallocate(u8)
            return rows
        x = ttnn.mesh_partition(image_rows(decoded), dim=2, cluster_axis=self._tp_axis)
        concat_dims: list[int | None] = [None, None]
        concat_dims[self._vae.h_axis] = 1
        concat_dims[self._tp_axis] = 2
        rows = tensor.fast_device_to_host(
            x, self._device, concat_dims, ccl_manager=self._vae.ccl_manager, root=0, pre_transfer_fn=rows_to_uint8
        )
        ttnn.deallocate(x)
        return None if rows is None else rows[0]

    def _upload_rows(self, rows: torch.Tensor | ttnn.Tensor, sp_factor: int) -> ttnn.Tensor:
        if isinstance(rows, ttnn.Tensor):
            # replicated device rows (the text encoding) are split over SP on device
            if rows.shape[-2] != padded_length(rows.shape[-2], sp_factor):
                raise ValueError(f"{rows.shape[-2]} device rows are not a multiple of {sp_factor} tiles")
            return ttnn.mesh_partition(rows, dim=1, cluster_axis=self._sp_axis)
        rows = pad_rows(rows.to(torch.bfloat16), padded_length(rows.shape[0], sp_factor))
        return tensor.from_torch(
            rows[None], device=self._device, dtype=ttnn.bfloat16, mesh_axes=[None, self._sp_axis, None]
        )

    def _scalar(self, value: torch.Tensor) -> ttnn.Tensor:
        return tensor.from_torch(value, device=self._device, dtype=ttnn.float32, mesh_axes=[None, None])

    def _plan_samples(
        self, model: str, h: int, w: int, text: tuple, cond_ids: torch.Tensor | None, guided: bool
    ) -> list[_PlanSample]:
        """The plan of each denoise branch (conditional, null text with guidance). No device access: runs on the helper
        thread; parts already in the parts cache are not built."""
        sp = self._transformers[model].sp_factor
        ids = image_ids(h, w)
        # the reference's tokens pad to their bucket: every reference size of a bucket runs the same programs
        cond_valid = 0 if cond_ids is None else cond_ids.shape[0]
        cond_rows = 0 if cond_ids is None else reference_bucket(cond_valid)
        samples = []
        for branch, (text_rows, text_ids, text_valid) in enumerate([text, self._null_text] if guided else [text]):
            lengths = {"image": h * w, "image_cond": cond_rows, "text": text_rows.shape[-2]}
            layout = Flux3KleinSequenceLayout(sp, lengths, text_valid, cond_valid)
            sample_ids = {"image": ids, "text": text_ids}
            if cond_ids is not None:
                sample_ids["image_cond"] = pad_rows(cond_ids, cond_rows)
            # one plan per branch, mode lengths and image / text ids (the branch keeps the conditional and null-text
            # samples apart); its parts per prompt length and reference size and ids
            key = (model, branch, sp, tuple(sorted(lengths.items())), _ids_digest({"image": ids, "text": text_ids}))
            parts_key = (
                key,
                text_valid,
                cond_valid,
                _ids_digest({k: v for k, v in sample_ids.items() if k == "image_cond"}),
            )
            parts = (
                None
                if parts_key in self._plan_parts_cache
                else self._transformers[model].plan_parts(layout, sample_ids)
            )
            samples.append(_PlanSample(key, parts_key, layout, sample_ids, parts))
        return samples

    def _denoise(
        self,
        model: str,
        noise: torch.Tensor,
        plans: list[_PlanSample],
        text: tuple[torch.Tensor, torch.Tensor, int],
        cond_tokens: torch.Tensor | None,
        steps: int,
        guidance: float | None,
        cache_config: DBCacheConfig | None,
        x_host: ttnn.Tensor | None = None,
    ) -> tuple[ttnn.Tensor, int]:
        transformer = self._transformers[model]
        sp = transformer.sp_factor
        cond_embedded = None
        if cond_tokens is not None:
            tokens = self._upload_rows(pad_rows(cond_tokens, plans[0].layout.lengths["image_cond"]), sp)
            cond_embedded = transformer.embed_image_cond(tokens)
            ttnn.deallocate(tokens)

        def sample(spec: _PlanSample, text_rows: ttnn.Tensor):
            tokens = self._upload_rows(text_rows, sp)
            embedded = transformer.embed_text(tokens)
            ttnn.deallocate(tokens)
            # least recently used first in both caches
            plan = self._plan_cache.pop(spec.key, None) or transformer.prepare_plan(spec.layout, spec.ids)
            self._plan_cache[spec.key] = plan
            while len(self._plan_cache) > PLAN_CACHE_SIZE:
                self._plan_cache.pop(next(iter(self._plan_cache))).deallocate()
            parts = self._plan_parts_cache.pop(spec.parts_key, None)
            if parts is None:
                parts = transformer.upload_plan_parts(spec.parts or transformer.plan_parts(spec.layout, spec.ids))
            self._plan_parts_cache[spec.parts_key] = parts
            while len(self._plan_parts_cache) > PLAN_PARTS_CACHE_SIZE:
                self._plan_parts_cache.pop(next(iter(self._plan_parts_cache))).deallocate()
            plan.parts = parts
            return plan, embedded

        samples = [sample(plans[0], text[0])]
        if guidance is not None:
            samples.append(sample(plans[1], self._null_text[0]))
        image_axes = [None, self._sp_axis, None]
        rows, rows_padded = noise.shape[0], padded_length(noise.shape[0], sp)
        if x_host is not None:
            x = ttnn.to_device(x_host, self._device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        else:
            x = tensor.from_torch(
                pad_rows(noise, rows_padded)[None], device=self._device, dtype=ttnn.float32, mesh_axes=image_axes
            )
        mask = self._mask_cache.get((rows, rows_padded))
        if mask is None:
            mask = torch.zeros(1, rows_padded, IN_CHANNELS["image"])
            mask[:, :rows] = 1.0
            mask = tensor.from_torch(mask, device=self._device, dtype=ttnn.float32, mesh_axes=image_axes)
            self._mask_cache[(rows, rows_padded)] = mask
        guidance_tt = self._scalar(torch.tensor([[guidance]])) if guidance is not None else None
        dbcache = Flux3KleinDBCache(cache_config, self._device, num_branches=len(samples)) if cache_config else None

        scheduler = Flux3KleinScheduler(steps)
        for index in range(len(scheduler)):
            step = scheduler.step(index)
            if dbcache is not None:
                dbcache.context.mark_step_begin()
            t_embed, t_ref_embed = timestep_inputs(
                step.embedding_t, step.embedding_t_next if transformer.flow_map else None
            )
            t_embed = self._scalar(t_embed)
            t_ref_embed = self._scalar(t_ref_embed) if t_ref_embed is not None else None
            dt = self._scalar(torch.full((1, 1), step.dt))

            image_tokens = ttnn.typecast(x, ttnn.bfloat16)
            velocities = [
                transformer(
                    image_tokens=image_tokens,
                    text_embedded=embedded,
                    image_cond_embedded=cond_embedded,
                    t_embed=t_embed,
                    t_ref_embed=t_ref_embed,
                    plan=plan,
                    cache=dbcache.runner(branch) if dbcache is not None else None,
                )
                for branch, (plan, embedded) in enumerate(samples)
            ]
            ttnn.deallocate(image_tokens)
            if guidance is not None:
                velocity = combine_cfg(*velocities, guidance_tt)
                for v in velocities:
                    ttnn.deallocate(v)
            else:
                (velocity,) = velocities
            x_new = euler_update(x, velocity, dt, mask)
            for t in (x, velocity, t_embed, t_ref_embed, dt):
                if t is not None:
                    ttnn.deallocate(t)
            x = x_new
        ttnn.synchronize_device(self._device)

        cached_steps = 0
        if dbcache is not None:
            cached_steps = dbcache.context.num_cached_steps(0)
            dbcache.close()
        for _, embedded in samples:
            ttnn.deallocate(embedded)
        for t in (cond_embedded, guidance_tt):
            if t is not None:
                ttnn.deallocate(t)
        return x, cached_steps

    def close(self) -> None:
        if getattr(self, "_prestep_executor", None) is not None:
            self._prestep_executor.shutdown(wait=True)
            self._prestep_executor = None
        self.clear_constant_caches()
        for module in (*self._transformers.values(), self._text_encoder):
            module.deallocate_weights()
        self._vae.deallocate_weights()

    def clear_constant_caches(self) -> None:
        for plan in self._plan_cache.values():
            plan.deallocate()
        for parts in self._plan_parts_cache.values():
            parts.deallocate()
        self._plan_parts_cache.clear()
        for mask in self._mask_cache.values():
            ttnn.deallocate(mask)
        self._plan_cache.clear()
        self._mask_cache.clear()


def _ids_digest(ids: dict[str, torch.Tensor]) -> tuple:
    """A key for a request's token ids (shape + content), so a cached plan is reused only for identical ids."""
    return tuple((name, *ids_digest(t)) for name, t in sorted(ids.items()))


def _load_dit_state(snapshot: Path) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    state = load_file(str(Path(snapshot) / "model.safetensors"))
    return {k: v.to(torch.bfloat16) if v.is_floating_point() else v for k, v in state.items()}
