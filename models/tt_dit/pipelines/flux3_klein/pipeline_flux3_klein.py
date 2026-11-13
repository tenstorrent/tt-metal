# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from PIL import Image, ImageOps

import ttnn

from ...encoders.qwen3vl.loader_flux3_klein import build_flux3_klein_text_encoder
from ...models.transformers.flux3_klein.transformer_flux3_klein import (
    EARLY_DEPTH,
    IN_CHANNELS,
    SINGLE_DEPTH,
    TILE,
    Flux3KleinSequenceLayout,
    Flux3KleinTransformer,
    timestep_inputs,
)
from ...models.vae.vae_flux3_klein import Flux3KleinVae
from ...parallel.config import DiTGParallelConfigNoCFG, EncoderParallelConfig, ParallelFactor
from ...parallel.manager import CCLManager
from ...utils import cache, tensor
from ...utils.dbcache import DBCacheConfig
from .dbcache_flux3_klein import Flux3KleinDBCache
from .packing import DOWNSAMPLING_RATIO, MAX_TEXT_LENGTH, image_ids, latent_shape, pad_rows, reference_ids
from .prompt_encoder import Flux3KleinPromptEncoder
from .scheduler import GUIDANCE, STEPS, Flux3KleinScheduler
from .upsampler_flux3_klein import Flux3KleinSharedUpsampler, reserve_upsampler_ranges
from .weights_flux3_klein import resolve_weights_dir

SIZES = (1024, 2048, 4096)
DEFAULT_SIZE = 2048
MAX_REFERENCES = 1
MATCH_MAX_PIXELS = 4 * 1024 * 1024
MIN_SIDE, MAX_SIDE = 64, 4096
REFERENCE_TOKEN_BUCKETS = (4096, MATCH_MAX_PIXELS // DOWNSAMPLING_RATIO**2)
WARMUP_PROMPT = "warmup"


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
    cache: DBCacheConfig | None = None


@dataclass
class Flux3KleinResult:
    image: Image.Image
    prompt: str
    seed: int
    timings: dict[str, float]
    cached_steps: int = 0


def make_noise(seed: int, h: int, w: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn((h * w, IN_CHANNELS["image"]), generator=generator)
    return noise.to(torch.bfloat16).float()


def caption_seed(caption: str) -> int:
    return int(hashlib.sha256(caption.encode()).hexdigest(), 16) % 2**32


def reference_canvas(width: int, height: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """(fitted size, canvas) of a single reference as BFL's pipeline.reference_canvas: scaled down to fit 4 MP and a
    4096 side, the canvas the fitted size rounded up to 16 (sides 64-4096) and at most 4 MP."""
    scale = min(1.0, math.sqrt(MATCH_MAX_PIXELS / (width * height)), MAX_SIDE / max(width, height))
    while True:
        fit = (max(1, round(width * scale)), max(1, round(height * scale)))
        canvas = tuple(min(MAX_SIDE, max(MIN_SIDE, -(-v // DOWNSAMPLING_RATIO) * DOWNSAMPLING_RATIO)) for v in fit)
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
    return next(b for b in REFERENCE_TOKEN_BUCKETS if tokens <= b)


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
    delta = ttnn.multiply(p, mask)
    ttnn.deallocate(p)
    x_new = ttnn.add(x, delta)
    ttnn.deallocate(delta)
    return x_new


def image_to_uint8(image: ttnn.Tensor) -> ttnn.Tensor:
    """The reference's ((x + 1) * 127.5).clamp().to(uint8) on device, as one [1, H, W * 3] row per image row."""
    n, h, w, c = (int(d) for d in image.shape)
    x = ttnn.to_layout(image, ttnn.ROW_MAJOR_LAYOUT)
    x = ttnn.reshape(x, (n, h, w * c))
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
        sp_axis: int = 1,
        tp_axis: int = 0,
        topology: ttnn.Topology = ttnn.Topology.Ring,
        num_links: int = 2,
        upsample: bool = False,
    ) -> None:
        """With ``upsample``, the prompt upsampler shares the first chip of ``mesh_device``, which must be opened
        with upsampler_flux3_klein.SHARED_MESH_PARAMS, and every prompt is rewritten before it is encoded."""
        self._upsampler_ranges = reserve_upsampler_ranges(mesh_device) if upsample else None
        self._device = mesh_device
        self._text_length = text_length
        self._sp_axis = sp_axis
        mesh_shape = tuple(mesh_device.shape)
        shared = next(iter(snapshots.values()))

        encoder_parallel = EncoderParallelConfig(
            tensor_parallel=ParallelFactor(factor=mesh_shape[tp_axis], mesh_axis=tp_axis)
        )
        encoder, text_config = build_flux3_klein_text_encoder(
            shared,
            mesh_device=mesh_device,
            parallel_config=encoder_parallel,
            ccl_manager=CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology),
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
                flow_map=model == "distilled", device=mesh_device, ccl_manager=self._ccl, parallel_config=parallel
            )
            cache.load_model(
                transformer,
                model_name=f"flux3_klein_{model}_{cache.checkpoint_id(Path(snapshot) / 'model.safetensors')}",
                subfolder=f"transformer_D{EARLY_DEPTH}_S{SINGLE_DEPTH}{'_flowmap' if transformer.flow_map else ''}",
                parallel_config=parallel,
                mesh_shape=mesh_shape,
                mesh_device=mesh_device,
                get_torch_state_dict=lambda snapshot=snapshot: _load_dit_state(snapshot),
            )
            self._transformers[model] = transformer

        self._vae = Flux3KleinVae(
            shared,
            device=mesh_device,
            ccl_manager=CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology),
            dit_ccl_manager=self._ccl,
            dit_sp_axis=sp_axis,
            sizes=SIZES,
        )
        self._null_text = self._prompt_encoder.encode("") if "base" in snapshots else None
        self._upsampler = Flux3KleinSharedUpsampler(mesh_device) if upsample else None

    @classmethod
    def create(cls, mesh_device: ttnn.MeshDevice, *, models: tuple[str, ...] = ("distilled",), **kwargs):
        return cls(mesh_device, snapshots={model: resolve_weights_dir(model) for model in models}, **kwargs)

    def warmup(self, sizes: tuple[int, ...] = SIZES) -> None:
        """Compile every program of the served shapes: each size without a reference and with one per reference token
        bucket (a square reference that fills the bucket). A reference of another shape reuses its bucket's DiT
        programs; only the VAE encoder compiles for a new reference shape."""
        for model in self._transformers:
            for size in sizes:
                for bucket in (None, *REFERENCE_TOKEN_BUCKETS):
                    images = ()
                    if bucket is not None:
                        side = math.isqrt(bucket) * DOWNSAMPLING_RATIO
                        images = (Image.new("RGB", (side, side)),)
                    self.generate(
                        Flux3KleinRequest(WARMUP_PROMPT, model=model, width=size, height=size, seed=0, images=images)
                    )

    def validate_request(self, request: Flux3KleinRequest) -> None:
        if request.model not in self._transformers:
            raise ValueError(f"model {request.model!r} is not loaded ({tuple(self._transformers)})")
        if request.height != request.width or request.width not in SIZES:
            raise ValueError(f"{request.width}x{request.height} is not one of the served sizes {SIZES} (square)")
        if len(request.images) > MAX_REFERENCES:
            raise ValueError(f"at most {MAX_REFERENCES} reference image")
        if request.model == "distilled" and request.guidance not in (None, 1.0):
            raise ValueError("the distilled model takes no guidance")

    def generate(self, request: Flux3KleinRequest) -> Flux3KleinResult:
        self.validate_request(request)
        size = request.width
        steps = request.steps or STEPS[request.model]
        guidance = GUIDANCE[request.model] if request.guidance is None else request.guidance
        guidance = None if request.model == "distilled" else guidance
        reference = None
        if request.images:
            reference = request.images[0]
            if not isinstance(reference, Image.Image):
                reference = Image.open(reference)
        timings = {}

        t0 = time.time()
        prompt = request.prompt if self._upsampler is None else self._upsampler(request.prompt, reference)
        timings["upsample"] = time.time() - t0
        seed = caption_seed(prompt) if request.seed is None else request.seed

        t0 = time.time()
        text = self._prompt_encoder.encode(prompt, self._text_length)
        timings["encode"] = time.time() - t0

        t0 = time.time()
        image_cond = None
        if reference is not None:
            canvas = prepare_reference(reference)
            h, w = latent_shape(canvas.height, canvas.width)
            image_cond = (self._vae.encode(canvas), reference_ids(h, w, 0))
        timings["encode_reference"] = time.time() - t0

        t0 = time.time()
        h, w = latent_shape(size, size)
        state, cached_steps = self._denoise(
            request.model, make_noise(seed, h, w), image_ids(h, w), text, image_cond, steps, guidance, request.cache
        )
        timings["denoise"] = time.time() - t0

        t0 = time.time()
        decoded = self._vae.decode(state, size=size)
        ttnn.synchronize_device(self._device)
        timings["decode"] = time.time() - t0

        t0 = time.time()
        u8 = image_to_uint8(decoded)
        rows = ttnn.to_torch(ttnn.get_device_tensors(u8)[0])[0]
        image = Image.fromarray(rows.reshape(size, size, 3).numpy())
        ttnn.deallocate(u8)
        ttnn.deallocate(decoded)
        ttnn.deallocate(state)
        timings["to_host"] = time.time() - t0
        logger.info(f"FLUX 3 Klein {request.model} {size}x{size} refs {len(request.images)}: {timings}")
        return Flux3KleinResult(image=image, prompt=prompt, seed=seed, timings=timings, cached_steps=cached_steps)

    def _upload_rows(self, rows: torch.Tensor, sp_factor: int) -> ttnn.Tensor:
        rows = pad_rows(rows.to(torch.bfloat16), ttnn.core.roundup(rows.shape[0], sp_factor * TILE))
        return tensor.from_torch(
            rows[None], device=self._device, dtype=ttnn.bfloat16, mesh_axes=[None, self._sp_axis, None]
        )

    def _scalar(self, value: torch.Tensor) -> ttnn.Tensor:
        return tensor.from_torch(value, device=self._device, dtype=ttnn.float32, mesh_axes=[None, None])

    def _denoise(
        self,
        model: str,
        noise: torch.Tensor,
        ids: torch.Tensor,
        text: tuple[torch.Tensor, torch.Tensor, int],
        image_cond: tuple[torch.Tensor, torch.Tensor] | None,
        steps: int,
        guidance: float | None,
        cache_config: DBCacheConfig | None,
    ) -> tuple[ttnn.Tensor, int]:
        transformer = self._transformers[model]
        sp = transformer.sp_factor
        cond_valid = 0 if image_cond is None else image_cond[0].shape[0]
        cond_rows = 0 if image_cond is None else reference_bucket(cond_valid)
        cond_embedded = None
        if image_cond is not None:
            tokens = self._upload_rows(pad_rows(image_cond[0], cond_rows), sp)
            cond_embedded = transformer.embed_image_cond(tokens)
            ttnn.deallocate(tokens)

        def sample(index: int, text_rows: torch.Tensor, text_ids: torch.Tensor, text_valid: int):
            lengths = {"image": noise.shape[0], "image_cond": cond_rows, "text": text_rows.shape[0]}
            layout = Flux3KleinSequenceLayout(sp, lengths, text_valid, cond_valid)
            sample_ids = {"image": ids, "text": text_ids}
            if image_cond is not None:
                sample_ids["image_cond"] = image_cond[1]
            tokens = self._upload_rows(text_rows, sp)
            embedded = transformer.embed_text(tokens)
            ttnn.deallocate(tokens)
            return transformer.prepare_plan(layout, sample_ids, sample=index), embedded

        samples = [sample(0, *text)]
        if guidance is not None:
            samples.append(sample(1, *self._null_text))
        image_axes = [None, self._sp_axis, None]
        rows, rows_padded = noise.shape[0], ttnn.core.roundup(noise.shape[0], sp * TILE)
        x = tensor.from_torch(
            pad_rows(noise, rows_padded)[None], device=self._device, dtype=ttnn.float32, mesh_axes=image_axes
        )
        mask = torch.zeros(1, rows_padded, IN_CHANNELS["image"])
        mask[:, :rows] = 1.0
        mask = tensor.from_torch(mask, device=self._device, dtype=ttnn.float32, mesh_axes=image_axes)
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
        for plan, embedded in samples:
            plan.deallocate()
            ttnn.deallocate(embedded)
        for t in (cond_embedded, mask, guidance_tt):
            if t is not None:
                ttnn.deallocate(t)
        return x, cached_steps

    def close(self) -> None:
        if self._upsampler is not None:
            self._upsampler.close()
            for t in self._upsampler_ranges:
                ttnn.deallocate(t)
        for module in (*self._transformers.values(), self._text_encoder):
            module.deallocate_weights()
        self._vae.deallocate_weights()


def _load_dit_state(snapshot: Path) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    state = load_file(str(Path(snapshot) / "model.safetensors"))
    return {k: v.to(torch.bfloat16) if v.is_floating_point() else v for k, v in state.items()}
