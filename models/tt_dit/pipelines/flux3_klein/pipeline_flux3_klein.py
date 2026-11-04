# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from loguru import logger
from PIL import Image, ImageOps

import ttnn

from ...encoders.qwen3vl.loader_flux3_klein import build_flux3_klein_text_encoder
from ...models.transformers.flux3_klein.transformer_flux3_klein import (
    EARLY_DEPTH,
    IN_CHANNELS,
    SINGLE_DEPTH,
    Flux3KleinSequenceLayout,
    Flux3KleinTransformer,
    padded_length,
    timestep_inputs,
)
from ...parallel.config import DiTGParallelConfigNoCFG, EncoderParallelConfig, ParallelFactor
from ...parallel.manager import CCLManager
from ...utils import cache, tensor
from ...utils.dbcache import DBCacheConfig
from .dbcache_flux3_klein import Flux3KleinDBCache
from .packing import MAX_TEXT_LENGTH, image_ids, latent_shape, pad_rows, reference_ids
from .prompt_encoder import Flux3KleinPromptEncoder
from .scheduler import GUIDANCE, STEPS, Flux3KleinScheduler
from .upsampler_flux3_klein import Flux3KleinUpsampler
from .vae_flux3_klein import Flux3KleinVae
from .weights_flux3_klein import resolve_weights_dir

# The first release serves fixed square sizes, all warmed up at startup, and at most one reference image,
# center-cropped and resized to the output size.
SIZES = (1024, 2048, 4096)
DEFAULT_SIZE = 2048
MAX_REFERENCES = 1


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
    prompt_upsampling: bool = True
    return_final_tokens: bool = False


@dataclass
class Flux3KleinResult:
    image: Image.Image
    caption: str
    seed: int
    timings: dict[str, float]
    cached_steps: int = 0
    final_tokens: torch.Tensor | None = None


def make_noise(seed: int, h: int, w: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn((h * w, IN_CHANNELS["image"]), generator=generator)
    return noise.to(torch.bfloat16).float()


def caption_seed(caption: str) -> int:
    return int(hashlib.sha256(caption.encode()).hexdigest(), 16) % 2**32


def prepare_reference(image: Image.Image, size: int) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    side = min(image.size)
    left, top = (image.width - side) // 2, (image.height - side) // 2
    image = image.crop((left, top, left + side, top + side))
    return image if side == size else image.resize((size, size), Image.Resampling.LANCZOS)


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
        upsampler: Flux3KleinUpsampler | None = None,
    ) -> None:
        self._device = mesh_device
        self._upsampler = upsampler
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
                model_name=f"flux3_klein_{model}_{Path(snapshot).name}",
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
        )
        # the base model's unconditional sample keeps the reference's own text length
        self._null_text = self._prompt_encoder.encode("") if "base" in snapshots else None

    @classmethod
    def create(
        cls, mesh_device: ttnn.MeshDevice, *, models: tuple[str, ...] = ("distilled",), upsample: bool = True, **kwargs
    ):
        # the upsampler runs on one chip of the mesh, next to the DiT
        upsampler = Flux3KleinUpsampler(mesh_device.create_submesh(ttnn.MeshShape(1, 1))) if upsample else None
        snapshots = {model: resolve_weights_dir(model) for model in models}
        return cls(mesh_device, snapshots=snapshots, upsampler=upsampler, **kwargs)

    def warmup(self, sizes: tuple[int, ...] = SIZES) -> None:
        """Compile every program of the served shapes: each size with and without a reference."""
        for model in self._transformers:
            for size in sizes:
                for references in range(MAX_REFERENCES + 1):
                    image = Image.new("RGB", (size, size))
                    self.generate(
                        Flux3KleinRequest(
                            "warmup",
                            model=model,
                            width=size,
                            height=size,
                            seed=0,
                            images=(image,) * references,
                            prompt_upsampling=False,
                        )
                    )

    def generate(self, request: Flux3KleinRequest) -> Flux3KleinResult:
        if request.model not in self._transformers:
            raise ValueError(f"model {request.model!r} is not loaded ({tuple(self._transformers)})")
        size = request.width
        if request.height != size or size not in SIZES:
            raise ValueError(f"{request.width}x{request.height} is not one of the served sizes {SIZES} (square)")
        if len(request.images) > MAX_REFERENCES:
            raise ValueError(f"at most {MAX_REFERENCES} reference image")
        steps = request.steps or STEPS[request.model]
        guidance = GUIDANCE[request.model] if request.guidance is None else request.guidance
        if request.model == "distilled" and guidance not in (None, 1.0):
            raise ValueError("the distilled model takes no guidance")
        guidance = None if request.model == "distilled" else guidance
        timings = {}
        reference = None
        if request.images:
            reference = request.images[0]
            if not isinstance(reference, Image.Image):
                reference = Image.open(reference)
            reference = prepare_reference(reference, size)

        t0 = time.time()
        caption = request.prompt
        if request.prompt_upsampling and self._upsampler is not None:
            caption = self._upsampler(request.prompt, reference)
        timings["upsample"] = time.time() - t0
        seed = caption_seed(caption) if request.seed is None else request.seed

        t0 = time.time()
        text = self._prompt_encoder.encode(caption, self._text_length)
        timings["encode"] = time.time() - t0

        t0 = time.time()
        image_cond = None
        if reference is not None:
            h, w = latent_shape(*reference.size)
            image_cond = (self._vae.encode(reference), reference_ids(h, w, 0))
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
        final_tokens = None
        if request.return_final_tokens:
            final_tokens = tensor.to_torch(state, mesh_axes=[None, self._sp_axis, None])[0][: h * w].clone()
        ttnn.deallocate(state)
        timings["to_host"] = time.time() - t0
        logger.info(f"FLUX 3 Klein {request.model} {size}x{size} refs {len(request.images)}: {timings}")
        return Flux3KleinResult(
            image=image,
            caption=caption,
            seed=seed,
            timings=timings,
            cached_steps=cached_steps,
            final_tokens=final_tokens,
        )

    def _upload_rows(self, rows: torch.Tensor, sp_factor: int) -> ttnn.Tensor:
        rows = pad_rows(rows.to(torch.bfloat16), padded_length(rows.shape[0], sp_factor))
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
        cond_rows = 0 if image_cond is None else image_cond[0].shape[0]
        cond_embedded = None
        if image_cond is not None:
            tokens = self._upload_rows(image_cond[0], sp)
            cond_embedded = transformer.embed_image_cond(tokens)
            ttnn.deallocate(tokens)

        def sample(text_rows: torch.Tensor, text_ids: torch.Tensor, text_valid: int):
            lengths = {"image": noise.shape[0], "image_cond": cond_rows, "text": text_rows.shape[0]}
            layout = Flux3KleinSequenceLayout(sp, lengths, text_valid)
            sample_ids = {"image": ids, "text": text_ids}
            if image_cond is not None:
                sample_ids["image_cond"] = image_cond[1]
            tokens = self._upload_rows(text_rows, sp)
            embedded = transformer.embed_text(tokens)
            ttnn.deallocate(tokens)
            return transformer.prepare_plan(layout, sample_ids), embedded

        samples = [sample(*text)]
        if guidance is not None:
            samples.append(sample(*self._null_text))
        image_axes = [None, self._sp_axis, None]
        rows, rows_padded = noise.shape[0], padded_length(noise.shape[0], sp)
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
        for module in (*self._transformers.values(), self._text_encoder):
            module.deallocate_weights()
        self._vae.deallocate_weights()
        if self._upsampler is not None:
            self._upsampler.close()


def _load_dit_state(snapshot: Path) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    state = load_file(str(Path(snapshot) / "model.safetensors"))
    return {k: v.to(torch.bfloat16) if v.is_floating_point() else v for k, v in state.items()}
