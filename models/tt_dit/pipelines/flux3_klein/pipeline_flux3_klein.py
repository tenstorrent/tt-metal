# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""FLUX 3 Image Klein text-to-image on one tt_dit mesh: ``reference/flux3_klein/pipeline.py::Pipeline.generate``
with every model stage on device (design record ``docs/M6_M7_DESIGN.md`` section 5).

Stages per request (``pipeline.py:55-83``), host work limited to noise, ids, the request schema and the PNG:
  1. :meth:`Flux3KleinRequest.resolve` (the ``generate.py:36-42`` defaults and checks) and ``policy.admit`` (host,
     before any device work; a rejection raises :class:`Flux3KleinAdmissionError`).
  2. ``h, w = latent_shape(height, width)``; ``noise = make_noise(seed, h, w)`` = ``pipeline.py:69-70`` (CPU
     generator, ``randn`` rounded to bf16, fp32 state; hashes to the goldens' ``noise_sha256``); ``image_ids(h, w)``.
  3. encode (device): ``Flux3KleinPromptEncoder.encode(prompt)`` on the co-resident Qwen3-VL encoder; the base model
     also takes ``encode('')`` (computed once per pipeline and kept on the host). The rows go to the host
     (``to_host``) and into the denoise request: the values are bf16 either way, so a device handoff is a later
     zero-numerics optimisation (M10).
  4. denoise (traced): ``Flux3KleinDenoiser.prepare`` + :meth:`Flux3KleinDenoiser.denoise_on_device` on the resident
     transformer of ``model`` (M4; with recorded text this is bit-identical to the M4 traced run).
  5. decode (device, traced): :class:`Flux3KleinVaeDecode` (M6) on the fp32 state slot.
  6. host: ``to_torch(out)[0].permute(2, 0, 1)`` and the reference ``to_pil`` (``pipeline.py:208-210``, truncation);
     ``final_tokens`` is read AFTER the decode so no host read sits between replays.

Residency (decision 8 of the design): BOTH DiTs (1.5 GB per chip each), the text encoder (TP8 x FSDP4, 0.36 GB per
chip) and the VAE decoder (0.1 GB) live on the 4x8 ring from ``__init__``; 34 GB per chip. Traces: one per denoiser
(``max_live_traces=1``, base 25.6 MB + distilled 13.0 MB) plus the VAE's 2.3 MB of the 64 MB region.

Trace aliasing rule (the M4 contract, ``denoise_flux3_klein.py`` docstring): a buffer allocated while a trace is
live may sit where that trace's replay writes. Each denoiser releases ITS traces before allocating a new layout's
slots; this pipeline extends the rule across modules: before a ``prepare`` that allocates (``has_layout`` False) and
before building the fallback VAE decoder, EVERY live trace (both denoisers and the VAE) is released and recaptured on
its next use (base capture 4.2 s, distilled 2.0 s, VAE 1.0 s). The text encoder's persistent CCL buffers and the VAE's
kernels are warmed in ``__init__`` for the same reason.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import torch
from loguru import logger

import ttnn

from ...encoders.qwen3vl.loader_flux3_klein import build_flux3_klein_text_encoder
from ...models.transformers.flux3_klein.transformer_flux3_klein import Flux3KleinTransformer
from ...parallel.config import DiTGParallelConfigNoCFG, EncoderParallelConfig, ParallelFactor
from ...parallel.manager import CCLManager
from ...utils import cache, tensor
from .denoise_flux3_klein import Flux3KleinDenoiser, Flux3KleinDenoiseRequest, Flux3KleinPreparedRequest
from .packing import FLUX3_KLEIN_IN_CHANNELS, image_ids, latent_shape, padded_length
from .policy import Admission, Flux3KleinAdmissionError, admit
from .prompt_encoder import Flux3KleinPromptEncoder
from .scheduler import FLUX3_KLEIN_ALPHA, FLUX3_KLEIN_GUIDANCE, FLUX3_KLEIN_STEPS
from .vae_flux3_klein import Flux3KleinVaeDecode, build_flux3_klein_vae_decoder, load_flux3_klein_vae_weights
from .weights_flux3_klein import dit_checkpoint, resolve_weights_dir

# reference pipeline.py:23-26 (the weights entry is resolved by weights_flux3_klein).
MODELS = {
    "base": {"steps": FLUX3_KLEIN_STEPS["base"], "guidance": FLUX3_KLEIN_GUIDANCE["base"]},
    "distilled": {"steps": FLUX3_KLEIN_STEPS["distilled"], "guidance": FLUX3_KLEIN_GUIDANCE["distilled"]},
}
OUTPUT_FORMATS = ("png", "jpeg")
TextPair = tuple[torch.Tensor, torch.Tensor]


# --- host helpers (the reference's formulas) ---------------------------------------------------------------------------


def make_noise(seed: int, h: int, w: int) -> torch.Tensor:
    """``pipeline.py:69-70``: CPU generator, ``randn((h*w, 128))``, rounded to bf16, held in fp32."""
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn((h * w, FLUX3_KLEIN_IN_CHANNELS["image"]), generator=generator)
    return noise.to(torch.bfloat16).float()


def noise_sha256(noise: torch.Tensor) -> str:
    return hashlib.sha256(noise.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def to_pil(image: torch.Tensor):
    """``pipeline.py:208-210`` verbatim: ``[3, H, W]`` in [-1, 1] -> PIL by ``(clamp + 1) * 127.5`` TRUNCATED to uint8."""
    from PIL import Image

    array = ((image.float().clamp(-1, 1) + 1) * 127.5).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(array)


# --- request / result ------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Flux3KleinRequest:
    """One text-to-image request with BFL's field names (``generate.py`` / the API schema). ``images`` is accepted
    for schema parity and rejected by ``policy.admit`` until M8. ``text_override`` (tests only) replaces the device
    text encoder with recorded encodings: ``((tokens, ids), (null_tokens, null_ids) | None)``."""

    prompt: str
    model: str = "base"
    width: int = 1024
    height: int = 1024
    seed: int = 0
    steps: int | None = None
    guidance: float | None = None
    images: tuple = ()
    alpha: float = FLUX3_KLEIN_ALPHA
    output_format: str = "png"
    text_override: tuple | None = None

    def resolve(self) -> Flux3KleinResolved:
        """``generate.py:36-42``: ``steps = steps or defaults``; ``guidance = defaults if None``; a flow-map model
        accepts only ``None`` / ``1.0``. Raises ``ValueError``."""
        if self.model not in MODELS:
            raise ValueError(f"unknown model {self.model!r}; expected one of {tuple(MODELS)}")
        defaults = MODELS[self.model]
        flow_map = defaults["guidance"] is None
        if flow_map and self.guidance not in (None, 1.0):
            raise ValueError(f"{self.model} is a flow map: guidance must be None or 1.0, got {self.guidance}")
        steps = self.steps or defaults["steps"]
        if steps < 1:
            raise ValueError(f"steps must be at least 1, got {steps}")
        guidance = defaults["guidance"] if self.guidance is None else float(self.guidance)
        if self.output_format not in OUTPUT_FORMATS:
            raise ValueError(f"output_format {self.output_format!r} not in {OUTPUT_FORMATS}")
        if self.text_override is not None:
            text, null = self.text_override
            if (null is None) != flow_map:
                raise ValueError(
                    "text_override needs the empty-prompt encoding for the base model and none for a flow map"
                )
        return Flux3KleinResolved(
            prompt=self.prompt,
            model=self.model,
            flow_map=flow_map,
            width=int(self.width),
            height=int(self.height),
            seed=int(self.seed),
            steps=int(steps),
            guidance=None if flow_map else guidance,
            images=tuple(self.images),
            alpha=float(self.alpha),
            output_format=self.output_format,
            text_override=self.text_override,
        )


@dataclass(frozen=True)
class Flux3KleinResolved:
    prompt: str
    model: str
    flow_map: bool
    width: int
    height: int
    seed: int
    steps: int
    guidance: float | None
    images: tuple
    alpha: float
    output_format: str
    text_override: tuple | None = None
    text_length: int | None = None
    """The padded text length (a multiple of 80) once the tokenizer has seen the prompt; ``None`` before."""


@dataclass
class Flux3KleinResult:
    image: object
    """PIL image ``(width, height)`` RGB."""
    final_tokens: torch.Tensor
    """fp32 ``[h*w, 128]``, the state after the last step (host read after the decode)."""
    timings: dict[str, float | bool]
    layout_key: tuple
    model: str
    seed: int
    steps: int
    guidance: float | None
    admission: Admission = field(default_factory=lambda: Admission(True))
    text_length: int | None = None
    text: TextPair | None = None
    """The ``((L, 20480) bf16, ids)`` encoding the denoise consumed (the device encoder's output, or the override):
    the input a CPU reference run of the same request must take to be the device trajectory's proper reference."""


# --- weights ---------------------------------------------------------------------------------------------------------------


def transformer_cache_key(variant: str, snapshot: Path, model: Flux3KleinTransformer) -> tuple[str, str]:
    """The M3 convention (``tests/models/flux3_klein/test_transformer_flux3_klein.py::load_real_weights``)."""
    model_name = f"flux3_klein_{variant}_{snapshot.name}"
    subfolder = f"transformer_D{model.depth}_S{model.depth_single}{'_flowmap' if model.flow_map else ''}"
    return model_name, subfolder


def load_dit_torch_state(snapshot: Path) -> dict[str, torch.Tensor]:
    """``model.safetensors`` (BFL layout) as the reference ``load_checkpoint`` reads it (floats to bf16)."""
    from safetensors.torch import load_file

    state = load_file(str(dit_checkpoint(snapshot)))
    return {k: v.to(torch.bfloat16) if v.is_floating_point() else v for k, v in state.items()}


def load_flux3_klein_transformer_weights(
    model: Flux3KleinTransformer,
    variant: str,
    snapshot: Path,
    *,
    parallel_config: DiTGParallelConfigNoCFG,
    mesh_device: ttnn.MeshDevice,
) -> str:
    model_name, subfolder = transformer_cache_key(variant, snapshot, model)
    cache_dir = (
        cache.model_cache_dir(
            model_name=model_name,
            subfolder=subfolder,
            parallel_config=parallel_config,
            mesh_shape=tuple(mesh_device.shape),
            mesh_device=mesh_device,
            required=False,
        )
        if cache.cache_dir_is_set()
        else None
    )
    state = (
        "HIT"
        if cache_dir is not None and (Path(cache_dir) / cache.CACHE_DICT_FILE).is_file()
        else ("MISS" if cache_dir is not None else "NO CACHE DIR")
    )
    t0 = time.time()
    cache.load_model(
        model,
        model_name=model_name,
        subfolder=subfolder,
        parallel_config=parallel_config,
        mesh_shape=tuple(mesh_device.shape),
        mesh_device=mesh_device,
        get_torch_state_dict=lambda: load_dit_torch_state(snapshot),
    )
    ttnn.synchronize_device(mesh_device)
    logger.info(f"RECORD transformer[{variant}] weights: cache {state} ({cache_dir}) loaded in {time.time() - t0:.1f}s")
    return state


def dram_per_bank(device: ttnn.MeshDevice) -> str:
    view = ttnn.get_memory_view(device, ttnn.BufferType.DRAM)
    trace = ttnn.get_memory_view(device, ttnn.BufferType.TRACE)
    return (
        f"DRAM {view.total_bytes_allocated_per_bank / 1e6:.1f} MB per bank x {view.num_banks} banks = "
        f"{view.total_bytes_allocated_per_bank * view.num_banks / 1e9:.2f} GB per chip; "
        f"TRACE {trace.total_bytes_allocated_per_bank} B per bank"
    )


# --- the pipeline --------------------------------------------------------------------------------------------------------------


class Flux3KleinPipeline:
    """Module docstring. ``snapshots`` maps each resident variant to its checkpoint directory; the text encoder and
    the VAE are read from the first resident variant's snapshot (both snapshots link the same blobs)."""

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        *,
        snapshots: dict[str, Path],
        resident: tuple[str, ...] = ("base", "distilled"),
        sp_axis: int = 0,
        tp_axis: int = 1,
        topology=ttnn.Topology.Ring,
        num_links: int = 2,
        vae_tp_axis: int | None = 0,
        vae_h_axis: int | None = 1,
        text_encoder_fsdp: bool = True,
        max_live_traces: int = 1,
        warmup_size: tuple[int, int] | None = (1024, 1024),
    ) -> None:
        if not resident:
            raise ValueError("at least one resident variant is required")
        for variant in resident:
            if variant not in MODELS:
                raise ValueError(f"unknown variant {variant!r}")
            if variant not in snapshots:
                raise ValueError(f"no snapshot for resident variant {variant!r}")
        self._device = mesh_device
        self._shape = tuple(int(s) for s in mesh_device.shape)
        self._sp_axis, self._tp_axis = sp_axis, tp_axis
        self._snapshots = {v: Path(snapshots[v]) for v in resident}
        self._resident = tuple(resident)
        self.timings_init: dict[str, float] = {}
        t_init = time.time()

        # DiTs (one CCL manager, both resident).
        self._dit_ccl = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)
        self._dit_parallel = DiTGParallelConfigNoCFG(
            tensor_parallel=ParallelFactor(factor=self._shape[tp_axis], mesh_axis=tp_axis),
            sequence_parallel=ParallelFactor(factor=self._shape[sp_axis], mesh_axis=sp_axis),
        )
        self._transformers: dict[str, Flux3KleinTransformer] = {}
        self._denoisers: dict[str, Flux3KleinDenoiser] = {}
        for variant in self._resident:
            t0 = time.time()
            model = Flux3KleinTransformer(
                flow_map=variant == "distilled",
                device=mesh_device,
                ccl_manager=self._dit_ccl,
                parallel_config=self._dit_parallel,
            )
            load_flux3_klein_transformer_weights(
                model, variant, self._snapshots[variant], parallel_config=self._dit_parallel, mesh_device=mesh_device
            )
            self._transformers[variant] = model
            self._denoisers[variant] = Flux3KleinDenoiser(
                model, device=mesh_device, sp_axis=sp_axis, max_live_traces=max_live_traces
            )
            self.timings_init[f"load_{variant}_s"] = time.time() - t0
            logger.info(f"RECORD after transformer[{variant}]: {dram_per_bank(mesh_device)}")

        # Text encoder (co-resident, TP over the DiT's TP axis, FSDP over the other).
        shared = self._snapshots[self._resident[0]]
        t0 = time.time()
        self._enc_ccl = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)
        self._enc_parallel = EncoderParallelConfig(
            tensor_parallel=ParallelFactor(factor=self._shape[tp_axis], mesh_axis=tp_axis)
        )
        encoder, text_config = build_flux3_klein_text_encoder(
            shared,
            mesh_device=mesh_device,
            parallel_config=self._enc_parallel,
            ccl_manager=self._enc_ccl,
            is_fsdp=text_encoder_fsdp,
            load_weights=True,
        )
        self._encoder = encoder
        self._prompt_encoder = Flux3KleinPromptEncoder(
            encoder, shared, mesh_device, text_config=text_config, output_sp_axis=sp_axis
        )
        self.timings_init["load_text_encoder_s"] = time.time() - t0
        logger.info(f"RECORD after text encoder: {dram_per_bank(mesh_device)}")

        # VAE decoder (its own CCL manager, Flux2's 4x8 row: channels TP over axis 0, latent rows H over axis 1).
        t0 = time.time()
        self._vae_ccl = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)
        decoder, self._vae_parallel = build_flux3_klein_vae_decoder(
            mesh_device, tp_axis=vae_tp_axis, h_axis=vae_h_axis, ccl_manager=self._vae_ccl
        )
        load_flux3_klein_vae_weights(decoder, shared, parallel_config=self._vae_parallel, mesh_device=mesh_device)
        self._vae = Flux3KleinVaeDecode(
            decoder,
            device=mesh_device,
            dit_ccl=self._dit_ccl,
            sp_axis=sp_axis,
            h_axis=vae_h_axis,
            tp_axis=vae_tp_axis,
            vae_ccl=self._vae_ccl,
            snapshot=shared,
        )
        self.timings_init["load_vae_s"] = time.time() - t0
        logger.info(f"RECORD after vae: {dram_per_bank(mesh_device)}")

        # Warm-ups BEFORE any trace exists: the empty prompt (base CFG; also allocates the encoder's persistent CCL
        # buffers) and one eager VAE decode of a zero state (kernel compile, ~95 s on a cold TT_METAL_CACHE).
        t0 = time.time()
        self._null_text: TextPair = self._prompt_encoder.encode_host("")
        self.timings_init["encode_null_s"] = time.time() - t0
        if warmup_size is not None:
            t0 = time.time()
            self._warm_vae(*warmup_size)
            self.timings_init["vae_warmup_s"] = time.time() - t0
        ttnn.synchronize_device(mesh_device)
        self.timings_init["total_s"] = time.time() - t_init
        logger.info(f"RECORD pipeline ready in {self.timings_init['total_s']:.1f}s: {dram_per_bank(mesh_device)}")

    @classmethod
    def create(cls, mesh_device: ttnn.MeshDevice, *, resident: tuple[str, ...] = ("base", "distilled"), **kw):
        """Snapshots from ``weights_flux3_klein.resolve_weights_dir`` per resident variant."""
        snapshots = {variant: resolve_weights_dir(variant) for variant in resident}
        return cls(mesh_device, snapshots=snapshots, resident=resident, **kw)

    # --- accessors -------------------------------------------------------------------------------------------------------

    @property
    def device(self) -> ttnn.MeshDevice:
        return self._device

    @property
    def mesh_shape(self) -> tuple[int, int]:
        return self._shape

    @property
    def resident(self) -> tuple[str, ...]:
        return self._resident

    @property
    def denoisers(self) -> dict[str, Flux3KleinDenoiser]:
        return self._denoisers

    @property
    def transformers(self) -> dict[str, Flux3KleinTransformer]:
        return self._transformers

    @property
    def prompt_encoder(self) -> Flux3KleinPromptEncoder:
        return self._prompt_encoder

    @property
    def vae(self) -> Flux3KleinVaeDecode:
        return self._vae

    @property
    def null_text(self) -> TextPair:
        """The empty prompt's encoding ``((80, 20480) bf16, ids)``, computed once."""
        return self._null_text

    # --- internals -------------------------------------------------------------------------------------------------------

    def _warm_vae(self, height: int, width: int) -> None:
        h, w = latent_shape(height, width)
        rows = padded_length(h * w, self._shape[self._sp_axis])
        zero = tensor.from_torch(
            torch.zeros(1, rows, FLUX3_KLEIN_IN_CHANNELS["image"]),
            device=self._device,
            dtype=ttnn.float32,
            mesh_axes=[None, self._sp_axis, None],
        )
        out = self._vae.decode(zero, num_image_rows=h * w, height=height, width=width, traced=False)
        ttnn.synchronize_device(self._device)
        ttnn.deallocate(out)
        ttnn.deallocate(zero)

    def release_traces(self) -> None:
        """Release every live trace (both denoisers and the VAE); each is recaptured on its next traced use."""
        for denoiser in self._denoisers.values():
            denoiser.release_traces()
        self._vae.release_trace()

    def _encode(self, prompt: str) -> TextPair:
        return self._prompt_encoder.encode_host(prompt)

    def text_length(self, prompt: str) -> int:
        return self._prompt_encoder.tokenize(prompt)[2]

    # --- the request -----------------------------------------------------------------------------------------------------

    def generate(self, request: Flux3KleinRequest, *, traced: bool = True) -> Flux3KleinResult:
        t_total = time.time()
        resolved = request.resolve()
        if resolved.model not in self._denoisers:
            raise ValueError(f"model {resolved.model!r} is not resident (resident: {self._resident})")
        if resolved.text_override is not None:
            text_length = int(resolved.text_override[0][0].shape[0])
        else:
            text_length = self.text_length(resolved.prompt)
        resolved = replace(resolved, text_length=text_length)
        admission = admit(resolved, self._shape)
        if not admission.ok:
            raise Flux3KleinAdmissionError(admission)

        h, w = latent_shape(resolved.height, resolved.width)
        noise = make_noise(resolved.seed, h, w)
        ids = image_ids(h, w)

        # 3. encode
        t0 = time.time()
        if resolved.text_override is not None:
            text, null = resolved.text_override
        else:
            text = self._encode(resolved.prompt)
            null = None if resolved.flow_map else self._null_text
        encode_s = time.time() - t0

        denoise_request = Flux3KleinDenoiseRequest(
            noise=noise,
            image_ids=ids,
            text=text,
            num_steps=resolved.steps,
            guidance=resolved.guidance,
            text_null=None if resolved.flow_map else null,
            alpha=resolved.alpha,
        )
        denoiser = self._denoisers[resolved.model]

        # 4. denoise (the aliasing rule of the module docstring before any long-lived allocation)
        needs_fallback_vae = not self._vae.partitioned(resolved.height, resolved.width) and self._vae._fallback is None
        if not denoiser.has_layout(denoise_request) or needs_fallback_vae:
            self.release_traces()
        t0 = time.time()
        prepared: Flux3KleinPreparedRequest = denoiser.prepare(denoise_request)
        prepare_s = time.time() - t0
        captured_before = prepared.key in denoiser.captured_keys
        t0 = time.time()
        state = denoiser.denoise_on_device(prepared, traced=traced)
        denoise_s = time.time() - t0

        # 5. decode
        vae_captured_before = self._vae.trace_captured()
        t0 = time.time()
        out = self._vae.decode(state, num_image_rows=h * w, height=resolved.height, width=resolved.width, traced=traced)
        ttnn.synchronize_device(self._device)
        decode_s = time.time() - t0

        # 6. host
        t0 = time.time()
        img = tensor.to_torch(out)[0].permute(2, 0, 1).float()
        if not traced:
            ttnn.deallocate(out)
        image = to_pil(img)
        final_tokens = denoiser.read_state(prepared)
        to_host_s = time.time() - t0

        timings = {
            "encode_s": encode_s,
            "prepare_s": prepare_s,
            "denoise_s": denoise_s,
            "decode_s": decode_s,
            "to_host_s": to_host_s,
            "total_s": time.time() - t_total,
            "trace_captured": bool(traced and not captured_before),
            "vae_trace_captured": bool(traced and not vae_captured_before),
        }
        logger.info(
            f"RECORD generate[{resolved.model}] {resolved.width}x{resolved.height} seed {resolved.seed} steps "
            f"{resolved.steps} guidance {resolved.guidance} text L {text_length}: encode {encode_s:.3f}s prepare "
            f"{prepare_s:.3f}s denoise {denoise_s:.3f}s decode {decode_s:.3f}s to_host {to_host_s:.3f}s total "
            f"{timings['total_s']:.3f}s (denoise trace captured {timings['trace_captured']}, vae trace captured "
            f"{timings['vae_trace_captured']}); {dram_per_bank(self._device)}"
        )
        return Flux3KleinResult(
            image=image,
            final_tokens=final_tokens,
            timings=timings,
            layout_key=prepared.key,
            model=resolved.model,
            seed=resolved.seed,
            steps=resolved.steps,
            guidance=resolved.guidance,
            admission=admission,
            text_length=text_length,
            text=text,
        )

    def close(self) -> None:
        """Traces, then slots, then weights; the caller closes the mesh."""
        self.release_traces()
        for denoiser in self._denoisers.values():
            denoiser.deallocate()
        ttnn.synchronize_device(self._device)
        for module in (*self._transformers.values(), self._encoder, self._vae.decoder, self._vae._fallback):
            if module is not None and hasattr(module, "deallocate_weights"):
                module.deallocate_weights()
        ttnn.synchronize_device(self._device)


__all__ = [
    "MODELS",
    "Flux3KleinAdmissionError",
    "Flux3KleinPipeline",
    "Flux3KleinRequest",
    "Flux3KleinResolved",
    "Flux3KleinResult",
    "load_flux3_klein_transformer_weights",
    "make_noise",
    "noise_sha256",
    "to_pil",
    "transformer_cache_key",
]
