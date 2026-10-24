# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""FLUX.2 Klein text-to-image and reference editing on one tt_dit mesh: the diffusers ``Flux2KleinPipeline.__call__``
(``pipeline_flux2_klein.py``, cited as ``P``; BFL ``sampling.py`` ``SA`` as the second citation) with every model
stage on device. The structure is the FLUX 3 Klein pipeline (``pipelines/flux3_klein/pipeline_flux3_klein.py``) with
everything FLUX-3-specific removed.

Stages per request (``P:700-925``), host work limited to the request schema, the reference sizing, noise, ids and
the PNG:
  1. :meth:`Flux2KleinRequest.resolve`: the defaults (``P:622-624`` / BFL ``util.py``: distilled 4 steps and no
     guidance, base 50 steps and CFG 4.0), the checks, then the reference sizing of ``references.prepare_references``
     (``P:764-787``: a 1 MP cap, the 16-px floor, the centre crop; the canvas defaults to the FIRST prepared
     reference's size, else 1024 x 1024; a path source is served from the pipeline's ``references.ReferenceCache``,
     keyed by path + mtime + size, so a reference seen before costs a stat instead of the PNG decode and the fp32
     conversion, ``timings["resolve_s"]``) and ``policy.admit`` with the SP factor of the layout the request renders on
     (host, before any device work; a rejection, including a canvas whose per-device spatial rows equal the prompt
     stream's, raises :class:`Flux2KleinAdmissionError` and nothing has been allocated on the device).
  2. ``h, w = latent_shape(height, width)``; ``noise = make_noise(seed, h, w)`` (``P:496-510``: ``randn_tensor`` in
     bf16 on a CPU generator, packed) drawn on the pipeline's one worker thread while steps 3-4 run on the device
     (20-60 ms of host time at 1K / 2K; the explicit ``Generator`` makes it deterministic on any thread) and
     memoised per ``(seed, h, w)`` (``packing.make_noise_cached``; ``noise_cache=False`` draws every time);
     ``latent_ids(h, w)`` (``P:306-313``).
  3. encode text (device): the co-resident Qwen3 encoder through ``prompt_encoder.Flux2KleinTextEncoding.
     encode_device`` (``P:208-263``: chat template, 512 tokens, hidden states 9 / 18 / 27 concatenated); the rows
     stay on the device, ``ttnn.mesh_partition`` ed over the layout's SP axis into the placement the denoiser's
     hoisted ``embed_prompt`` reads (no host round trip; ``Flux2KleinRequest.return_text`` adds the one-replica host
     read for ``Flux2KleinResult.text``). The base model also takes the empty prompt's encoding (``P:749-760``),
     computed once per layout at start-up (host rows, re-uploaded per request).
  4. encode references (device): :class:`Flux2KleinVaeEncode` per prepared reference (``P:514-540``: the posterior
     mean, patchify, BatchNorm, pack) -> bf16 ``[h_i * w_i, 128]`` tokens on the host with ``reference_ids`` into
     ``Flux2KleinDenoiseRequest.image_cond``; shared by both CFG samples.
  5. denoise (traced): :meth:`Flux2KleinDenoiser.prepare` + :meth:`Flux2KleinDenoiser.denoise_on_device` on the
     resident transformer of ``model`` (``P:826-880``; the state slot holds ``[latents | references | pad]``).
  6. decode (device, traced): :class:`Flux2KleinVaeDecode` on the fp32 state slot (``P:905-921``: the image rows,
     the inverse BatchNorm, unpatchify, ``AutoencoderKLFlux2.decode``); a replay of the trace captured before the DiT
     trace (capture order below); a canvas whose latent height is not a multiple of the H partition takes the
     replicated-rows decoder.
  7. host: ``image_processor.postprocess`` (``P:922``) = ``(x * 0.5 + 0.5).clamp(0, 1)`` in the decoder's bf16,
     ``.float() * 255`` ROUNDED to uint8 (:func:`image_to_uint8_on_device`, one replica read); ``final_tokens`` is
     read AFTER the decode so no host read sits between replays.

Trace aliasing rule (``denoise_flux2_klein.py`` docstring): a buffer allocated while a trace is live may sit where
that trace's replay writes. Each denoiser releases ITS traces before allocating a new layout's slots; this pipeline
extends the rule across modules: before a ``prepare`` that allocates (``has_layout`` False), before building the
fallback VAE decoder, before the first reference encode of a size the encoder has not seen (its CCL manager
allocates persistent buffers per shape) and before a VAE capture of a size while a DiT trace is live, EVERY live
trace (every denoiser and the VAE) is released and recaptured on its next use. Capture ORDER per layout: the
denoiser's slots are allocated by ``prepare``, then the VAE decode trace of the canvas size is captured on a zero
state (``_warm_vae(traced=True)``: its captured input buffer and the VAE CCL manager's gather buffers are allocated
HERE), and only then the DiT trace is captured by the first step; the per-request decode is then a replay whose only
allocations are temporaries freed before the next DiT replay. The text encoder's persistent CCL buffers and the VAE
decoder's and encoder's kernels (and the encoder's prepared conv weights) are warmed in ``__init__`` for the same
reason.
"""

from __future__ import annotations

import os
import secrets
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path

import torch
from loguru import logger

import ttnn

from ...models.transformers.flux2_klein.block_flux2_klein import (
    FLUX2_KLEIN_DIT_MATH_FIDELITY_DEFAULT,
    FLUX2_KLEIN_SDPA_EXP_APPROX_DEFAULT,
)
from ...models.transformers.flux2_klein.config_flux2_klein import FLUX2_KLEIN_SIZES, Flux2KleinConfig
from ...models.transformers.flux2_klein.transformer_flux2_klein import Flux2KleinTransformer
from ...parallel.config import DiTGParallelConfigNoCFG, EncoderParallelConfig, ParallelFactor
from ...parallel.manager import CCLManager
from ...utils import tensor
from .denoise_flux2_klein import Flux2KleinDenoiser, Flux2KleinDenoiseRequest, Flux2KleinPreparedRequest
from .packing import (
    FLUX2_KLEIN_DOWNSAMPLING_RATIO,
    FLUX2_KLEIN_IN_CHANNELS,
    FLUX2_KLEIN_MAX_REFERENCES,
    FLUX2_KLEIN_TEXT_LENGTH,
    latent_ids,
    latent_shape,
    make_noise,
    make_noise_cached,
    padded_length,
    text_ids,
)
from .policy import Admission, Flux2KleinAdmissionError, admit
from .prompt_encoder import Flux2KleinPromptEncoder, Flux2KleinTextEncoding, build_flux2_klein_text_encoder
from .references import TOO_MANY_REFERENCES, PreparedReferences, ReferenceCache, image_to_uint8, prepare_references
from .scheduler import FLUX2_KLEIN_GUIDANCE, FLUX2_KLEIN_STEPS
from .vae_encode_flux2_klein import (
    Flux2KleinVaeEncode,
    build_flux2_klein_vae_encoder,
    load_flux2_klein_vae_encoder_weights,
)
from .vae_flux2_klein import Flux2KleinVaeDecode, build_flux2_klein_vae_decoder, load_flux2_klein_vae_weights
from .weights_dit_flux2_klein import load_flux2_klein_transformer_weights
from .weights_flux2_klein import resolve_weights_dir

# BFL util.py / README: the two released variants; the size ("4b" | "9b") is orthogonal.
MODELS = {
    "distilled": {"steps": FLUX2_KLEIN_STEPS["distilled"], "guidance": FLUX2_KLEIN_GUIDANCE["distilled"]},
    "base": {"steps": FLUX2_KLEIN_STEPS["base"], "guidance": FLUX2_KLEIN_GUIDANCE["base"]},
}
SIZES = tuple(FLUX2_KLEIN_SIZES)
OUTPUT_FORMATS = ("png", "jpeg")
TextPair = tuple[torch.Tensor, torch.Tensor]
CondPair = tuple[torch.Tensor, torch.Tensor]

SDPA_EXP_APPROX_ENV = "FLUX2_KLEIN_SDPA_EXP_APPROX"
TEXT_ENCODER_FIDELITY_ENV = "FLUX2_KLEIN_TEXT_ENCODER_FIDELITY"
DIT_FIDELITY_ENV = "FLUX2_KLEIN_DIT_FIDELITY"
TEXT_ENCODER_FSDP_ENV = "FLUX2_KLEIN_TEXT_ENCODER_FSDP"
RECORD_MEMORY_ENV = "FLUX2_KLEIN_RECORD_MEMORY"
# Whether the per-request RECORD line ends with :func:`dram_per_bank` (two ``ttnn.get_memory_view`` calls over the
# mesh, host time inside ``total_s`` after the stages). The init-time RECORD lines always carry it.
RECORD_MEMORY_DEFAULT = False
# Whether the text encoder's weights are sharded over the mesh axis its tensor parallelism does not use (FSDP: every
# linear all-gathers its weight shard on every forward, 5 collectives per layer) or replicated over it (no gathers;
# the 27 layers cost 1.4 GB per chip on TP4 instead of 0.17 GB). Same math either way: the gathered weight is the
# replicated weight, so the encodings are bit-equal.
TEXT_ENCODER_FSDP_DEFAULT = True
_FIDELITIES = {
    "lofi": ttnn.MathFidelity.LoFi,
    "hifi2": ttnn.MathFidelity.HiFi2,
    "hifi3": ttnn.MathFidelity.HiFi3,
    "hifi4": ttnn.MathFidelity.HiFi4,
}
# The text encoder's linear fidelity when the constructor is given nothing and the environment says nothing.
TEXT_ENCODER_MATH_FIDELITY_DEFAULT: ttnn.MathFidelity | None = None
# The DiT linears' matmul fidelity when the constructor is given nothing and the environment says nothing: None = the
# block module's default (``FLUX2_KLEIN_DIT_MATH_FIDELITY_DEFAULT``, HiFi4); ``hifi2`` is the opt-in (the
# layers/linear.py default FLUX 3's exact profile runs on). Only these two are admitted: both keep every mantissa bit
# of the bf16 weights' operand pairs that the exact rule allows (HiFi2 drops the low bits of ONE operand per product).
DIT_MATH_FIDELITY_DEFAULT: ttnn.MathFidelity | None = None
_DIT_FIDELITIES = {"hifi2": ttnn.MathFidelity.HiFi2, "hifi4": ttnn.MathFidelity.HiFi4}


def sdpa_exp_approx_from_env(default: bool | None = None) -> bool:
    """``FLUX2_KLEIN_SDPA_EXP_APPROX`` (``1`` | ``0``) over ``default`` (``None``: the port's default)."""
    raw = os.environ.get(SDPA_EXP_APPROX_ENV)
    if raw is not None and raw.strip() != "":
        value = raw.strip().lower()
        if value not in ("0", "1", "true", "false", "yes", "no"):
            raise ValueError(f"{SDPA_EXP_APPROX_ENV} must be 0 or 1, got {raw!r}")
        return value in ("1", "true", "yes")
    return FLUX2_KLEIN_SDPA_EXP_APPROX_DEFAULT if default is None else bool(default)


def text_encoder_fsdp_from_env(default: bool | None = None) -> bool:
    """``FLUX2_KLEIN_TEXT_ENCODER_FSDP`` (``1`` | ``0``) over ``default`` (``None``: :data:`TEXT_ENCODER_FSDP_DEFAULT`)."""
    raw = os.environ.get(TEXT_ENCODER_FSDP_ENV)
    if raw is not None and raw.strip() != "":
        value = raw.strip().lower()
        if value not in ("0", "1", "true", "false", "yes", "no"):
            raise ValueError(f"{TEXT_ENCODER_FSDP_ENV} must be 0 or 1, got {raw!r}")
        return value in ("1", "true", "yes")
    return TEXT_ENCODER_FSDP_DEFAULT if default is None else bool(default)


def record_memory_from_env(default: bool | None = None) -> bool:
    """``FLUX2_KLEIN_RECORD_MEMORY`` (``1`` | ``0``) over ``default`` (``None``: :data:`RECORD_MEMORY_DEFAULT`)."""
    raw = os.environ.get(RECORD_MEMORY_ENV)
    if raw is not None and raw.strip() != "":
        value = raw.strip().lower()
        if value not in ("0", "1", "true", "false", "yes", "no"):
            raise ValueError(f"{RECORD_MEMORY_ENV} must be 0 or 1, got {raw!r}")
        return value in ("1", "true", "yes")
    return RECORD_MEMORY_DEFAULT if default is None else bool(default)


def dit_math_fidelity_from_env(default: ttnn.MathFidelity | None = None) -> ttnn.MathFidelity | None:
    """``FLUX2_KLEIN_DIT_FIDELITY`` (``hifi4`` | ``hifi2`` | ``default``) over ``default`` (``None``:
    :data:`DIT_MATH_FIDELITY_DEFAULT`); ``None`` returned = the block module's default (HiFi4)."""
    raw = os.environ.get(DIT_FIDELITY_ENV)
    if raw is not None and raw.strip() != "":
        value = raw.strip().lower()
        if value == "default":
            return None
        if value not in _DIT_FIDELITIES:
            raise ValueError(f"{DIT_FIDELITY_ENV} must be one of {tuple(_DIT_FIDELITIES)} or default, got {raw!r}")
        return _DIT_FIDELITIES[value]
    if default is not None and default not in _DIT_FIDELITIES.values():
        raise ValueError(f"dit_math_fidelity must be one of {tuple(_DIT_FIDELITIES.values())}, got {default!r}")
    return DIT_MATH_FIDELITY_DEFAULT if default is None else default


def text_encoder_math_fidelity_from_env(default: ttnn.MathFidelity | None = None) -> ttnn.MathFidelity | None:
    """``FLUX2_KLEIN_TEXT_ENCODER_FIDELITY`` (``hifi2`` | ``hifi4`` | ``default``) over ``default``."""
    raw = os.environ.get(TEXT_ENCODER_FIDELITY_ENV)
    if raw is not None and raw.strip() != "":
        value = raw.strip().lower()
        if value == "default":
            return None
        if value not in _FIDELITIES:
            raise ValueError(f"{TEXT_ENCODER_FIDELITY_ENV} must be one of {tuple(_FIDELITIES)} or default, got {raw!r}")
        return _FIDELITIES[value]
    return TEXT_ENCODER_MATH_FIDELITY_DEFAULT if default is None else default


# --- host helpers (the reference's formulas) ---------------------------------------------------------------------------


def random_seed() -> int:
    """A fresh 32-bit seed for an unseeded request (the reference draws from an unseeded generator; the seed used is
    reported on the result so the render is reproducible)."""
    return secrets.randbits(32)


def to_pil(image: torch.Tensor):
    """``image_processor.postprocess(output_type="pil")`` (``P:922``) for one decoded ``[3, H, W]`` image in the
    decoder's dtype: ``references.image_to_uint8`` (denormalize in the input dtype, ``.float() * 255`` rounded)."""
    from PIL import Image

    return Image.fromarray(image_to_uint8(image).cpu().numpy())


def image_to_uint8_on_device(image: ttnn.Tensor) -> ttnn.Tensor:
    """:func:`to_pil`'s arithmetic ON THE DEVICE: bf16 ``[1, H, W, 3]`` -> uint8 ROW_MAJOR ``[1, H, W * 3]``.
    ``denormalize`` runs in the decoder's bf16 (``(x * 0.5 + 0.5).clamp(0, 1)``: the halving is exact, the add is
    rounded to bf16 by a ``typecast(bf16) -> typecast(fp32)`` round trip), ``pt_to_numpy`` widens to fp32 and
    ``numpy_to_pil`` takes ``round(x * 255)`` (``VaeImageProcessor.denormalize`` / ``pt_to_numpy`` / ``numpy_to_pil``),
    then a typecast that cannot round because the values are already integral.

    The bf16 image is untilized FIRST (dropping the TILE padding of the 3 channels to 32) and reshaped to
    ``[1, H, W * 3]`` (one image row per page) before the fp32 arithmetic, so the fp32 temporary is 200 MB per chip
    at 4096x4096 instead of the 2 GiB a typecast of the padded ``[1, H, W, 32]`` tensor takes; the readback then
    moves 1 byte per channel and needs no host untilize. ``image_rows_to_hwc`` restores ``[H, W, 3]`` on the host."""
    n, h, w, c = (int(d) for d in image.shape)
    x = ttnn.to_layout(image, ttnn.ROW_MAJOR_LAYOUT)
    x = ttnn.reshape(x, (n, h, w * c))
    x = ttnn.typecast(x, ttnn.float32)
    x = ttnn.multiply(x, 0.5)
    x = ttnn.add(x, 0.5)
    x = ttnn.typecast(ttnn.typecast(x, ttnn.bfloat16), ttnn.float32)
    x = ttnn.clamp(x, min=0.0, max=1.0)
    x = ttnn.multiply(x, 255.0)
    x = ttnn.round(x)
    return ttnn.typecast(x, ttnn.uint8)


def image_rows_to_hwc(rows: torch.Tensor, channels: int = 3) -> torch.Tensor:
    """``[H, W * C]`` (one replica of :func:`image_to_uint8_on_device`, batch dim dropped) -> ``[H, W, C]``."""
    h, wc = rows.shape[-2], rows.shape[-1]
    return rows.reshape(h, wc // channels, channels)


# --- request / result ------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Flux2KleinRequest:
    """One request. ``model`` is ``"distilled"`` (4 steps, no guidance) or ``"base"`` (50 steps, CFG 4.0); ``size``
    ``"4b"`` | ``"9b"``. ``width`` / ``height`` default to ``None`` = the first prepared reference's size, else 1024 x
    1024 (``P:782-787``). ``images`` holds up to ``FLUX2_KLEIN_MAX_REFERENCES`` references as ``PIL.Image.Image`` s
    or paths. ``seed`` ``None`` (the default) draws a fresh 32-bit seed, reported on the result. ``text_override``
    (tests only) replaces the device text encoder with recorded encodings: ``((tokens, ids), (null_tokens,
    null_ids) | None)``; ``image_cond_override`` (tests only) replaces the device VAE encoder with recorded
    ``(cond_tokens, cond_ids)`` whose row count must equal the prepared references' token count."""

    prompt: str
    model: str = "distilled"
    size: str = "4b"
    width: int | None = None
    height: int | None = None
    seed: int | None = None
    steps: int | None = None
    guidance: float | None = None
    images: tuple = ()
    output_format: str = "png"
    text_override: tuple | None = None
    image_cond_override: CondPair | None = None
    return_final_tokens: bool = True
    """Read the final latent state back to the host (``Flux2KleinResult.final_tokens``; the tests need it). A
    serving request does not: ``False`` leaves ``final_tokens`` None and the ``to_host`` stage shorter."""
    return_text: bool = True
    """Read the prompt encoding back to the host (``Flux2KleinResult.text``, the ``[512, joint]`` bf16 rows a CPU
    reference run of the request must take; the tests need it): one replica read before the rows are partitioned
    for the DiT. A serving request / the benchmark does not: ``False`` leaves ``text`` None (unless the request
    carried ``text_override``) and the ``encode`` stage shorter."""

    def resolve(self, *, reference_cache: ReferenceCache | None = None) -> Flux2KleinResolved:
        """``steps = steps or defaults``; ``guidance = defaults if None``; the distilled model accepts only ``None``
        / ``1.0`` (``P:587-588``: ``guidance_scale > 1`` on a distilled model is not CFG); at most 10 references.
        Then the references are prepared (sized) on the host, path sources through ``reference_cache`` when given
        (the pipeline passes its own), and the canvas is ``(width or default_w, height or default_h)``. Raises
        ``ValueError``."""
        if self.model not in MODELS:
            raise ValueError(f"unknown model {self.model!r}; expected one of {tuple(MODELS)}")
        if self.size not in SIZES:
            raise ValueError(f"unknown size {self.size!r}; expected one of {SIZES}")
        if len(self.images) > FLUX2_KLEIN_MAX_REFERENCES:
            raise ValueError(TOO_MANY_REFERENCES)
        defaults = MODELS[self.model]
        cfg = defaults["guidance"] is not None
        if not cfg and self.guidance not in (None, 1.0):
            raise ValueError(f"{self.model} is distilled: guidance must be None or 1.0, got {self.guidance}")
        steps = self.steps or defaults["steps"]
        if steps < 1:
            raise ValueError(f"steps must be at least 1, got {steps}")
        guidance = defaults["guidance"] if self.guidance is None else float(self.guidance)
        if self.output_format not in OUTPUT_FORMATS:
            raise ValueError(f"output_format {self.output_format!r} not in {OUTPUT_FORMATS}")
        if self.text_override is not None:
            _, null = self.text_override
            if (null is None) == cfg:
                raise ValueError(
                    "text_override needs the empty-prompt encoding for the base model and none for the distilled model"
                )
        # ``width or default_width``: 0 (and None) means "not given", as in the reference.
        for name, side in (("width", self.width), ("height", self.height)):
            if side is not None and int(side) < 0:
                raise ValueError(f"{name} must not be negative, got {side}")
        width_given = int(self.width) if self.width else None
        height_given = int(self.height) if self.height else None
        prepared = prepare_references(self.images, width=width_given, height=height_given, cache=reference_cache)
        default_w, default_h = prepared.default_size
        width = width_given if width_given else default_w
        height = height_given if height_given else default_h
        if self.image_cond_override is not None:
            tokens, ids = self.image_cond_override
            if tokens.shape[0] != prepared.cond_tokens() or ids.shape[0] != tokens.shape[0]:
                raise ValueError(
                    f"image_cond_override has {tokens.shape[0]} rows ({ids.shape[0]} ids); the prepared references "
                    f"{prepared.sizes} need {prepared.cond_tokens()}"
                )
        return Flux2KleinResolved(
            prompt=self.prompt,
            model=self.model,
            size=self.size,
            cfg=cfg,
            width=width,
            height=height,
            seed=None if self.seed is None else int(self.seed),
            steps=int(steps),
            guidance=guidance if cfg else None,
            images=tuple(self.images),
            output_format=self.output_format,
            text_override=self.text_override,
            image_cond_override=self.image_cond_override,
            prepared=prepared,
        )


@dataclass(frozen=True)
class Flux2KleinResolved:
    prompt: str
    model: str
    size: str
    cfg: bool
    """Classifier-free guidance (the base model: two forwards per step with the empty prompt, ``P:595-596``)."""
    width: int
    height: int
    seed: int | None
    """``None`` until ``generate()`` draws one (:func:`random_seed`)."""
    steps: int
    guidance: float | None
    images: tuple
    output_format: str
    text_override: tuple | None = None
    image_cond_override: CondPair | None = None
    prepared: PreparedReferences | None = None
    """The references after the host sizing (``None`` only for a resolved object built by hand)."""
    text_length: int = FLUX2_KLEIN_TEXT_LENGTH
    """The prompt stream's row count: always 512 (``P:204``, ``padding="max_length"``)."""


@dataclass
class Flux2KleinResult:
    image: object
    """PIL image ``(width, height)`` RGB."""
    final_tokens: torch.Tensor | None
    """fp32 ``[h*w, 128]`` (bf16-valued), the state after the last step (host read after the decode)."""
    timings: dict[str, float | bool]
    layout_key: tuple
    model: str
    size: str
    seed: int
    steps: int
    guidance: float | None
    admission: Admission = field(default_factory=lambda: Admission(True))
    text: TextPair | None = None
    """The ``([512, joint] bf16, [512, 4] int64)`` encoding the denoise consumed (the device encoder's output, or the
    override): the input a CPU reference run of the same request must take to be the device trajectory's proper
    reference. ``None`` when the request said ``return_text=False`` (the rows went to the DiT on the device and were
    never read)."""
    image_cond: CondPair | None = None
    """The ``([N, 128] bf16, [N, 4] int64)`` reference rows the denoise consumed (device encoder or override)."""
    references: tuple[tuple[int, int], ...] = ()
    """``(width, height)`` of every prepared reference."""
    input_prompt: str = ""
    """The prompt as typed."""
    layout: str = ""
    """The DiT mesh layout that rendered it (``"sp8tp4"`` / ``"sp4tp8"``, :func:`layout_name`)."""


# --- layouts -------------------------------------------------------------------------------------------------------------


def layout_name(sp_axis: int, tp_axis: int, shape: tuple[int, ...] = (4, 8)) -> str:
    """``"sp8tp4"`` for SP over axis 1 x TP over axis 0 of a 4x8 mesh; ``"sp4tp8"`` for the transposed assignment."""
    return f"sp{shape[sp_axis]}tp{shape[tp_axis]}"


LARGE_LAYOUT_MIN_TOKENS = 16384
"""Requests with at least this many DiT tokens (image + reference + text rows) take the large-canvas layout when one is
resident (the FLUX 3 Klein threshold, kept for layouts that may diverge: reference-heavy 4K, base CFG)."""


def pick_layout(
    total_tokens: int, *, primary: str, large: str | None, min_tokens: int = LARGE_LAYOUT_MIN_TOKENS
) -> str:
    """The layout name a request with ``total_tokens`` DiT rows renders on: ``large`` from ``min_tokens`` up (when it
    is resident), else ``primary``."""
    if large is not None and int(total_tokens) >= int(min_tokens):
        return large
    return primary


@dataclass
class _Layout:
    """One DiT mesh layout resident in the pipeline: its transformers and denoisers (per variant), the text encoding
    wrapper, the decode helper that gathers the state from its SP axis, and the empty prompt's encoding for the base
    CFG."""

    name: str
    sp_axis: int
    tp_axis: int
    ccl: CCLManager
    parallel: DiTGParallelConfigNoCFG
    transformers: dict[str, Flux2KleinTransformer]
    denoisers: dict[str, Flux2KleinDenoiser]
    prompt_encoder: Flux2KleinTextEncoding
    vae: Flux2KleinVaeDecode
    null_text: TextPair | None = None


def dram_per_bank(device: ttnn.MeshDevice) -> str:
    view = ttnn.get_memory_view(device, ttnn.BufferType.DRAM)
    trace = ttnn.get_memory_view(device, ttnn.BufferType.TRACE)
    return (
        f"DRAM {view.total_bytes_allocated_per_bank / 1e6:.1f} MB per bank x {view.num_banks} banks = "
        f"{view.total_bytes_allocated_per_bank * view.num_banks / 1e9:.2f} GB per chip; "
        f"TRACE {trace.total_bytes_allocated_per_bank} B per bank"
    )


# --- the pipeline --------------------------------------------------------------------------------------------------------------


class Flux2KleinPipeline:
    """Module docstring. ``snapshots`` maps each resident variant to its diffusers snapshot directory; the text
    encoder and the VAE are read from the first resident variant's snapshot (every Klein variant of a size ships the
    same Qwen3 encoder and the FLUX.2 autoencoder)."""

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        *,
        snapshots: dict[str, Path],
        size: str = "4b",
        resident: tuple[str, ...] = ("distilled",),
        sp_axis: int = 1,
        tp_axis: int = 0,
        topology=ttnn.Topology.Ring,
        num_links: int = 2,
        vae_tp_axis: int | None = 0,
        vae_h_axis: int | None = 1,
        text_encoder_fsdp: bool | None = None,
        max_live_traces: int = 1,
        warmup_size: tuple[int, int] | None = (1024, 1024),
        shard_prompt: bool = True,
        large_layout: tuple[int, int] | None = None,
        large_layout_min_tokens: int = LARGE_LAYOUT_MIN_TOKENS,
        sdpa_exp_approx: bool | None = None,
        text_encoder_math_fidelity: ttnn.MathFidelity | None = None,
        dit_math_fidelity: ttnn.MathFidelity | None = None,
        noise_cache: bool = True,
        record_memory: bool | None = None,
        reference_cache: bool = True,
    ) -> None:
        """``sp_axis`` x ``tp_axis`` is the primary DiT layout (SP8 x TP4 on a 4x8 mesh by default). ``large_layout``
        = ``(sp_axis, tp_axis)`` of a second resident layout that requests with at least ``large_layout_min_tokens``
        DiT rows render on (:func:`pick_layout`); it costs a second set of transformer weights per resident variant
        and a second warm-up. ``shard_prompt`` shards the 512 prompt rows over SP (required from SP 4 up).
        ``sdpa_exp_approx`` / ``text_encoder_math_fidelity`` are the two exact-profile levers (the DiT ring SDPA exp and
        the Qwen3 linears' matmul fidelity); ``None`` reads ``FLUX2_KLEIN_SDPA_EXP_APPROX`` /
        ``FLUX2_KLEIN_TEXT_ENCODER_FIDELITY`` and falls back to the port's defaults. ``dit_math_fidelity`` is the matmul
        fidelity of every DiT linear (HiFi4 or HiFi2; the ring SDPA compute stays HiFi4); ``None`` reads
        ``FLUX2_KLEIN_DIT_FIDELITY`` and falls back to the block module's default (HiFi4). ``text_encoder_fsdp`` shards the
        text encoder's weights over the mesh axis its TP does not use (gathered per linear on every encode) or
        replicates them there (``False``: no weight gathers, more DRAM per chip, bit-equal encodings); ``None``
        reads ``FLUX2_KLEIN_TEXT_ENCODER_FSDP`` and falls back to :data:`TEXT_ENCODER_FSDP_DEFAULT`. ``noise_cache``:
        memoise the initial latents per ``(seed, h, w)`` (``packing.make_noise_cached``; ``False`` redraws on every
        request, for timing a fresh seed honestly). ``record_memory``: end every per-request RECORD line with the
        mesh memory view (``None``: ``FLUX2_KLEIN_RECORD_MEMORY``, else :data:`RECORD_MEMORY_DEFAULT`).
        ``reference_cache``: keep the prepared image + fp32 pixels of every PATH reference (``references.
        ReferenceCache``, last ``FLUX2_KLEIN_REFERENCE_CACHE_SIZE`` files, keyed by path + mtime + size) so a request
        that names a file again skips its PNG decode and conversion; ``False`` prepares every reference per request
        (for timing a never-seen file honestly). PIL images are prepared per request either way."""
        if not resident:
            raise ValueError("at least one resident variant is required")
        if size not in SIZES:
            raise ValueError(f"unknown size {size!r}; expected one of {SIZES}")
        if large_layout is not None:
            large_layout = (int(large_layout[0]), int(large_layout[1]))
            if sorted(large_layout) != [0, 1]:
                raise ValueError(f"large_layout must name the two mesh axes, got {large_layout}")
        self.readback = "fast"
        """How the decoded image reaches the host: ``"fast"`` (uint8 on the device, one replica read; the default) or
        ``"legacy"`` (``tensor.to_torch`` of the bf16 image, every replica; kept for the equality test)."""
        for variant in resident:
            if variant not in MODELS:
                raise ValueError(f"unknown variant {variant!r}")
            if variant not in snapshots:
                raise ValueError(f"no snapshot for resident variant {variant!r}")
        self._device = mesh_device
        self._shape = tuple(int(s) for s in mesh_device.shape)
        self._size = size
        self._sp_axis, self._tp_axis = sp_axis, tp_axis
        self._snapshots = {v: Path(snapshots[v]) for v in resident}
        self._resident = tuple(resident)
        self._shard_prompt = bool(shard_prompt)
        self.sdpa_exp_approx = sdpa_exp_approx_from_env(sdpa_exp_approx)
        self.text_encoder_math_fidelity = text_encoder_math_fidelity_from_env(text_encoder_math_fidelity)
        self.dit_math_fidelity = dit_math_fidelity_from_env(dit_math_fidelity)
        self.text_encoder_fsdp = text_encoder_fsdp_from_env(text_encoder_fsdp)
        self.noise_cache = bool(noise_cache)
        self.record_memory = record_memory_from_env(record_memory)
        self._reference_cache: ReferenceCache | None = ReferenceCache() if reference_cache else None
        # One worker: the initial latents of a request are drawn here while its encodes run on the device.
        self._noise_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="flux2-klein-noise")
        logger.info(
            f"RECORD exact-profile levers: DiT ring SDPA exp {'approx' if self.sdpa_exp_approx else 'precise'}; "
            f"DiT linear fidelity {self.dit_math_fidelity_effective} (SDPA compute HiFi4); "
            f"text encoder linear fidelity {self.text_encoder_math_fidelity or 'HiFi2 (layer default)'}; "
            f"text encoder weights {'FSDP-sharded' if self.text_encoder_fsdp else 'replicated'} over the non-TP axis"
        )
        self.timings_init: dict[str, float] = {}
        t_init = time.time()

        self._num_links, self._topology, self._max_live_traces = num_links, topology, max_live_traces
        self._vae_tp_axis, self._vae_h_axis = vae_tp_axis, vae_h_axis
        self._large_layout_min_tokens = int(large_layout_min_tokens)

        # The DiT configuration of every resident snapshot must be the requested size (transformer/config.json).
        self._configs: dict[str, Flux2KleinConfig] = {}
        for variant in self._resident:
            config = Flux2KleinConfig.from_snapshot(self._snapshots[variant])
            if config.size != size:
                raise ValueError(f"{self._snapshots[variant]} holds a {config.size} transformer, requested {size}")
            self._configs[variant] = config

        # Text encoder (co-resident, TP over the primary layout's TP axis, FSDP over the other or replicated there).
        # Built first: each layout's text encoding wrapper shares it.
        shared = self._snapshots[self._resident[0]]
        other_axis = 1 - tp_axis
        t0 = time.time()
        self._enc_ccl = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)
        self._enc_parallel = EncoderParallelConfig(
            tensor_parallel=ParallelFactor(factor=self._shape[tp_axis], mesh_axis=tp_axis),
            fsdp=ParallelFactor(factor=self._shape[other_axis], mesh_axis=other_axis)
            if self.text_encoder_fsdp
            else None,
        )
        self._encoder: Flux2KleinPromptEncoder = build_flux2_klein_text_encoder(
            shared,
            mesh_device=mesh_device,
            parallel_config=self._enc_parallel,
            ccl_manager=self._enc_ccl,
            is_fsdp=self.text_encoder_fsdp,
            load_weights=True,
            linear_math_fidelity=self.text_encoder_math_fidelity,
        )
        self.timings_init["load_text_encoder_s"] = time.time() - t0
        logger.info(f"RECORD after text encoder: {dram_per_bank(mesh_device)}")

        # VAE decoder (its own CCL manager; channels TP over axis 0, latent rows H over axis 1); the weights are
        # loaded once and every layout's decode helper shares the module.
        t0 = time.time()
        self._vae_ccl = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)
        decoder, self._vae_parallel = build_flux2_klein_vae_decoder(
            mesh_device, tp_axis=vae_tp_axis, h_axis=vae_h_axis, ccl_manager=self._vae_ccl
        )
        load_flux2_klein_vae_weights(decoder, shared, parallel_config=self._vae_parallel, mesh_device=mesh_device)
        self._decoder = decoder
        self.timings_init["load_vae_s"] = time.time() - t0
        logger.info(f"RECORD after vae: {dram_per_bank(mesh_device)}")

        # DiT layouts: the primary, then the large-canvas layout (each with its own CCL manager and weight shards).
        self._layouts: dict[str, _Layout] = {}
        self._primary = layout_name(sp_axis, tp_axis, self._shape)
        self._layouts[self._primary] = self._build_layout(sp_axis, tp_axis, shared)
        self._large: str | None = None
        if large_layout is not None and large_layout != (sp_axis, tp_axis):
            self._large = layout_name(large_layout[0], large_layout[1], self._shape)
            self._layouts[self._large] = self._build_layout(large_layout[0], large_layout[1], shared)
        primary = self._layouts[self._primary]
        self._dit_ccl, self._dit_parallel = primary.ccl, primary.parallel
        self._transformers, self._denoisers = primary.transformers, primary.denoisers
        self._prompt_encoder, self._vae = primary.prompt_encoder, primary.vae
        for variant in self._resident:  # the pre-layout timing keys name the primary layout's loads
            self.timings_init[f"load_{variant}_s"] = self.timings_init[f"load_{variant}_{self._primary}_s"]

        # VAE encoder (references): channels TP over the decoder's TP axis, spatial axes replicated; same CCL manager.
        t0 = time.time()
        encoder_vae, self._vae_encoder_parallel = build_flux2_klein_vae_encoder(
            mesh_device, tp_axis=vae_tp_axis, ccl_manager=self._vae_ccl
        )
        load_flux2_klein_vae_encoder_weights(
            encoder_vae, shared, parallel_config=self._vae_encoder_parallel, mesh_device=mesh_device
        )
        self._vae_encode = Flux2KleinVaeEncode(encoder_vae, device=mesh_device)
        self._encoded_sizes: set[tuple[int, int]] = set()
        self.timings_init["load_vae_encoder_s"] = time.time() - t0
        logger.info(f"RECORD after vae encoder: {dram_per_bank(mesh_device)}")

        # Warm-ups BEFORE any trace exists, per layout: the empty prompt (the base CFG's unconditional text; it also
        # allocates the text encoder's persistent CCL buffers) and one eager VAE decode of a zero state (kernel
        # compile; the second layout's decode only adds its state gather).
        for name, layout in self._layouts.items():
            t0 = time.time()
            layout.null_text = layout.prompt_encoder.encode_host("")
            self.timings_init[f"encode_null_{name}_s"] = time.time() - t0
            if warmup_size is not None:
                t0 = time.time()
                self._warm_vae(*warmup_size, layout=layout)
                self.timings_init[f"vae_warmup_{name}_s"] = time.time() - t0
        self._null_text: TextPair = primary.null_text
        self.timings_init["encode_null_s"] = self.timings_init[f"encode_null_{self._primary}_s"]
        if warmup_size is not None:
            self.timings_init["vae_warmup_s"] = self.timings_init[f"vae_warmup_{self._primary}_s"]
            t0 = time.time()
            self._warm_vae_encoder(*warmup_size)
            self.timings_init["vae_encoder_warmup_s"] = time.time() - t0
        ttnn.synchronize_device(mesh_device)
        self.timings_init["total_s"] = time.time() - t_init
        logger.info(f"RECORD pipeline ready in {self.timings_init['total_s']:.1f}s: {dram_per_bank(mesh_device)}")

    @classmethod
    def create(
        cls, mesh_device: ttnn.MeshDevice, *, size: str = "4b", resident: tuple[str, ...] = ("distilled",), **kw
    ):
        """Snapshots from ``weights_flux2_klein.resolve_weights_dir`` per resident variant of ``size``."""
        snapshots = {variant: resolve_weights_dir(variant, size) for variant in resident}
        return cls(mesh_device, snapshots=snapshots, size=size, resident=resident, **kw)

    # --- accessors -------------------------------------------------------------------------------------------------------

    @property
    def device(self) -> ttnn.MeshDevice:
        return self._device

    @property
    def mesh_shape(self) -> tuple[int, int]:
        return self._shape

    @property
    def size(self) -> str:
        return self._size

    @property
    def resident(self) -> tuple[str, ...]:
        return self._resident

    @property
    def configs(self) -> dict[str, Flux2KleinConfig]:
        return self._configs

    @property
    def denoisers(self) -> dict[str, Flux2KleinDenoiser]:
        return self._denoisers

    @property
    def transformers(self) -> dict[str, Flux2KleinTransformer]:
        return self._transformers

    @property
    def encoder(self) -> Flux2KleinPromptEncoder:
        return self._encoder

    @property
    def prompt_encoder(self) -> Flux2KleinTextEncoding:
        return self._prompt_encoder

    @property
    def vae(self) -> Flux2KleinVaeDecode:
        return self._vae

    @property
    def vae_encode(self) -> Flux2KleinVaeEncode:
        return self._vae_encode

    @property
    def null_text(self) -> TextPair:
        """The empty prompt's encoding ``([512, joint] bf16, ids)``, computed once (the primary layout's)."""
        return self._null_text

    @property
    def layouts(self) -> tuple[str, ...]:
        """The resident DiT layouts, primary first."""
        return tuple(self._layouts)

    @property
    def primary_layout(self) -> str:
        return self._primary

    @property
    def large_layout(self) -> str | None:
        return self._large

    def layout_for(self, total_tokens: int) -> str:
        """The layout a request with ``total_tokens`` DiT rows (image + reference + text) renders on."""
        return pick_layout(
            total_tokens, primary=self._primary, large=self._large, min_tokens=self._large_layout_min_tokens
        )

    def denoisers_of(self, layout: str) -> dict[str, Flux2KleinDenoiser]:
        return self._layouts[layout].denoisers

    # --- internals -------------------------------------------------------------------------------------------------------

    @property
    def dit_math_fidelity_effective(self) -> ttnn.MathFidelity:
        """The DiT linears' matmul fidelity in force (``dit_math_fidelity`` or the block module's default)."""
        return self.dit_math_fidelity if self.dit_math_fidelity is not None else FLUX2_KLEIN_DIT_MATH_FIDELITY_DEFAULT

    def _build_layout(self, sp_axis: int, tp_axis: int, shared: Path) -> _Layout:
        """The transformers (weights loaded), denoisers, text encoding wrapper and decode helper of one layout."""
        name = layout_name(sp_axis, tp_axis, self._shape)
        ccl = CCLManager(mesh_device=self._device, num_links=self._num_links, topology=self._topology)
        parallel = DiTGParallelConfigNoCFG(
            tensor_parallel=ParallelFactor(factor=self._shape[tp_axis], mesh_axis=tp_axis),
            sequence_parallel=ParallelFactor(factor=self._shape[sp_axis], mesh_axis=sp_axis),
        )
        transformers: dict[str, Flux2KleinTransformer] = {}
        denoisers: dict[str, Flux2KleinDenoiser] = {}
        for variant in self._resident:
            t0 = time.time()
            model = Flux2KleinTransformer(
                self._configs[variant],
                device=self._device,
                ccl_manager=ccl,
                parallel_config=parallel,
                shard_prompt=self._shard_prompt,
                sdpa_exp_approx=self.sdpa_exp_approx,
                math_fidelity=self.dit_math_fidelity,
            )
            load_flux2_klein_transformer_weights(
                model,
                variant,
                self._size,
                self._snapshots[variant],
                parallel_config=parallel,
                mesh_device=self._device,
            )
            transformers[variant] = model
            denoisers[variant] = Flux2KleinDenoiser(
                model, device=self._device, sp_axis=sp_axis, max_live_traces=self._max_live_traces
            )
            self.timings_init[f"load_{variant}_{name}_s"] = time.time() - t0
            logger.info(f"RECORD after transformer[{variant}/{self._size}] {name}: {dram_per_bank(self._device)}")
        # The encoder hands the rows to this layout's DiT on the device, partitioned over its SP axis under
        # shard_prompt (replicated at SP 1 / without it: embed_prompt's other placement).
        prompt_encoder = Flux2KleinTextEncoding(
            self._encoder,
            text_ids=text_ids(FLUX2_KLEIN_TEXT_LENGTH),
            sp_axis=sp_axis if self._shard_prompt else None,
        )
        vae = Flux2KleinVaeDecode(
            self._decoder,
            device=self._device,
            dit_ccl=ccl,
            sp_axis=sp_axis,
            h_axis=self._vae_h_axis,
            tp_axis=self._vae_tp_axis,
            vae_ccl=self._vae_ccl,
            snapshot=shared,
        )
        return _Layout(
            name=name,
            sp_axis=sp_axis,
            tp_axis=tp_axis,
            ccl=ccl,
            parallel=parallel,
            transformers=transformers,
            denoisers=denoisers,
            prompt_encoder=prompt_encoder,
            vae=vae,
        )

    def _warm_vae(self, height: int, width: int, *, traced: bool = False, layout: _Layout | None = None) -> None:
        """One decode of a zero state at ``height x width``: eager (``__init__``) compiles the kernels and allocates
        the gather buffers; traced (``generate``, before the DiT capture of a layout) CAPTURES the decoder trace of
        the size, so its captured input buffer and the VAE CCL buffers exist before any DiT trace (module docstring).
        The traced output is the trace's buffer (never deallocated here). ``layout``: the primary when ``None``."""
        layout = layout or self._layouts[self._primary]
        h, w = latent_shape(height, width)
        rows = padded_length(h * w, self._shape[layout.sp_axis])
        zero = tensor.from_torch(
            torch.zeros(1, rows, FLUX2_KLEIN_IN_CHANNELS),
            device=self._device,
            dtype=ttnn.float32,
            mesh_axes=[None, layout.sp_axis, None],
        )
        out = layout.vae.decode(zero, num_image_rows=h * w, height=height, width=width, traced=traced)
        ttnn.synchronize_device(self._device)
        if not traced:
            ttnn.deallocate(out)
        ttnn.deallocate(zero)

    def _warm_vae_encoder(self, height: int, width: int) -> None:
        """One eager encode of a zero image: compiles the encoder's kernels and prepares its conv weights (long-lived
        device buffers) before any trace exists."""
        from PIL import Image

        h, w = latent_shape(height, width)
        tokens = self._vae_encode.encode(Image.new("RGB", (width, height)))
        if tuple(tokens.shape) != (h * w, FLUX2_KLEIN_IN_CHANNELS):
            raise RuntimeError(f"VAE encoder warm-up returned {tuple(tokens.shape)} for {width}x{height}")
        self._encoded_sizes.add((width, height))

    def traces_live(self) -> bool:
        """Whether any denoiser or VAE trace is captured (the condition under which a long-lived allocation needs the
        aliasing rule)."""
        return any(
            any(d.captured_keys for d in layout.denoisers.values()) or layout.vae.trace_captured()
            for layout in self._layouts.values()
        )

    def release_traces(self) -> None:
        """Release every live trace (every layout's denoisers and decode helper, and the VAE encoder); each is
        recaptured on its next traced use."""
        for layout in self._layouts.values():
            for denoiser in layout.denoisers.values():
                denoiser.release_traces()
            layout.vae.release_trace()
        self._vae_encode.release_trace()

    def _encode(self, prompt: str) -> TextPair:
        return self._prompt_encoder.encode_host(prompt)

    def _layout_of(self, resolved: Flux2KleinResolved) -> _Layout:
        """The DiT layout a resolved request renders on: :func:`pick_layout` of its image + reference + text rows
        (an off-grid canvas counts no image rows; ``admit`` refuses it before anything runs)."""
        r = FLUX2_KLEIN_DOWNSAMPLING_RATIO
        image_rows = (
            (resolved.height // r) * (resolved.width // r)
            if resolved.height % r == 0 and resolved.width % r == 0
            else 0
        )
        prepared = resolved.prepared
        cond_rows = prepared.cond_tokens() if prepared is not None and prepared.images else 0
        return self._layouts[self.layout_for(image_rows + cond_rows + resolved.text_length)]

    # --- the request -----------------------------------------------------------------------------------------------------

    @property
    def reference_cache(self) -> ReferenceCache | None:
        """The prepared-reference cache path sources go through (``None`` under ``reference_cache=False``)."""
        return self._reference_cache

    def _draw_noise(self, seed: int, h: int, w: int) -> torch.Tensor:
        """The initial latents (``packing.make_noise``), memoised per ``(seed, h, w)`` under ``noise_cache``."""
        return make_noise_cached(seed, h, w) if self.noise_cache else make_noise(seed, h, w)

    def _submit_noise(self, seed: int, h: int, w: int) -> Future[torch.Tensor]:
        """Queue the draw on the worker thread (a ``Future``; ``result()`` joins it, re-raising its error)."""
        return self._noise_pool.submit(self._draw_noise, seed, h, w)

    def generate(self, request: Flux2KleinRequest, *, traced: bool = True) -> Flux2KleinResult:
        t_total = time.time()
        cache = self._reference_cache
        hits_before, misses_before = (cache.hits, cache.misses) if cache is not None else (0, 0)
        resolved = request.resolve(reference_cache=cache)
        resolve_s = time.time() - t_total
        cache_hits = cache.hits - hits_before if cache is not None else 0
        cache_misses = cache.misses - misses_before if cache is not None else len(request.images)
        if resolved.model not in self._denoisers:
            raise ValueError(f"model {resolved.model!r} is not resident (resident: {self._resident})")
        if resolved.size != self._size:
            raise ValueError(f"size {resolved.size!r} is not resident (resident size: {self._size})")
        # 1. the seed, the DiT layout for this request's row count (image + reference + text rows;
        #    :func:`pick_layout`) and the admission (host, before any device work; the stream-rows check of
        #    ``policy.admit`` needs the layout's SP factor).
        seed = resolved.seed if resolved.seed is not None else random_seed()
        resolved = replace(resolved, seed=seed)
        layout = self._layout_of(resolved)
        admission = admit(resolved, self._shape, sp_factor=self._shape[layout.sp_axis], shard_prompt=self._shard_prompt)
        if not admission.ok:
            # Nothing has touched the device for this request: no upload, no encode, no slot allocation.
            raise Flux2KleinAdmissionError(admission)

        # 2. noise and ids (P:496-510, P:306-313). The noise (a bf16 randn of 128 x h x w on a CPU generator: 20-60
        #    ms of host time at 1K / 2K) is drawn on the worker thread while steps 3-4 run on the device and joined
        #    before the denoise request is built; the explicit Generator makes torch.randn deterministic on any
        #    thread, so the values are make_noise's bit for bit (memoised per (seed, h, w) under noise_cache).
        h, w = latent_shape(resolved.height, resolved.width)
        noise_future = self._submit_noise(resolved.seed, h, w)
        ids = latent_ids(h, w)
        prepared_refs = resolved.prepared
        cond_rows = prepared_refs.cond_tokens() if prepared_refs is not None and prepared_refs.images else 0

        # 3. encode text (P:208-263): the rows stay on the device in the denoiser's placement (module docstring);
        #    the host copy only when the request asks for it. The base model also takes the empty prompt (P:749-760;
        #    cached host rows). The device work is enqueued asynchronously, so the stage ends on a sync.
        t0 = time.time()
        text_rows: TextPair | None
        if resolved.text_override is not None:
            text, null = resolved.text_override
            text_rows = text
        else:
            device_text = layout.prompt_encoder.encode_device(resolved.prompt, with_host=request.return_text)
            text = device_text.pair
            text_rows = device_text.host
            null = layout.null_text if resolved.cfg else None
            ttnn.synchronize_device(self._device)
        encode_s = time.time() - t0

        # 4. encode references (device VAE encoder; the override is for the bit-comparable golden rows only). The
        # first encode of a size the encoder has not seen allocates persistent CCL buffers (aliasing rule).
        t0 = time.time()
        image_cond: CondPair | None = None
        if prepared_refs is not None and prepared_refs.images:
            if resolved.image_cond_override is not None:
                image_cond = resolved.image_cond_override
            else:
                new_sizes = {tuple(s) for s in prepared_refs.sizes} - self._encoded_sizes
                if new_sizes and self.traces_live():
                    logger.info(
                        f"releasing every trace before the first encode of reference size(s) {sorted(new_sizes)}"
                    )
                    self.release_traces()
                image_cond = self._vae_encode.encode_references(prepared_refs)
                self._encoded_sizes |= {tuple(s) for s in prepared_refs.sizes}
        encode_refs_s = time.time() - t0

        t0 = time.time()
        noise = noise_future.result()
        noise_wait_s = time.time() - t0

        denoise_request = Flux2KleinDenoiseRequest(
            noise=noise,
            image_ids=ids,
            text=text,
            num_steps=resolved.steps,
            guidance=resolved.guidance,
            text_null=null if resolved.cfg else None,
            image_cond=image_cond,
        )
        denoiser = layout.denoisers[resolved.model]
        vae = layout.vae

        # 5. denoise (the aliasing rule of the module docstring before any long-lived allocation): slots first, then
        # the VAE trace of the canvas size, then the DiT trace (capture order; module docstring).
        needs_fallback_vae = not vae.partitioned(resolved.height, resolved.width) and vae._fallback is None
        vae_captured_before = (not traced) or vae.trace_captured_for(resolved.height, resolved.width)
        if (
            not denoiser.has_layout(denoise_request)
            or needs_fallback_vae
            or (not vae_captured_before and self.traces_live())
        ):
            self.release_traces()
            vae_captured_before = (not traced) or vae.trace_captured_for(resolved.height, resolved.width)
        plan_cache_before = layout.transformers[resolved.model].plan_cache_info()
        t0 = time.time()
        prepared: Flux2KleinPreparedRequest = denoiser.prepare(denoise_request)
        prepare_s = time.time() - t0
        plan_cache_after = layout.transformers[resolved.model].plan_cache_info()
        plan_hits = plan_cache_after["hits"] - plan_cache_before["hits"]
        plan_misses = plan_cache_after["misses"] - plan_cache_before["misses"]
        t0 = time.time()
        if not vae_captured_before:
            if self.traces_live():
                raise RuntimeError("a VAE capture must precede every DiT trace (aliasing rule)")
            self._warm_vae(resolved.height, resolved.width, traced=True, layout=layout)
        vae_capture_s = time.time() - t0
        captured_before = prepared.key in denoiser.captured_keys
        t0 = time.time()
        state = denoiser.denoise_on_device(prepared, traced=traced)
        denoise_s = time.time() - t0

        # 6. decode (a replay when traced: the trace was captured above, before the DiT trace)
        t0 = time.time()
        out = vae.decode(state, num_image_rows=h * w, height=resolved.height, width=resolved.width, traced=traced)
        ttnn.synchronize_device(self._device)
        decode_s = time.time() - t0

        # 7. host. ``out`` is replicated on every chip (the decoder ends with its spatial all-gather). "fast": the
        # uint8 conversion runs on the device and ONE replica is read (``ttnn.get_device_tensors(x)[0]``).
        t0 = time.time()
        if self.readback == "fast":
            from PIL import Image

            u8 = image_to_uint8_on_device(out)
            array = image_rows_to_hwc(ttnn.to_torch(ttnn.get_device_tensors(u8)[0])[0]).numpy()
            ttnn.deallocate(u8)
            if not traced:
                ttnn.deallocate(out)
            image = Image.fromarray(array)
        else:
            img = tensor.to_torch(out)[0].permute(2, 0, 1)
            if not traced:
                ttnn.deallocate(out)
            image = to_pil(img)
        final_tokens = denoiser.read_state(prepared) if request.return_final_tokens else None
        to_host_s = time.time() - t0

        timings = {
            "resolve_s": resolve_s,
            "reference_cache_hits": cache_hits,
            "reference_cache_misses": cache_misses,
            "encode_s": encode_s,
            "encode_refs_s": encode_refs_s,
            "prepare_s": prepare_s,
            "plan_cache_hits": plan_hits,
            "plan_cache_misses": plan_misses,
            "vae_capture_s": vae_capture_s,
            "denoise_s": denoise_s,
            "decode_s": decode_s,
            "to_host_s": to_host_s,
            "noise_wait_s": noise_wait_s,
            "total_s": time.time() - t_total,
            "trace_captured": bool(traced and not captured_before),
            "vae_trace_captured": bool(traced and not vae_captured_before),
        }
        references = prepared_refs.sizes if prepared_refs is not None else ()
        logger.info(
            f"RECORD generate[{resolved.model}/{resolved.size}] {layout.name} {resolved.width}x{resolved.height} seed "
            f"{resolved.seed} steps {resolved.steps} guidance {resolved.guidance} references {list(references)} cond "
            f"rows {cond_rows}: resolve {resolve_s:.3f}s (reference cache hits {cache_hits} misses {cache_misses}) "
            f"encode {encode_s:.3f}s encode_refs {encode_refs_s:.3f}s prepare {prepare_s:.3f}s (plan cache hits "
            f"{plan_hits} misses {plan_misses}) "
            f"vae_capture {vae_capture_s:.3f}s denoise {denoise_s:.3f}s decode {decode_s:.3f}s to_host "
            f"{to_host_s:.3f}s total {timings['total_s']:.3f}s (noise wait {noise_wait_s:.3f}s; denoise trace captured "
            f"{timings['trace_captured']}, vae trace captured {timings['vae_trace_captured']}); "
            f"{dram_per_bank(self._device) if self.record_memory else f'memory view off ({RECORD_MEMORY_ENV}=1 records it)'}"
        )
        return Flux2KleinResult(
            image=image,
            final_tokens=final_tokens,
            timings=timings,
            layout_key=prepared.key,
            model=resolved.model,
            size=resolved.size,
            seed=resolved.seed,
            steps=resolved.steps,
            guidance=resolved.guidance,
            admission=admission,
            text=text_rows,
            image_cond=image_cond,
            references=tuple(references),
            input_prompt=resolved.prompt,
            layout=layout.name,
        )

    def close(self) -> None:
        """Traces, then slots, then weights; the caller closes the mesh."""
        self._noise_pool.shutdown(wait=True)
        if self._reference_cache is not None:
            self._reference_cache.clear()
        self.release_traces()
        for layout in self._layouts.values():
            for denoiser in layout.denoisers.values():
                denoiser.deallocate()
        ttnn.synchronize_device(self._device)
        for module in (
            *(m for layout in self._layouts.values() for m in layout.transformers.values()),
            self._encoder,
            self._decoder,
            *(layout.vae._fallback for layout in self._layouts.values()),
            self._vae_encode.encoder,
        ):
            if module is not None and hasattr(module, "deallocate_weights"):
                module.deallocate_weights()
        ttnn.synchronize_device(self._device)


__all__ = [
    "LARGE_LAYOUT_MIN_TOKENS",
    "MODELS",
    "OUTPUT_FORMATS",
    "SIZES",
    "Flux2KleinAdmissionError",
    "Flux2KleinPipeline",
    "Flux2KleinRequest",
    "Flux2KleinResolved",
    "Flux2KleinResult",
    "ReferenceCache",
    "dram_per_bank",
    "image_rows_to_hwc",
    "image_to_uint8_on_device",
    "layout_name",
    "make_noise",
    "make_noise_cached",
    "pick_layout",
    "record_memory_from_env",
    "random_seed",
    "to_pil",
]
