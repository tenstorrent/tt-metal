# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""FLUX 3 Image Klein text-to-image and reference editing on one tt_dit mesh: ``reference/flux3_klein/pipeline.py::
Pipeline.generate`` with every model stage on device (design records ``docs/M6_M7_DESIGN.md`` section 5 and
``docs/M8_DESIGN.md`` section 5).

Stages per request (``pipeline.py:55-83``, ``generate.py:65-77``), host work limited to the request schema, the
reference sizing, noise, ids and the PNG:
  0. :func:`resolve_caption` (``generate.py:52-63``, M9): a finished upsampler answer (no boxes and a box table fully
     cited as ``<id>`` s, ``bbox.is_finished``) goes to the DiT as it is; otherwise, with ``prompt_upsampling`` and an
     ``upsampler`` attached to the pipeline (``Flux3KleinUpsampler`` / its subprocess client / a test stub, anything
     with ``__call__(prompt, images, boxes) -> str``), the prompt is upsampled on the pup's own chip BEFORE any Klein
     device work; a ``ValueError`` from the upsampler falls back to the prompt as typed; with boxes and no upsampling
     the ``to_canvas`` table is appended to the prompt. No upsampler attached (the default, every recorded-caption
     gate) = the prompt is the caption.
  1. :meth:`Flux3KleinRequest.resolve` (the ``generate.py:36-42`` defaults and checks, then the reference sizing of
     ``references.prepare_references`` = ``generate.py:65-72``: one reference fitted under 4 MiP and padded to the
     16-px grid, several each under 1 MiP; the canvas defaults to the first PREPARED reference's size, else 1024 x
     1024) and ``policy.admit`` (host, before any device work; a rejection raises :class:`Flux3KleinAdmissionError`
     and nothing has been allocated on the device).
  2. ``h, w = latent_shape(height, width)``; ``noise = make_noise(seed, h, w)`` = ``pipeline.py:69-70`` (CPU
     generator, ``randn`` rounded to bf16, fp32 state; hashes to the goldens' ``noise_sha256``); ``image_ids(h, w)``.
  3. encode text (device): ``Flux3KleinPromptEncoder.encode(prompt)`` on the co-resident Qwen3-VL encoder; the base
     model also takes ``encode('')`` (computed once per pipeline and kept on the host). The rows go to the host
     (``to_host``) and into the denoise request: the values are bf16 either way, so a device handoff is a later
     zero-numerics optimisation (M10).
  4. encode references (device): :class:`Flux3KleinVaeEncode` (``Flux2VaeEncoder``, channels TP4) per prepared
     reference -> bf16 ``[h_i * w_i, 128]`` tokens on the host, concatenated with ``packing.reference_ids(h_i, w_i,
     i)`` into ``Flux3KleinDenoiseRequest.image_cond`` (``pipeline.py:84-95``); shared by both CFG samples.
  5. denoise (traced): ``Flux3KleinDenoiser.prepare`` + :meth:`Flux3KleinDenoiser.denoise_on_device` on the resident
     transformer of ``model`` (M4; with recorded text and recorded reference tokens this is bit-identical to the M4
     traced run of the golden's request).
  6. decode (device, traced): :class:`Flux3KleinVaeDecode` (M6) on the fp32 state slot (a replay of the trace captured
     before the DiT trace, see the capture order below); a canvas whose latent height is not a multiple of the H
     partition (896 x 1120: h 70) takes the replicated-rows decoder.
  7. host: ``to_torch(out)[0].permute(2, 0, 1)`` and the reference ``to_pil`` (``pipeline.py:208-210``, truncation);
     the single-reference crop-back (``generate.py:76-77``) when neither side was given; ``final_tokens`` is read AFTER
     the decode so no host read sits between replays.

Residency (decision 8 of the design), measured 2026-10-17 on the 4x8 ring with everything resident
(logs/m8_verifier_device_probe_4x8_run2.log:225-299, run3:236-268): after both DiTs 362.0 MB per bank (2.90 GB per
chip), the text encoder (TP8 x FSDP4) +38.9 MB per bank, the VAE decoder weights +0.1 MB, the VAE encoder weights
+0.2 MB, and the warm-ups (the empty-prompt encode = the text encoder's persistent CCL buffers, one eager decode and
one eager encode at ``warmup_size`` = the VAE kernels, the encoder's prepared conv weights and the VAE CCL manager's
persistent buffers at that size) +91.7 MB, of which the encoder warm-up is 66.6 MB per bank = 0.53 GB per chip
(M7 without the encoder: 426.3 MB per bank, logs/m67_builder_pipeline_4x8.log:94). Pipeline ready: 492.9 MB per
bank = 3.94 GB per chip of 34 GB; a 1-reference 1024 request adds 20 MB per bank (513.1 after it, TRACE 2,121,728 B
per bank); after ``close()`` 85-96 MB per bank stay allocated (persistent CCL buffers of the three managers, not
released by ``deallocate_weights``). Traces: one per denoiser (``max_live_traces=1``, base 25.6 MB + distilled 13.0
MB) plus the VAE decoder's 2.3 MB of the 64 MB region; the encoder runs eager.

Trace aliasing rule (the M4 contract, ``denoise_flux3_klein.py`` docstring): a buffer allocated while a trace is
live may sit where that trace's replay writes. Each denoiser releases ITS traces before allocating a new layout's
slots; this pipeline extends the rule across modules: before a ``prepare`` that allocates (``has_layout`` False),
before building the fallback VAE decoder, before the first reference encode of a size the encoder has not seen (its
CCL manager allocates persistent buffers per shape) and before a VAE capture of a size while a DiT trace is live,
EVERY live trace (both denoisers and the VAE) is released and recaptured on its next use (base capture 4.2 s,
distilled 2.0 s, VAE 1.0 s). Capture ORDER per layout (the M8 fix of the two-generate tracker gate, found 2026-10-17
by the M8 device verifier, logs/m8_verifier_device_tracker_4x8.log:37-49,114,180-186,252-266): the denoiser's slots
are allocated by ``prepare``, then the VAE decode trace of the canvas size is captured on a zero state
(``_warm_vae(traced=True)``: its captured input buffer ``z`` and the VAE CCL manager's gather buffers are allocated
HERE), and only then the DiT trace is captured by the first step; the per-request decode is then a replay whose
only allocations are temporaries freed before the next DiT replay. With the M7 order (DiT first, VAE captured by the
first decode) the second request's first DiT replay found the two VAE buffers allocated after its capture
(``Found 2 device buffer(s) still alive before trace replay``). The text encoder's persistent CCL buffers and the VAE
decoder's and encoder's kernels (and the encoder's prepared conv weights) are warmed in ``__init__`` for the same
reason.
"""

from __future__ import annotations

import hashlib
import json
import os
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
from ...reference.flux3_klein import bbox
from ...utils import cache, tensor
from .denoise_flux3_klein import Flux3KleinDenoiser, Flux3KleinDenoiseRequest, Flux3KleinPreparedRequest
from .packing import FLUX3_KLEIN_IN_CHANNELS, FLUX3_KLEIN_MAX_REFERENCES, image_ids, latent_shape, padded_length
from .policy import Admission, Flux3KleinAdmissionError, admit
from .prompt_encoder import Flux3KleinPromptEncoder
from .references import TOO_MANY_REFERENCES, CropBox, PreparedReferences, open_image, prepare_references
from .scheduler import FLUX3_KLEIN_ALPHA, FLUX3_KLEIN_GUIDANCE, FLUX3_KLEIN_STEPS
from .vae_encode_flux3_klein import (
    Flux3KleinVaeEncode,
    build_flux3_klein_vae_encoder,
    load_flux3_klein_vae_encoder_weights,
)
from .vae_flux3_klein import Flux3KleinVaeDecode, build_flux3_klein_vae_decoder, load_flux3_klein_vae_weights
from .weights_flux3_klein import dit_checkpoint, resolve_weights_dir

# reference pipeline.py:23-26 (the weights entry is resolved by weights_flux3_klein).
MODELS = {
    "base": {"steps": FLUX3_KLEIN_STEPS["base"], "guidance": FLUX3_KLEIN_GUIDANCE["base"]},
    "distilled": {"steps": FLUX3_KLEIN_STEPS["distilled"], "guidance": FLUX3_KLEIN_GUIDANCE["distilled"]},
}
OUTPUT_FORMATS = ("png", "jpeg")
TextPair = tuple[torch.Tensor, torch.Tensor]
CondPair = tuple[torch.Tensor, torch.Tensor]


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


def image_to_uint8_on_device(image: ttnn.Tensor) -> ttnn.Tensor:
    """``to_pil``'s arithmetic ON THE DEVICE: bf16 ``[1, H, W, 3]`` in [-1, 1] -> uint8 ROW_MAJOR ``[1, H, W, 3]`` by
    ``floor(clamp((x + 1) * 127.5, 0, 255))`` computed in fp32 (bf16 -> fp32 typecast is exact, and ``(clamp(x, -1, 1)
    + 1) * 127.5`` equals ``clamp((x + 1) * 127.5, 0, 255)`` for every x: the map is monotonic), then a typecast that
    cannot round because the values are already integral. Bit-identical to ``to_pil`` of the bf16 host copy
    (``test_readback_fast_equals_legacy``). The readback then moves 1 byte per channel instead of 2 and needs no host
    untilize.

    The result is ``[1, H, W * 3]``, one image row per page: a row-major page is the last dim, and the ``[1, H, W, 3]``
    form read 3-byte pages (0.63 s for the 50 MB of a 4096x4096 image, 0.17 s for 12.6 MB at 2048x2048). The bf16
    image is untilized FIRST (dropping the TILE padding of the 3 channels to 32) and reshaped to ``[1, H, W * 3]``
    before the fp32 arithmetic, so the fp32 temporary is 200 MB per chip at 4096x4096 instead of the 2 GiB a typecast
    of the padded ``[1, 4096, 4096, 32]`` tensor took (that allocation is what ran the two-model sweep out of DRAM at
    4096x4096 + 6 references). Conversion plus read: 0.16 s at 4096x4096 and 0.04 s at 2048x2048, bit-identical to
    the padded-first order and to ``to_pil`` (bin/probes/probe_readback_pages.py, probe_readback_mem.py, 2026-10-21).
    ``image_rows_to_hwc`` restores ``[H, W, 3]`` on the host."""
    n, h, w, c = (int(d) for d in image.shape)
    x = ttnn.to_layout(image, ttnn.ROW_MAJOR_LAYOUT)
    x = ttnn.reshape(x, (n, h, w * c))
    x = ttnn.typecast(x, ttnn.float32)
    x = ttnn.add(x, 1.0)
    x = ttnn.multiply(x, 127.5)
    x = ttnn.clamp(x, min=0.0, max=255.0)
    x = ttnn.floor(x)
    return ttnn.typecast(x, ttnn.uint8)


def image_rows_to_hwc(rows: torch.Tensor, channels: int = 3) -> torch.Tensor:
    """``[H, W * C]`` (one replica of :func:`image_to_uint8_on_device`, batch dim dropped) -> ``[H, W, C]``."""
    h, wc = rows.shape[-2], rows.shape[-1]
    return rows.reshape(h, wc // channels, channels)


# --- request / result ------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Flux3KleinRequest:
    """One request with BFL's field names (``generate.py`` / the API schema). ``width`` / ``height`` default to
    ``None`` = ``default_size(references)`` (the first prepared reference's size, else 1024 x 1024). ``images`` holds
    up to ``FLUX3_KLEIN_MAX_REFERENCES`` references as ``PIL.Image.Image`` s or paths. ``seed`` ``None`` (the default)
    is ``generate.py:73``: ``sha256(caption) % 2**32`` of the caption the DiT sees (computed after the upsampler, so
    the same prompt renders the same image as upstream's CLI without ``--seed``; ``caption_seed``), reported on the
    result. ``prompt`` is the prompt as typed (the API's ``prompt``); ``prompt_upsampling`` (``pup``, on by default as upstream) lets the pipeline's
    upsampler expand it unless it is already a finished caption; ``boxes`` are the layout boxes of the API (0-1
    ``[y_min, x_min, y_max, x_max]`` rows with ``id`` / ``desc`` / ``bbox`` or ``from`` / ``src_bbox`` / ``tgt_bbox``),
    validated through ``bbox.to_canvas`` on the host. ``text_override`` (tests only) replaces the device text encoder
    with recorded encodings: ``((tokens, ids), (null_tokens, null_ids) | None)``; ``image_cond_override`` (tests only)
    replaces the device VAE encoder with recorded ``(cond_tokens, cond_ids)`` whose row count must equal the prepared
    references' token count."""

    prompt: str
    model: str = "base"
    width: int | None = None
    height: int | None = None
    seed: int | None = None
    steps: int | None = None
    guidance: float | None = None
    images: tuple = ()
    alpha: float = FLUX3_KLEIN_ALPHA
    output_format: str = "png"
    text_override: tuple | None = None
    image_cond_override: CondPair | None = None
    prompt_upsampling: bool = True
    boxes: tuple | None = None
    return_final_tokens: bool = True
    cache: object = None
    """DBCache (cache-dit) for the DiT single stage, eager only: ``None`` (off), a ``Flux3KleinCacheConfig``, a dict
    of its fields, or a spec string such as ``"F8B0:0.08:w8"`` (``cache_flux3_klein.Flux3KleinCacheConfig.from_spec``)."""
    """Read the final latent state back to the host (``Flux3KleinResult.final_tokens``; the tests and the sweep's
    replay check need it). A serving request does not: at 4096x4096 that read is ~1.5 s of the ``to_host`` stage
    (sweep 2026-10-21: to_host 1.94 s, of which the uint8 image is ~0.3 s). ``False`` leaves ``final_tokens`` None."""

    def resolve(self) -> Flux3KleinResolved:
        """``generate.py:36-42``: ``steps = steps or defaults``; ``guidance = defaults if None``; a flow-map model
        accepts only ``None`` / ``1.0``; at most 10 references. Then ``generate.py:65-72`` on the host: the references
        are prepared (sized), the canvas is ``(width or default_w, height or default_h)`` and the single-reference
        crop-back box is kept only when neither side was given. Raises ``ValueError``."""
        if self.model not in MODELS:
            raise ValueError(f"unknown model {self.model!r}; expected one of {tuple(MODELS)}")
        if len(self.images) > FLUX3_KLEIN_MAX_REFERENCES:
            raise ValueError(TOO_MANY_REFERENCES)
        boxes = tuple(dict(box) for box in self.boxes) if self.boxes else None
        if boxes:
            bbox.to_canvas(list(boxes))  # a malformed box raises ValueError here, before anything else
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
        # generate.py:72 ``width or default_width``: 0 (and None) means "not given", as in the reference API; the
        # crop-back rule of generate.py:70 (``not (width or height)``) reads it the same way (M8 verifier finding).
        for name, side in (("width", self.width), ("height", self.height)):
            if side is not None and int(side) < 0:
                raise ValueError(f"{name} must not be negative, got {side}")
        width_given = int(self.width) if self.width else None
        height_given = int(self.height) if self.height else None
        prepared = prepare_references(self.images, width=width_given, height=height_given)
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
        return Flux3KleinResolved(
            prompt=self.prompt,
            model=self.model,
            flow_map=flow_map,
            width=width,
            height=height,
            seed=None if self.seed is None else int(self.seed),
            steps=int(steps),
            guidance=None if flow_map else guidance,
            images=tuple(self.images),
            alpha=float(self.alpha),
            output_format=self.output_format,
            text_override=self.text_override,
            image_cond_override=self.image_cond_override,
            prepared=prepared,
            crop=prepared.crop,
            prompt_upsampling=bool(self.prompt_upsampling),
            boxes=boxes,
        )


@dataclass(frozen=True)
class Flux3KleinResolved:
    prompt: str
    """The prompt as typed (``input_prompt`` of the record); the caption the DiT sees comes from ``resolve_caption``."""
    model: str
    flow_map: bool
    width: int
    height: int
    seed: int | None
    """``None`` until ``generate()`` derives it from the caption (``caption_seed``)."""
    steps: int
    guidance: float | None
    images: tuple
    alpha: float
    output_format: str
    text_override: tuple | None = None
    image_cond_override: CondPair | None = None
    prepared: PreparedReferences | None = None
    """The references after the host sizing (``None`` only for a resolved object built by hand)."""
    crop: CropBox | None = None
    """The single-reference crop-back box applied to the decoded image (``generate.py:76-77``)."""
    text_length: int | None = None
    """The padded text length (a multiple of 80) once the tokenizer has seen the caption; ``None`` before."""
    prompt_upsampling: bool = True
    boxes: tuple | None = None
    """The 0-1 layout boxes (validated), or ``None``."""


def caption_seed(caption: str) -> int:
    """``generate.py:73``: the seed of an unseeded request, ``int(sha256(caption).hexdigest(), 16) % 2**32`` of the
    caption the DiT sees (the upsampler's answer, the prompt + box table, or the prompt itself)."""
    return int(hashlib.sha256(caption.encode()).hexdigest(), 16) % 2**32


def resolve_caption(resolved: Flux3KleinResolved, upsampler) -> tuple[str, bool]:
    """``generate.py:52-63`` verbatim: ``(caption, upsampled)``. A finished upsampler answer (``not boxes and
    bbox.is_finished(prompt)``) is used as it is; otherwise ``prompt_upsampling`` with an ``upsampler`` runs it on
    the ORIGINAL images (the pup's own ``policy_reference`` sizing is independent of the DiT's) and the boxes; a
    ``ValueError`` from the upsampler (too many references, an aspect over 200:1, a prompt over its cap) is logged
    and the prompt as typed is used; with boxes and no upsampling the ``to_canvas`` table is appended."""
    boxes = list(resolved.boxes) if resolved.boxes else None
    finished = not boxes and bbox.is_finished(resolved.prompt)
    use_pup = bool(resolved.prompt_upsampling) and upsampler is not None and not finished
    caption = None
    if use_pup:
        try:
            caption = upsampler(resolved.prompt, [open_image(image) for image in resolved.images], boxes)
        except ValueError as error:
            logger.warning(f"upsampling skipped: {error}")
            use_pup = False
    if caption is None:
        caption = (
            f"{resolved.prompt.rstrip()} {json.dumps(bbox.to_canvas(boxes), ensure_ascii=False)}"
            if boxes
            else resolved.prompt
        )
    return caption, use_pup


@dataclass
class Flux3KleinResult:
    image: object
    """PIL image ``(width, height)`` RGB."""
    final_tokens: torch.Tensor | None
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
    image_cond: CondPair | None = None
    """The ``([N, 128] bf16, [N, 4] fp32)`` reference rows the denoise consumed (device encoder or override)."""
    references: tuple[tuple[int, int], ...] = ()
    """``(width, height)`` of every prepared reference."""
    crop: CropBox | None = None
    """The crop-back box applied to ``image`` (``None`` when nothing was cropped)."""
    input_prompt: str = ""
    """The prompt as typed (the record's ``input_prompt``)."""
    caption: str = ""
    """The caption the DiT saw (the record's ``prompt``): the upsampler's answer, the prompt + box table, or the
    prompt itself."""
    upsampled: bool = False
    """Whether the upsampler produced ``caption`` (the record's ``upsampled``)."""
    boxes: tuple | None = None
    """The request's 0-1 boxes (the record's ``boxes``)."""
    layout: str = ""
    """The DiT mesh layout that rendered it (``"sp4tp8"`` / ``"sp8tp4"``, :func:`layout_name`)."""


# --- layouts -------------------------------------------------------------------------------------------------------------


def layout_name(sp_axis: int, tp_axis: int, shape: tuple[int, ...] = (4, 8)) -> str:
    """``"sp4tp8"`` for SP over axis 0 x TP over axis 1 of a 4x8 mesh; ``"sp8tp4"`` for the transposed assignment."""
    return f"sp{shape[sp_axis]}tp{shape[tp_axis]}"


LARGE_LAYOUT_MIN_TOKENS = 16384
"""Requests with at least this many DiT tokens (image + reference + text rows) take the large-canvas layout when one is
resident. Measured 2026-10-21 (distilled, 4x8 ring, 160 W; docs/STAGE_TIMING.md), with the ring SDPA worker grid
chosen per sequence length: SP8 x TP4 against SP4 x TP8 is +19 % denoise at 1024² (4.2K tokens), equal at 2048²
(16.4K) and 4096x2304 (37K), and within 1 % at 4096² (66K). The first measurement of the day (-7 to -10 % at 2K/4K)
was the grid pinned by each process's first forward, not the layout. So a second resident layout buys nothing today;
the threshold is where SP8 x TP4 stops costing, kept for layouts that may diverge (reference-heavy 4K, base CFG)."""


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
    """One DiT mesh layout resident in the pipeline: its transformers and denoisers (per variant), the prompt encoder
    wrapper that shards the text rows on its SP axis, the decode helper that gathers the state from that axis, and the
    empty prompt's encoding for the base CFG."""

    name: str
    sp_axis: int
    tp_axis: int
    ccl: CCLManager
    parallel: DiTGParallelConfigNoCFG
    transformers: dict[str, Flux3KleinTransformer]
    denoisers: dict[str, Flux3KleinDenoiser]
    prompt_encoder: Flux3KleinPromptEncoder
    vae: Flux3KleinVaeDecode
    null_text: TextPair | None = None


# --- weights ---------------------------------------------------------------------------------------------------------------


def transformer_cache_key(variant: str, snapshot: Path, model: Flux3KleinTransformer) -> tuple[str, str]:
    """The M3 convention (``tests/models/flux3_klein/test_transformer_flux3_klein.py::load_real_weights``)."""
    model_name = f"flux3_klein_{variant}_{snapshot.name}"
    subfolder = f"transformer_D{model.depth}_S{model.depth_single}{'_flowmap' if model.flow_map else ''}"
    if _dit_weight_override()[0] is not None:  # measurement profile: its own cache, the bf16 cache stays intact
        subfolder += f"_{os.environ['FLUX3_KLEIN_DIT_WEIGHTS'].replace(',', '_')}"
    return model_name, subfolder


def _dit_weight_override():
    """``FLUX3_KLEIN_DIT_WEIGHTS=bfp8[,lofi]`` (MEASUREMENT ONLY, 2026-10-22: the GPU's fp8 column has no exact-profile
    counterpart): the DiT's linears take bfloat8_b weights (and LoFi matmuls with ``,lofi``); attention stays bf16 /
    HiFi2 with the approximate exp. Returns (dtype, fidelity) or (None, None)."""
    spec = os.environ.get("FLUX3_KLEIN_DIT_WEIGHTS", "")
    if not spec:
        return None, None
    parts = [p.strip() for p in spec.split(",")]
    dtype = {"bfp8": ttnn.bfloat8_b, "bf16": ttnn.bfloat16}[parts[0]]
    fidelity = {"lofi": ttnn.MathFidelity.LoFi, "hifi2": ttnn.MathFidelity.HiFi2}[parts[1]] if len(parts) > 1 else None
    return dtype, fidelity


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
        sp_axis: int = 1,
        tp_axis: int = 0,
        topology=ttnn.Topology.Ring,
        num_links: int = 2,
        vae_tp_axis: int | None = 0,
        vae_h_axis: int | None = 1,
        text_encoder_fsdp: bool = True,
        max_live_traces: int = 1,
        warmup_size: tuple[int, int] | None = (1024, 1024),
        upsampler=None,
        large_layout: tuple[int, int] | None = None,
        large_layout_min_tokens: int = LARGE_LAYOUT_MIN_TOKENS,
    ) -> None:
        """``sp_axis`` x ``tp_axis`` is the primary DiT layout (SP4 x TP8 on a 4x8 mesh). ``large_layout`` =
        ``(sp_axis, tp_axis)`` of a second resident layout (``(1, 0)`` = SP8 x TP4) that requests with at least
        ``large_layout_min_tokens`` DiT rows render on (:func:`pick_layout`); it costs a second set of transformer
        weights per resident variant (TP4: 2.8 GB per chip for one variant) and a second warm-up."""
        if not resident:
            raise ValueError("at least one resident variant is required")
        if large_layout is not None:
            large_layout = (int(large_layout[0]), int(large_layout[1]))
            if sorted(large_layout) != [0, 1]:
                raise ValueError(f"large_layout must name the two mesh axes, got {large_layout}")
        self.upsampler = upsampler
        """Anything with ``__call__(prompt, images, boxes) -> str`` (``resolve_caption``); ``None`` = no upsampling."""
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
        self._sp_axis, self._tp_axis = sp_axis, tp_axis
        self._snapshots = {v: Path(snapshots[v]) for v in resident}
        self._resident = tuple(resident)
        self.timings_init: dict[str, float] = {}
        t_init = time.time()

        self._num_links, self._topology, self._max_live_traces = num_links, topology, max_live_traces
        self._vae_tp_axis, self._vae_h_axis = vae_tp_axis, vae_h_axis
        self._large_layout_min_tokens = int(large_layout_min_tokens)

        # Text encoder (co-resident, TP over the primary layout's TP axis, FSDP over the other). Built first: each
        # layout's prompt encoder wraps it with its own output row axis.
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
        self._text_config = text_config
        self.timings_init["load_text_encoder_s"] = time.time() - t0
        logger.info(f"RECORD after text encoder: {dram_per_bank(mesh_device)}")

        # VAE decoder (its own CCL manager, Flux2's 4x8 row: channels TP over axis 0, latent rows H over axis 1); the
        # weights are loaded once and every layout's decode helper shares the module.
        t0 = time.time()
        self._vae_ccl = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)
        decoder, self._vae_parallel = build_flux3_klein_vae_decoder(
            mesh_device, tp_axis=vae_tp_axis, h_axis=vae_h_axis, ccl_manager=self._vae_ccl
        )
        load_flux3_klein_vae_weights(decoder, shared, parallel_config=self._vae_parallel, mesh_device=mesh_device)
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
        encoder_vae, self._vae_encoder_parallel = build_flux3_klein_vae_encoder(
            mesh_device, tp_axis=vae_tp_axis, ccl_manager=self._vae_ccl
        )
        load_flux3_klein_vae_encoder_weights(
            encoder_vae, shared, parallel_config=self._vae_encoder_parallel, mesh_device=mesh_device
        )
        self._vae_encode = Flux3KleinVaeEncode(encoder_vae, device=mesh_device)
        self._encoded_sizes: set[tuple[int, int]] = set()
        self.timings_init["load_vae_encoder_s"] = time.time() - t0
        logger.info(f"RECORD after vae encoder: {dram_per_bank(mesh_device)}")

        # Warm-ups BEFORE any trace exists, per layout: the empty prompt (base CFG; also allocates the encoder's
        # persistent CCL buffers) and one eager VAE decode of a zero state (kernel compile, ~95 s on a cold
        # TT_METAL_CACHE; the second layout's decode only adds its state gather).
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
    def vae_encode(self) -> Flux3KleinVaeEncode:
        return self._vae_encode

    @property
    def null_text(self) -> TextPair:
        """The empty prompt's encoding ``((80, 20480) bf16, ids)``, computed once (the primary layout's)."""
        return self._null_text

    @property
    def layouts(self) -> tuple[str, ...]:
        """The resident DiT layouts, primary first (``"sp4tp8"``, then ``"sp8tp4"`` when ``large_layout`` is set)."""
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

    def denoisers_of(self, layout: str) -> dict[str, Flux3KleinDenoiser]:
        return self._layouts[layout].denoisers

    # --- internals -------------------------------------------------------------------------------------------------------

    def _build_layout(self, sp_axis: int, tp_axis: int, shared: Path) -> _Layout:
        """The transformers (weights loaded), denoisers, prompt encoder wrapper and decode helper of one layout."""
        name = layout_name(sp_axis, tp_axis, self._shape)
        ccl = CCLManager(mesh_device=self._device, num_links=self._num_links, topology=self._topology)
        parallel = DiTGParallelConfigNoCFG(
            tensor_parallel=ParallelFactor(factor=self._shape[tp_axis], mesh_axis=tp_axis),
            sequence_parallel=ParallelFactor(factor=self._shape[sp_axis], mesh_axis=sp_axis),
        )
        transformers: dict[str, Flux3KleinTransformer] = {}
        denoisers: dict[str, Flux3KleinDenoiser] = {}
        for variant in self._resident:
            t0 = time.time()
            from ...layers import linear as _linear

            _linear.DTYPE_OVERRIDE, _linear.FIDELITY_OVERRIDE = _dit_weight_override()
            if _linear.DTYPE_OVERRIDE is not None:
                logger.warning(f"MEASUREMENT PROFILE: DiT linears {_linear.DTYPE_OVERRIDE} fidelity {_linear.FIDELITY_OVERRIDE} (FLUX3_KLEIN_DIT_WEIGHTS)")
            try:
                model = Flux3KleinTransformer(
                    flow_map=variant == "distilled", device=self._device, ccl_manager=ccl, parallel_config=parallel
                )
            finally:
                _linear.DTYPE_OVERRIDE = _linear.FIDELITY_OVERRIDE = None
            load_flux3_klein_transformer_weights(
                model, variant, self._snapshots[variant], parallel_config=parallel, mesh_device=self._device
            )
            transformers[variant] = model
            denoisers[variant] = Flux3KleinDenoiser(
                model, device=self._device, sp_axis=sp_axis, max_live_traces=self._max_live_traces
            )
            self.timings_init[f"load_{variant}_{name}_s"] = time.time() - t0
            logger.info(f"RECORD after transformer[{variant}] {name}: {dram_per_bank(self._device)}")
        prompt_encoder = Flux3KleinPromptEncoder(
            self._encoder, shared, self._device, text_config=self._text_config, output_sp_axis=sp_axis
        )
        vae = Flux3KleinVaeDecode(
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
            torch.zeros(1, rows, FLUX3_KLEIN_IN_CHANNELS["image"]),
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

        tokens = self._vae_encode.encode(Image.new("RGB", (width, height)))
        assert tuple(tokens.shape) == (latent_shape(height, width)[0] * latent_shape(height, width)[1], 128)
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

    def text_length(self, prompt: str) -> int:
        return self._prompt_encoder.tokenize(prompt)[2]

    # --- the request -----------------------------------------------------------------------------------------------------

    def generate(self, request: Flux3KleinRequest, *, traced: bool = True) -> Flux3KleinResult:
        t_total = time.time()
        resolved = request.resolve()
        if resolved.model not in self._denoisers:
            raise ValueError(f"model {resolved.model!r} is not resident (resident: {self._resident})")
        # 0. the caption (generate.py:52-63): the upsampler runs on its own chip before any Klein device work.
        t0 = time.time()
        caption, upsampled = resolve_caption(resolved, self.upsampler)
        upsample_s = time.time() - t0
        if resolved.text_override is not None:
            text_length = int(resolved.text_override[0][0].shape[0])
        else:
            text_length = self.text_length(caption)
        # generate.py:73: an unseeded request hashes the caption (after the upsampler), as upstream's CLI.
        seed = resolved.seed if resolved.seed is not None else caption_seed(caption)
        resolved = replace(resolved, text_length=text_length, seed=seed)
        admission = admit(resolved, self._shape)
        if not admission.ok:
            # Nothing has touched the device for this request: no upload, no encode, no slot allocation.
            raise Flux3KleinAdmissionError(admission)

        h, w = latent_shape(resolved.height, resolved.width)
        noise = make_noise(resolved.seed, h, w)
        ids = image_ids(h, w)
        prepared_refs = resolved.prepared

        # 2b. the DiT layout for this request's row count (image + reference + text rows; :func:`pick_layout`).
        cond_rows = prepared_refs.cond_tokens() if prepared_refs is not None and prepared_refs.images else 0
        layout = self._layouts[self.layout_for(h * w + cond_rows + text_length)]

        # 3. encode text (sharded on the layout's SP axis)
        t0 = time.time()
        if resolved.text_override is not None:
            text, null = resolved.text_override
        else:
            text = layout.prompt_encoder.encode_host(caption)
            null = None if resolved.flow_map else layout.null_text
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

        denoise_request = Flux3KleinDenoiseRequest(
            noise=noise,
            image_ids=ids,
            text=text,
            num_steps=resolved.steps,
            guidance=resolved.guidance,
            text_null=None if resolved.flow_map else null,
            image_cond=image_cond,
            alpha=resolved.alpha,
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
        t0 = time.time()
        prepared: Flux3KleinPreparedRequest = denoiser.prepare(denoise_request)
        prepare_s = time.time() - t0
        t0 = time.time()
        if not vae_captured_before:
            assert not self.traces_live(), "a VAE capture must precede every DiT trace (aliasing rule)"
            self._warm_vae(resolved.height, resolved.width, traced=True, layout=layout)
        vae_capture_s = time.time() - t0
        captured_before = prepared.key in denoiser.captured_keys
        from .cache_flux3_klein import Flux3KleinCache, Flux3KleinCacheConfig

        cache_config = Flux3KleinCacheConfig.from_spec(request.cache)
        cache_stats = None
        denoiser._transformer.cache = Flux3KleinCache(cache_config, self._device) if cache_config is not None else None
        t0 = time.time()
        try:
            state = denoiser.denoise_on_device(prepared, traced=traced)
        finally:
            if denoiser._transformer.cache is not None:
                cache_stats = denoiser._transformer.cache.stats
                denoiser._transformer.cache.close()
                denoiser._transformer.cache = None
        denoise_s = time.time() - t0
        if cache_stats is not None:
            logger.info(
                f"RECORD dbcache {cache_config.tag()}: cached {len(cache_stats.cached_steps)}/{cache_stats.steps} steps "
                f"{cache_stats.cached_steps}; diffs {[(i, round(d, 4)) for i, d in cache_stats.diffs]}"
            )

        # 6. decode (a replay when traced: the trace was captured above, before the DiT trace)
        t0 = time.time()
        out = vae.decode(state, num_image_rows=h * w, height=resolved.height, width=resolved.width, traced=traced)
        ttnn.synchronize_device(self._device)
        decode_s = time.time() - t0

        # 6. host. ``out`` is replicated on every chip (the decoder ends with its spatial all-gather); the legacy
        # ``tensor.to_torch`` read ALL 32 replicas in TILE layout and untilized each on the host (2.1 s at 1024x1024,
        # 8.4 s at 2048x2048, sweep 2026-10-20). "fast": the uint8 conversion runs on the device and ONE replica is
        # read (``ttnn.get_device_tensors(x)[0]``, the pattern of the Flux2 prompt encoder / Ideogram decode).
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
            img = tensor.to_torch(out)[0].permute(2, 0, 1).float()
            if not traced:
                ttnn.deallocate(out)
            image = to_pil(img)
        if resolved.crop is not None:
            image = image.crop(resolved.crop)  # generate.py:76-77
        final_tokens = denoiser.read_state(prepared) if request.return_final_tokens else None
        to_host_s = time.time() - t0

        timings = {
            "upsample_s": upsample_s,
            "encode_s": encode_s,
            "encode_refs_s": encode_refs_s,
            "prepare_s": prepare_s,
            "vae_capture_s": vae_capture_s,
            "denoise_s": denoise_s,
            "cached_steps": float(len(cache_stats.cached_steps)) if cache_stats is not None else 0.0,
            "decode_s": decode_s,
            "to_host_s": to_host_s,
            "total_s": time.time() - t_total,
            "trace_captured": bool(traced and not captured_before),
            "vae_trace_captured": bool(traced and not vae_captured_before),
        }
        references = prepared_refs.sizes if prepared_refs is not None else ()
        logger.info(
            f"RECORD generate[{resolved.model}] {layout.name} {resolved.width}x{resolved.height} seed {resolved.seed} steps "
            f"{resolved.steps} guidance {resolved.guidance} upsampled {upsampled} caption L {text_length} references "
            f"{list(references)} cond rows {0 if image_cond is None else int(image_cond[0].shape[0])} crop "
            f"{resolved.crop}: upsample {upsample_s:.3f}s encode "
            f"{encode_s:.3f}s encode_refs {encode_refs_s:.3f}s prepare {prepare_s:.3f}s vae_capture {vae_capture_s:.3f}s "
            f"denoise {denoise_s:.3f}s decode {decode_s:.3f}s to_host {to_host_s:.3f}s total {timings['total_s']:.3f}s "
            f"(denoise trace captured {timings['trace_captured']}, vae trace captured {timings['vae_trace_captured']}); "
            f"{dram_per_bank(self._device)}"
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
            image_cond=image_cond,
            references=tuple(references),
            crop=resolved.crop,
            input_prompt=resolved.prompt,
            caption=caption,
            upsampled=upsampled,
            boxes=resolved.boxes,
            layout=layout.name,
        )

    def close(self) -> None:
        """Traces, then slots, then weights; the caller closes the mesh."""
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
    "MODELS",
    "Flux3KleinAdmissionError",
    "Flux3KleinPipeline",
    "Flux3KleinRequest",
    "Flux3KleinResolved",
    "Flux3KleinResult",
    "LARGE_LAYOUT_MIN_TOKENS",
    "layout_name",
    "pick_layout",
    "load_flux3_klein_transformer_weights",
    "make_noise",
    "noise_sha256",
    "caption_seed",
    "resolve_caption",
    "to_pil",
    "transformer_cache_key",
]
