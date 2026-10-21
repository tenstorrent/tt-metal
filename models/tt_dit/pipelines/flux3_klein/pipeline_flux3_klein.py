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
    untilize."""
    x = ttnn.typecast(image, ttnn.float32)
    x = ttnn.add(x, 1.0)
    x = ttnn.multiply(x, 127.5)
    x = ttnn.clamp(x, min=0.0, max=255.0)
    x = ttnn.floor(x)
    x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
    return ttnn.typecast(x, ttnn.uint8)


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
        upsampler=None,
    ) -> None:
        if not resident:
            raise ValueError("at least one resident variant is required")
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

        # Warm-ups BEFORE any trace exists: the empty prompt (base CFG; also allocates the encoder's persistent CCL
        # buffers) and one eager VAE decode of a zero state (kernel compile, ~95 s on a cold TT_METAL_CACHE).
        t0 = time.time()
        self._null_text: TextPair = self._prompt_encoder.encode_host("")
        self.timings_init["encode_null_s"] = time.time() - t0
        if warmup_size is not None:
            t0 = time.time()
            self._warm_vae(*warmup_size)
            self.timings_init["vae_warmup_s"] = time.time() - t0
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
        """The empty prompt's encoding ``((80, 20480) bf16, ids)``, computed once."""
        return self._null_text

    # --- internals -------------------------------------------------------------------------------------------------------

    def _warm_vae(self, height: int, width: int, *, traced: bool = False) -> None:
        """One decode of a zero state at ``height x width``: eager (``__init__``) compiles the kernels and allocates
        the gather buffers; traced (``generate``, before the DiT capture of a layout) CAPTURES the decoder trace of
        the size, so its captured input buffer and the VAE CCL buffers exist before any DiT trace (module docstring).
        The traced output is the trace's buffer (never deallocated here)."""
        h, w = latent_shape(height, width)
        rows = padded_length(h * w, self._shape[self._sp_axis])
        zero = tensor.from_torch(
            torch.zeros(1, rows, FLUX3_KLEIN_IN_CHANNELS["image"]),
            device=self._device,
            dtype=ttnn.float32,
            mesh_axes=[None, self._sp_axis, None],
        )
        out = self._vae.decode(zero, num_image_rows=h * w, height=height, width=width, traced=traced)
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
        return any(d.captured_keys for d in self._denoisers.values()) or self._vae.trace_captured()

    def release_traces(self) -> None:
        """Release every live trace (both denoisers and the VAE); each is recaptured on its next traced use."""
        for denoiser in self._denoisers.values():
            denoiser.release_traces()
        self._vae.release_trace()
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

        # 3. encode text
        t0 = time.time()
        if resolved.text_override is not None:
            text, null = resolved.text_override
        else:
            text = self._encode(caption)
            null = None if resolved.flow_map else self._null_text
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
        denoiser = self._denoisers[resolved.model]

        # 5. denoise (the aliasing rule of the module docstring before any long-lived allocation): slots first, then
        # the VAE trace of the canvas size, then the DiT trace (capture order; module docstring).
        needs_fallback_vae = not self._vae.partitioned(resolved.height, resolved.width) and self._vae._fallback is None
        vae_captured_before = (not traced) or self._vae.trace_captured_for(resolved.height, resolved.width)
        if (
            not denoiser.has_layout(denoise_request)
            or needs_fallback_vae
            or (not vae_captured_before and self.traces_live())
        ):
            self.release_traces()
            vae_captured_before = (not traced) or self._vae.trace_captured_for(resolved.height, resolved.width)
        t0 = time.time()
        prepared: Flux3KleinPreparedRequest = denoiser.prepare(denoise_request)
        prepare_s = time.time() - t0
        t0 = time.time()
        if not vae_captured_before:
            assert not self.traces_live(), "a VAE capture must precede every DiT trace (aliasing rule)"
            self._warm_vae(resolved.height, resolved.width, traced=True)
        vae_capture_s = time.time() - t0
        captured_before = prepared.key in denoiser.captured_keys
        t0 = time.time()
        state = denoiser.denoise_on_device(prepared, traced=traced)
        denoise_s = time.time() - t0

        # 6. decode (a replay when traced: the trace was captured above, before the DiT trace)
        t0 = time.time()
        out = self._vae.decode(state, num_image_rows=h * w, height=resolved.height, width=resolved.width, traced=traced)
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
            array = ttnn.to_torch(ttnn.get_device_tensors(u8)[0])[0].numpy()
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
            "decode_s": decode_s,
            "to_host_s": to_host_s,
            "total_s": time.time() - t_total,
            "trace_captured": bool(traced and not captured_before),
            "vae_trace_captured": bool(traced and not vae_captured_before),
        }
        references = prepared_refs.sizes if prepared_refs is not None else ()
        logger.info(
            f"RECORD generate[{resolved.model}] {resolved.width}x{resolved.height} seed {resolved.seed} steps "
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
        )

    def close(self) -> None:
        """Traces, then slots, then weights; the caller closes the mesh."""
        self.release_traces()
        for denoiser in self._denoisers.values():
            denoiser.deallocate()
        ttnn.synchronize_device(self._device)
        for module in (
            *self._transformers.values(),
            self._encoder,
            self._vae.decoder,
            self._vae._fallback,
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
    "load_flux3_klein_transformer_weights",
    "make_noise",
    "noise_sha256",
    "caption_seed",
    "resolve_caption",
    "to_pil",
    "transformer_cache_key",
]
