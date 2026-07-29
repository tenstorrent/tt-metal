# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Bytes streamed per unit of work, computed ANALYTICALLY from the checkpoint.

The roofline ceiling is peak_DRAM_bandwidth / bytes-per-unit-of-work. Getting the numerator right is
the whole problem, and two earlier attempts were both wrong:

    checkpoint file size    counts the dtype the weights are STORED in. Llama-3.1-8B is 16.06 GB of
                            bf16 on disk but streams 6.09 GB as bfp4/bfp8 on device -- a ceiling of
                            31.9 instead of 84.0 tok/s/u.

    profile per-op bytes    measured, but there is no reliable per-unit divisor: the call counts in
                            one window implied 51 tokens from the FFN matmuls, 25 from QKV and 376
                            from the LM head. Dividing by a guessed iteration count produced a figure
                            that happened to look right once, which is worse than being obviously
                            wrong.

Here every tensor's shape and dtype is read from the safetensors header -- exact, no architecture
formulas, no per-family shape arithmetic, works for any checkpoint -- and the on-device width is
applied per tensor by name pattern. Nothing is inferred from the model's identity.

THE UNIT MATTERS AS MUCH AS THE BYTES. `peak_BW / weight_bytes` is only a rate if the whole weight
set is read once per unit and the work is memory-bound. That holds for an autoregressive token, a
diffusion step and a single forward pass -- three different units, one formula. It does NOT hold for
prefill or large-batch work (compute-bound), for long-sequence encoders (activations dominate), or
for a mixed multimodal pipeline (each stage has its own bound). Those get no ceiling from here, and
the caller falls back to the per-op roofline floor, which picks the binding term per op.
"""
from __future__ import annotations

import json
import os
import re
import struct
from pathlib import Path

# Bytes per element as STORED in a checkpoint. TT block formats are not checkpoint dtypes -- they are
# what the device reads -- so they live in _DEVICE_WIDTHS below.
_STORED_WIDTHS = {
    "F64": 8.0,
    "I64": 8.0,
    "F32": 4.0,
    "I32": 4.0,
    "U32": 4.0,
    "BF16": 2.0,
    "F16": 2.0,
    "I16": 2.0,
    "U16": 2.0,
    "F8_E4M3": 1.0,
    "F8_E5M2": 1.0,
    "I8": 1.0,
    "U8": 1.0,
    "BOOL": 1.0,
}

# TT block float widths: one shared exponent per 16 elements, so 8 -> 8.5 bits, 4 -> 4.5 bits.
_DEVICE_WIDTHS = {
    "bfloat16": 2.0,
    "bf16": 2.0,
    "float32": 4.0,
    "bfloat8_b": (8 + 8 / 16) / 8,
    "bfp8": (8 + 8 / 16) / 8,
    "bfloat4_b": (4 + 8 / 16) / 8,
    "bfp4": (4 + 8 / 16) / 8,
}

# HF pipeline_tag -> the unit of work whose cost the ceiling describes. Keyed on the unit rather than
# on a model-type taxonomy: HF publishes 47 tags and adds more, so a category list goes stale (the
# planner's map is missing 19 of them today). A tag absent here yields no unit, and no ceiling, which
# is the safe direction -- a wrong ceiling reads as a target and can stop a run early.
_UNIT_BY_TAG = {
    # autoregressive decode: one token reads every weight
    "text-generation": "token",
    "text2text-generation": "token",
    "summarization": "token",
    "translation": "token",
    "conversational": "token",
    "image-text-to-text": "token",
    "video-text-to-text": "token",
    "audio-text-to-text": "token",
    "visual-question-answering": "token",
    "document-question-answering": "token",
    "image-to-text": "token",
    "any-to-any": "token",
    "text-to-speech": "token",
    "text-to-audio": "token",
    "text-to-music": "token",
    "music-generation": "token",
    "automatic-speech-recognition": "token",
    # iterative denoising: one step reads every weight
    "text-to-image": "step",
    "image-to-image": "step",
    "image-text-to-image": "step",
    "unconditional-image-generation": "step",
    "text-to-video": "step",
    "image-to-video": "step",
    "image-text-to-video": "step",
    "video-to-video": "step",
    "text-to-3d": "step",
    "image-to-3d": "step",
    # single forward pass
    "feature-extraction": "inference",
    "image-feature-extraction": "inference",
    "sentence-similarity": "inference",
    "text-classification": "inference",
    "token-classification": "inference",
    "text-ranking": "inference",
    "zero-shot-classification": "inference",
    "fill-mask": "inference",
    "question-answering": "inference",
    "image-classification": "inference",
    "object-detection": "inference",
    "image-segmentation": "inference",
    "depth-estimation": "inference",
    "keypoint-detection": "inference",
    "mask-generation": "inference",
    "zero-shot-image-classification": "inference",
    "zero-shot-object-detection": "inference",
    "video-classification": "inference",
    "visual-document-retrieval": "inference",
    "audio-classification": "inference",
    # A source-separation / speech-enhancement model runs one forward pass over an audio segment --
    # a perfectly well-defined unit. It was missing, so an audio-to-audio model got no ceiling at all.
    "audio-to-audio": "inference",
    # TAPAS-style table QA is an encoder forward pass over the flattened table + question.
    "table-question-answering": "inference",
}

# Tags with NO defined unit of work, listed deliberately so the set cannot grow by accident:
#   reinforcement-learning   a policy rollout has no fixed weight-read-per-unit; the episode length is
#                            the workload and it is not a model property.
#   tabular-classification   these are gradient-boosted trees / sklearn estimators far more often than
#   tabular-regression       neural nets, so there is no weight stream to bound at all.
# Anything here yields no unit, no conditions and no ceiling -- the same safe direction as everywhere
# else in this module: publish nothing rather than a number that reads like a target.
NO_UNIT_TAGS = ("reinforcement-learning", "tabular-classification", "tabular-regression")

_UNIT_LABEL = {"token": "tok/s/u", "step": "steps/s", "inference": "inferences/s"}

# Tensors an autoregressive step does NOT stream in full: a token reads ONE embedding row (a few KB),
# not the table. The output projection is read in full and is deliberately absent from this list.
_LOOKUP_ONLY = re.compile(r"(^|\.)(embed_tokens|wte|word_embeddings|token_embedding|embeddings?\.weight$)", re.I)


# HF class-name suffixes -> unit. config.json carries `architectures` for every model but usually NOT
# a pipeline_tag (that lives on the model card), so keying only on the tag meant the analytic path
# never fired for a local checkpoint and silently fell back to the file size -- a 2.4x wrong ceiling.
# Longest suffix first: ForConditionalGeneration must beat ForCausalLM-style generic matching.
_UNIT_BY_ARCH_SUFFIX = (
    ("forconditionalgeneration", "token"),
    ("forcausallm", "token"),
    ("lmheadmodel", "token"),
    ("forspeechseq2seq", "token"),
    ("forvision2seq", "token"),
    ("fortexttospeech", "token"),
    ("forsequenceclassification", "inference"),
    ("fortokenclassification", "inference"),
    ("forquestionanswering", "inference"),
    ("forimageclassification", "inference"),
    ("forobjectdetection", "inference"),
    ("forsemanticsegmentation", "inference"),
    ("forimagesegmentation", "inference"),
    ("fordepthestimation", "inference"),
    ("formaskedlm", "inference"),
    ("formaskedimagemodeling", "inference"),
    ("unet2dconditionmodel", "step"),
    ("unet2dmodel", "step"),
    ("transformer2dmodel", "step"),
    ("dittransformer2dmodel", "step"),
    ("fluxtransformer2dmodel", "step"),
)


def unit_for_architectures(architectures, model_type: str = "") -> str:
    """The unit of work implied by an HF `architectures` entry, or "".

    A class name states the head, and the head states what one unit of work is: a CausalLM emits a
    token, a UNet takes a denoise step, a SequenceClassification does one forward pass. Unrecognised
    heads return "" so the caller publishes no ceiling rather than assuming decode.
    """
    for arch in list(architectures or []) + ([model_type] if model_type else []):
        a = str(arch or "").strip().lower()
        if not a:
            continue
        for suffix, unit in _UNIT_BY_ARCH_SUFFIX:
            if a.endswith(suffix) or suffix in a:
                return unit
    return ""


def unit_from_config(cfg: dict) -> str:
    """The unit for an HF config: its pipeline_tag when present, else its architecture head."""
    cfg = cfg if isinstance(cfg, dict) else {}
    return unit_for_tag(cfg.get("pipeline_tag") or "") or unit_for_architectures(
        cfg.get("architectures"), cfg.get("model_type") or ""
    )


def unit_for_tag(pipeline_tag: str) -> str:
    """The unit of work for an HF pipeline tag, or "" when there is no single well-defined one."""
    return _UNIT_BY_TAG.get(str(pipeline_tag or "").strip().lower(), "")


def unit_label(unit: str) -> str:
    """How the rate reads in the report: tok/s/u, steps/s, inferences/s."""
    return _UNIT_LABEL.get(str(unit or "").strip().lower(), "")


def _headers(path: Path):
    """{tensor_name: {dtype, shape}} from a safetensors file, reading only its header."""
    with path.open("rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        if n <= 0 or n > 200_000_000:
            return {}
        hdr = json.loads(fh.read(n))
    return {k: v for k, v in hdr.items() if k != "__metadata__" and isinstance(v, dict)}


def _numel(shape) -> int:
    total = 1
    for d in shape or []:
        total *= int(d)
    return total if shape else 0


def device_width(dtype_name) -> float | None:
    """Bytes per element for a TT dtype name, or None when unrecognised."""
    return _DEVICE_WIDTHS.get(str(dtype_name or "").strip().lower())


def weight_bytes(
    snapshot_dir,
    *,
    unit: str = "token",
    overrides=(),
    default_device_dtype: str = "",
) -> dict:
    """Bytes streamed per unit of work, per tensor, from the checkpoint's own headers.

    ``overrides`` is a sequence of (name_regex, tt_dtype) applied in order -- how a build states that
    it serves a tensor group narrower than the checkpoint stores it, e.g. FF1/FF3 at bfloat4_b. Any
    tensor no pattern matches keeps its stored width (or ``default_device_dtype`` when given), so the
    result is never better than what is actually known.

    Returns {bytes, tensors, skipped_lookup_bytes, by_pattern, shards} or {} when nothing was read.
    """
    d = Path(snapshot_dir or "")
    files = sorted(d.glob("*.safetensors")) if d.is_dir() else []
    if not files:
        return {}
    compiled = [(re.compile(pat), dt) for pat, dt in (overrides or ())]
    dflt = device_width(default_device_dtype)

    total, skipped, count = 0.0, 0.0, 0
    # Params on the SAME read set as `total` (lookup-only tensors excluded when unit=token). The
    # params-based ceiling needs this, and unlike bytes a param count does not depend on the width
    # the device serves -- so it costs no per-model investigation.
    params = 0
    by_pattern: dict = {}
    for f in files:
        try:
            hdr = _headers(f)
        except Exception:  # noqa: BLE001
            continue
        for name, meta in hdr.items():
            n = _numel(meta.get("shape"))
            if n <= 0:
                continue
            width = None
            key = "stored:%s" % meta.get("dtype")
            for rx, dt in compiled:
                if rx.search(name):
                    w = device_width(dt)
                    if w is not None:
                        width, key = w, "%s:%s" % (rx.pattern, dt)
                    break
            if width is None:
                width = dflt if dflt is not None else _STORED_WIDTHS.get(str(meta.get("dtype")), 2.0)
            b = n * width
            count += 1
            if unit == "token" and _LOOKUP_ONLY.search(name):
                skipped += b
                continue
            total += b
            params += n
            e = by_pattern.setdefault(key, {"bytes": 0.0, "tensors": 0})
            e["bytes"] += b
            e["tensors"] += 1
    if total <= 0:
        return {}
    return {
        "bytes": int(round(total)),
        "params": int(params),
        "tensors": count,
        "skipped_lookup_bytes": int(round(skipped)),
        "by_pattern": by_pattern,
        "shards": len(files),
        "unit": unit,
    }


# DEFAULT MEASUREMENT CONDITIONS, keyed on the unit of work rather than on a model family.
#
# The distinction that decides these: a condition the model's own config STATES is read, never
# defaulted; a condition it does NOT state needs a tool default, because otherwise whoever writes the
# perf test picks one and nobody records it.
#
#   token      ISL and OSL are runtime choices, absent from every config.json -- which is exactly how
#              a generated test ended up on a six-token prompt. 128 in / 128 out is the standard
#              short-context benchmark point, so that is the fallback.
#   step       50 denoise steps -- diffusers' own documented default for
#              `StableDiffusionPipeline.__call__(num_inference_steps: int = 50)`, so the number has a
#              citable source rather than being a round figure someone liked. The RATE is per step, so
#              the count only bounds how long the measurement runs.
#   inference  one forward pass at batch 1. Input size is a model property -- an image processor's
#              `image_size` (ViT: 224) and a feature extractor's `chunk_length` (Whisper: 30 s) are
#              read from the config, never defaulted. Only the TEXT case has no such property:
#              `max_position_embeddings` is a cap, not a workload, and HF pipelines pad to the batch,
#              so there is no HF number to inherit. 384 is MLPerf's BERT inference sequence length --
#              a published reference, chosen over a figure picked for internal consistency.
_DEFAULT_CONDITIONS = {
    "token": {"isl": 128, "osl": 128, "batch": 1},
    "step": {"steps": 50, "batch": 1},
    "inference": {"batch": 1, "seq_len": 384},
}


def default_conditions(unit: str, cfg: dict = None) -> dict:
    """The conditions a perf measurement should default to for this unit of work.

    Anything the config states wins over the fallback: `sample_size` fixes a diffusion model's
    resolution, `max_position_embeddings` caps a text model's sequence length. Returns {} for a unit
    with no defined unit of work -- the same safe direction as the ceiling, where no unit means no
    published number rather than an invented one.
    """
    unit = str(unit or "").strip().lower()
    base = dict(_DEFAULT_CONDITIONS.get(unit) or {})
    if not base:
        return {}
    cfg = cfg if isinstance(cfg, dict) else {}
    # RESOLUTION. A UNet's `sample_size` is the LATENT size, not pixels: diffusers documents
    # height as `unet.config.sample_size * vae_scale_factor`, so SD-1.5's sample_size=64 is a
    # 512px image. Reporting 64px would understate the workload by 8x per side. Only convert when
    # the scale factor is actually known -- otherwise say latent, because a guessed multiplier is
    # how a plausible-looking wrong number gets into a report.
    latent = cfg.get("sample_size")
    pixels = (cfg.get("vision_config") or {}).get("image_size") or cfg.get("image_size")
    scale = cfg.get("vae_scale_factor") or (cfg.get("vae_config") or {}).get("scale_factor")
    if unit in ("step", "inference"):
        if isinstance(pixels, (int, float)) and pixels > 0:
            base["resolution"] = int(pixels)
        elif isinstance(latent, (int, float)) and latent > 0:
            if isinstance(scale, (int, float)) and scale > 0:
                base["resolution"] = int(latent * scale)
            else:
                base["latent"] = int(latent)
    if "resolution" in base or "latent" in base:
        # seq_len is the TEXT fallback for a single forward pass; a model that states an image size is
        # not a text model, and reporting both would describe a workload that does not exist.
        base.pop("seq_len", None)
    # AUDIO. The workload for an audio model is a DURATION -- reporting "seq_len 128" for a speech
    # enhancer describes nothing. Read it from the feature extractor's own config (Whisper carries
    # chunk_length=30); when absent, publish no duration rather than invent one, since a segment
    # length is a preprocessing choice and not something to guess.
    secs = cfg.get("chunk_length_s") or cfg.get("chunk_length") or cfg.get("max_length_s")
    if isinstance(secs, (int, float)) and secs > 0 and unit == "inference":
        base["seconds"] = float(secs)
        base.pop("seq_len", None)
    cap = cfg.get("max_position_embeddings")
    if isinstance(cap, (int, float)) and cap > 0:
        for k in ("isl", "osl", "seq_len"):
            if k in base and base[k] > cap:
                base[k] = int(cap)
    return base


def conditions_label(conds: dict) -> str:
    """How the conditions read in a report: "ISL 128 / OSL 128, batch 1"."""
    c = conds if isinstance(conds, dict) else {}
    parts = []
    if "isl" in c:
        parts.append("ISL %d" % c["isl"])
    if "osl" in c:
        parts.append("OSL %d" % c["osl"])
    if "steps" in c:
        parts.append("%d steps" % c["steps"])
    if "seq_len" in c and "isl" not in c:
        parts.append("seq_len %d" % c["seq_len"])
    if "resolution" in c:
        parts.append("%dpx" % c["resolution"])
    if "latent" in c:
        parts.append("latent %d" % c["latent"])
    if "seconds" in c:
        parts.append("%gs audio" % c["seconds"])
    if "batch" in c:
        parts.append("batch %d" % c["batch"])
    return ", ".join(parts)


def parse_overrides(spec: str):
    """``"pattern=dtype,pattern=dtype"`` -> the overrides sequence. Malformed entries are skipped
    rather than guessed at: a wrong width silently moves the ceiling."""
    out = []
    for part in str(spec or "").split(","):
        part = part.strip()
        if not part or "=" not in part:
            continue
        pat, _, dt = part.partition("=")
        pat, dt = pat.strip(), dt.strip()
        if not pat or device_width(dt) is None:
            continue
        try:
            re.compile(pat)
        except re.error:
            continue
        out.append((pat, dt))
    return out


def overrides_from_env() -> list:
    """TT_PERF_WEIGHT_DTYPES, e.g. "gate_proj|up_proj=bfloat4_b,down_proj=bfloat4_b"."""
    return parse_overrides(os.environ.get("TT_PERF_WEIGHT_DTYPES", ""))
