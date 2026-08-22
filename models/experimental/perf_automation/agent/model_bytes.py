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

# The output projection, under every name checkpoints give it. Its PRESENCE is what makes the input
# embedding lookup-only: with a separate head, a token reads one row of the table and streams the head
# in full. TIED checkpoints ship no head at all -- the table IS the projection -- so skipping it there
# deletes a tensor the step really does stream (1.007 B params on gemma-3-12b-it, which is why its read
# set came out 11.18 B instead of 11.77 B). Tying is the norm in Gemma, Qwen, Phi and the small Llamas,
# so this is not one model's quirk.
_OUTPUT_PROJ = re.compile(r"(^|\.)(lm_head|output|embed_out|score|proj_out)\.weight$", re.I)

# Encoders that run once per image or audio clip, never per generated token. gemma-3-12b-it carries 437
# such tensors, 0.411 B params, and every one of them was charged to every token. Anchored to a NAME
# COMPONENT (start of string or after a dot, then a dot) so a decoder weight can never be dropped for
# containing one of these words by accident -- losing a real streamed tensor is the worse failure.
_TOWER_ONLY = re.compile(
    r"(^|\.)(vision_tower|vision_model|vision_encoder|visual|image_encoder|image_tower"
    r"|audio_tower|audio_encoder|speech_encoder|multi_modal_projector|mm_projector)\.",
    re.I,
)


def resolution_from_config(cfg: dict):
    """The spatial size one unit of work is measured at, or None.

    emit-e2e already reads this to build its PCC input (e2e_emitter: vision_config.image_size ->
    torch.randn(1, 3, H, W)), but the value never reached the PERF side, so a steps/s or vision
    inferences/s figure could not state the resolution it described -- and resolution IS the work: a
    denoise step at 1024 is roughly 4x the step at 512, so two runs of one model differ ~4x with
    nothing in the report distinguishing them.

    Two shapes, because the two model families store it differently:
      * vision encoders publish `image_size` (in vision_config for a multimodal config), which is the
        PIXEL size fed to the tower;
      * latent diffusion publishes `sample_size`, the LATENT size, whose pixel size is
        sample_size * vae_scale_factor (8 for every SD-family VAE, so that is the default).

    None when neither is present, which is every text model -- and None must stay None rather than
    becoming a number, because a resolution printed for a model that has none is a claim about a
    condition that did not exist.
    """
    cfg = cfg if isinstance(cfg, dict) else {}
    vc = cfg.get("vision_config") if isinstance(cfg.get("vision_config"), dict) else {}
    for src in (vc, cfg):
        px = src.get("image_size")
        if isinstance(px, (int, float)) and px > 0:
            return int(px)
    latent = cfg.get("sample_size")
    if isinstance(latent, (int, float)) and latent > 0:
        scale = cfg.get("vae_scale_factor") or (cfg.get("vae_config") or {}).get("scale_factor") or 8
        try:
            return int(latent) * int(scale)
        except (TypeError, ValueError):
            return int(latent)
    return None


def unit_for_tag(pipeline_tag: str) -> str:
    """The unit of work for an HF pipeline tag, or "" when there is no single well-defined one.

    ITS ONLY REMAINING JOB is the lookup-only tensor exclusion in the param-count walk: a token unit
    reads its embedding table by INDEX, one row per token, so counting the whole table as streamed
    bytes overstates what a decode step moves. Nothing else consults it.

    IT NO LONGER FEEDS THE CEILING, and it no longer picks measurement conditions -- default_conditions
    and its table are deleted. They had no production caller: a run takes ISL/OSL/seq_len/batch from
    TT_PERF_* and its resolution from resolution_from_config, and the two disagreed anyway (384 vs the
    128 that TT_PERF_SEQ_LEN actually uses), which is what a second unused source of the same fact
    does.

    THE CEILING SIDE. A tag names the TASK and cannot state whether a model loops:
    `text-to-speech` covers XTTS, which emits tokens, and Kokoro-82M, which is StyleTTS2 and produces
    a whole waveform in one pass. One tag, two units, so the table had to pick and was wrong for the
    other -- and a wrong unit does not degrade a ceiling, it puts it in the wrong currency, taking the
    band, the at-floor verdict and the headline rate with it. The unit now comes from what the built
    pipeline actually does (perf_adapter.headline_unit), and unit_from_config / unit_for_architectures
    -- the class-name fallback -- are deleted rather than left as a second answer to the same
    question.
    """
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

    # ONE PASS FIRST, to learn whether a separate output projection exists. That fact decides whether
    # the embedding table is lookup-only or is itself the projection, and it cannot be known from the
    # embedding tensor alone.
    shards = []
    for f in files:
        try:
            shards.append(_headers(f))
        except Exception:  # noqa: BLE001
            continue
    tied = not any(_OUTPUT_PROJ.search(n) for hdr in shards for n in hdr)

    total, skipped, count = 0.0, 0.0, 0
    # Params on the SAME read set as `total` (lookup-only tensors excluded when unit=token). The
    # params-based ceiling needs this, and unlike bytes a param count does not depend on the width
    # the device serves -- so it costs no per-model investigation.
    params = 0
    by_pattern: dict = {}
    for hdr in shards:
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
            if unit == "token" and _TOWER_ONLY.search(name):
                # Not streamed per token at all: an encoder pass is per image/clip. Not "skipped
                # lookup" either -- that counter means "read one row of", which this is not.
                continue
            if unit == "token" and not tied and _LOOKUP_ONLY.search(name):
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


def untowered_sections(snapshot_dir, stage_roots: dict) -> list:
    """Sections a NON-recurring stage runs that _TOWER_ONLY did not exclude. Empty when consistent.

    DETECTION, NOT CORRECTION. _TOWER_ONLY keeps an encoder's parameters out of a per-token read set
    by matching names (`audio_tower`, `vision_tower`, ...). It is a good list and it is still a list:
    a model naming its encoder off-list has that encoder charged to every token, inflating the
    divisor, lowering the ceiling, and so making the run look closer to the wall than it is -- the
    direction that stops a run early.

    What this does NOT do is fix it by substituting stage_roots for the name list. That was tried and
    is worse: stage_roots names the section holding a stage's BLOCK STACK, and on any untied model
    `lm_head` is a sibling of that stack, streamed every token and named by no stage_roots entry --
    so excluding everything outside the map silently drops it (1.0 B params on gemma-3-12b-it). One
    silent error traded for a larger one in the opposite direction.

    So this only reports the disagreement, which needs no guess: stage_roots says a stage OTHER than
    the recurring one runs out of section X -- that is derived from the checkpoint's own key
    structure, not from anyone's naming -- while _TOWER_ONLY did not exclude X. Both statements are
    evidence, and when they conflict a human can see it instead of the divisor quietly moving.

    The honest fix is a measurement that does not exist yet: the profile records nothing about which
    phase an op ran in ("_top_ops keys on (op_code, shape, memory)"), so there is no observation of
    what a token actually read. Until buckets carry a stage, every route here is a guess, and the
    least-bad guess is the one whose failure mode is known.
    """
    try:
        _roots = {str(k).strip().lower(): str(v) for k, v in (stage_roots or {}).items() if v}
        if not _roots:
            return []
        _recurring = {v for k, v in _roots.items() if k in ("decode", "prefill")}
        _others = {v for k, v in _roots.items() if k not in ("decode", "prefill")} - _recurring
        if not _others:
            return []
        d = Path(snapshot_dir or "")
        files = sorted(d.glob("*.safetensors")) if d.is_dir() else []
        _present = set()
        for f in files:
            try:
                with open(f, "rb") as fh:
                    n = struct.unpack("<Q", fh.read(8))[0]
                    if 0 < n <= 200_000_000:
                        for k in json.loads(fh.read(n)):
                            if k != "__metadata__" and "." in str(k):
                                _present.add(str(k).split(".", 1)[0])
            except Exception:  # noqa: BLE001
                continue
        # A section only counts as missed if it is REALLY in this checkpoint and the list lets it
        # through: `_TOWER_ONLY` is anchored to a name component, so testing "<section>." is the same
        # question the byte walk asks of every tensor in it.
        return sorted(x for x in _others if x in _present and not _TOWER_ONLY.search(x + "."))
    except Exception:  # noqa: BLE001 -- a detector that raises must not cost a ceiling
        return []


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
