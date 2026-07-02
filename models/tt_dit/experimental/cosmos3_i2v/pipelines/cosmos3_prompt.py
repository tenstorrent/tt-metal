# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""JSON-object prompt handling for the unified Cosmos3 pipeline.

Cosmos3 was trained on structured "director's JSON" captions carrying the
generation specs inline (resolution / aspect_ratio / duration / fps), not just
free text. NVIDIA's inference path (`cosmos_framework/inference/inference.py`)
detects a JSON-object prompt and injects those specs into the object — the
actual generation specs are the source of truth and overwrite whatever the
input JSON claimed — then serializes it back for the tokenizer. Flat English
metadata sentences are appended only to free-text prompts; a JSON prompt already
carries the schema, so the flat templates are skipped for it.

`parse_json_object_prompt` and `format_json_prompt_with_template` mirror the
NVIDIA functions of the same shape byte-for-byte. `install_json_prompt_parsing`
wraps `pipe.tokenize_prompt` so the transform runs at the one point the pipeline
turns strings into tokens, covering both the native and native-cfg builders (the
cfg pipe reuses the native builder, so a single install point serves both).
"""

from __future__ import annotations

import json
from typing import Any

# Nearest-match bins for deriving aspect_ratio from the target canvas. NVIDIA
# passes aspect_ratio as a first-class sample arg; this pipeline's tokenize_prompt
# only carries height/width, so we recover the same string the payload specifies
# (1280x720 -> "16,9") from the resolved canvas.
_ASPECT_RATIO_BINS = ("1,1", "4,3", "3,4", "16,9", "9,16")


def parse_json_object_prompt(prompt: str) -> dict | None:
    """Return the parsed dict iff ``prompt`` is a JSON object string; else ``None``.

    JSON arrays / numbers / strings / nulls are NOT considered "JSON-object
    prompts" and return ``None`` so they continue down the free-text path.
    """
    try:
        obj = json.loads(prompt)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


def _derive_aspect_ratio(width: int, height: int) -> str:
    ratio = width / height if height > 0 else 1.0
    return min(
        _ASPECT_RATIO_BINS,
        key=lambda r: abs(int(r.split(",")[0]) / int(r.split(",")[1]) - ratio),
    )


def format_json_prompt_with_template(
    prompt_obj: dict,
    *,
    fps: float,
    num_frames: int,
    aspect_ratio: str | None,
    height: int,
    width: int,
    include_temporal_metadata: bool | None = None,
) -> str:
    """Inject the generation specs into a JSON-object prompt and serialize it.

    Mirrors NVIDIA's ``_format_json_prompt_with_template`` (``h``/``w`` there are
    ``height``/``width`` here): resolution ``{"H", "W"}`` and ``aspect_ratio`` for
    every sample, plus ``duration``/``fps`` for video only. Existing keys are
    overwritten — the generation specs, not the input JSON, are the source of
    truth. ``json.dumps`` uses default separators, matching the training-time
    schema the tokenizer expects (this is *not* the compact payload encoding).

    ``include_temporal_metadata`` defaults to ``num_frames > 1``; when False the
    stale ``duration``/``fps`` keys are dropped so a single-frame prompt carries
    no temporal specs.
    """
    if include_temporal_metadata is None:
        include_temporal_metadata = num_frames > 1

    metadata: dict[str, Any] = {}
    if include_temporal_metadata:
        duration_seconds = int(num_frames / fps) if fps > 0 else 0
        metadata.update({"duration": f"{duration_seconds}s", "fps": float(fps)})
    else:
        prompt_obj.pop("duration", None)
        prompt_obj.pop("fps", None)
    metadata["resolution"] = {"H": int(height), "W": int(width)}
    if aspect_ratio is not None:
        metadata["aspect_ratio"] = aspect_ratio

    prompt_obj.update(metadata)
    return json.dumps(prompt_obj)


def _append(base: str, addition: str) -> str:
    """Reference ``_append`` semantics: join metadata sentences with ``. ``."""
    base = base.rstrip(".")
    return f"{base}. {addition}" if base else addition


def _apply_inverse_flat_templates(
    pipe, text: str, *, is_image: bool, num_frames, height, width, fps, add_duration: bool, add_resolution: bool
) -> str:
    """Append the reference's inverse duration/resolution sentences to a negative.

    The JSON-positive path forces ``add_*_template=False`` to keep the flat
    English sentences off the serialized object, but that flag is shared with the
    negative branch inside ``tokenize_prompt``. Re-apply the negative's inverse
    templates here so the negative prompt is byte-identical whether or not the
    positive is JSON — honoring each flag independently, as the reference does.
    Uses the pipe's own template strings so any change to the vendored templates
    is tracked automatically.
    """
    if not is_image and add_duration:
        text = _append(text, pipe.inverse_duration_template.format(duration=num_frames / fps, fps=fps))
    if add_resolution:
        inverse_res = pipe.inverse_image_resolution_template if is_image else pipe.inverse_video_resolution_template
        text = _append(text, inverse_res.format(height=height, width=width))
    return text


def install_json_prompt_parsing(pipe):
    """Wrap ``pipe.tokenize_prompt`` to JSON-format object prompts before tokenizing.

    Idempotent. Free-text prompts and action-mode calls pass straight through to
    the original method. For a JSON-object positive prompt: inject the generation
    specs, drop the flat templates for the positive, and pre-apply the negative's
    inverse templates so the negative is unchanged.
    """
    orig = pipe.tokenize_prompt
    if getattr(orig, "_json_prompt_wrapped", False):
        return pipe

    def wrapped(prompt, negative_prompt=None, **kw):
        num_frames = kw.get("num_frames", 189)
        height = kw.get("height", 720)
        width = kw.get("width", 1280)
        fps = kw.get("fps", 24.0)

        # Action mode builds its own structured caption downstream; leave it be.
        if kw.get("action_mode") is None:
            prompt_obj = parse_json_object_prompt(prompt)
            if prompt_obj is not None:
                is_image = num_frames == 1
                prompt = format_json_prompt_with_template(
                    prompt_obj,
                    fps=fps,
                    num_frames=num_frames,
                    aspect_ratio=_derive_aspect_ratio(width, height),
                    height=height,
                    width=width,
                )
                add_duration = kw.get("add_duration_template", True)
                add_resolution = kw.get("add_resolution_template", True)
                negative_prompt = _apply_inverse_flat_templates(
                    pipe,
                    "" if negative_prompt is None else negative_prompt,
                    is_image=is_image,
                    num_frames=num_frames,
                    height=height,
                    width=width,
                    fps=fps,
                    add_duration=add_duration,
                    add_resolution=add_resolution,
                )
                kw = {**kw, "add_duration_template": False, "add_resolution_template": False}

        return orig(prompt, negative_prompt, **kw)

    wrapped._json_prompt_wrapped = True
    pipe.tokenize_prompt = wrapped
    return pipe
