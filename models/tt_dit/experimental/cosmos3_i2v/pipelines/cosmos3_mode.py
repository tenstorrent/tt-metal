# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Data-shape-driven mode dispatch for the unified Cosmos3 pipeline.

A single native(-cfg) pipeline serves text2image / text2video / image2video /
audio_image2video. The mode is a function of the call arguments, not a separate
factory — `resolve_mode` reads (image, num_frames, enable_sound, action) and the
reference `Cosmos3OmniPipeline` already branches internally on the same signals
(`is_image = num_frames == 1`, `image is None` → zero vision conditioning).

`run_cosmos3` is the thin call-time seam that classifies the run, merges per-mode
defaults (`cosmos3_defaults/<mode>/`) including the recommended negative prompt,
and forwards to `pipe.__call__`. JSON-prompt parsing hangs off this seam later.
"""

from __future__ import annotations

import inspect
import json
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from PIL.Image import Image

_DEFAULTS_DIR = Path(__file__).parent / "cosmos3_defaults"


class ModelMode(str, Enum):
    TEXT2IMAGE = "text2image"
    TEXT2VIDEO = "text2video"
    IMAGE2VIDEO = "image2video"
    AUDIO_IMAGE2VIDEO = "audio_image2video"


# Modes wired through the native device trunk today. Audio (Phase 5) resolves
# correctly but is not yet runnable on device: the device proj_out
# `_native_forward` raises on sound tokens. run_cosmos3 rejects it here so a
# mis-shaped call fails on host with a clear message instead of deep in the trunk.
_SUPPORTED_MODES = frozenset({ModelMode.TEXT2IMAGE, ModelMode.TEXT2VIDEO, ModelMode.IMAGE2VIDEO})


def resolve_mode(
    image: "Image | Any | None" = None,
    num_frames: int | None = None,
    enable_sound: bool = False,
    action: Any | None = None,
) -> ModelMode:
    """Classify a run from its conditioning shape.

    `num_frames is None` means "unspecified" — the reference defaults it to 189,
    so it counts as video, not a single frame. `image` is ignored for the
    single-frame case (the reference drops it when `num_frames == 1`).
    """
    if action is not None:
        raise NotImplementedError("action-conditioned modes are out of scope for the unified pipeline")
    if enable_sound:
        return ModelMode.AUDIO_IMAGE2VIDEO
    is_single_frame = num_frames == 1
    if image is None:
        return ModelMode.TEXT2IMAGE if is_single_frame else ModelMode.TEXT2VIDEO
    return ModelMode.TEXT2IMAGE if is_single_frame else ModelMode.IMAGE2VIDEO


@lru_cache(maxsize=None)
def load_modality_defaults(mode: ModelMode) -> dict[str, Any]:
    """Load the per-mode defaults from `cosmos3_defaults/<mode>/`.

    Returns `{"sample_args": {...}, "negative_prompt": str}`. `sample_args` keys
    are reference `Cosmos3OmniPipeline.__call__` parameter names; the recommended
    negative prompt is the NVIDIA quality-control string for that modality (empty
    for text2image, which ships no default negative).
    """
    mode_dir = _DEFAULTS_DIR / mode.value
    if not mode_dir.is_dir():
        raise FileNotFoundError(f"no defaults for mode {mode.value!r} at {mode_dir}")
    sample_args = json.loads((mode_dir / "sample_args.json").read_text())
    negative_prompt = json.loads((mode_dir / "neg_prompts.json").read_text())["negative_prompt"]
    return {"sample_args": sample_args, "negative_prompt": negative_prompt}


def _call_param_names(pipe) -> set[str]:
    """Named parameters accepted by the pipeline's real `__call__`.

    The cfg pipeline wraps `__call__` as `(*args, **kwargs)` and forwards to
    `Cosmos3OmniPipeline.__call__`; that wrapper would accept anything, so walk
    the MRO past any var-keyword forwarder to the first concrete signature. Used
    to drop merged defaults the pipeline can't take (e.g. the build-time
    `flow_shift`) before they reach `pipe(...)` as a `TypeError`.
    """
    for klass in type(pipe).__mro__:
        call = klass.__dict__.get("__call__")
        if call is None:
            continue
        params = inspect.signature(call).parameters.values()
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params):
            continue
        return {p.name for p in params if p.name != "self" and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)}
    raise TypeError(f"{type(pipe).__name__} exposes no concrete __call__ signature to filter against")


def run_cosmos3(
    pipe,
    *,
    prompt: str,
    image: "Image | Any | None" = None,
    num_frames: int | None = None,
    enable_sound: bool = False,
    negative_prompt: str | None = None,
    action: Any | None = None,
    mode: ModelMode | None = None,
    **call_kwargs,
):
    """Resolve the mode, merge per-mode defaults, gate unsupported modes, forward.

    A caller-supplied `mode` is validated against the data-derived mode rather
    than trusted — the conditioning shape is the source of truth, so a mismatch
    is a bug, not an override.

    Per-mode `sample_args` seed the call; explicit `call_kwargs` override them
    (caller wins). `negative_prompt=None` falls back to the mode's recommended
    default. The merged kwargs are filtered to the pipeline's real `__call__`
    signature so defaults carrying build-time knobs don't crash the forward.
    """
    resolved = resolve_mode(image=image, num_frames=num_frames, enable_sound=enable_sound, action=action)
    if mode is not None and mode != resolved:
        raise ValueError(f"mode={mode.value!r} contradicts the call arguments, which resolve to {resolved.value!r}")
    if resolved not in _SUPPORTED_MODES:
        raise NotImplementedError(f"mode {resolved.value!r} is not wired through the native trunk yet")

    defaults = load_modality_defaults(resolved)
    if negative_prompt is None:
        negative_prompt = defaults["negative_prompt"]

    merged = {**defaults["sample_args"], **call_kwargs}
    # Conditioning shape is source of truth: always set the resolved signals.
    merged["prompt"] = prompt
    merged["image"] = image
    merged["enable_sound"] = enable_sound
    merged["negative_prompt"] = negative_prompt
    if num_frames is not None:
        merged["num_frames"] = num_frames

    accepted = _call_param_names(pipe)
    return pipe(**{k: v for k, v in merged.items() if k in accepted})
