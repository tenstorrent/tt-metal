# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Fuse LoRA deltas into LTX transformer state dicts (bf16 base weights only).

FP8 / scaled-FP8 base weights are intentionally unsupported — we do not use
them on device.

Supported LoRA conventions (normalized to ``<prefix>.lora_A/B.weight`` before
fusing):
- diffusers / PEFT ``<prefix>.lora_A.weight`` + ``<prefix>.lora_B.weight``,
  including the PEFT adapter-name infix ``<prefix>.lora_A.<adapter>.weight``.
- kohya ``<prefix>.lora_down.weight`` + ``<prefix>.lora_up.weight``.
- optional ``<prefix>.alpha`` rank scaling (``strength * alpha / rank``);
  alpha-less LoRAs keep the legacy ``×strength`` behavior.
- full-weight deltas ``<prefix>.diff`` (weight) and ``<prefix>.diff_b`` (bias).
Leading ``model.diffusion_model.`` / ``diffusion_model.`` / ``transformer.``
prefixes are stripped so the LoRA keyspace aligns with the base.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import torch
from loguru import logger
from safetensors.torch import load_file

# Tried longest-first; the first prefix any key carries is removed.
_LORA_PREFIXES = ("model.diffusion_model.", "diffusion_model.", "transformer.")
# PEFT writes ``<prefix>.lora_A.<adapter>.weight``; collapse the adapter infix.
_PEFT_ADAPTER = re.compile(r"\.lora_(A|B)\.[^.]+\.weight$")


@dataclass(frozen=True)
class LoraSpec:
    """A LoRA safetensors file with a fuse strength."""

    path: str
    strength: float = 1.0


def _strip_prefix(state_dict: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    """Return ``{k.removeprefix(prefix): v}`` if any key starts with ``prefix``, else input."""
    if any(k.startswith(prefix) for k in state_dict):
        return {k[len(prefix) :] if k.startswith(prefix) else k: v for k, v in state_dict.items()}
    return state_dict


def _normalize_lora_keys(lora_sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Strip the diffusion-model prefix and normalize LoRA key conventions.

    Maps kohya ``lora_down/up`` to ``lora_A/B`` and collapses the PEFT
    adapter-name infix, so every low-rank pair ends up as
    ``<prefix>.lora_A.weight`` / ``<prefix>.lora_B.weight`` (``.alpha`` /
    ``.diff`` / ``.diff_b`` keys pass through under ``<prefix>``).
    """
    for prefix in _LORA_PREFIXES:
        if any(k.startswith(prefix) for k in lora_sd):
            lora_sd = _strip_prefix(lora_sd, prefix)
            break

    normalized: dict[str, torch.Tensor] = {}
    for k, v in lora_sd.items():
        nk = _PEFT_ADAPTER.sub(lambda m: f".lora_{m.group(1)}.weight", k)
        nk = nk.replace(".lora_down.weight", ".lora_A.weight").replace(".lora_up.weight", ".lora_B.weight")
        normalized[nk] = v
    return normalized


def _weight_delta(lsd: dict[str, torch.Tensor], prefix: str, strength: float) -> torch.Tensor | None:
    """fp32 delta for base ``<prefix>.weight`` from this LoRA, or ``None`` if it
    carries no matching low-rank pair or ``.diff``."""
    a = lsd.get(f"{prefix}.lora_A.weight")
    b = lsd.get(f"{prefix}.lora_B.weight")
    diff = lsd.get(f"{prefix}.diff")
    if a is None and b is None and diff is None:
        return None

    delta: torch.Tensor | None = None
    if a is not None and b is not None:
        a = a.to(torch.float32)
        b = b.to(torch.float32)
        alpha = lsd.get(f"{prefix}.alpha")
        if alpha is None:
            # Alpha-less: keep the pre-generalization expression verbatim so
            # existing LTX variants (e.g. the distilled LoRA) fuse bit-identically.
            delta = torch.matmul(b * strength, a)
        else:
            # rank = in-dim of B = out-dim of A = a.shape[0]
            scale = strength * float(alpha) / a.shape[0]
            delta = torch.matmul(b, a) * scale
    if diff is not None:
        contrib = diff.to(torch.float32) * strength
        delta = contrib if delta is None else delta + contrib
    return delta


def fuse_loras_into(
    base_sd: dict[str, torch.Tensor],
    loras: list[LoraSpec],
    *,
    strict: bool = False,
) -> dict[str, torch.Tensor]:
    """Return a new state dict with each base weight/bias plus its fused LoRA delta.

    For each base ``<prefix>.weight`` we accumulate matching low-rank
    (``lora_A/B``) and ``.diff`` deltas; each ``<prefix>.bias`` accumulates
    ``.diff_b`` deltas. Deltas are summed in fp32 and cast back to the base
    dtype; keys without a LoRA contribution pass through unchanged. A fuse count
    of 0 means the LoRA keyspace did not align with the base — logged as a
    warning, or raised when ``strict``.
    """
    if not loras:
        return base_sd

    lora_sds = [(_normalize_lora_keys(load_file(s.path)), s.strength, s.path) for s in loras]

    fused: dict[str, torch.Tensor] = {}
    fuse_counts = [0] * len(lora_sds)

    for key, w in base_sd.items():
        if key.endswith(".weight"):
            prefix = key[: -len(".weight")]
            delta: torch.Tensor | None = None
            for idx, (lsd, strength, _path) in enumerate(lora_sds):
                contrib = _weight_delta(lsd, prefix, strength)
                if contrib is None:
                    continue
                delta = contrib if delta is None else delta + contrib
                fuse_counts[idx] += 1
            fused[key] = w if delta is None else (w.to(torch.float32) + delta).to(w.dtype)
        elif key.endswith(".bias"):
            prefix = key[: -len(".bias")]
            delta = None
            for idx, (lsd, strength, _path) in enumerate(lora_sds):
                diff_b = lsd.get(f"{prefix}.diff_b")
                if diff_b is None:
                    continue
                contrib = diff_b.to(torch.float32) * strength
                delta = contrib if delta is None else delta + contrib
                fuse_counts[idx] += 1
            fused[key] = w if delta is None else (w.to(torch.float32) + delta).to(w.dtype)
        else:
            fused[key] = w

    for (_lsd, strength, path), count in zip(lora_sds, fuse_counts):
        if count == 0:
            msg = (
                f"LoRA {path}: 0 tensors fused — prefix mismatch? "
                f"Inspect a few base/lora keys to confirm alignment."
            )
            if strict:
                raise ValueError(msg)
            logger.warning(msg)
        else:
            logger.info(f"LoRA {path}: fused {count} tensors (strength={strength})")

    return fused
