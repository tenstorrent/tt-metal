# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Fuse LoRA deltas into LTX transformer state dicts (bf16 base weights only).

FP8 / scaled-FP8 base weights are intentionally unsupported — we do not use
them on device.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from loguru import logger
from safetensors.torch import load_file


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
    """Strip the diffusion-model prefix so LoRA keys align with the base keyspace.

    HF LoRAs ship as ``model.diffusion_model.<rest>`` or the shorter
    ``diffusion_model.<rest>``; try both (longest first) so either becomes
    ``<rest>.lora_A.weight``.
    """
    for prefix in ("model.diffusion_model.", "diffusion_model."):
        if any(k.startswith(prefix) for k in lora_sd):
            return _strip_prefix(lora_sd, prefix)
    return lora_sd


def fuse_loras_into(
    base_sd: dict[str, torch.Tensor],
    loras: list[LoraSpec],
) -> dict[str, torch.Tensor]:
    """Return a new state dict with ``W += sum_i strength_i * (B_i @ A_i)``.

    For each base ``<prefix>.weight`` we accumulate matching
    ``<prefix>.lora_A/B.weight`` deltas in fp32 and cast back to the base dtype;
    keys without a LoRA pair pass through unchanged. A fuse count of 0 is logged
    as a warning since it means the LoRA keyspace did not align with the base.
    """
    if not loras:
        return base_sd

    lora_sds = [(_normalize_lora_keys(load_file(s.path)), s.strength, s.path) for s in loras]

    fused: dict[str, torch.Tensor] = {}
    fuse_counts = [0] * len(lora_sds)

    for key, w in base_sd.items():
        if not key.endswith(".weight"):
            fused[key] = w
            continue

        prefix = key[: -len(".weight")]
        a_key = f"{prefix}.lora_A.weight"
        b_key = f"{prefix}.lora_B.weight"

        delta: torch.Tensor | None = None
        for idx, (lsd, strength, _path) in enumerate(lora_sds):
            if a_key not in lsd or b_key not in lsd:
                continue
            a = lsd[a_key].to(dtype=torch.float32)
            b = lsd[b_key].to(dtype=torch.float32)
            contrib = torch.matmul(b * strength, a)
            delta = contrib if delta is None else delta + contrib
            fuse_counts[idx] += 1

        if delta is None:
            fused[key] = w
        else:
            fused[key] = (w.to(torch.float32) + delta).to(w.dtype)

    for (_lsd, _strength, path), count in zip(lora_sds, fuse_counts):
        if count == 0:
            logger.warning(
                f"LoRA {path}: 0 weights fused — prefix mismatch? "
                f"Inspect a few base/lora keys to confirm alignment."
            )
        else:
            logger.info(f"LoRA {path}: fused {count} weights (strength={_strength})")

    return fused
