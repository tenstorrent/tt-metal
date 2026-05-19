# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""LoRA fusion utilities for Wan2.2 pipelines.

Loads LoRA `.safetensors` adapters on CPU and fuses them into the base
PyTorch transformer state dict before TT conversion. The rank-r delta is
computed manually -- ``W_fused = W + scale * B @ A`` -- producing a single
fused weight matrix per linear with no LoRA-specific runtime cost.

Supports stacking multiple LoRAs per expert (e.g. LightX2V LoRA + SVI LoRA)
via :func:`fuse_lora_stack`. Each stack entry is a :class:`LoRASpec` with its
own scale, applied in order so later LoRAs see earlier ones' deltas already
folded into the base.

Adapter conventions detected automatically:

1. Low-rank pairs at ``<base>.lora_A.weight``/``<base>.lora_B.weight`` OR
   ``<base>.lora_down.weight``/``<base>.lora_up.weight``.
2. Bias deltas at ``<base>.diff_b``: added to the base ``.bias``.
3. Full param deltas at ``<base>.diff``: added to the base ``.weight``
   (e.g. RMSNorm gammas).
4. Per-module alpha scaling via ``<base>.alpha`` (effective_scale =
   scale * alpha / rank).

Key namespaces supported on input: lightx2v native (``blocks.N.attn.q.lora_A``),
diffusers prefixes (``diffusion_model.``, ``transformer.``, ``unet.``,
``model.``) stripped automatically, kohya/A1111 (``lora_unet_blocks_N_...``)
remapped to lightx2v style.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Union

import torch
from loguru import logger
from safetensors.torch import load_file

from models.tt_dit.experimental.utils.lightx2v_loader import wan_lightx2v_to_diffusers_key

_STRIP_PREFIXES = ("diffusion_model.", "transformer.", "unet.", "model.")


@dataclass(frozen=True)
class LoRASpec:
    """A single LoRA adapter and its blend strength."""

    path: str
    scale: float = 1.0


LoRAArg = Union[LoRASpec, str, Iterable[Union[LoRASpec, str]], None]


def normalize_lora_arg(arg: LoRAArg) -> list[LoRASpec]:
    """Coerce LoRA constructor arguments into a list of specs.

    Accepts None, a single path string, a single :class:`LoRASpec`, or any
    iterable of those. Bare strings default to ``scale=1.0``.
    """
    if arg is None:
        return []
    if isinstance(arg, LoRASpec):
        return [arg]
    if isinstance(arg, str):
        return [LoRASpec(arg)]
    out: list[LoRASpec] = []
    for item in arg:
        if isinstance(item, LoRASpec):
            out.append(item)
        elif isinstance(item, str):
            out.append(LoRASpec(item))
        else:
            raise TypeError(f"Expected LoRASpec or str in LoRA list, got {type(item).__name__}")
    return out


def _strip_known_prefixes(key: str) -> str:
    for prefix in _STRIP_PREFIXES:
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def _kohya_to_lightx2v(key: str) -> str:
    """Convert kohya/A1111-style keys to lightx2v-style keys.

    ``lora_unet_blocks_0_cross_attn_k.lora_down.weight``
    → ``blocks.0.cross_attn.k.lora_down.weight``
    """
    if not key.startswith("lora_unet_"):
        return key
    parts = key.split(".", 1)
    module_path = parts[0]
    suffix = f".{parts[1]}" if len(parts) > 1 else ""

    module_path = module_path[len("lora_unet_") :]

    m = re.match(r"blocks_(\d+)_(cross_attn|self_attn)_([a-z]+)", module_path)
    if m:
        return f"blocks.{m.group(1)}.{m.group(2)}.{m.group(3)}{suffix}"

    m = re.match(r"blocks_(\d+)_(ffn)_(\d+)", module_path)
    if m:
        return f"blocks.{m.group(1)}.{m.group(2)}.{m.group(3)}{suffix}"

    logger.warning(f"Unrecognized kohya key structure: {key}")
    return key


def _diffusers_target(lightx2v_base_path: str, suffix: str) -> str:
    """Map ``<lightx2v_base>.weight``/``.bias`` to the diffusers key."""
    weight_key = wan_lightx2v_to_diffusers_key(f"{lightx2v_base_path}.weight")
    if suffix == ".weight":
        return weight_key
    return weight_key[: -len(".weight")] + suffix


def fuse_lora_state_dict(
    base_state_dict: dict[str, torch.Tensor],
    lora_state_dict: dict[str, torch.Tensor],
    *,
    scale: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Return a new state dict with LoRA deltas fused into ``base_state_dict``.

    ``base_state_dict`` is not mutated. The returned dict shares unchanged
    tensors with the input by reference; modified entries are fresh tensors.

    Raises:
        KeyError: a LoRA low-rank pair is missing one half.
    """
    # PEFT-style adapters insert an extra ``.<adapter_name>`` segment between
    # ``lora_X`` and the trailing ``.weight`` (e.g. SVI uses
    # ``...lora_A.default.weight``). Match it optionally.
    LOW_RANK_RE = re.compile(r"^(?P<base>.*)\.lora_(?P<slot>A|B|down|up)(?:\.[^.]+)?\.weight$")
    SLOT_MAP = {"A": "A", "down": "A", "B": "B", "up": "B"}
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    direct_deltas: list[tuple[str, str, torch.Tensor]] = []
    skipped_unknown: list[str] = []
    alphas: dict[str, float] = {}

    for raw_key, tensor in lora_state_dict.items():
        key = _strip_known_prefixes(raw_key)
        key = _kohya_to_lightx2v(key)
        m = LOW_RANK_RE.match(key)
        if m:
            pairs.setdefault(m.group("base"), {})[SLOT_MAP[m.group("slot")]] = tensor
            continue
        if key.endswith(".diff_b"):
            direct_deltas.append((key[: -len(".diff_b")], ".bias", tensor))
        elif key.endswith(".diff"):
            direct_deltas.append((key[: -len(".diff")], ".weight", tensor))
        elif key.endswith(".alpha"):
            base_path = key[: -len(".alpha")]
            alphas[base_path] = tensor.item()
        else:
            skipped_unknown.append(raw_key)

    if skipped_unknown:
        logger.warning(
            f"LoRA fusion: {len(skipped_unknown)} unrecognized key suffixes ignored (first: {skipped_unknown[0]})"
        )

    fused = dict(base_state_dict)
    applied_pairs = 0
    skipped_unmapped: list[str] = []

    for base_path, ab in pairs.items():
        if "A" not in ab or "B" not in ab:
            raise KeyError(f"LoRA pair incomplete for '{base_path}': have {list(ab)}")
        diffusers_key = _diffusers_target(base_path, ".weight")
        if diffusers_key not in fused:
            skipped_unmapped.append(diffusers_key)
            continue
        base_weight = fused[diffusers_key]
        rank = ab["A"].shape[0]
        alpha = alphas.get(base_path, float(rank))
        effective_scale = scale * (alpha / rank)
        delta = effective_scale * (ab["B"].to(torch.float32) @ ab["A"].to(torch.float32))
        # `+` always allocates a new tensor; avoids the .to(fp32)+.add_()
        # alias trap when base_weight is already fp32 (would mutate the
        # caller's dict on a subsequent stack pass).
        fused[diffusers_key] = (base_weight.float() + delta).to(base_weight.dtype)
        applied_pairs += 1

    applied_direct = 0
    for base_path, suffix, tensor in direct_deltas:
        diffusers_key = _diffusers_target(base_path, suffix)
        if diffusers_key not in fused:
            skipped_unmapped.append(diffusers_key)
            continue
        base = fused[diffusers_key]
        if base.shape != tensor.shape:
            logger.warning(
                f"LoRA direct delta shape mismatch for '{diffusers_key}': "
                f"base {tuple(base.shape)} vs delta {tuple(tensor.shape)}; skipping."
            )
            continue
        fused[diffusers_key] = (base.float() + scale * tensor.to(torch.float32)).to(base.dtype)
        applied_direct += 1

    if skipped_unmapped:
        sample = skipped_unmapped[:5]
        logger.warning(
            f"LoRA fusion: {len(skipped_unmapped)} adapter targets not present in base model; skipped. "
            f"Examples: {sample}"
        )

    logger.info(f"Fused {applied_pairs} low-rank pairs and {applied_direct} direct deltas (scale={scale})")
    return fused


def fuse_lora_stack(
    base_state_dict: dict[str, torch.Tensor],
    specs: Iterable[LoRASpec],
) -> dict[str, torch.Tensor]:
    """Fold an ordered stack of LoRAs into a base state dict.

    Each spec is loaded from disk and applied on top of the previous result,
    so later LoRAs see earlier deltas already folded into the base. Used for
    e.g. LightX2V LoRA + SVI LoRA stacking where the SVI adapter was trained
    expecting the LightX2V deltas to already be present.
    """
    fused = base_state_dict
    for spec in specs:
        lora_sd = load_file(str(spec.path))
        fused = fuse_lora_state_dict(fused, lora_sd, scale=spec.scale)
    return fused


def verify_fusion_changed_weights(
    base_sd: dict[str, torch.Tensor],
    fused_sd: dict[str, torch.Tensor],
    *,
    min_changed: int = 3,
    label: str,
) -> None:
    """Sanity check: at least ``min_changed`` weights must differ post-fusion.

    Raises ``RuntimeError`` if no weights changed at all -- the canonical
    "LoRA silently failed to apply" failure mode (usually a key-mapping bug
    that left every fused target absent from the base).
    """
    changed: list[tuple[str, float]] = []
    max_diff = 0.0
    for k, base in base_sd.items():
        fused = fused_sd.get(k)
        if fused is None or fused.shape != base.shape:
            continue
        if fused.data_ptr() == base.data_ptr():
            continue
        diff = (fused.to(torch.float32) - base.to(torch.float32)).norm().item()
        if diff > 0.0:
            changed.append((k, diff))
            max_diff = max(max_diff, diff)

    if max_diff == 0.0:
        raise RuntimeError(
            f"LoRA silently failed to apply -- weights are unchanged for '{label}'. "
            f"Verified {len(base_sd)} keys against fused dict; max L2 diff is 0.0."
        )

    sample = changed[: max(min_changed, 5)]
    logger.info(
        f"LoRA fusion verified for '{label}': {len(changed)} tensors changed, "
        f"max L2 diff={max_diff:.4f}. Sample diffs:"
    )
    for k, d in sample:
        logger.info(f"  {k}: L2={d:.4f}")

    if len(changed) < min_changed:
        raise RuntimeError(
            f"LoRA fusion changed only {len(changed)} weights for '{label}' "
            f"(require >= {min_changed}). Likely indicates partial LoRA load."
        )


def lora_stack_cache_namespace(specs_by_expert: dict[int, list[LoRASpec]]) -> str:
    """Stable short hash so distinct LoRA stacks cache separately.

    Hashes ordered ``(resolved_path, scale)`` tuples per expert index. Two
    stacks with the same files in different order get different namespaces.
    """
    h = hashlib.sha1()
    for idx in sorted(specs_by_expert.keys()):
        h.update(f"expert_{idx}\x00".encode())
        for spec in specs_by_expert[idx]:
            h.update(str(Path(spec.path).resolve()).encode())
            h.update(b"\x00")
            h.update(f"{spec.scale:.6f}".encode())
            h.update(b"\x00")
    return f"Wan2.2-I2V-LoRA-{h.hexdigest()[:12]}"
