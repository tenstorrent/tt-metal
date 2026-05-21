# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Wan2.2 I2V pipeline with LoRA hot-swap.

Subclass of :class:`WanPipelineI2V` that fuses a pair of LoRA adapters
(high-noise + low-noise expert) into the base PyTorch transformer state dict
before TT conversion. Output behaves like a vanilla Wan2.2 I2V model with the
LoRA permanently applied -- no per-step LoRA evaluation on device.

LoRA keys are detected and fused into the base PyTorch weights, then handed
to ``cache.load_model``. The rank-r delta is computed manually
(``W_fused = W + scale * B @ A``) producing a single fused weight matrix
per linear with no LoRA-specific runtime cost.

Cache keying: each (high_path, low_path, scale) combination produces a
distinct namespace via :func:`_lora_cache_namespace`, so the TT cache for one
LoRA never aliases another.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import torch
from loguru import logger
from safetensors.torch import load_file

from models.tt_dit.experimental.utils.lightx2v_loader import wan_lightx2v_to_diffusers_key
from models.tt_dit.pipelines.wan.pipeline_wan import WanPipeline
from models.tt_dit.pipelines.wan.pipeline_wan_i2v import WanPipelineI2V
from models.tt_dit.utils import cache

_STRIP_PREFIXES = ("diffusion_model.", "transformer.", "unet.", "model.")


def _strip_known_prefixes(key: str) -> str:
    for prefix in _STRIP_PREFIXES:
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def _kohya_to_lightx2v(key: str) -> str:
    """Convert kohya/A1111-style keys to lightx2v-style keys.

    ``lora_unet_blocks_0_cross_attn_k.lora_down.weight``
    → ``blocks.0.cross_attn.k.lora_down.weight``

    Also handles ``.alpha`` keys.
    """
    import re

    if not key.startswith("lora_unet_"):
        return key
    parts = key.split(".", 1)
    module_path = parts[0]  # e.g. "lora_unet_blocks_0_cross_attn_k"
    suffix = f".{parts[1]}" if len(parts) > 1 else ""

    module_path = module_path[len("lora_unet_") :]

    # Wan2.2 kohya keys follow: blocks_{N}_{attn_type}_{param} or blocks_{N}_ffn_{idx}
    # attn_type: cross_attn or self_attn, param: q/k/v/o
    m = re.match(r"blocks_(\d+)_(cross_attn|self_attn)_([a-z]+)", module_path)
    if m:
        return f"blocks.{m.group(1)}.{m.group(2)}.{m.group(3)}{suffix}"

    m = re.match(r"blocks_(\d+)_(ffn)_(\d+)", module_path)
    if m:
        return f"blocks.{m.group(1)}.{m.group(2)}.{m.group(3)}{suffix}"

    logger.warning(f"Unrecognized kohya key structure: {key}")
    return key


def _diffusers_target(lightx2v_base_path: str, suffix: str) -> str:
    """Map a lightx2v base path + suffix (.weight/.bias) to the diffusers key.

    The lightx2v-to-diffusers map operates on full parameter names ending in
    ``.weight``, so we feed it ``<base>.weight``, swap to the requested suffix
    afterwards.
    """
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

    Supports three lightx2v-style adapter conventions, all keyed under a
    ``diffusion_model.`` prefix that is stripped before mapping:

    1. **Low-rank pairs** at ``<base>.lora_A.weight``/``<base>.lora_B.weight``
       OR ``<base>.lora_down.weight``/``<base>.lora_up.weight``. Adds
       ``scale * B @ A`` to the base ``.weight``.
    2. **Bias deltas** at ``<base>.diff_b``: adds the tensor to the base
       ``.bias`` directly.
    3. **Full param deltas** at ``<base>.diff``: adds to the base ``.weight``.
       Used for params without a low-rank factorization (e.g. RMSNorm gammas).

    Keys whose remapped target is absent from ``base_state_dict`` are skipped
    with a warning -- this happens for adapter modules that don't exist in
    the diffusers Wan2.2 I2V architecture (e.g. ``cross_attn.k_img`` from a
    Wan2.1 I2V-trained adapter; Wan2.2 I2V uses channel-concat conditioning
    instead). ``base_state_dict`` is not mutated.

    Raises:
        KeyError: a LoRA low-rank pair is missing one half (lora_A without
            lora_B or vice versa).
    """
    LOW_RANK_SUFFIXES = {
        ".lora_A.weight": "A",
        ".lora_B.weight": "B",
        ".lora_down.weight": "A",
        ".lora_up.weight": "B",
    }
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    direct_deltas: list[tuple[str, str, torch.Tensor]] = []  # (base_path, suffix, tensor)
    skipped_unknown: list[str] = []

    alphas: dict[str, float] = {}

    for raw_key, tensor in lora_state_dict.items():
        key = _strip_known_prefixes(raw_key)
        key = _kohya_to_lightx2v(key)
        matched = False
        for suffix, slot in LOW_RANK_SUFFIXES.items():
            if key.endswith(suffix):
                base_path = key[: -len(suffix)]
                pairs.setdefault(base_path, {})[slot] = tensor
                matched = True
                break
        if matched:
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
        # Per-module alpha scaling: effective_scale = scale * (alpha / rank)
        rank = ab["A"].shape[0]
        alpha = alphas.get(base_path, float(rank))
        effective_scale = scale * (alpha / rank)
        delta = effective_scale * (ab["B"].to(torch.float32) @ ab["A"].to(torch.float32))
        fused[diffusers_key] = (base_weight.to(torch.float32) + delta).to(base_weight.dtype)
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
                f"LoRA direct delta shape mismatch for '{diffusers_key}': base {tuple(base.shape)} vs delta {tuple(tensor.shape)}; skipping."
            )
            continue
        fused[diffusers_key] = (base.to(torch.float32) + scale * tensor.to(torch.float32)).to(base.dtype)
        applied_direct += 1

    if skipped_unmapped:
        sample = skipped_unmapped[:5]
        logger.warning(
            f"LoRA fusion: {len(skipped_unmapped)} adapter targets not present in base model; skipped. "
            f"Examples: {sample}"
        )

    logger.info(f"Fused {applied_pairs} low-rank pairs and {applied_direct} direct deltas (scale={scale})")
    return fused


def _verify_fusion_changed_weights(
    base_sd: dict[str, torch.Tensor],
    fused_sd: dict[str, torch.Tensor],
    *,
    min_changed: int = 3,
    label: str,
) -> None:
    """Sanity check: at least ``min_changed`` weights must differ post-fusion.

    Logs the L2 norm of the diff for the first few changed tensors. Raises
    ``RuntimeError`` if no weights changed at all -- the canonical "LoRA
    silently failed to apply" failure mode.
    """
    changed: list[tuple[str, float]] = []
    max_diff = 0.0
    for k, base in base_sd.items():
        fused = fused_sd.get(k)
        if fused is None or fused.shape != base.shape:
            continue
        if fused.data_ptr() == base.data_ptr():
            continue  # untouched (same tensor object)
        diff = (fused.to(torch.float32) - base.to(torch.float32)).norm().item()
        if diff > 0.0:
            changed.append((k, diff))
            max_diff = max(max_diff, diff)
            if len(changed) >= min_changed:
                # Keep scanning past min to surface the max accurately, but
                # stop logging beyond a small sample.
                pass

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
            f"LoRA fusion changed only {len(changed)} weights for '{label}' (require >= {min_changed}). "
            f"Likely indicates partial LoRA load."
        )


def _lora_cache_namespace(high_path: str, low_path: str | None, scale: float) -> str:
    """Stable short hash so distinct (paths, scale) combos cache separately."""
    h = hashlib.sha1()
    h.update(str(Path(high_path).resolve()).encode())
    h.update(b"\x00")
    h.update(str(Path(low_path).resolve()).encode() if low_path else b"none")
    h.update(b"\x00")
    h.update(f"{scale:.6f}".encode())
    return f"Wan2.2-I2V-LoRA-{h.hexdigest()[:12]}"


class WanLoraPipelineI2V(WanPipelineI2V):
    """Wan2.2 I2V with LoRA adapters fused into the base PyTorch weights.

    Each expert transformer can get its own LoRA file, or a single LoRA
    can be applied to the high-noise expert only. Inference parameters
    (steps, CFG, boundary_ratio) are left to the caller.

    Args:
        lora_high_path: Path to ``.safetensors`` LoRA for the high-noise
            expert (``transformer``). May also be passed via the
            ``LORA_HIGH_PATH`` env var.
        lora_low_path: Path to ``.safetensors`` LoRA for the low-noise
            expert (``transformer_2``). May also be passed via
            ``LORA_LOW_PATH``.
        lora_scale: Scalar multiplier applied to the fused delta
            ``W' = W + scale * B @ A``. Default 1.0.
    """

    def __init__(
        self,
        *args,
        lora_high_path: str | None = None,
        lora_low_path: str | None = None,
        lora_scale: float = 1.0,
        **kwargs,
    ):
        # Resolve LoRA paths/scale and (optional) boundary_ratio from env.
        # NOTE: WanPipeline.create_pipeline drops unknown kwargs and hardcodes
        # boundary_ratio=0.875, so the test funnels per-run overrides through
        # env vars (LORA_HIGH_PATH/LORA_LOW_PATH/LORA_SCALE/BOUNDARY_RATIO).
        lora_high_path = lora_high_path or os.environ.get("LORA_HIGH_PATH")
        lora_low_path = lora_low_path or os.environ.get("LORA_LOW_PATH") or None
        if "LORA_SCALE" in os.environ:
            lora_scale = float(os.environ["LORA_SCALE"])
        if "BOUNDARY_RATIO" in os.environ:
            kwargs["boundary_ratio"] = float(os.environ["BOUNDARY_RATIO"])
        if not lora_high_path:
            raise ValueError(
                "WanLoraPipelineI2V requires at least lora_high_path "
                "(or LORA_HIGH_PATH env var). LORA_LOW_PATH is optional — "
                "if omitted, the low-noise expert uses unmodified base weights."
            )
        for p, name in [(lora_high_path, "lora_high_path")] + (
            [(lora_low_path, "lora_low_path")] if lora_low_path else []
        ):
            if not Path(p).is_file():
                raise FileNotFoundError(f"{name}: file does not exist: {p}")

        self._lora_paths = (lora_high_path, lora_low_path)
        self._lora_scale = float(lora_scale)
        self._cache_namespace = _lora_cache_namespace(lora_high_path, lora_low_path, self._lora_scale)
        # Lazily-built fused state dicts keyed by transformer index. Cleared
        # after handoff to TT cache to free CPU memory.
        self._fused_state_dicts: dict[int, dict[str, torch.Tensor] | None] = {0: None, 1: None}

        super().__init__(*args, **kwargs)

    def prepare_text_conditioning(self, tt_model, prompt_embeds, buffer, traced=False):
        # When guidance_scale=1.0, encode_prompt returns negative_prompt_embeds=None.
        # The base loop still calls this for the negative buffer; forwarding None
        # would hit NoneType in Linear. Safe to skip since combined_step
        # short-circuits on do_classifier_free_guidance=False.
        if prompt_embeds is None:
            return buffer
        return super().prepare_text_conditioning(tt_model, prompt_embeds, buffer, traced)

    def _build_fused_state_dict(self, idx: int) -> dict[str, torch.Tensor] | None:
        state = self.transformer_states[idx]
        lora_path = self._lora_paths[idx]
        if lora_path is None:
            logger.info(f"No LoRA for expert idx={idx} ('{state.subfolder}') — using base weights")
            return None
        label = state.subfolder
        logger.info(f"Loading LoRA for '{label}' from {lora_path}")
        lora_sd = load_file(str(lora_path))
        has_lora = any(
            ("lora_A" in k)
            or ("lora_B" in k)
            or ("lora_down" in k)
            or ("lora_up" in k)
            or k.endswith(".diff")
            or k.endswith(".diff_b")
            or k.startswith("lora_unet_")
            for k in lora_sd
        )
        if not has_lora:
            raise RuntimeError(
                f"No LoRA-style keys (lora_A/lora_B, lora_down/lora_up, diff/diff_b) found in {lora_path}"
            )

        base_sd = state.torch_model.state_dict()
        fused_sd = fuse_lora_state_dict(base_sd, lora_sd, scale=self._lora_scale)
        _verify_fusion_changed_weights(base_sd, fused_sd, label=label)
        return fused_sd

    def _prepare_transformer(self, idx: int):
        state = self.transformer_states[idx]

        if self._lora_paths[idx] is None:
            # No LoRA for this expert — use the base class path (vanilla weights).
            super()._prepare_transformer(idx)
            return

        def _get_state_dict(idx_=idx):
            cached = self._fused_state_dicts.get(idx_)
            if cached is not None:
                return cached
            sd = self._build_fused_state_dict(idx_)
            self._fused_state_dicts[idx_] = sd
            return sd

        cache.load_model(
            state.model,
            model_name=self._cache_namespace,
            subfolder=state.subfolder,
            parallel_config=self.parallel_config,
            mesh_shape=tuple(self.mesh_device.shape),
            is_fsdp=self.is_fsdp,
            get_torch_state_dict=_get_state_dict,
        )
        self._fused_state_dicts[idx] = None

    @staticmethod
    def create_pipeline(*args, **kwargs):
        kwargs["checkpoint_name"] = kwargs.get("checkpoint_name") or "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
        return WanPipeline.create_pipeline(*args, pipeline_class=WanLoraPipelineI2V, **kwargs)
