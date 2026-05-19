# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Wan2.2 I2V pipeline with LoRA adapters fused into the base PyTorch weights.

Each expert transformer (high-noise + low-noise) gets its own ordered LoRA
stack. Stacks are fused on CPU before TT conversion, so inference uses
vanilla I2V machinery with no LoRA-specific runtime cost. The TT cache is
keyed by a SHA1 hash of the ordered ``(path, scale)`` tuples per expert,
so distinct stacks never alias.

Single-LoRA usage::

    pipe = WanPipelineI2VLora.create_pipeline(
        mesh_device=...,
        lora_high="/path/high.safetensors",
        lora_low="/path/low.safetensors",
    )

Multi-LoRA stacking (e.g. LightX2V LoRA + style LoRA, applied in order)::

    pipe = WanPipelineI2VLora.create_pipeline(
        mesh_device=...,
        lora_high=[
            LoRASpec("/path/lightx2v_high.safetensors", scale=1.0),
            LoRASpec("/path/style_high.safetensors", scale=0.5),
        ],
        lora_low=[LoRASpec("/path/lightx2v_low.safetensors", scale=1.0)],
    )

Adapter conventions detected automatically inside :func:`fuse_lora_state_dict`:

1. Low-rank pairs at ``<base>.lora_A.weight`` / ``<base>.lora_B.weight`` or
   ``<base>.lora_down.weight`` / ``<base>.lora_up.weight``. PEFT-style
   adapter-name segments (``<base>.lora_A.default.weight``) are tolerated.
2. Bias deltas at ``<base>.diff_b`` (added to the base ``.bias``).
3. Full param deltas at ``<base>.diff`` (added to the base ``.weight``,
   used e.g. for RMSNorm gammas).
4. Per-module alpha scaling via ``<base>.alpha`` (effective_scale =
   scale * alpha / rank).

Key namespaces accepted on input: lightx2v native (``blocks.N.attn.q.lora_A``),
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
from models.tt_dit.pipelines.wan.pipeline_wan import WanPipeline
from models.tt_dit.pipelines.wan.pipeline_wan_i2v import WanPipelineI2V
from models.tt_dit.utils import cache

# ---------------------------------------------------------------------------
# Public API: types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LoRASpec:
    """A single LoRA adapter file and its blend strength."""

    path: str
    scale: float = 1.0


LoRAArg = Union[LoRASpec, str, Iterable[Union[LoRASpec, str]], None]


def normalize_lora_arg(arg: LoRAArg) -> list[LoRASpec]:
    """Coerce a LoRA constructor argument into a list of :class:`LoRASpec`.

    Accepts None, a bare path string (scale defaults to 1.0), a single
    LoRASpec, or any iterable mixing the two. Returns an empty list for None.
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


# ---------------------------------------------------------------------------
# Internal: key normalization / detection helpers
# ---------------------------------------------------------------------------


_STRIP_PREFIXES = ("diffusion_model.", "transformer.", "unet.", "model.")

# PEFT-style adapters insert an extra ``.<adapter_name>`` segment between
# ``lora_X`` and the trailing ``.weight`` (e.g. SVI uses
# ``...lora_A.default.weight``). The optional non-capturing group matches it.
_LOW_RANK_RE = re.compile(r"^(?P<base>.*)\.lora_(?P<slot>A|B|down|up)(?:\.[^.]+)?\.weight$")
_SLOT_MAP = {"A": "A", "down": "A", "B": "B", "up": "B"}


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
    module_path = parts[0][len("lora_unet_") :]
    suffix = f".{parts[1]}" if len(parts) > 1 else ""

    m = re.match(r"blocks_(\d+)_(cross_attn|self_attn)_([a-z]+)", module_path)
    if m:
        return f"blocks.{m.group(1)}.{m.group(2)}.{m.group(3)}{suffix}"

    m = re.match(r"blocks_(\d+)_(ffn)_(\d+)", module_path)
    if m:
        return f"blocks.{m.group(1)}.{m.group(2)}.{m.group(3)}{suffix}"

    logger.warning(f"Unrecognized kohya key structure: {key}")
    return key


def _diffusers_target(lightx2v_base_path: str, suffix: str) -> str:
    """Map ``<lightx2v_base>.weight``/``.bias`` to the diffusers parameter key."""
    weight_key = wan_lightx2v_to_diffusers_key(f"{lightx2v_base_path}.weight")
    if suffix == ".weight":
        return weight_key
    return weight_key[: -len(".weight")] + suffix


def _has_lora_keys(state_dict: dict) -> bool:
    """Return True iff the safetensors file contains at least one LoRA-style key."""
    return any(
        ("lora_A" in k)
        or ("lora_B" in k)
        or ("lora_down" in k)
        or ("lora_up" in k)
        or k.endswith(".diff")
        or k.endswith(".diff_b")
        or k.startswith("lora_unet_")
        for k in state_dict
    )


# ---------------------------------------------------------------------------
# Public API: fusion + verification
# ---------------------------------------------------------------------------


def fuse_lora_state_dict(
    base_state_dict: dict[str, torch.Tensor],
    lora_state_dict: dict[str, torch.Tensor],
    *,
    scale: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Return a new state dict with LoRA deltas fused into ``base_state_dict``.

    Computes ``W_fused = W + scale * B @ A`` for each rank-r low-rank pair and
    adds direct deltas for ``.diff`` / ``.diff_b`` keys. ``base_state_dict``
    is not mutated; modified entries are fresh tensors so the caller can
    chain fusions safely.

    Raises:
        KeyError: a LoRA low-rank pair is missing one half.
    """
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    direct_deltas: list[tuple[str, str, torch.Tensor]] = []  # (base_path, suffix, tensor)
    alphas: dict[str, float] = {}
    skipped_unknown: list[str] = []

    for raw_key, tensor in lora_state_dict.items():
        key = _kohya_to_lightx2v(_strip_known_prefixes(raw_key))
        m = _LOW_RANK_RE.match(key)
        if m:
            pairs.setdefault(m.group("base"), {})[_SLOT_MAP[m.group("slot")]] = tensor
            continue
        if key.endswith(".diff_b"):
            direct_deltas.append((key[: -len(".diff_b")], ".bias", tensor))
        elif key.endswith(".diff"):
            direct_deltas.append((key[: -len(".diff")], ".weight", tensor))
        elif key.endswith(".alpha"):
            alphas[key[: -len(".alpha")]] = tensor.item()
        else:
            skipped_unknown.append(raw_key)

    if skipped_unknown:
        logger.warning(
            f"LoRA fusion: {len(skipped_unknown)} unrecognized key suffixes ignored " f"(first: {skipped_unknown[0]})"
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
        # ``+`` always allocates a new tensor; avoids the .to(fp32)+.add_()
        # alias trap when base_weight is already fp32 (would mutate the
        # caller's dict on a subsequent stacked-fusion pass).
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
            f"LoRA fusion: {len(skipped_unmapped)} adapter targets not present in base model; "
            f"skipped. Examples: {sample}"
        )

    logger.info(f"Fused {applied_pairs} low-rank pairs and {applied_direct} direct deltas (scale={scale})")
    return fused


def verify_fusion_changed_weights(
    base_sd: dict[str, torch.Tensor],
    fused_sd: dict[str, torch.Tensor],
    *,
    min_changed: int = 3,
    label: str,
) -> None:
    """Sanity check that at least ``min_changed`` weights actually differ post-fusion.

    Catches the canonical "LoRA silently failed to apply" failure mode where
    a key-mapping bug leaves every fused target absent from the base. Logs a
    sample of changed tensors with their L2 diff for visibility.

    Raises ``RuntimeError`` if no weights changed at all, or if fewer than
    ``min_changed`` weights changed (likely a partial load).
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


# ---------------------------------------------------------------------------
# Internal: cache key
# ---------------------------------------------------------------------------


def _lora_stack_cache_namespace(specs_by_expert: dict[int, list[LoRASpec]]) -> str:
    """Stable short hash so distinct LoRA stacks cache separately on disk.

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


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class WanPipelineI2VLora(WanPipelineI2V):
    """Wan2.2 I2V with LoRA stacks fused into the base PyTorch weights."""

    def __init__(
        self,
        *args,
        lora_high: LoRAArg = None,
        lora_low: LoRAArg = None,
        **kwargs,
    ):
        high_specs = normalize_lora_arg(lora_high)
        low_specs = normalize_lora_arg(lora_low)

        if not high_specs and not low_specs:
            raise ValueError(
                "WanPipelineI2VLora requires at least one LoRA. "
                "Pass lora_high and/or lora_low as a path, LoRASpec, or list."
            )

        for label, specs in [("lora_high", high_specs), ("lora_low", low_specs)]:
            for spec in specs:
                if not Path(spec.path).is_file():
                    raise FileNotFoundError(f"{label}: file does not exist: {spec.path}")

        self._lora_specs: dict[int, list[LoRASpec]] = {0: high_specs, 1: low_specs}
        self._cache_namespace = _lora_stack_cache_namespace(self._lora_specs)
        # Lazily-built fused state dicts keyed by transformer index. Cleared
        # after handoff to TT cache to free CPU memory.
        self._fused_state_dicts: dict[int, dict[str, torch.Tensor] | None] = {0: None, 1: None}

        super().__init__(*args, **kwargs)

    def prepare_text_conditioning(self, tt_model, prompt_embeds, buffer, traced=False):
        # When guidance_scale=1.0 the encoder returns negative_prompt_embeds=None.
        # The base loop still calls this for the negative buffer; forwarding
        # None would hit NoneType in Linear. combined_step short-circuits when
        # do_classifier_free_guidance is False, so leaving the buffer untouched
        # is safe.
        if prompt_embeds is None:
            return buffer
        return super().prepare_text_conditioning(tt_model, prompt_embeds, buffer, traced)

    def _build_fused_state_dict(self, idx: int) -> dict[str, torch.Tensor] | None:
        specs = self._lora_specs[idx]
        state = self.transformer_states[idx]
        if not specs:
            logger.info(f"No LoRA for expert idx={idx} ('{state.subfolder}') -- using base weights")
            return None

        base_sd = state.torch_model.state_dict()
        fused_sd = base_sd
        for spec in specs:
            logger.info(f"Loading LoRA for '{state.subfolder}' from {spec.path} (scale={spec.scale})")
            lora_sd = load_file(str(spec.path))
            if not _has_lora_keys(lora_sd):
                raise RuntimeError(
                    f"No LoRA-style keys (lora_A/lora_B, lora_down/lora_up, diff/diff_b) " f"found in {spec.path}"
                )
            fused_sd = fuse_lora_state_dict(fused_sd, lora_sd, scale=spec.scale)

        verify_fusion_changed_weights(
            base_sd,
            fused_sd,
            label=f"{state.subfolder} (stack of {len(specs)})",
        )
        return fused_sd

    def _prepare_transformer(self, idx: int):
        state = self.transformer_states[idx]

        if not self._lora_specs[idx]:
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
        return WanPipeline.create_pipeline(*args, pipeline_class=WanPipelineI2VLora, **kwargs)
