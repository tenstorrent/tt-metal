# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Wan2.2 I2V pipeline with LoRA adapters fused into the base weights.

Each expert (high/low noise) takes an ordered LoRA stack; stacks are fused
on CPU before TT conversion so inference has no LoRA-specific runtime cost.
See ``experimental/models/Wan2_2_LoRA.md`` for the adapter-key formats
detected by ``fuse_lora_state_dict`` and the supported namespaces.
"""
import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from loguru import logger
from safetensors.torch import load_file

import ttnn
from models.tt_dit.experimental.utils.lightx2v_loader import wan_lightx2v_to_diffusers_key
from models.tt_dit.pipelines.wan.pipeline_wan import WanPipelineConfig
from models.tt_dit.pipelines.wan.pipeline_wan_i2v import WanPipelineI2V
from models.tt_dit.utils import cache


@dataclass(frozen=True)
class LoRASpec:
    """A single LoRA adapter file and its blend strength."""

    path: str
    scale: float = 1.0


LoRAArg = LoRASpec | str | Sequence[LoRASpec | str] | None


@dataclass(frozen=True)
class _FusionStats:
    applied_pairs: int = 0
    applied_direct: int = 0
    skipped_unknown: int = 0
    skipped_unmapped: int = 0
    skipped_shape_mismatch: int = 0

    @property
    def applied(self) -> int:
        return self.applied_pairs + self.applied_direct


def _normalize_lora_arg(arg: LoRAArg) -> list[LoRASpec]:
    """Coerce None / path / LoRASpec / sequence-of-either into LoRASpecs."""
    if arg is None:
        return []
    if isinstance(arg, LoRASpec):
        return [arg]
    if isinstance(arg, str):
        return [LoRASpec(arg)]
    if not isinstance(arg, Sequence):
        raise TypeError(f"Expected LoRASpec, str, sequence, or None; got {type(arg).__name__}")
    out: list[LoRASpec] = []
    for item in arg:
        if isinstance(item, LoRASpec):
            out.append(item)
        elif isinstance(item, str):
            out.append(LoRASpec(item))
        else:
            raise TypeError(f"Expected LoRASpec or str in LoRA list, got {type(item).__name__}")
    return out


_STRIP_PREFIXES = ("diffusion_model.", "transformer.", "unet.", "model.")

# PEFT may insert an adapter name before ``.weight``, e.g. ``lora_A.default.weight``.
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


def _is_lora_key(raw_key: str) -> bool:
    key = _kohya_to_lightx2v(_strip_known_prefixes(raw_key))
    return bool(_LOW_RANK_RE.match(key)) or key.endswith((".diff", ".diff_b"))


def _has_lora_keys(state_dict: dict[str, torch.Tensor]) -> bool:
    """Return True iff the safetensors file contains at least one LoRA-style key."""
    return any(_is_lora_key(k) for k in state_dict)


def fuse_lora_state_dict(
    base_state_dict: dict[str, torch.Tensor],
    lora_state_dict: dict[str, torch.Tensor],
    *,
    scale: float = 1.0,
    return_stats: bool = False,
) -> dict[str, torch.Tensor] | tuple[dict[str, torch.Tensor], _FusionStats]:
    """Return a fused state dict, optionally with stats describing what applied.

    The base dict is not mutated; entries that were touched are fresh tensors
    so the caller can chain fusions across a stack safely. See
    ``experimental/models/Wan2_2_LoRA.md`` for the supported adapter key
    formats. Raises ``KeyError`` when a low-rank pair is missing one half.
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
            f"LoRA fusion: {len(skipped_unknown)} unrecognized keys ignored. Examples: {skipped_unknown[:5]}"
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
    skipped_shape_mismatch = 0
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
            skipped_shape_mismatch += 1
            continue
        fused[diffusers_key] = (base.float() + scale * tensor.to(torch.float32)).to(base.dtype)
        applied_direct += 1

    if skipped_unmapped:
        logger.warning(
            f"LoRA fusion: {len(skipped_unmapped)} adapter targets not present in base; skipped. "
            f"Examples: {skipped_unmapped[:5]}"
        )

    logger.info(f"Fused {applied_pairs} low-rank pairs and {applied_direct} direct deltas (scale={scale})")
    stats = _FusionStats(
        applied_pairs=applied_pairs,
        applied_direct=applied_direct,
        skipped_unknown=len(skipped_unknown),
        skipped_unmapped=len(skipped_unmapped),
        skipped_shape_mismatch=skipped_shape_mismatch,
    )
    return (fused, stats) if return_stats else fused


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

    sample = changed[:5]
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


def _lora_stack_cache_namespace(specs_by_expert: dict[int, list[LoRASpec]]) -> str:
    """Hash ordered ``(resolved_path, scale)`` per expert so distinct stacks cache separately."""
    h = hashlib.sha1()
    for idx in sorted(specs_by_expert.keys()):
        h.update(f"expert_{idx}\x00".encode())
        for spec in specs_by_expert[idx]:
            h.update(str(Path(spec.path).resolve()).encode())
            h.update(b"\x00")
            h.update(f"{spec.scale:.6f}".encode())
            h.update(b"\x00")
    return f"Wan2.2-I2V-LoRA-{h.hexdigest()[:12]}"


class WanPipelineI2VLora(WanPipelineI2V):
    """Wan2.2 I2V with LoRA stacks fused into the base PyTorch weights."""

    BASE_DIFFUSERS_REPO = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"

    def __init__(
        self,
        *,
        device: ttnn.MeshDevice,
        config: WanPipelineConfig,
        lora_high: LoRAArg = None,
        lora_low: LoRAArg = None,
    ) -> None:
        high_specs = _normalize_lora_arg(lora_high)
        low_specs = _normalize_lora_arg(lora_low)

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
        # Cleared after TT cache handoff (see _prepare_transformer) to free CPU memory.
        self._fused_state_dicts: dict[int, dict[str, torch.Tensor] | None] = {0: None, 1: None}

        super().__init__(device=device, config=config)

    def prepare_text_conditioning(self, tt_model, prompt_embeds, buffer, traced=False):
        # guidance_scale=1.0 → encoder returns None for negative embeds; combined_step
        # skips CFG so the untouched buffer is fine.
        if prompt_embeds is None:
            return buffer
        return super().prepare_text_conditioning(tt_model, prompt_embeds, buffer, traced)

    def _build_fused_state_dict(self, idx: int) -> dict[str, torch.Tensor] | None:
        specs = self._lora_specs[idx]
        state = self.transformer_states[idx]
        subfolder = state.checkpoint.subfolder
        if not specs:
            logger.info(f"No LoRA for expert idx={idx} ('{subfolder}') -- using base weights")
            return None

        base_sd = state.checkpoint.state_dict()
        fused_sd = base_sd
        for spec in specs:
            logger.info(f"Loading LoRA for '{subfolder}' from {spec.path} (scale={spec.scale})")
            lora_sd = load_file(str(spec.path))
            if not _has_lora_keys(lora_sd):
                raise RuntimeError(
                    f"No LoRA-style keys (lora_A/lora_B, lora_down/lora_up, diff/diff_b) found in {spec.path}"
                )
            previous_sd = fused_sd
            fused_sd, stats = fuse_lora_state_dict(previous_sd, lora_sd, scale=spec.scale, return_stats=True)
            label = f"{subfolder}: {Path(spec.path).name}"
            if stats.applied == 0:
                raise RuntimeError(
                    f"LoRA fusion applied no tensors for '{label}'. "
                    f"Skipped unmapped={stats.skipped_unmapped}, unknown={stats.skipped_unknown}, "
                    f"shape_mismatch={stats.skipped_shape_mismatch}."
                )
            verify_fusion_changed_weights(previous_sd, fused_sd, label=label)
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
            subfolder=state.checkpoint.subfolder,
            parallel_config=self.parallel_config,
            mesh_shape=tuple(self.mesh_device.shape),
            is_fsdp=self.is_fsdp,
            get_torch_state_dict=_get_state_dict,
        )
        self._fused_state_dicts[idx] = None

    @classmethod
    def create_pipeline(
        cls,
        *,
        mesh_device: ttnn.MeshDevice,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_links: int | None = None,
        dynamic_load: bool | None = None,
        topology: ttnn.Topology | None = None,
        is_fsdp: bool | None = None,
        boundary_ratio: float | None = 0.875,
        lora_high: LoRAArg = None,
        lora_low: LoRAArg = None,
    ) -> WanPipelineI2VLora:
        config = WanPipelineConfig.default(
            mesh_shape=mesh_device.shape,
            checkpoint_name=cls.BASE_DIFFUSERS_REPO,
            height=height,
            width=width,
            num_frames=num_frames,
            num_links=num_links,
            topology=topology,
            dynamic_load=dynamic_load,
            is_fsdp=is_fsdp,
            boundary_ratio=boundary_ratio,
            model_type="i2v",
        )
        return cls(
            device=mesh_device,
            config=config,
            lora_high=lora_high,
            lora_low=lora_low,
        )
