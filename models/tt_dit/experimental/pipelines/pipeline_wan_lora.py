# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Wan2.2 I2V pipeline with LoRA adapters fused into the base weights.

Each expert (high/low noise) takes an ordered LoRA stack; stacks are fused
on CPU before TT conversion so inference has no LoRA-specific runtime cost.
See ``experimental/models/Wan2_2_LoRA.md`` for the adapter-key formats
detected by ``fuse_lora_state_dict`` and the supported namespaces.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from diffusers.schedulers import UniPCMultistepScheduler
from loguru import logger
from PIL import Image
from safetensors.torch import load_file

import ttnn
from models.tt_dit.experimental.utils.lightx2v_loader import wan_lightx2v_to_diffusers_key
from models.tt_dit.pipelines.wan.pipeline_wan import WanPipeline, WanPipelineConfig
from models.tt_dit.pipelines.wan.pipeline_wan_i2v import ImagePrompt, WanPipelineI2V
from models.tt_dit.solvers import UniPCSolver, UniPCVariant
from models.tt_dit.utils import cache
from models.tt_dit.utils.conv3d import conv_pad_height, conv_pad_in_channels, conv_pad_width
from models.tt_dit.utils.tensor import bf16_tensor_2dshard, fast_device_to_host


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


# encoder_t_chunk_size per mesh. None = full-T single pass, N = chunked with feat_cache;
# the two are numerically identical and full-T is faster.
_ENCODER_T_CHUNK_BY_MESH = {
    (4, 8): 16,
    (4, 32): None,
}


def _resolve_checkpoint(repo_id: str) -> str:
    """Prefer a local checkpoint directory over a hub repo id.

    All ranks share one HF cache on NFS; resolving by repo id makes each rank hit the hub
    and write cache metadata concurrently, and a rank that loses that race fails with a
    spurious missing-weights error. A directory path skips hub resolution.

    WAN22_I2V_CHECKPOINT_DIR must share the repo id's basename: the tt_dit cache namespace
    is os.path.basename(checkpoint_name), so a different basename orphans the compiled
    cache. Falls back to the repo id when unset, missing, or mismatched.
    """
    local = os.environ.get("WAN22_I2V_CHECKPOINT_DIR")
    if not local:
        return repo_id
    if not os.path.isdir(local):
        logger.warning(f"WAN22_I2V_CHECKPOINT_DIR={local!r} is not a directory; falling back to {repo_id}")
        return repo_id
    if os.path.basename(local.rstrip("/")) != os.path.basename(repo_id):
        logger.warning(
            f"WAN22_I2V_CHECKPOINT_DIR={local!r} basename does not match {repo_id!r}; "
            "using it would orphan the tt_dit cache. Falling back to the repo id."
        )
        return repo_id
    logger.info(f"Resolving {repo_id} from local checkpoint dir {local} (no hub lookup)")
    return local


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
        flow_shift: float | None = None,
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

        # Distilled few-step LoRAs need a different flow shift than the base schedule,
        # which WanPipelineConfig pins at 12.0. Rebuilt here instead of plumbing it through
        # the shared config. __call__ re-derives sigmas per call so a post-init swap takes
        # effect, and this runs before the first traced call. Solver order/variant are
        # unchanged by flow_shift, so the compiled graph is identical.
        if flow_shift is not None:
            self._scheduler = UniPCMultistepScheduler.from_pretrained(
                self.checkpoint_name, subfolder="scheduler", flow_shift=flow_shift
            )
            self._solver = UniPCSolver(
                order=self._scheduler.config.solver_order,
                variant=UniPCVariant(self._scheduler.config.solver_type),
            )
            logger.info(f"WanPipelineI2VLora: scheduler flow_shift overridden to {flow_shift}")

    def prepare_latents(
        self,
        batch_size: int,
        image_prompt,
        num_channels_latents: int = 16,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        """I2V conditioning that uploads only the conditioned frames.

        The base implementation materializes the whole conditioning video on the host
        (896 MB float32 at 81x720x1280) and uploads all of it, though every frame but the
        conditioned one is zero. Here the real frames are padded and uploaded individually
        and the zero runs are built on device, then encoded in one pass.
        """
        assert batch_size == 1, "Only batch size 1 is currently supported for I2V"

        if isinstance(image_prompt, ImagePrompt):
            image_prompt = [image_prompt]
        elif isinstance(image_prompt, Image.Image):
            image_prompt = [ImagePrompt(image=image_prompt, frame_pos=0)]

        # Skip WanPipelineI2V.prepare_latents (the path being replaced) and take the
        # plain noise-latent allocation from the grandparent.
        latents, _ = WanPipeline.prepare_latents(
            self,
            batch_size=batch_size,
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            num_frames=num_frames,
            dtype=dtype,
            device=device,
        )
        latent_shape = latents.shape
        h_lat, w_lat = latent_shape[-2], latent_shape[-1]

        # ---- host: preprocess + pad ONLY the conditioned frames --------------
        cond_by_pos: dict[int, ttnn.Tensor] = {}
        logical_h = None
        seen: set[int] = set()
        for image, frame_pos in image_prompt:
            assert frame_pos not in seen, f"Frame position {frame_pos} already processed."
            seen.add(frame_pos)
            img = self.video_processor.preprocess(image, height=height, width=width).to(device, dtype=torch.float32)
            # (B,C,H,W) -> (B,T=1,H,W,C)
            frame_BTHWC = img.unsqueeze(2).permute(0, 2, 3, 4, 1)
            frame_BTHWC = conv_pad_in_channels(frame_BTHWC)
            frame_BTHWC, logical_h = conv_pad_height(
                frame_BTHWC, self.vae_parallel_config.height_parallel.factor * self.vae_scale_factor_spatial
            )
            frame_BTHWC, _logical_w = conv_pad_width(
                frame_BTHWC, self.vae_parallel_config.width_parallel.factor * self.vae_scale_factor_spatial
            )
            cond_by_pos[frame_pos] = bf16_tensor_2dshard(
                frame_BTHWC,
                self.mesh_device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                shard_mapping={
                    self.vae_parallel_config.height_parallel.mesh_axis: 2,
                    self.vae_parallel_config.width_parallel.mesh_axis: 3,
                },
            )
            logical_w = _logical_w

        tt_zero_1 = None

        def _zero_run(n: int) -> ttnn.Tensor:
            nonlocal tt_zero_1
            if tt_zero_1 is None:
                tt_zero_1 = ttnn.zeros_like(next(iter(cond_by_pos.values())))
            z, built = tt_zero_1, 1
            while built * 2 <= n:
                z = ttnn.concat([z, z], dim=1)
                built *= 2
            if built < n:
                z = ttnn.concat([z, z[:, : n - built, :, :, :]], dim=1)
            return z

        segments: list[ttnn.Tensor] = []
        zero_start = None
        for i in range(num_frames):
            if i in cond_by_pos:
                if zero_start is not None:
                    segments.append(_zero_run(i - zero_start))
                    zero_start = None
                segments.append(cond_by_pos[i])
            elif zero_start is None:
                zero_start = i
        if zero_start is not None:
            segments.append(_zero_run(num_frames - zero_start))

        tt_video_BTHWC = segments[0] if len(segments) == 1 else ttnn.concat(segments, dim=1)

        chunk = _ENCODER_T_CHUNK_BY_MESH.get(tuple(self.mesh_device.shape))

        encoded_BCTHW, new_logical_h, new_logical_w = self.tt_vae_encoder(
            tt_video_BTHWC, logical_h, logical_w=logical_w, encoder_t_chunk_size=chunk
        )

        # tt_video_BTHWC may alias a conditioned frame or the zero tensor when there is
        # only one segment, so guard against a double free.
        owned = list(cond_by_pos.values()) + ([tt_zero_1] if tt_zero_1 is not None else [])
        if all(tt_video_BTHWC is not t for t in owned):
            ttnn.deallocate(tt_video_BTHWC)
        for tt in owned:
            ttnn.deallocate(tt)

        # ---- host: replicate + normalize + mask ------------------------------
        concat_dims = [None, None]
        concat_dims[self.vae_parallel_config.height_parallel.mesh_axis] = 3
        concat_dims[self.vae_parallel_config.width_parallel.mesh_axis] = 4
        encoded = fast_device_to_host(encoded_BCTHW, self.mesh_device, concat_dims, ccl_manager=self.vae_ccl_manager)
        ttnn.deallocate(encoded_BCTHW)
        ttnn.synchronize_device(self.mesh_device)
        # Same crop convention as WanPipelineI2V.prepare_latents.
        encoded = encoded[:, :, :, :new_logical_h, :new_logical_w].to(dtype=dtype)

        f_lat_full = latent_shape[2]
        if encoded.shape[2] != f_lat_full:
            encoded = encoded[:, :, :f_lat_full, :, :]

        latents_mean = (
            torch.tensor(self._vae.config.latents_mean)
            .view(1, self._vae.config.z_dim, 1, 1, 1)
            .to(encoded.device, encoded.dtype)
        )
        latents_std = 1.0 / torch.tensor(self._vae.config.latents_std).view(1, self._vae.config.z_dim, 1, 1, 1).to(
            encoded.device, encoded.dtype
        )
        encoded = (encoded - latents_mean) * latents_std

        msk = torch.zeros(batch_size, num_frames, h_lat, w_lat)
        for pos in cond_by_pos:
            msk[:, pos, :, :] = 1
        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, h_lat, w_lat)
        msk = msk.transpose(1, 2)

        y = torch.cat([msk, encoded], dim=1)
        return latents, y

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
            mesh_device=self.mesh_device,
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
        flow_shift: float | None = None,
        lora_high: LoRAArg = None,
        lora_low: LoRAArg = None,
    ) -> WanPipelineI2VLora:
        config = WanPipelineConfig.default(
            mesh_shape=mesh_device.shape,
            checkpoint_name=_resolve_checkpoint(cls.BASE_DIFFUSERS_REPO),
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
            flow_shift=flow_shift,
        )
