# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Parity tests for Mistral3MultiModalProjector.

Validates ``Mistral3MultiModalProjector.forward_prefill`` against a hand-rolled
PyTorch reference that mirrors ``references/model.py::Mistral3MultiModalProjector``.

Two test variants:
  1. ``test_multi_modal_projector_synthetic`` – tiny synthetic config and random
     weights; always runs (no checkpoint required).
  2. ``test_multi_modal_projector_checkpoint`` – loads real projector weights from
     ``models/mistral_small_4/``; skipped when the snapshot is absent.

Config (tiny, used in synthetic test):
    vision_hidden   = 32   (patch_size=4, spatial_merge_size=2)
    text_hidden     = 64
    image: 8×8 pixels  →  2×2 patch grid  →  1 merged token (after 2×2 merge)

PCC requirement: ≥ 0.97 for bfloat16 / bfloat8 weights.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import List, Tuple

import pytest
import torch
import torch.nn.functional as F
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.mistral_small_4_119B.tt.multi_modal_projector.multi_modal_projector import Mistral3MultiModalProjector
from models.demos.mistral_small_4_119B.tt_utils.run_config import create_run_config

# ─── Tiny config factory ─────────────────────────────────────────────────────


def _tiny_config(
    vision_hidden: int = 32,
    text_hidden: int = 64,
    patch_size: int = 4,
    spatial_merge_size: int = 2,
):
    """Build a SimpleNamespace config that satisfies Mistral3MultiModalProjector's
    attribute accesses without requiring a real HuggingFace Mistral3Config."""
    vision_cfg = SimpleNamespace(
        hidden_size=vision_hidden,
        patch_size=patch_size,
        rms_norm_eps=1e-6,
    )
    text_cfg = SimpleNamespace(
        hidden_size=text_hidden,
        rms_norm_eps=1e-6,
    )
    return SimpleNamespace(
        vision_config=vision_cfg,
        text_config=text_cfg,
        spatial_merge_size=spatial_merge_size,
        vision_feature_layer=-1,  # int → num_feature_layers = 1
        multimodal_projector_bias=False,
        projector_hidden_act="gelu",
    )


# ─── Reference (torch) implementations ───────────────────────────────────────


def _rms_norm_ref(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Numerically equivalent to Mistral3RMSNorm."""
    variance = x.float().pow(2).mean(-1, keepdim=True)
    x_normed = x.float() * torch.rsqrt(variance + eps)
    return (weight.float() * x_normed).to(x.dtype)


def _patch_merge_ref(
    x: torch.Tensor,
    merging_weight: torch.Tensor,
    image_sizes: List[Tuple[int, int]],
    patch_size: int,
    s: int,
) -> torch.Tensor:
    """Reference patch merger matching Mistral3PatchMerger.forward."""
    grid_sizes = [(h // patch_size, w // patch_size) for h, w in image_sizes]
    tokens_per_image = [gh * gw for gh, gw in grid_sizes]
    d = x.shape[-1]

    parts: list[torch.Tensor] = []
    offset = 0
    for idx, (gh, gw) in enumerate(grid_sizes):
        n = tokens_per_image[idx]
        tokens = x[offset : offset + n]  # [gh*gw, d]
        offset += n
        grid = tokens.view(gh, gw, d).permute(2, 0, 1).unsqueeze(0)  # [1,d,gh,gw]
        unfolded = F.unfold(grid.float(), kernel_size=s, stride=s)  # [1, d*s², merged]
        merged = unfolded.view(d * s * s, -1).t().to(x.dtype)  # [merged, d*s²]
        parts.append(merged)

    merged_all = torch.cat(parts, dim=0)  # [total_merged, d*s²]
    return F.linear(merged_all, merging_weight)  # [total_merged, d]


def _reference_forward(
    vision_features: torch.Tensor,
    image_sizes: List[Tuple[int, int]],
    weights: dict,
    cfg,
) -> torch.Tensor:
    """Full reference forward matching Mistral3MultiModalProjector.forward."""
    eps = cfg.text_config.rms_norm_eps
    s = cfg.spatial_merge_size
    ps = cfg.vision_config.patch_size

    # 1. RMSNorm
    x = _rms_norm_ref(vision_features, weights["norm.weight"], eps)

    # 2. Patch merge
    x = _patch_merge_ref(x, weights["patch_merger.merging_layer.weight"], image_sizes, ps, s)

    # 3. linear_1 + GELU + linear_2
    x = F.linear(x, weights["linear_1.weight"])
    x = F.gelu(x)
    x = F.linear(x, weights["linear_2.weight"])
    return x


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _build_random_weights(cfg, seed: int = 42) -> dict:
    torch.manual_seed(seed)
    vh = cfg.vision_config.hidden_size
    th = cfg.text_config.hidden_size
    s = cfg.spatial_merge_size
    return {
        "norm.weight": torch.ones(vh, dtype=torch.bfloat16),
        "patch_merger.merging_layer.weight": torch.randn(vh, vh * s * s, dtype=torch.bfloat16) * 0.02,
        "linear_1.weight": torch.randn(th, vh, dtype=torch.bfloat16) * 0.02,
        "linear_2.weight": torch.randn(th, th, dtype=torch.bfloat16) * 0.02,
    }


def _iter_weight_tensors(weight_config):
    stack = [weight_config]
    while stack:
        node = stack.pop()
        if isinstance(node, ttnn.Tensor):
            yield node
        elif isinstance(node, dict):
            stack.extend(node.values())
        elif isinstance(node, (list, tuple)):
            stack.extend(node)


def _assert_pcc(tt_out: torch.Tensor, ref_out: torch.Tensor, *, pcc_required: float) -> None:
    tt = tt_out.cpu().float()
    ref = ref_out.cpu().float()

    # Align dims
    while tt.ndim < ref.ndim:
        tt = tt.unsqueeze(0)
    while ref.ndim < tt.ndim:
        ref = ref.unsqueeze(0)

    # Align seq (tt may be tile-padded)
    seq = min(tt.shape[-2], ref.shape[-2])
    tt = tt[..., :seq, :]
    ref = ref[..., :seq, :]

    passing, pcc = comp_pcc(tt, ref, pcc_required)
    logger.info(f"multi_modal_projector PCC: {pcc:.6f}")
    assert passing, f"PCC {pcc:.6f} < required {pcc_required}"


def _snapshot_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "models" / "mistral_small_4"


# ─── Synthetic test ───────────────────────────────────────────────────────────


def test_multi_modal_projector_synthetic(device, tmp_path):
    """``forward_prefill`` vs torch reference – tiny config, random weights.

    Image layout used:
        Two images: 8×8 pixels and 8×8 pixels.
        patch_size=4 → 2×2 = 4 patches each.
        spatial_merge_size=2 → 1 merged token each → 2 total merged tokens.
    """
    cfg = _tiny_config(vision_hidden=32, text_hidden=64, patch_size=4, spatial_merge_size=2)
    weights = _build_random_weights(cfg, seed=7)

    # Two 8×8 images → 4 patches each → 8 total patches
    image_sizes = [(8, 8), (8, 8)]
    total_patches = sum(
        (h // cfg.vision_config.patch_size) * (w // cfg.vision_config.patch_size) for h, w in image_sizes
    )
    vision_h = cfg.vision_config.hidden_size

    torch.manual_seed(0)
    vision_features = torch.randn(total_patches, vision_h, dtype=torch.bfloat16)

    # ── Reference ──────────────────────────────────────────────────────
    with torch.no_grad():
        ref_out = _reference_forward(vision_features, image_sizes, weights, cfg)
    # ref_out: [total_merged_patches, text_hidden]

    # ── TT path ────────────────────────────────────────────────────────
    weight_cfg = Mistral3MultiModalProjector.convert_weights(cfg, (weights,), tmp_path / "proj", device)
    model_cfg = Mistral3MultiModalProjector.prefill_model_config(cfg, device)
    model_state = Mistral3MultiModalProjector.create_state(cfg, device)
    run_cfg = create_run_config(model_cfg, weight_cfg, model_state, {})

    # Inject runtime image sizes
    run_cfg["image_sizes"] = image_sizes

    # Input: [1, 1, total_patches, vision_h] replicated
    tt_input = ttnn.from_torch(
        vision_features.unsqueeze(0).unsqueeze(0),
        device=device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    tt_output = None
    try:
        tt_output = Mistral3MultiModalProjector.forward_prefill(tt_input, run_cfg)

        R, C = tuple(device.shape)
        tt_out_torch = ttnn.to_torch(
            tt_output,
            mesh_composer=ttnn.ConcatMesh2dToTensor(device, dims=(0, -1), mesh_shape=(R, C)),
        )
        # [R, 1, merged_padded, text_h*C]  → take first replica
        text_h = cfg.text_config.hidden_size
        tt_out_torch = tt_out_torch[0, 0, :, :text_h]  # [merged_padded, text_h]

        logger.info(f"TT output shape: {tt_out_torch.shape}, ref shape: {ref_out.shape}")
        _assert_pcc(tt_out_torch, ref_out, pcc_required=0.97)
    finally:
        if tt_input is not None:
            ttnn.deallocate(tt_input)
        if tt_output is not None:
            ttnn.deallocate(tt_output)
        for t in _iter_weight_tensors(weight_cfg):
            ttnn.deallocate(t)


# ─── Synthetic test – single image ───────────────────────────────────────────


def test_multi_modal_projector_single_image(device, tmp_path):
    """Same as synthetic test but with a single non-square image (16×8 pixels)."""
    cfg = _tiny_config(vision_hidden=32, text_hidden=64, patch_size=4, spatial_merge_size=2)
    weights = _build_random_weights(cfg, seed=13)

    # 16×8 image: 4×2 = 8 patches → 2 merged tokens (2×1 after 2×2 merge, non-square)
    image_sizes = [(16, 8)]
    gh, gw = 16 // 4, 8 // 4  # 4×2 grid
    total_patches = gh * gw
    vision_h = cfg.vision_config.hidden_size

    torch.manual_seed(1)
    vision_features = torch.randn(total_patches, vision_h, dtype=torch.bfloat16)

    with torch.no_grad():
        ref_out = _reference_forward(vision_features, image_sizes, weights, cfg)

    weight_cfg = Mistral3MultiModalProjector.convert_weights(cfg, (weights,), tmp_path / "proj_single", device)
    model_cfg = Mistral3MultiModalProjector.prefill_model_config(cfg, device)
    model_state = Mistral3MultiModalProjector.create_state(cfg, device)
    run_cfg = create_run_config(model_cfg, weight_cfg, model_state, {})
    run_cfg["image_sizes"] = image_sizes

    tt_input = ttnn.from_torch(
        vision_features.unsqueeze(0).unsqueeze(0),
        device=device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    tt_output = None
    try:
        tt_output = Mistral3MultiModalProjector.forward_prefill(tt_input, run_cfg)

        R, C = tuple(device.shape)
        tt_out_torch = ttnn.to_torch(
            tt_output,
            mesh_composer=ttnn.ConcatMesh2dToTensor(device, dims=(0, -1), mesh_shape=(R, C)),
        )
        text_h = cfg.text_config.hidden_size
        tt_out_torch = tt_out_torch[0, 0, :, :text_h]

        logger.info(f"TT output shape: {tt_out_torch.shape}, ref shape: {ref_out.shape}")
        _assert_pcc(tt_out_torch, ref_out, pcc_required=0.97)
    finally:
        if tt_input is not None:
            ttnn.deallocate(tt_input)
        if tt_output is not None:
            ttnn.deallocate(tt_output)
        for t in _iter_weight_tensors(weight_cfg):
            ttnn.deallocate(t)


# ─── Checkpoint test ──────────────────────────────────────────────────────────


def _load_checkpoint_projector_weights(snapshot_dir: Path) -> dict | None:
    """Load multi_modal_projector weights from a HuggingFace sharded snapshot.

    Returns None if the snapshot is incomplete or the required tensors are absent.
    Requires ``safetensors`` and ``transformers``.
    """
    try:
        import json

        from safetensors import safe_open
    except ImportError:
        return None

    index_path = snapshot_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        return None

    with open(index_path) as f:
        index = json.load(f)

    prefix = "model.multi_modal_projector."
    keys_needed = {
        "norm.weight",
        "patch_merger.merging_layer.weight",
        "linear_1.weight",
        "linear_2.weight",
    }
    full_keys = {f"{prefix}{k}": k for k in keys_needed}

    # Find which shard files contain the needed keys
    shard_map: dict[str, list[str]] = {}
    for full_key, short_key in full_keys.items():
        shard_file = index.get("weight_map", {}).get(full_key)
        if shard_file is None:
            return None
        shard_map.setdefault(shard_file, []).append(full_key)

    loaded: dict[str, torch.Tensor] = {}
    for shard_file, needed_full_keys in shard_map.items():
        shard_path = snapshot_dir / shard_file
        if not shard_path.is_file():
            return None
        with safe_open(str(shard_path), framework="pt", device="cpu") as f:
            for full_key in needed_full_keys:
                loaded[full_keys[full_key]] = f.get_tensor(full_key)

    return loaded if len(loaded) == len(keys_needed) else None


def _load_checkpoint_hf_config(snapshot_dir: Path):
    """Load Mistral3Config from snapshot; return None if unavailable."""
    try:
        from transformers.models.mistral3.configuration_mistral3 import Mistral3Config

        config_path = snapshot_dir / "config.json"
        if not config_path.is_file():
            return None
        return Mistral3Config.from_pretrained(str(snapshot_dir))
    except Exception:
        return None


def test_multi_modal_projector_checkpoint(device, tmp_path):
    """``forward_prefill`` with real checkpoint weights (skip if snapshot absent).

    Uses random vision activations since running the full vision tower is out of
    scope here.  The test validates that the weight-conversion and linear-projection
    pipeline produces output that matches the torch reference with the same weights.
    """
    snap = _snapshot_dir()
    if not (snap / "config.json").is_file():
        pytest.skip("No config.json under models/mistral_small_4/ (install snapshot per download_model.py)")

    hf_config = _load_checkpoint_hf_config(snap)
    if hf_config is None:
        pytest.skip("Could not load Mistral3Config from snapshot")

    ckpt_weights = _load_checkpoint_projector_weights(snap)
    if ckpt_weights is None:
        pytest.skip("Could not load multi_modal_projector weights from snapshot")

    vh = hf_config.vision_config.hidden_size  # 1024
    th = hf_config.text_config.hidden_size  # 4096
    ps = hf_config.vision_config.patch_size  # 14
    s = hf_config.spatial_merge_size  # 2

    # Synthetic image: 28×28 pixels → 2×2 patch grid → 1 merged token
    image_sizes = [(28, 28)]
    total_patches = (28 // ps) ** 2  # 4

    torch.manual_seed(99)
    vision_features = torch.randn(total_patches, vh, dtype=torch.bfloat16) * 0.1

    # Reference forward with checkpoint weights
    with torch.no_grad():
        ref_out = _reference_forward(vision_features, image_sizes, ckpt_weights, hf_config)

    weight_cfg = Mistral3MultiModalProjector.convert_weights(hf_config, (ckpt_weights,), tmp_path / "proj_ckpt", device)
    model_cfg = Mistral3MultiModalProjector.prefill_model_config(hf_config, device)
    model_state = Mistral3MultiModalProjector.create_state(hf_config, device)
    run_cfg = create_run_config(model_cfg, weight_cfg, model_state, {})
    run_cfg["image_sizes"] = image_sizes

    tt_input = ttnn.from_torch(
        vision_features.unsqueeze(0).unsqueeze(0),
        device=device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    tt_output = None
    try:
        tt_output = Mistral3MultiModalProjector.forward_prefill(tt_input, run_cfg)

        R, C = tuple(device.shape)
        tt_out_torch = ttnn.to_torch(
            tt_output,
            mesh_composer=ttnn.ConcatMesh2dToTensor(device, dims=(0, -1), mesh_shape=(R, C)),
        )
        tt_out_torch = tt_out_torch[0, 0, :, :th]

        logger.info(f"[checkpoint] TT shape {tt_out_torch.shape}, ref shape {ref_out.shape}")
        _assert_pcc(tt_out_torch, ref_out, pcc_required=0.97)
    finally:
        if tt_input is not None:
            ttnn.deallocate(tt_input)
        if tt_output is not None:
            ttnn.deallocate(tt_output)
        for t in _iter_weight_tensors(weight_cfg):
            ttnn.deallocate(t)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
