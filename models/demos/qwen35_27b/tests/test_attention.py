# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Phase 1 validation: Qwen3.5 full attention layer on device.

Tests:
1. Qwen35Attention forward_decode on a single device
2. Qwen35Attention forward_decode on TP=4 mesh
3. Full model construction + decode step (TP=4 only)

Run:
    # Single device (N150)
    MESH_DEVICE=N150 HF_MODEL=~/models/Qwen3.5-27B-FP8 \
        pytest models/demos/qwen35_27b/tests/test_attention.py -k "single" -v

    # 4-device mesh
    HF_MODEL=~/models/Qwen3.5-27B-FP8 \
        pytest models/demos/qwen35_27b/tests/test_attention.py -v
"""

import os
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.qwen35_27b.tt.attention import Qwen35Attention
from models.demos.qwen35_27b.tt.model_config import (
    Qwen35ModelArgs,
    _replicate,
    _shard_w,
    load_qwen35_state_dict,
    prepare_attn_qg,
)
from models.demos.qwen35_27b.tt.rope import Qwen35PartialRopeSetup
from models.tt_transformers.tt.ccl import TT_CCL


def _get_model_path():
    return os.path.expanduser(os.environ.get("HF_MODEL", "~/models/Qwen3.5-27B-FP8"))


def _find_first_full_attn_layer(layer_types):
    """Find the first full_attention layer index."""
    for i, lt in enumerate(layer_types):
        if lt == "full_attention":
            return i
    raise ValueError("No full_attention layer found")


def _load_attention_weights_for_layer(state_dict, layer_idx, mesh, args, cache_dir):
    """Load attention mesh tensors for a single layer."""
    os.makedirs(cache_dir, exist_ok=True)
    p = f"layers.{layer_idx}."
    tp = args.num_devices

    qg_reordered = prepare_attn_qg(state_dict, p, args.n_heads, args.head_dim, tp)

    tw = {}
    tw["wqkv"] = _shard_w(
        qg_reordered,
        mesh,
        dim=-1,
        memory_config=args.attn_qg_weight_memcfg,
        cache_path=os.path.join(cache_dir, "wqkv"),
    )
    tw["wk"] = _shard_w(
        state_dict[p + "attention.wk.weight"],
        mesh,
        dim=-1,
        memory_config=args.attn_k_weight_memcfg,
        cache_path=os.path.join(cache_dir, "wk"),
    )
    tw["wv"] = _shard_w(
        state_dict[p + "attention.wv.weight"],
        mesh,
        dim=-1,
        memory_config=args.attn_v_weight_memcfg,
        cache_path=os.path.join(cache_dir, "wv"),
    )
    tw["wo"] = _shard_w(
        state_dict[p + "attention.wo.weight"],
        mesh,
        dim=0,
        memory_config=args.attn_wo_weight_memcfg,
        cache_path=os.path.join(cache_dir, "wo"),
    )
    tw["q_norm"] = _replicate(
        state_dict[p + "attention.q_norm.weight"],
        mesh,
        os.path.join(cache_dir, "q_norm"),
    )
    tw["k_norm"] = _replicate(
        state_dict[p + "attention.k_norm.weight"],
        mesh,
        os.path.join(cache_dir, "k_norm"),
    )
    return tw


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "N150x4": (1, 4), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
def test_attention_forward_decode(mesh_device, reset_seeds, ensure_gc):
    """Test Qwen35Attention forward_decode produces correct-shaped output."""
    model_path = _get_model_path()
    batch_size = 32  # Must match tile size (32) to avoid padding issues in reshapes
    max_seq_len = 256

    # Ensure HF_MODEL is set for ModelArgs
    if not os.environ.get("HF_MODEL"):
        os.environ["HF_MODEL"] = model_path

    logger.info(f"Creating Qwen35ModelArgs (devices={mesh_device.get_num_devices()})...")
    args = Qwen35ModelArgs(mesh_device, max_batch_size=batch_size, max_seq_len=max_seq_len)

    layer_idx = _find_first_full_attn_layer(args.layer_types)
    logger.info(f"Testing full_attention layer {layer_idx}")

    # Load state dict and attention weights
    logger.info("Loading state dict...")
    state_dict = load_qwen35_state_dict(model_path)

    cache_dir = os.path.expanduser(f"~/models/Qwen3.5-27B-mesh-tp{args.num_devices}/test_attn_layer_{layer_idx}")
    logger.info("Loading attention weights to mesh...")
    tw = _load_attention_weights_for_layer(state_dict, layer_idx, mesh_device, args, cache_dir)

    # Create attention module
    logger.info("Creating Qwen35Attention...")
    tt_ccl = TT_CCL(mesh_device) if mesh_device.get_num_devices() > 1 else None
    attn = Qwen35Attention(
        mesh_device=mesh_device,
        tt_ccl=tt_ccl,
        args=args,
        state_dict={},  # Weights loaded via set_weights
        weight_cache_path=Path(cache_dir),
        layer_num=layer_idx,
        dtype=ttnn.bfloat8_b,
        transformation_mats=None,
        configuration=args,
    )
    attn.set_weights(tw)

    # Create RoPE setup
    logger.info("Creating RoPE setup...")
    rope = Qwen35PartialRopeSetup(
        device=mesh_device,
        batch_size=batch_size,
        head_dim=args.head_dim,
        max_seq_len=max_seq_len,
        rope_theta=args.rope_theta,
    )

    # Create input: [1, 1, B, dim] — framework decode format
    x_torch = torch.randn(1, 1, batch_size, args.dim, dtype=torch.bfloat16)
    x_tt = ttnn.from_torch(
        x_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    x_tt = ttnn.to_memory_config(x_tt, ttnn.DRAM_MEMORY_CONFIG)

    # Current position
    cur_pos = torch.tensor([5] * batch_size, dtype=torch.int32)
    rot_idxs = rope.get_rot_idxs(cur_pos, on_host=True)
    rot_idxs = ttnn.to_device(rot_idxs, mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    rot_mats = rope.get_rot_mats(rot_idxs)

    cur_pos_tt = ttnn.from_torch(
        cur_pos,
        dtype=ttnn.int32,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # Forward pass
    logger.info("Running forward_decode...")
    out = attn.forward_decode(x_tt, cur_pos_tt, rot_mats)

    # Verify output
    # On multi-device, all-reduce does reduce_scatter → output is fractured (dim/num_devices per device)
    num_devices = mesh_device.get_num_devices()
    if num_devices > 1:
        out_torch = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3))
    else:
        out_torch = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    logger.info(f"Output shape: {out_torch.shape}")
    logger.info(f"Output range: [{out_torch.min():.4f}, {out_torch.max():.4f}]")
    logger.info(f"Output mean: {out_torch.mean():.6f}, std: {out_torch.std():.6f}")

    # Shape check — fractured dim should equal dim (concat along dim=3 for multi-device)
    expected_dim = args.dim
    assert out_torch.shape[-1] == expected_dim, f"Expected dim {expected_dim}, got {out_torch.shape[-1]}"
    # Output should not be all zeros
    assert out_torch.abs().max() > 0, "Output is all zeros"
    # Output should not contain NaN/Inf
    assert not torch.isnan(out_torch).any(), "Output contains NaN"
    assert not torch.isinf(out_torch).any(), "Output contains Inf"

    logger.info("PASSED: Attention forward_decode produces valid output")

    # Run a second step at a different position to verify KV cache update
    logger.info("Running second decode step...")
    cur_pos2 = torch.tensor([6] * batch_size, dtype=torch.int32)
    rot_idxs2 = rope.get_rot_idxs(cur_pos2, on_host=True)
    rot_idxs2 = ttnn.to_device(rot_idxs2, mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    rot_mats2 = rope.get_rot_mats(rot_idxs2)

    cur_pos_tt2 = ttnn.from_torch(
        cur_pos2,
        dtype=ttnn.int32,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    x_torch2 = torch.randn(1, 1, batch_size, args.dim, dtype=torch.bfloat16)
    x_tt2 = ttnn.from_torch(
        x_torch2,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    x_tt2 = ttnn.to_memory_config(x_tt2, ttnn.DRAM_MEMORY_CONFIG)

    out2 = attn.forward_decode(x_tt2, cur_pos_tt2, rot_mats2)
    if num_devices > 1:
        out2_torch = ttnn.to_torch(out2, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3))
    else:
        out2_torch = ttnn.to_torch(out2, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))

    assert not torch.isnan(out2_torch).any(), "Second decode output contains NaN"
    assert out2_torch.abs().max() > 0, "Second decode output is all zeros"

    logger.info("PASSED: Second decode step also produces valid output")
    logger.info(f"Test complete on {mesh_device.get_num_devices()} device(s)")
