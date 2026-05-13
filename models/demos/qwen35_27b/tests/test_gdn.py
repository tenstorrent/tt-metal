# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Phase 2 validation: Qwen3.5 GDN (Gated DeltaNet) linear attention on device.

Tests:
1. TtGatedDeltaNet forward_decode on a single device or TP mesh

Run:
    MESH_DEVICE=N150 HF_MODEL=~/models/Qwen3.5-27B-FP8 \
        pytest models/demos/qwen35_27b/tests/test_gdn.py -v
"""

import os
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.qwen35_27b.tt.gdn import TtGatedDeltaNet
from models.demos.qwen35_27b.tt.model_config import (
    GDN_CONV_KERNEL_SIZE,
    Qwen35ModelArgs,
    _replicate,
    _shard_small,
    _shard_w,
    load_qwen35_state_dict,
    prepare_conv_taps,
    prepare_gdn_qkv,
)
from models.tt_transformers.tt.ccl import TT_CCL


def _get_model_path():
    return os.path.expanduser(os.environ.get("HF_MODEL", "~/models/Qwen3.5-27B-FP8"))


def _find_first_linear_attn_layer(layer_types):
    for i, lt in enumerate(layer_types):
        if lt == "linear_attention":
            return i
    raise ValueError("No linear_attention layer found")


def _load_gdn_weights_for_layer(state_dict, layer_idx, mesh, args, cache_dir):
    """Load GDN mesh tensors for a single layer."""
    os.makedirs(cache_dir, exist_ok=True)
    p = f"layers.{layer_idx}."
    tp = args.num_devices

    tw = {}

    # Fused QKV+Z
    qkv_reordered = prepare_gdn_qkv(state_dict, p, tp)
    z_weight = state_dict[p + "linear_attn.in_proj_z.weight"]
    qkv_per = args.gdn_qkv_dim_tp
    z_per = args.gdn_z_dim_tp
    fused_parts = []
    for d in range(tp):
        fused_parts.append(
            torch.cat(
                [
                    qkv_reordered[d * qkv_per : (d + 1) * qkv_per, :],
                    z_weight[d * z_per : (d + 1) * z_per, :],
                ],
                dim=0,
            )
        )
    qkvz_fused = torch.cat(fused_parts, dim=0)
    tw["qkvz"] = _shard_w(
        qkvz_fused,
        mesh,
        dim=-1,
        memory_config=args.gdn_qkvz_weight_memcfg,
        cache_path=os.path.join(cache_dir, "qkvz"),
    )

    # Fused A+B
    a_w = state_dict[p + "linear_attn.in_proj_a.weight"]
    b_w = state_dict[p + "linear_attn.in_proj_b.weight"]
    a_per = args.gdn_nv_tp
    b_per = args.gdn_nv_tp
    ab_parts = []
    for d in range(tp):
        ab_parts.append(
            torch.cat(
                [
                    a_w[d * a_per : (d + 1) * a_per, :],
                    b_w[d * b_per : (d + 1) * b_per, :],
                ],
                dim=0,
            )
        )
    ab_fused = torch.cat(ab_parts, dim=0)
    tw["ab"] = _shard_w(
        ab_fused,
        mesh,
        dim=-1,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        cache_path=os.path.join(cache_dir, "ab"),
    )

    # Output projection
    tw["out"] = _shard_w(
        state_dict[p + "linear_attn.out_proj.weight"],
        mesh,
        dim=0,
        memory_config=args.gdn_out_weight_memcfg,
        cache_path=os.path.join(cache_dir, "out"),
    )

    # Per-head params
    tw["A_log"] = _shard_small(
        state_dict[p + "linear_attn.A_log"].float(),
        mesh,
        os.path.join(cache_dir, "A_log"),
    )
    tw["dt_bias"] = _shard_small(
        state_dict[p + "linear_attn.dt_bias"].float(),
        mesh,
        os.path.join(cache_dir, "dt_bias"),
    )
    tw["norm_w"] = _replicate(
        state_dict[p + "linear_attn.norm.weight"].float(),
        mesh,
        os.path.join(cache_dir, "norm_w"),
    )

    # Conv taps
    taps = prepare_conv_taps(state_dict, p, tp)
    tw["conv_taps"] = [
        _shard_small(taps[j], mesh, os.path.join(cache_dir, f"conv_tap_{j}")) for j in range(GDN_CONV_KERNEL_SIZE)
    ]

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
def test_gdn_forward_decode(mesh_device, reset_seeds, ensure_gc):
    """Test TtGatedDeltaNet forward_decode produces correct-shaped output.

    Note: GDN requires TP>=4 due to L1 memory constraints (unsharded dims too large for single device).
    """
    model_path = _get_model_path()
    batch_size = 32
    max_seq_len = 256

    if mesh_device.get_num_devices() < 4:
        pytest.skip("GDN requires TP>=4 (L1 memory constraints)")

    if not os.environ.get("HF_MODEL"):
        os.environ["HF_MODEL"] = model_path

    logger.info(f"Creating Qwen35ModelArgs (devices={mesh_device.get_num_devices()})...")
    args = Qwen35ModelArgs(mesh_device, max_batch_size=batch_size, max_seq_len=max_seq_len)

    layer_idx = _find_first_linear_attn_layer(args.layer_types)
    logger.info(f"Testing linear_attention layer {layer_idx}")

    logger.info("Loading state dict...")
    state_dict = load_qwen35_state_dict(model_path)

    cache_dir = os.path.expanduser(f"~/models/Qwen3.5-27B-mesh-tp{args.num_devices}/test_gdn_layer_{layer_idx}")
    logger.info("Loading GDN weights to mesh...")
    tw = _load_gdn_weights_for_layer(state_dict, layer_idx, mesh_device, args, cache_dir)

    # Create GDN module
    logger.info("Creating TtGatedDeltaNet...")
    tt_ccl = TT_CCL(mesh_device) if mesh_device.get_num_devices() > 1 else None
    gdn = TtGatedDeltaNet(
        mesh_device=mesh_device,
        tt_ccl=tt_ccl,
        args=args,
        state_dict={},
        weight_cache_path=Path(cache_dir),
        layer_num=layer_idx,
        dtype=ttnn.bfloat8_b,
        transformation_mats=None,
        configuration=args,
    )
    gdn.set_weights(tw)

    # Create input: [1, 1, B, dim]
    x_torch = torch.randn(1, 1, batch_size, args.dim, dtype=torch.bfloat16)
    x_tt = ttnn.from_torch(
        x_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    x_tt = ttnn.to_memory_config(x_tt, ttnn.DRAM_MEMORY_CONFIG)

    # Forward pass (GDN doesn't use cur_pos or rot_mats)
    logger.info("Running forward_decode...")
    out = gdn.forward_decode(x_tt)

    # Verify output
    num_devices = mesh_device.get_num_devices()
    if num_devices > 1:
        out_torch = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3))
    else:
        out_torch = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    logger.info(f"Output shape: {out_torch.shape}")
    logger.info(f"Output range: [{out_torch.min():.4f}, {out_torch.max():.4f}]")
    logger.info(f"Output mean: {out_torch.mean():.6f}, std: {out_torch.std():.6f}")

    expected_dim = args.dim
    assert out_torch.shape[-1] == expected_dim, f"Expected dim {expected_dim}, got {out_torch.shape[-1]}"
    assert out_torch.abs().max() > 0, "Output is all zeros"
    assert not torch.isnan(out_torch).any(), "Output contains NaN"
    assert not torch.isinf(out_torch).any(), "Output contains Inf"

    logger.info("PASSED: GDN forward_decode produces valid output")

    # Run a second step to verify state update
    logger.info("Running second decode step...")
    x_torch2 = torch.randn(1, 1, batch_size, args.dim, dtype=torch.bfloat16)
    x_tt2 = ttnn.from_torch(
        x_torch2,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    x_tt2 = ttnn.to_memory_config(x_tt2, ttnn.DRAM_MEMORY_CONFIG)

    out2 = gdn.forward_decode(x_tt2)
    if num_devices > 1:
        out2_torch = ttnn.to_torch(out2, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3))
    else:
        out2_torch = ttnn.to_torch(out2, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))

    assert not torch.isnan(out2_torch).any(), "Second decode output contains NaN"
    assert out2_torch.abs().max() > 0, "Second decode output is all zeros"

    logger.info("PASSED: Second decode step also produces valid output")
    logger.info(f"Test complete on {mesh_device.get_num_devices()} device(s)")
