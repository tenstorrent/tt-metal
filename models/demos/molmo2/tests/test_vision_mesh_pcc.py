# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Test vision backbone PCC on mesh device vs single device reference.

Mesh shape: N300 -> (1, 2), T3K -> (1, 8). Set MESH_DEVICE=N300 or T3K.
Use this test with Tracy to profile full vision pipeline (ViT + pooling + projector).

Profile with Tracy: python -m tracy -r -p -v -m pytest models/demos/molmo2/tests/test_vision_mesh_pcc.py -v -s
"""

import os

import numpy as np
import torch
from loguru import logger

import ttnn

# Mesh shape for multi-device: N300=2 chips, T3K=8 chips.
# If MESH_DEVICE is unset, use (1, num_devices) so the test runs on whatever hardware is available (e.g. N300 with 2 devices).
MESH_SHAPE_BY_DEVICE = {"N300": (1, 2), "T3K": (1, 8)}


def calculate_pcc(ref, out):
    """Calculate Pearson Correlation Coefficient."""
    if ref.shape != out.shape:
        return -1.0, f"Shape mismatch: {ref.shape} vs {out.shape}"
    ref_flat = ref.flatten().float()
    out_flat = out.flatten().float()
    ref_mean = ref_flat.mean()
    out_mean = out_flat.mean()
    numerator = ((ref_flat - ref_mean) * (out_flat - out_mean)).sum()
    denominator = torch.sqrt(((ref_flat - ref_mean) ** 2).sum() * ((out_flat - out_mean) ** 2).sum())
    if denominator == 0:
        return 1.0 if numerator == 0 else 0.0, "zero std"
    pcc = numerator / denominator
    return pcc.item(), "ok"


def test_vision_backbone_mesh_pcc():
    """Test vision backbone on mesh device."""
    from models.demos.molmo2.tests.test_vision_full_pcc import ref_full_vision_backbone
    from models.demos.molmo2.tt.load_weights import load_state_dict_from_safetensors

    logger.info("Loading weights...")
    state_dict = load_state_dict_from_safetensors("allenai/Molmo2-8B")
    logger.info("Loaded weights")

    # Create random image input
    torch.manual_seed(42)
    batch_size = 1
    pixel_values = torch.randn(batch_size, 3, 378, 378)

    # Create pooled_patches_idx
    patches_per_side = 27
    pool_h, pool_w = 2, 2

    idx_arr = np.arange(patches_per_side * patches_per_side).reshape(patches_per_side, patches_per_side)

    h_pad = pool_h * ((idx_arr.shape[0] + pool_h - 1) // pool_h) - idx_arr.shape[0]
    w_pad = pool_w * ((idx_arr.shape[1] + pool_w - 1) // pool_w) - idx_arr.shape[1]
    idx_arr = np.pad(
        idx_arr,
        [[h_pad // 2, (h_pad + 1) // 2], [w_pad // 2, (w_pad + 1) // 2]],
        constant_values=-1,
    )

    h_out = idx_arr.shape[0] // pool_h
    w_out = idx_arr.shape[1] // pool_w
    idx_arr = idx_arr.reshape(h_out, pool_h, w_out, pool_w)
    idx_arr = idx_arr.transpose(0, 2, 1, 3).reshape(-1, pool_h * pool_w)

    pooled_patches_idx = torch.from_numpy(idx_arr).long().unsqueeze(0)
    logger.info(f"pooled_patches_idx shape: {pooled_patches_idx.shape}")

    # Reference forward (PyTorch)
    logger.info("=" * 80)
    logger.info("Reference forward (PyTorch)")
    logger.info("=" * 80)

    ref_output = ref_full_vision_backbone(
        pixel_values,
        pooled_patches_idx,
        state_dict,
    )
    logger.info(f"Reference output: shape={ref_output.shape}, min={ref_output.min():.4f}, max={ref_output.max():.4f}")

    # TTNN mesh device forward: use MESH_DEVICE env if set, else (1, num_devices) for current machine (e.g. N300 -> (1, 2)).
    mesh_env = os.environ.get("MESH_DEVICE", "").strip()
    if mesh_env in MESH_SHAPE_BY_DEVICE:
        mesh_tup = MESH_SHAPE_BY_DEVICE[mesh_env]
    else:
        num_devices = ttnn.get_num_devices()
        mesh_tup = (1, num_devices)
    logger.info("=" * 80)
    logger.info(f"TTNN forward (mesh device {mesh_tup[0]}x{mesh_tup[1]})")
    logger.info("=" * 80)

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh_shape = ttnn.MeshShape(*mesh_tup)
    mesh_device = ttnn.open_mesh_device(mesh_shape)
    logger.info(f"Mesh device opened with {mesh_device.get_num_devices()} devices")

    try:
        from models.demos.molmo2.tt.vision_backbone import VisionBackbone

        ttnn_backbone = VisionBackbone(
            mesh_device=mesh_device,
            state_dict=state_dict,
            dtype=ttnn.bfloat16,
        )

        ttnn_output = ttnn_backbone(
            images_embedded=pixel_values,
            pooled_patches_idx=pooled_patches_idx,
        )

        logger.info(f"TTNN output: shape={ttnn_output.shape}, min={ttnn_output.min():.4f}, max={ttnn_output.max():.4f}")

        # Compare
        pcc, _ = calculate_pcc(ref_output, ttnn_output)
        diff = (ref_output - ttnn_output).abs()

        logger.info("=" * 80)
        logger.info(f"Vision Backbone PCC (mesh device): {pcc:.6f}")
        logger.info(f"Max diff: {diff.max():.4f}, Mean diff: {diff.mean():.6f}")
        logger.info("=" * 80)

        return pcc

    finally:
        ttnn.close_mesh_device(mesh_device)


if __name__ == "__main__":
    test_vision_backbone_mesh_pcc()
