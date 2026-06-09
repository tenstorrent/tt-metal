# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
VLM Stage-2: full sequence-parallel vision encoder e2e on BH GLX 8x4.

Drives Qwen36VisionEncoder.forward(..., seq_parallel=True) (the default) for a
single image, two images, and a 2-frame video, comparing against the HF
Qwen3VLVisionModel.forward. The sequence is sharded across the 4 cols (all 32
chips active) and attention uses the cu_seqlens block-diagonal mask, so the
multi-image case is exact vs HF (the legacy replicated/global path is not).

Builds the encoder ONCE (loads 27 blocks) and loops the three scenarios.
"""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.qwen3_6_galaxy_v2.tt.vision_encoder import Qwen36VisionEncoder
from models.demos.qwen3_6_galaxy_v2.tt.vision_model_args import Qwen36VisionModelArgs
from models.tt_dit.parallel.manager import CCLManager


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4), "BH_GLX": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
@pytest.mark.parametrize("grid_h,grid_w", [(14, 14)])
def test_vision_encoder_seqp_e2e(grid_h, grid_w, mesh_device, reset_seeds, ensure_gc):
    if not os.environ.get("HF_MODEL"):
        os.environ["HF_MODEL"] = "Qwen/Qwen3.6-27B"

    max_seq = grid_h * grid_w * 2  # largest scenario (2 images / 2 frames)
    model_args = Qwen36VisionModelArgs(mesh_device, dummy_weights=False, max_batch_size=1, max_seq_len=max_seq)
    vc = model_args.hf_config.vision_config
    patch_feat_dim = vc.in_channels * vc.temporal_patch_size * vc.patch_size**2  # 1536

    ccl_manager = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)
    tt_encoder = Qwen36VisionEncoder(
        mesh_device=mesh_device, ccl_manager=ccl_manager, model_args=model_args, dtype=ttnn.bfloat16
    )
    reference_full = model_args.reference_vision_model()

    pcc_required = 0.99
    results = []
    # (n_images, n_frames, label)
    scenarios = [(1, 1, "1 image"), (2, 1, "2 images"), (1, 2, "1 video x2 frames")]
    for n_images, n_frames, label in scenarios:
        seq_len = grid_h * grid_w * n_frames * n_images
        torch.manual_seed(0)
        pixel_values = torch.randn(seq_len, patch_feat_dim, dtype=torch.float32)
        grid_thw = torch.tensor([[n_frames, grid_h, grid_w]] * n_images, dtype=torch.long)

        ref_features, ref_deepstack = reference_full(pixel_values, grid_thw=grid_thw)
        tt_features = tt_encoder.forward(pixel_values, grid_thw, seq_parallel=True)

        passing, pcc_message = comp_pcc(ref_features, tt_features, pcc_required)
        logger.info(
            f"[seqp encoder] {label}: seq_len={seq_len} shapes ref={tuple(ref_features.shape)} -> PCC {pcc_message}"
        )
        results.append((label, passing, pcc_message))

    for label, passing, msg in results:
        logger.info(f"  {label}: {'PASS' if passing else 'FAIL'} ({msg})")
    assert all(p for _, p, _ in results), f"seq-parallel encoder PCC<{pcc_required}: {results}"
