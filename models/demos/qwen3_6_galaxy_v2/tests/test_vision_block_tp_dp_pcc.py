# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
V2-VISION-V2 (7/N): VisionBlock TP=8 + DP=4 — 4 parallel frames on BH GLX 8x4.

Validates the full topology:
  - TP=8 across cluster_axis=0 (rows, 8 chips per frame)
  - DP=4 across cluster_axis=1 (cols, 1 frame per col)
  - All 32 chips do useful work

Layered on top of the working TP=8 VisionBlock (commit 3a6797f2dcb) by
changing the input mapper to `ShardTensor2dMesh(dims=(None, 0))` — shard
dim=0 (batch/frame) across cluster_axis=1, replicate across cluster_axis=0.
Each col's 8 chips process the same frame in TP=8 mode. The block
implementation itself doesn't change.
"""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.demos.qwen3_6_galaxy_v2.tt.vision_attention_tp import build_vision_rope_tensors
from models.demos.qwen3_6_galaxy_v2.tt.vision_block_tp import Qwen36VisionBlockTP
from models.demos.qwen3_6_galaxy_v2.tt.vision_model_args import Qwen36VisionModelArgs
from models.demos.qwen3_vl.reference.functional import qwen3_vision_transformer_preprocess
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
@pytest.mark.parametrize("n_frames", (4,))
@pytest.mark.parametrize("layer_num", (0,))
def test_vision_block_tp_dp(grid_h, grid_w, n_frames, layer_num, mesh_device, reset_seeds, ensure_gc):
    if not os.environ.get("HF_MODEL"):
        os.environ["HF_MODEL"] = "Qwen/Qwen3.6-27B"

    seq_len = grid_h * grid_w
    model_args = Qwen36VisionModelArgs(
        mesh_device,
        dummy_weights=False,
        max_batch_size=n_frames,
        max_seq_len=seq_len,
    )
    assert (
        model_args.cluster_shape[1] == n_frames
    ), f"DP={n_frames} expects mesh col count == n_frames; got cluster_shape={model_args.cluster_shape}"

    vc = model_args.hf_config.vision_config
    H, NH, HD = vc.hidden_size, vc.num_heads, vc.hidden_size // vc.num_heads
    I = vc.intermediate_size
    logger.info(f"qwen3.6 VisionBlock TP=8 + DP={n_frames} on {mesh_device.shape}: seq_len={seq_len}")

    reference_full = model_args.reference_vision_model()
    reference_block = reference_full.blocks[layer_num]
    hf_state = reference_block.state_dict()

    image_grid_thw = torch.tensor([[1, grid_h, grid_w]])
    cu_seqlens, (cos_ref, sin_ref) = qwen3_vision_transformer_preprocess(
        seq_len=seq_len,
        grid_thw=image_grid_thw,
        head_dim=HD,
        spatial_merge_size=vc.spatial_merge_size,
    )

    ccl_manager = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)

    tt_block = Qwen36VisionBlockTP(
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        state_dict=hf_state,
        hidden_size=H,
        intermediate_size=I,
        num_heads=NH,
        head_dim=HD,
        norm_eps=1e-6,
        tp_mesh_axis=0,
        dtype=ttnn.bfloat16,
    )

    cos_tt, sin_tt = build_vision_rope_tensors(
        seq_len=seq_len,
        grid_thw=image_grid_thw,
        head_dim=HD,
        padded_head_dim=tt_block.attn.padded_head_dim,
        spatial_merge_size=vc.spatial_merge_size,
        mesh_device=mesh_device,
    )

    # 4 distinct frames stacked on dim=0
    torch_input_frames = torch.randn(n_frames, seq_len, H, dtype=torch.float32)

    # Per-frame torch references
    reference_outputs = [
        reference_block(
            torch_input_frames[i],
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=None,
            position_embeddings=(cos_ref, sin_ref),
        )
        for i in range(n_frames)
    ]

    # DP=4 input layout: tensor dim=0 (frame) sharded across cluster_axis=1 (4 cols),
    # replicated across cluster_axis=0 (8 rows).
    # Input shape [n_frames, 1, seq_len, H] → per-chip [1, 1, seq_len, H].
    tt_input = ttnn.from_torch(
        torch_input_frames.view(n_frames, 1, seq_len, H),
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device,
            dims=(None, 0),
            mesh_shape=model_args.cluster_shape,
        ),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    logger.info(f"Running TT VisionBlock TP=8 + DP={n_frames} on {mesh_device.shape}")
    tt_output = tt_block.forward(tt_input, cos_tt, sin_tt)

    # Output composer: each col produces output for its frame. Collect all 32 chips
    # concatenated on dim=0. Row-major mesh order: indices 0..3 = mesh-row 0, cols 0..3
    # → the 4 distinct frame outputs.
    tt_output_torch = ttnn.to_torch(tt_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    logger.info(f"tt_output_torch shape: {tuple(tt_output_torch.shape)}")
    # shape [32, 1, S, H] or [32, S, H]; first 4 entries = mesh-row 0, cols 0..3
    if tt_output_torch.dim() == 4:
        tt_output_torch = tt_output_torch[:n_frames, 0, :seq_len, :H]
    else:
        tt_output_torch = tt_output_torch[:n_frames, :seq_len, :H]

    pcc_required = 0.98  # matches the single-frame VisionBlock threshold
    all_pass = True
    for col_idx in range(n_frames):
        passing, pcc_message = comp_pcc(reference_outputs[col_idx], tt_output_torch[col_idx], pcc_required)
        logger.info(
            f"  col {col_idx}: {comp_allclose(reference_outputs[col_idx], tt_output_torch[col_idx])} | PCC: {pcc_message}"
        )
        if not passing:
            all_pass = False
    assert all_pass, f"At least one column failed PCC>={pcc_required}"
