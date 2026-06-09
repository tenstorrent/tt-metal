# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
VLM Stage-1: sequence-parallel qwen3.6 VisionBlock on BH GLX 8x4.

The sequence is sharded across the col axis (cluster_axis=1); attention
all-gathers K/V across cols and applies the cu_seqlens block-diagonal mask.
Validates against the HF reference block for:
  - n_images=1: single segment (exercises seq padding 196 -> 256).
  - n_images=2: two segments (exercises the cu_seqlens mask — the OLD global
    attention path gets this wrong, so this is also the multi-image bug fix).

Layout: TP=8 heads on rows x seq-parallel x4 on cols => all 32 chips active.
"""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.demos.qwen3_6_galaxy_v2.tt.vision_attention_tp import build_vision_seq_parallel_tensors
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
# (n_images, n_frames): 1 image | 2 separate images | 1 video with 2 frames.
# A video is a SINGLE cu_seqlens segment (all frames attend within it); two
# images are two segments (the mask must block cross-image attention).
@pytest.mark.parametrize("n_images,n_frames", [(1, 1), (2, 1), (1, 2)])
@pytest.mark.parametrize("layer_num", (0,))
def test_vision_seqp_block(grid_h, grid_w, n_images, n_frames, layer_num, mesh_device, reset_seeds, ensure_gc):
    if not os.environ.get("HF_MODEL"):
        os.environ["HF_MODEL"] = "Qwen/Qwen3.6-27B"

    per_frame = grid_h * grid_w
    seq_len = per_frame * n_frames * n_images
    cols = mesh_device.shape[1]

    model_args = Qwen36VisionModelArgs(mesh_device, dummy_weights=False, max_batch_size=1, max_seq_len=seq_len)
    vc = model_args.hf_config.vision_config
    H, NH, HD, I = vc.hidden_size, vc.num_heads, vc.hidden_size // vc.num_heads, vc.intermediate_size
    logger.info(
        f"seq-parallel vision block layer={layer_num} n_images={n_images} n_frames={n_frames} "
        f"seq_len={seq_len} cols={cols}"
    )

    reference_block = model_args.reference_vision_model().blocks[layer_num]
    hf_state = reference_block.state_dict()

    # grid_thw: n_images items, each a [n_frames, grid_h, grid_w] image/video.
    image_grid_thw = torch.tensor([[n_frames, grid_h, grid_w]] * n_images)
    cu_seqlens, (cos_ref, sin_ref) = qwen3_vision_transformer_preprocess(
        seq_len=seq_len, grid_thw=image_grid_thw, head_dim=HD, spatial_merge_size=vc.spatial_merge_size
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

    cos_tt, sin_tt, mask_tt, S_pad = build_vision_seq_parallel_tensors(
        seq_len=seq_len,
        grid_thw=image_grid_thw,
        head_dim=HD,
        padded_head_dim=tt_block.attn.padded_head_dim,
        spatial_merge_size=vc.spatial_merge_size,
        mesh_device=mesh_device,
    )
    logger.info(f"S_pad={S_pad} (seq_len={seq_len}); mask={'yes' if mask_tt is not None else 'none'}")

    torch_input_2d = torch.randn(seq_len, H, dtype=torch.float32)

    # HF reference: cu_seqlens makes attention block-diagonal per image.
    reference_output = reference_block(
        torch_input_2d, cu_seqlens=cu_seqlens, rotary_pos_emb=None, position_embeddings=(cos_ref, sin_ref)
    )  # [seq_len, H]

    # TT input: pad sequence to S_pad, shard dim=2 (seq) across cols, replicate rows.
    x_pad = torch.zeros(S_pad, H, dtype=torch.float32)
    x_pad[:seq_len] = torch_input_2d
    tt_input = ttnn.from_torch(
        x_pad.view(1, 1, S_pad, H),
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 2), mesh_shape=tuple(mesh_device.shape)),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    logger.info(f"Running seq-parallel VisionBlock on {mesh_device.shape}")
    tt_output = tt_block.forward(tt_input, cos_tt, sin_tt, attn_mask=mask_tt, seq_parallel=True)

    # Output: seq-sharded on dim=2 across cols, replicated across rows.
    # ConcatMeshToTensor(dim=0) -> [32, 1, S_pad/cols, H] row-major (chip = row*cols+col).
    # Row 0 (chips 0..cols-1) holds the cols' contiguous seq shards.
    tt_full = ttnn.to_torch(tt_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    if tt_full.dim() == 4:
        tt_full = tt_full[:cols, 0]  # [cols, S_pad/cols, H]
    else:
        tt_full = tt_full[:cols]  # [cols, S_pad/cols, H]
    tt_output_torch = tt_full.reshape(S_pad, H)[:seq_len, :H]

    pcc_required = 0.99
    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc_required)
    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(f"seq-parallel n_images={n_images} PCC: {pcc_message}")
    assert passing, f"seq-parallel vision block PCC {pcc_required} not met (n_images={n_images}): {pcc_message}"
