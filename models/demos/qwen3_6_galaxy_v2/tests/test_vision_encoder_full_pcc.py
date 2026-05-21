# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
V2-VISION-V2 (6/N) — full 27-layer vision encoder PCC on BH GLX 8x4.

Composes 27 `Qwen36VisionBlockTP` instances in sequence. Tests end-to-end
through the block stack (excluding patch_embed, pos_embed, and the final
PatchMerger — those are small CPU ops for V2 first pass).

Input: random hidden states `[seq_len, hidden=1152]` as if the patch_embed
+ pos_embed had already produced them. Output: same shape after 27 blocks.

PCC > 0.95 required (bf16 quantization compounds through 27 layers — the
single block hit 0.9981, attention alone 0.987; expect some degradation).
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
def test_vision_encoder_full_qwen36(grid_h, grid_w, mesh_device, reset_seeds, ensure_gc):
    if not os.environ.get("HF_MODEL"):
        os.environ["HF_MODEL"] = "Qwen/Qwen3.6-27B"

    seq_len = grid_h * grid_w
    model_args = Qwen36VisionModelArgs(
        mesh_device,
        dummy_weights=False,
        max_batch_size=1,
        max_seq_len=seq_len,
    )
    vc = model_args.hf_config.vision_config
    H, NH, HD = vc.hidden_size, vc.num_heads, vc.hidden_size // vc.num_heads
    I = vc.intermediate_size
    DEPTH = vc.depth
    logger.info(
        f"qwen3.6 27-layer vision encoder TP=8 on {mesh_device.shape}: H={H}, NH={NH}, HD={HD}, I={I}, depth={DEPTH}"
    )

    # Reference encoder
    reference_full = model_args.reference_vision_model()

    # Vision RoPE for this input
    image_grid_thw = torch.tensor([[1, grid_h, grid_w]])
    cu_seqlens, (cos_ref, sin_ref) = qwen3_vision_transformer_preprocess(
        seq_len=seq_len,
        grid_thw=image_grid_thw,
        head_dim=HD,
        spatial_merge_size=vc.spatial_merge_size,
    )

    ccl_manager = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)

    # Build all 27 TT blocks
    tt_blocks = []
    for layer_num in range(DEPTH):
        reference_block = reference_full.blocks[layer_num]
        hf_state = reference_block.state_dict()
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
        tt_blocks.append(tt_block)
        if layer_num % 5 == 0:
            logger.info(f"  built TT block {layer_num + 1}/{DEPTH}")

    cos_tt, sin_tt = build_vision_rope_tensors(
        seq_len=seq_len,
        grid_thw=image_grid_thw,
        head_dim=HD,
        padded_head_dim=tt_blocks[0].attn.padded_head_dim,
        spatial_merge_size=vc.spatial_merge_size,
        mesh_device=mesh_device,
    )

    # Input as if patch_embed + pos_embed produced this
    torch_input_2d = torch.randn(seq_len, H, dtype=torch.float32)

    # Reference: run 27 blocks sequentially (skip patch_embed, pos_embed, merger)
    x_ref = torch_input_2d
    for layer_num in range(DEPTH):
        x_ref = reference_full.blocks[layer_num](
            x_ref, cu_seqlens=cu_seqlens, rotary_pos_emb=None, position_embeddings=(cos_ref, sin_ref)
        )
    logger.info(f"reference output stats: mean={x_ref.mean().item():.4f}, std={x_ref.std().item():.4f}")

    # TT: same chain
    tt_input = ttnn.from_torch(
        torch_input_2d.view(1, 1, seq_len, H),
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )
    logger.info(f"Running TT 27-layer encoder on {mesh_device.shape}")
    x_tt = tt_input
    for layer_num in range(DEPTH):
        x_tt = tt_blocks[layer_num].forward(x_tt, cos_tt, sin_tt)
        if layer_num % 5 == 0:
            logger.info(f"  ran TT block {layer_num + 1}/{DEPTH}")

    tt_output_torch = ttnn.to_torch(x_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    logger.info(f"tt_output_torch shape: {tuple(tt_output_torch.shape)}")
    if tt_output_torch.dim() == 4:
        tt_output_torch = tt_output_torch[0, 0, :seq_len, :H]
    else:
        tt_output_torch = tt_output_torch[0, :seq_len, :H]
    logger.info(f"tt output stats: mean={tt_output_torch.mean().item():.4f}, std={tt_output_torch.std().item():.4f}")

    # Lower threshold for the full 27-layer chain (bf16 compounds)
    pcc_required = 0.95
    passing, pcc_message = comp_pcc(x_ref, tt_output_torch, pcc_required)
    logger.info(comp_allclose(x_ref, tt_output_torch))
    logger.info(f"PCC: {pcc_message}")
    assert passing, f"qwen3.6 27-layer vision encoder PCC {pcc_required} not met: {pcc_message}"
