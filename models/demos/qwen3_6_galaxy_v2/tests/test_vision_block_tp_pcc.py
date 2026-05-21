# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
V2-VISION-V2 (5/N) PCC: qwen3.6 VisionBlock composite TP=8 on BH GLX 8x4.
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
@pytest.mark.parametrize("layer_num", (0,))
def test_vision_block_tp_qwen36(grid_h, grid_w, layer_num, mesh_device, reset_seeds, ensure_gc):
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
    logger.info(f"qwen3.6 vision block TP=8 layer={layer_num} on {mesh_device.shape}: H={H}, NH={NH}, HD={HD}, I={I}")

    # Reference block (HF blocks[layer]) — strict-loaded with qwen3.6 weights
    reference_full = model_args.reference_vision_model()
    reference_block = reference_full.blocks[layer_num]
    hf_state = reference_block.state_dict()
    logger.info(f"reference block state-dict top-level keys: {sorted(set(k.split('.')[0] for k in hf_state.keys()))}")

    # Vision RoPE for this input
    image_grid_thw = torch.tensor([[1, grid_h, grid_w]])
    cu_seqlens, (cos_ref, sin_ref) = qwen3_vision_transformer_preprocess(
        seq_len=seq_len,
        grid_thw=image_grid_thw,
        head_dim=HD,
        spatial_merge_size=vc.spatial_merge_size,
    )

    # CCL manager
    ccl_manager = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)

    # Build TT block with TP=8
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

    # Build TT cos/sin
    cos_tt, sin_tt = build_vision_rope_tensors(
        seq_len=seq_len,
        grid_thw=image_grid_thw,
        head_dim=HD,
        padded_head_dim=tt_block.attn.padded_head_dim,
        spatial_merge_size=vc.spatial_merge_size,
        mesh_device=mesh_device,
    )

    # Random input
    torch_input_2d = torch.randn(seq_len, H, dtype=torch.float32)

    # Reference: HF block forward signature matches qwen3_vl test_vision_block.py
    reference_output = reference_block(
        torch_input_2d,
        cu_seqlens=cu_seqlens,
        rotary_pos_emb=None,
        position_embeddings=(cos_ref, sin_ref),
    )  # [seq_len, H]

    # TT input: replicated [1, 1, S, H]
    tt_input = ttnn.from_torch(
        torch_input_2d.view(1, 1, seq_len, H),
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )
    logger.info(f"Running TT VisionBlock TP=8 on {mesh_device.shape}")
    tt_output = tt_block.forward(tt_input, cos_tt, sin_tt)

    tt_output_torch = ttnn.to_torch(tt_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    logger.info(f"tt_output_torch shape: {tuple(tt_output_torch.shape)}")
    # Output is either [num_devices, 1, S, H] (4D) or [num_devices, S, H] (3D). Slice chip 0.
    if tt_output_torch.dim() == 4:
        tt_output_torch = tt_output_torch[0, 0, :seq_len, :H]
    else:
        tt_output_torch = tt_output_torch[0, :seq_len, :H]

    pcc_required = 0.98  # match the attention block's relaxed threshold (bf16 floor)
    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc_required)
    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(f"PCC: {pcc_message}")
    assert passing, f"qwen3.6 vision Block TP=8 PCC {pcc_required} not met: {pcc_message}"
