# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
V2-VISION-V2 (3/N): qwen3.6 vision Block (norm1 + attention + norm2 + MLP) PCC
on BH GLX 8x4 (replicated mode).

Mirrors `models/demos/qwen3_vl/tests/test_vision_block.py` but:
  1. Uses qwen3.6 weights via the `Qwen36VisionModelArgs` bridge
  2. Runs replicated across the 32-chip mesh (input via ReplicateTensorToMesh)
  3. Tests just layer 0 first (V1 strict-load gives us per-layer key
     equivalence so layer 0 is representative)

If this passes, the vision encoder forward (attention + MLP + RoPE + norms)
works end-to-end with qwen3.6 weights on the BH GLX mesh.
"""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.demos.qwen3_6_galaxy_v2.tt.vision_model_args import Qwen36VisionModelArgs
from models.demos.qwen3_vl.reference.functional import qwen3_vision_transformer_preprocess
from models.demos.qwen3_vl.tt.vision_block import VisionBlock
from models.tt_transformers.tt.common import get_rot_transformation_mat
from models.tt_transformers.tt.load_checkpoints import convert_hf_to_meta, standardize_hf_keys_multimodal


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
@pytest.mark.parametrize("layer_num", (0,))
def test_vision_block_qwen36_pcc(layer_num, mesh_device, reset_seeds, ensure_gc):
    if not os.environ.get("HF_MODEL"):
        os.environ["HF_MODEL"] = "Qwen/Qwen3.6-27B"
    dtype = ttnn.bfloat8_b
    batch_size = 1

    # sample image: 98 x 146 patches (matches qwen3_vl test_vision_block.py defaults)
    image_grid_thw = torch.tensor([[1, 98, 146]])
    ref_seq_len = int(image_grid_thw[0, 1] * image_grid_thw[0, 2])
    seq_len = ((ref_seq_len // 128) + 1) * 128  # pad to MAX_QKV_MM_SEQ_LEN multiple

    model_args = Qwen36VisionModelArgs(
        mesh_device,
        dummy_weights=False,
        max_batch_size=batch_size,
        max_seq_len=seq_len,
    )
    logger.info(
        f"qwen3.6 vision block layer={layer_num} on {mesh_device.shape} mesh "
        f"(ref_seq={ref_seq_len}, padded_seq={seq_len})"
    )

    reference_full = model_args.reference_vision_model()
    reference_block = reference_full.blocks[layer_num]

    state_dict = standardize_hf_keys_multimodal(reference_block.state_dict())
    state_dict = convert_hf_to_meta(state_dict, model_args.head_dim)
    state_dict_prefix = model_args.get_state_dict_prefix("VisionBlock", layer_num)
    state_dict = {f"{state_dict_prefix}{k}": v for k, v in state_dict.items()}

    pt_input = torch.randn(1, 1, ref_seq_len, model_args.dim)
    cu_seqlens, position_embeddings = qwen3_vision_transformer_preprocess(
        seq_len=ref_seq_len,
        grid_thw=image_grid_thw,
        head_dim=model_args.head_dim,
        spatial_merge_size=model_args.hf_config.vision_config.spatial_merge_size,
    )

    cos, sin = position_embeddings
    cos = torch.nn.functional.pad(cos, (0, 0, 0, seq_len - ref_seq_len)).unsqueeze(0).unsqueeze(0)
    sin = torch.nn.functional.pad(sin, (0, 0, 0, seq_len - ref_seq_len)).unsqueeze(0).unsqueeze(0)
    cos_tt = ttnn.from_torch(
        cos,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    sin_tt = ttnn.from_torch(
        sin,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    rot_mats = [cos_tt, sin_tt]

    transformation_mat_torch = get_rot_transformation_mat(model_args.head_dim)
    transformation_mats = {
        "prefill": ttnn.as_tensor(
            transformation_mat_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
    }

    tt_model = VisionBlock(
        mesh_device=mesh_device,
        state_dict=state_dict,
        weight_cache_path=None,
        layer_num=layer_num,
        dtype=dtype,
        transformation_mats=transformation_mats,
        args=model_args,
    )

    # Replicated input (every chip gets the same data)
    tt_input = pt_input.clone()
    tt_input = torch.nn.functional.pad(tt_input, (0, 0, 0, seq_len - ref_seq_len))
    tt_input_tt = ttnn.from_torch(
        tt_input,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    tt_out = tt_model(tt_input_tt, rot_mats=rot_mats)

    # Output: every chip has the same data (replicated). Concat dim=0 → [32, ...]; take chip 0.
    tt_out_torch = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    tt_output_torch = tt_out_torch[0:1, 0:1, :ref_seq_len, : model_args.dim]
    tt_output_torch = tt_output_torch.view(batch_size, ref_seq_len, -1)

    reference_output = reference_block(
        pt_input.squeeze(0).squeeze(0),
        cu_seqlens=cu_seqlens,
        rotary_pos_emb=None,
        position_embeddings=position_embeddings,
    )

    pcc_required = 0.99
    passing, pcc_message = comp_pcc(reference_output, tt_output_torch.squeeze(0), pcc_required)
    logger.info(comp_allclose(reference_output, tt_output_torch.squeeze(0)))
    logger.info(f"PCC: {pcc_message}")
    assert passing, f"qwen3.6 vision Block (layer {layer_num}) PCC {pcc_required} not met: {pcc_message}"
