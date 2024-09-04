# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import os
import torch
import pytest
from loguru import logger

import ttnn
from ttnn import ReplicateTensorToMesh, ConcatMeshToTensor
from models.demos.t3000.mixtral8x7b.tt.mixtral_attention import TtMixtralAttention
from models.demos.t3000.mixtral8x7b.tt.mixtral_common import (
    prepare_inputs_ttnn_prefill,
    get_prefill_rot_mat,
    get_rot_transformation_mat,
    set_model_args,
)
from models.demos.t3000.mixtral8x7b.reference.model import Attention, precompute_freqs_cis, RMSNorm
from models.demos.t3000.mixtral8x7b.tt.model_config import TtModelArgs
from models.utility_functions import (
    comp_pcc,
    comp_allclose,
)


@pytest.mark.parametrize(
    "seq_len",
    (
        128,
        1024,
        1024 * 8,
        1024 * 32,
    ),
)
@torch.no_grad()
def test_mixtral_attention_inference(t3k_mesh_device, use_program_cache, reset_seeds, seq_len):
    for device in t3k_mesh_device.get_device_ids():
        t3k_mesh_device.get_device(device).enable_async(True)

    pcc = 0.99
    dtype = ttnn.bfloat8_b
    model_args = TtModelArgs(t3k_mesh_device.get_device(0))
    model_args = set_model_args(model_args, seq_len)
    state_dict = model_args.load_state_dict()
    batch = 1  # Prefill only a single user

    # Load reference model
    # Ref model needs partial state dict, but ttnn model uses full state dict keys as cached weight names
    partial_state_dict = {k[19:]: v for k, v in state_dict.items() if (k.startswith("layers.0.attention."))}
    reference_model = Attention(args=model_args)
    reference_model.load_state_dict(partial_state_dict)

    # Prepare rotation matrices for RoPE
    rot_mats = get_prefill_rot_mat(model_args.head_dim, model_args.max_seq_len, t3k_mesh_device, seq_len=seq_len)
    head_dim = model_args.dim // model_args.n_heads
    transformation_mat_torch = get_rot_transformation_mat(head_dim)
    transformation_mats = ttnn.as_tensor(
        transformation_mat_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=t3k_mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ReplicateTensorToMesh(t3k_mesh_device),
    )

    # Load ttnn model
    tt_model = TtMixtralAttention(t3k_mesh_device, state_dict, args=model_args, layer_num=0, dtype=dtype)

    generation_start_pos = 0
    generation_length = 3
    all_tests_pass = True

    # Run Attention module
    for i in range(generation_length):
        pt_attention_input = (torch.rand(batch, seq_len, model_args.dim) * 2) - 1  # Random inputs
        tt_attention_input = pt_attention_input
        start_pos_ids = [generation_start_pos + i for _ in range(batch)]
        attention_input, attn_mask, attn_mask_torch = prepare_inputs_ttnn_prefill(
            tt_attention_input,
            t3k_mesh_device,
        )

        # Run ttnn attention module
        tt_out = tt_model(
            attention_input,
            start_pos_ids,
            attn_mask,
            rot_mats,
            transformation_mats,
            user_id=0,  # single-user prefill
            mode="prefill",
        )

        tt_output_torch = ttnn.to_torch(tt_out, mesh_composer=ConcatMeshToTensor(t3k_mesh_device, dim=0))[0].view(
            batch, seq_len, -1
        )  # [ batch, seq, hidden_dim]

        # Run reference attention module
        positions = torch.LongTensor(range(seq_len))
        freqs_cis_i = precompute_freqs_cis(model_args.head_dim, 128_000)[positions]
        reference_output = reference_model(pt_attention_input, freqs_cis_i, positions, mask=attn_mask_torch)

        # Validate PCC
        passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc)

        logger.info(comp_allclose(reference_output, tt_output_torch))
        logger.info(f"Output PCC: {pcc_message}")

        if passing:
            logger.info(f"[start_pos={start_pos_ids[0]}] Mixtral_Attention Passed!")
        else:
            logger.warning(f"[start_pos={start_pos_ids[0]}] Mixtral_Attention Failed!")
            all_tests_pass = False

        # Validate KV-cache
        tt_layer_present_all = [ttnn.from_device(lp) for lp in tt_model.layer_past]
        tt_layer_present_all = [
            ttnn.to_torch(lp, mesh_composer=ConcatMeshToTensor(t3k_mesh_device, dim=1)) for lp in tt_layer_present_all
        ]
        pytorch_layer_present = [
            reference_model.cache_k.clone().permute(0, 2, 1, 3),  # [batch, n_kv_heads, seq, head_dim]
            reference_model.cache_v.clone().permute(0, 2, 1, 3),  # [batch, n_kv_heads, seq, head_dim]
        ]
        for i, (cache_pt, cache_tt) in enumerate(zip(pytorch_layer_present, tt_layer_present_all)):
            current_cache = "K_cache" if i == 0 else "V_cache"
            cache_pt = cache_pt[:, :, :seq_len, :]
            cache_tt = cache_tt[:, :, :seq_len, :]
            does_pass, output_pcc = comp_pcc(cache_pt, cache_tt, pcc)
            logger.info(f"{current_cache} PCC: {output_pcc}")

            if does_pass:
                logger.info(f"{current_cache} Passed!")
            else:
                logger.warning(f"{current_cache} Failed! PCC value is lower than {pcc}")
                all_tests_pass = False

    if all_tests_pass:
        logger.info("Mixtral Attention output Passed!")
    else:
        logger.warning("Mixtral Attention output Failed!")
        assert all_tests_pass, f"PCC value is lower than {pcc} for some of the outputs. Check Warnings!"
