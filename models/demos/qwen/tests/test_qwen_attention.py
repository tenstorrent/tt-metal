# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.qwen.reference.model import Attention
from models.demos.qwen.tt.model_config import TtModelArgs
from models.demos.qwen.tt.qwen_attention import TtQwenAttention
from models.demos.qwen.tt.qwen_common import get_single_rot_mat, precompute_freqs
from models.utility_functions import comp_allclose, comp_pcc, skip_for_grayskull


@torch.no_grad()
@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("FAKE_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
def test_qwen_attention_inference(mesh_device, use_program_cache, reset_seeds, ensure_gc):
    if mesh_device.shape != (1, 1):
        pytest.skip("Only N150 is supported")
    dtype = ttnn.bfloat8_b
    pcc = 0.99

    model_args = TtModelArgs(mesh_device)
    model_args.n_layers = 1
    state_dict = model_args.load_state_dict()

    first_layer_prefix = model_args.get_state_dict_prefix("TtQwenAttention", 0) + "."
    # Ref model needs partial state dict, but our models use full state dict keys as cached weight names
    partial_state_dict = {
        k[len(first_layer_prefix) :]: v for k, v in state_dict.items() if (k.startswith(first_layer_prefix))
    }

    reference_model = Attention(args=model_args)
    reference_model.load_state_dict(partial_state_dict)

    batch = model_args.max_batch_size
    seq_len = 1

    generation_start_pos = 0
    generation_length = 10
    all_tests_pass = True

    # pre-compute the rotational embedding matrix and send to device
    current_rot_mat, rot_matrix = get_single_rot_mat(
        model_args.head_dim,
        mesh_device,
        model_args.num_devices,
        start_pos=0,
    )

    tt_model = TtQwenAttention(
        mesh_device,
        state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
        layer_num=0,
        dtype=dtype,
        configuration=model_args,
    )

    cos, sin = precompute_freqs(model_args.head_dim, model_args.max_seq_len * 2)
    freqs_cis = torch.complex(cos, sin)
    for i in range(generation_length):
        # 70B attention block typically sees tensors with mean 0 and std 0.03 - 0.05 in layer 1
        pt_attention_input = torch.randn(batch, seq_len, model_args.dim) * 0.05

        tt_attention_input = pt_attention_input.clone()
        current_pos = generation_start_pos + i
        current_pos_tensor = ttnn.from_torch(
            torch.tensor([current_pos] * batch),
            device=mesh_device,
            dtype=ttnn.int32,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

        attention_input = model_args.prepare_inputs_ttnn_decode(
            tt_attention_input,
            model_args.model_config["SHARDED_ATTN_INPUT_MEMCFG"],
            force_replicated=True,
        )

        tt_out = tt_model(attention_input, current_pos_tensor, rot_mats=current_rot_mat, mode="decode")
        # multi-device attention module returns replicated output

        tt_output_torch = (
            ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1))[0, :, :, : model_args.dim]
            .view(1, -1, model_args.dim)
            .permute(1, 0, 2)[: model_args.max_batch_size, :, :]
        )  # [ batch, seq, hidden_dim]

        freqs_cis_i = freqs_cis[current_pos, :].unsqueeze(0)
        positions = torch.tensor([current_pos])

        reference_output = reference_model(pt_attention_input, freqs_cis_i, positions, mask=None)

        passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc)

        logger.info(comp_allclose(reference_output, tt_output_torch))
        logger.info(f"PCC: {pcc_message}")
        if passing:
            logger.info(f"[pos={current_pos}] Qwen_Attention Passed!")
        else:
            logger.warning(f"[pos={current_pos}] Qwen_Attention Failed!")
            all_tests_pass = False

        # Update rotation matrix for next iteration
        current_rot_mat = ttnn.linear(rot_matrix, current_rot_mat)

        check_kv_cache = False
        if check_kv_cache:
            # PyTorch output --------------------------------------------------------------------
            pytorch_layer_present = [
                reference_model.cache_k.clone().permute(0, 2, 1, 3),  # [batch, n_kv_heads, seq, head_dim]
                reference_model.cache_v.clone().permute(0, 2, 1, 3),  # [batch, n_kv_heads, seq, head_dim]
            ]
            # TT hardware execution -------------------------------------------------------------
            tt_layer_present = [
                ttnn.to_torch(cache, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=1))
                for cache in tt_model.layer_past
            ]

            for i, (cache_pt, cache_tt) in enumerate(zip(pytorch_layer_present, tt_layer_present)):
                cache_length_to_check = min(model_args.sliding_window, generation_start_pos + generation_length + 1)
                cache_pt = cache_pt[:, :, generation_start_pos:cache_length_to_check, :]
                cache_tt = cache_tt[:, :, generation_start_pos:cache_length_to_check, :]
                does_pass, output_pcc = comp_pcc(cache_pt, cache_tt, pcc)
                if i == 0:
                    logger.info(f"K cache output: {output_pcc}")
                else:
                    logger.info(f"V cache output: {output_pcc}")

                if does_pass:
                    logger.info(f"KV Cache Passed!")
                else:
                    logger.warning(f"KV Cache Failed! PCC value is lower than {pcc}")
                    all_tests_pass = False

    if all_tests_pass:
        logger.info("Qwen Attention output Passed!")
    else:
        logger.warning("Qwen Attention output Failed!")
        assert all_tests_pass, f"PCC value is lower than {pcc} for some of the outputs. Check Warnings!"
