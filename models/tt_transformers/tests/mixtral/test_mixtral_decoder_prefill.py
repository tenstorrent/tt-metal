# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from loguru import logger

import ttnn
from models.demos.t3000.mixtral8x7b.reference.model import TransformerBlock, precompute_freqs_cis
from models.tt_transformers.tt.ccl import TT_CCL
from models.tt_transformers.tt.common import get_prefill_rot_mat, get_rot_transformation_mat
from models.tt_transformers.tt.decoder import TransformerBlock as TtTransformerBlock
from models.tt_transformers.tt.model_config import ModelArgs
from models.utility_functions import comp_allclose, comp_pcc

# pytest models/tt_transformers/tests/mixtral/test_mixtral_decoder_prefill.py


def convert2ref(state_dict):
    out = {}
    for key, value in state_dict.items():
        if "block_sparse_moe" in key:
            new_key = key.replace("block_sparse_moe", "feed_forward")
            out[new_key] = value
        elif "feed_forward" not in key:
            out[key] = value
    return out


@pytest.mark.parametrize(
    "batch",
    (32,),
)
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
def test_mixtral_decoder_inference(t3k_mesh_device, reset_seeds, batch, device_params):
    """
    b: batch
    s: sequence length
    h: hidden size
    """

    pcc = 0.99
    dtype = ttnn.bfloat8_b
    mode = "prefill"
    batch = 1
    max_seq_len = 4096

    model_args = ModelArgs(t3k_mesh_device, max_seq_len=max_seq_len, max_batch_size=batch)
    model_args.n_layers = 1

    t3k_mesh_device.disable_and_clear_program_cache()

    state_dict = model_args.load_state_dict()

    first_layer_prefix = model_args.get_state_dict_prefix("TransformerBlock", 0)
    partial_state_dict = {
        k[len(first_layer_prefix) :]: v for k, v in state_dict.items() if (k.startswith(first_layer_prefix))
    }

    reference_model = TransformerBlock(args=model_args)
    reference_model.load_state_dict(convert2ref(partial_state_dict))

    # Initialize TT model
    rot_mats = get_prefill_rot_mat(
        model_args.head_dim,
        t3k_mesh_device,
        max_seq_len,
        model_args.rope_theta,
        model_args.rope_scaling.factor if model_args.rope_scaling else None,
        model_args.max_context_len,
    )
    transformation_mat_torch = get_rot_transformation_mat(model_args.head_dim)
    transformation_mats_prefill = ttnn.as_tensor(
        transformation_mat_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=t3k_mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(t3k_mesh_device),
    )
    transformation_mats = {"prefill": transformation_mats_prefill}

    tt_ccl = TT_CCL(t3k_mesh_device)
    tt_model = TtTransformerBlock(
        mesh_device=t3k_mesh_device,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
        layer_num=0,
        dtype=dtype,
        transformation_mats=transformation_mats,
        args=model_args,
        tt_ccl=tt_ccl,
    )

    generation_length = 10
    all_tests_pass = True
    generation_start_pos = 0

    for i in range(generation_length):
        logger.info(f"[Decoder] Generating token {i}")
        pt_decode_input_bsh = (torch.rand(batch, max_seq_len, model_args.dim) * 2) - 1
        tt_decode_input = pt_decode_input_bsh.clone()
        decode_input = model_args.prepare_residual_tensor_prefill(
            tt_decode_input,
        )

        positions = torch.LongTensor(range(max_seq_len))

        # Run TT model
        start_pos = generation_start_pos + i
        start_pos_ids = [start_pos for _ in range(batch)]
        tt_out_b1sh = tt_model(
            decode_input,
            None,
            rot_mats,
            user_id=0,
            mode=mode,
        )
        tt_out = (
            ttnn.to_torch(
                tt_out_b1sh,
                mesh_composer=ttnn.ConcatMesh2dToTensor(
                    t3k_mesh_device, dims=(1, 3), mesh_shape=model_args.cluster_shape
                ),
            )[0]
            .squeeze(0)
            .view(batch, max_seq_len, -1)
        )

        # Reference model
        attn_mask = torch.full((max_seq_len, max_seq_len), torch.finfo(torch.float32).min)
        attn_mask_torch = torch.triu(attn_mask, diagonal=1)
        positions = torch.LongTensor(range(max_seq_len))
        freqs_cis_i = precompute_freqs_cis(model_args.head_dim, 128_000)[positions]
        ref_output_bsh = reference_model(pt_decode_input_bsh, freqs_cis_i, positions, mask=attn_mask_torch)

        # Reference model
        passing, pcc_message = comp_pcc(ref_output_bsh, tt_out, pcc)

        logger.info(comp_allclose(ref_output_bsh, tt_out))
        logger.info(pcc_message)

        if passing:
            logger.info("Mixtral Decoder Block Passed!")
        else:
            logger.warning("Mixtral Decoder Block Failed!")
            all_tests_pass = False

    if all_tests_pass:
        logger.info(f"All {generation_length} Mixtral decode iterations Passed!")
    else:
        logger.warning("One or more iterations of Mixtral decode Failed!")
        assert all_tests_pass, f"PCC value is lower than {pcc} for some of the outputs. Check Warnings!"
