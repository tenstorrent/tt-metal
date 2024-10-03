# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
from loguru import logger
import os
import ttnn
from models.demos.wormhole.llama31_8b_N300.tt.llama_common import (
    get_prefill_rot_mat,
    prepare_inputs_ttnn_prefill,
    get_rot_transformation_mat,
)
from models.demos.wormhole.llama31_8b_N300.tt.llama_decoder import TtTransformerBlock
from models.demos.wormhole.llama31_8b_N300.tt.model_config import TtModelArgs
from models.demos.t3000.llama2_70b.reference.llama.llama31_8b.model import TransformerBlock, precompute_freqs_cis
from models.utility_functions import (
    comp_pcc,
    comp_allclose,
)
from models.utility_functions import skip_for_grayskull


@torch.no_grad()
@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "seq_len",
    (
        4096,
        128,
    ),
)
@pytest.mark.parametrize(
    "mesh_device",
    [{"N150": (1, 1), "N300": (1, 2), "T3K": (2, 4), "TG": (8, 4)}.get(os.environ.get("FAKE_DEVICE"), None)],
    indirect=True,
)
def test_llama_decoder_inference(mesh_device, seq_len, use_program_cache, reset_seeds):
    dtype = ttnn.bfloat8_b

    model_args = TtModelArgs(mesh_device)
    state_dict = torch.load(model_args.consolidated_weights_path, map_location=torch.device("cpu"))

    # Ref model needs partial state dict, but our models use full state dict keys as cached weight names
    first_layer_prefix = model_args.get_state_dict_prefix("TtTransformerBlock", 0)
    partial_state_dict = {
        k[len(first_layer_prefix) :]: v for k, v in state_dict.items() if (k.startswith(first_layer_prefix))
    }
    batch = 1
    reference_model = TransformerBlock(layer_id=0, args=model_args)
    reference_model.load_state_dict(partial_state_dict)

    generation_start_pos = 0
    generation_length = 1
    all_tests_pass = True

    # pre-compute the rotational embedding matrix and send to device
    rot_mats = get_prefill_rot_mat(model_args.head_dim, model_args.max_seq_len, mesh_device, seq_len=seq_len)
    transformation_mat_torch = get_rot_transformation_mat(model_args.head_dim)
    transformation_mats = ttnn.as_tensor(
        transformation_mat_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Initialize TT model
    tt_model = TtTransformerBlock(
        args=model_args,
        mesh_device=mesh_device,
        dtype=dtype,
        state_dict=state_dict,
        layer_num=0,
        weight_cache_path=model_args.weight_cache_path(dtype),
    )

    # TODO Update start_pos (check llama test for reference)
    for i in range(generation_length):
        print(f"[Decoder] Generating token {i}")
        pt_decode_input = (torch.rand(batch, seq_len, model_args.dim) * 2) - 1
        tt_decode_input = pt_decode_input.clone()
        decode_input = prepare_inputs_ttnn_prefill(
            tt_decode_input,
            tt_model.mesh_device,
        )
        positions = torch.LongTensor(range(seq_len))
        freqs_cis_i = precompute_freqs_cis(
            model_args.head_dim, model_args.max_seq_len * 2, model_args.rope_theta, model_args.use_scaled_rope
        )[positions]

        # Reference model
        attn_mask = torch.full((seq_len, seq_len), torch.finfo(torch.float32).min)
        attn_mask_torch = torch.triu(attn_mask, diagonal=1)
        ref_output = reference_model(pt_decode_input, positions[0], freqs_cis_i, mask=attn_mask_torch)
        # Run TT model
        tt_out = tt_model(decode_input, None, rot_mats, transformation_mats, user_id=0, mode="prefill")
        tt_output_torch = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=1))[
            :, 0, :, :
        ].view(
            batch, seq_len, -1
        )  # [ batch, seq, hidden_dim]
        passing, pcc_message = comp_pcc(ref_output, tt_output_torch)

        logger.info(comp_allclose(ref_output, tt_output_torch))
        logger.info(pcc_message)

        if passing:
            logger.info("Llama Decoder Block Passed!")
        else:
            logger.warning("Llama Decoder Block Failed!")
            all_tests_pass = False

    if all_tests_pass:
        logger.info(f"All Llama decode iterations Passed!")
    else:
        logger.warning("One or more iterations of Llama decode Failed!")
        assert all_tests_pass, f"PCC value is lower than {0.99} for some of the outputs. Check Warnings!"
