# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from loguru import logger

import ttnn
from models.demos.t3000.mixtral8x7b.reference.model import TransformerBlock, precompute_freqs_cis
from models.demos.t3000.mixtral8x7b.tt.mixtral_common import get_single_rot_mat
from models.tt_transformers.tt.decoder import TransformerBlock as TtTransformerBlock
from models.tt_transformers.tt.model_config import ModelArgs
from models.tt_transformers.tt.rope import RotarySetup
from models.utility_functions import comp_allclose, comp_pcc


@pytest.mark.parametrize(
    "batch",
    (
        32,
        16,
    ),
)
@pytest.mark.parametrize("mode", ["prefill", "decode"])
def test_mixtral_decoder_inference(t3k_mesh_device, reset_seeds, batch, mode):
    """
    b: batch
    s: sequence length
    h: hidden size
    """

    pcc = 0.99
    dtype = ttnn.bfloat8_b

    if batch == 32:
        generation_start_pos = 15000
        max_seq_len = 16384
    elif batch in [4, 8, 16]:
        generation_start_pos = 30000
        max_seq_len = 32768
    else:
        raise ValueError(f"Batch size {batch} not supported")

    model_args = ModelArgs(t3k_mesh_device, max_seq_len=max_seq_len, max_batch_size=batch)
    state_dict = model_args.load_state_dict()
    partial_state_dict = {k[9:]: v for k, v in state_dict.items() if (k.startswith("layers.0."))}
    reference_model = TransformerBlock(args=model_args)
    reference_model.load_state_dict(partial_state_dict)

    # Initialize TT model
    rope_setup = RotarySetup(
        t3k_mesh_device,
        model_args.max_batch_size,
        model_args.head_dim,
        model_args.max_seq_len,
        model_args.rope_theta,
        model_args.rope_scaling_factor,
        model_args.orig_context_len,
    )
    transformation_mats = rope_setup.get_both_trans_mats()

    tt_model = TtTransformerBlock(
        mesh_device=t3k_mesh_device,
        state_dict=state_dict,
        args=model_args,
        layer_num=0,
        dtype=dtype,
        weight_cache_path=model_args.weight_cache_path(dtype),
        transformation_mats=rope_setup.get_both_trans_mats(),
    )

    current_rot_mat, rot_matrix = get_single_rot_mat(
        model_args.head_dim,
        tt_model.mesh_device,
    )

    generation_length = 10
    all_tests_pass = True

    seqlen = 128

    for i in range(generation_length):
        logger.info(f"[Decoder] Generating token {i}")

        pt_decode_input_bsh = (torch.rand(batch, seqlen, model_args.dim) * 2) - 1

        start_pos = generation_start_pos + i
        start_pos_ids = torch.tensor([start_pos for _ in range(batch)])
        current_pos_tensor = ttnn.from_torch(
            start_pos_ids,
            device=t3k_mesh_device,
            dtype=ttnn.int32,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                t3k_mesh_device,
                dims=(None, None),
                mesh_shape=model_args.cluster_shape,
            ),
        )

        tt_decode_input = pt_decode_input_bsh.clone()

        if mode == "decode":
            decode_input = model_args.prepare_residual_tensor_decode(
                tt_decode_input,
                model_args.model_config["DECODE_RESIDUAL_MEMCFG"],
            )
        else:
            decode_input = ttnn.from_torch(
                tt_decode_input,
                device=t3k_mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    t3k_mesh_device,
                    dims=(None, None),
                    mesh_shape=model_args.cluster_shape,
                ),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        rot_mats = rope_setup.get_rot_mats(start_pos_ids)

        # Run TT model
        tt_out_b1sh = tt_model(
            decode_input,
            current_pos_tensor,
            rot_mats=rot_mats,
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
            .view(32, 1, -1)
        )

        # Reference model
        positions = torch.LongTensor([start_pos])
        freqs_cis_i = precompute_freqs_cis(model_args.head_dim, 128_000)[positions]
        ref_output_bsh = reference_model(pt_decode_input_bsh, freqs_cis_i, positions, mask=None)
        passing, pcc_message = comp_pcc(ref_output_bsh, tt_out, pcc)

        logger.info(comp_allclose(ref_output_bsh, tt_out))
        logger.info(pcc_message)

        if passing:
            logger.info("Mistral Decoder Block Passed!")
        else:
            logger.warning("Mistral Decoder Block Failed!")
            all_tests_pass = False

        current_rot_mat = ttnn.linear(rot_matrix, current_rot_mat)

    if all_tests_pass:
        logger.info(f"All {generation_length} Mistral decode iterations Passed!")
    else:
        logger.warning("One or more iterations of Mistral decode Failed!")
        assert all_tests_pass, f"PCC value is lower than {pcc} for some of the outputs. Check Warnings!"
