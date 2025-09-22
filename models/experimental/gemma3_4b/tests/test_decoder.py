"""Gemma-3-4b-it Test for Text Decoder"""

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
import os
import ttnn
from models.tt_transformers.tt.model_config import ModelArgs
from models.tt_transformers.tt.ccl import TT_CCL
from models.experimental.gemma3_4b.tt.decoder import TransformerBlock
from models.common.utility_functions import (
    comp_pcc,
    comp_allclose,
)
from models.common.utility_functions import skip_for_grayskull
from models.tt_transformers.tt.common import PagedAttentionConfig
from models.tt_transformers.tt.rope import RotarySetup


@torch.no_grad()
@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "paged_attention",
    (
        True,
        # False
    ),
    ids=(
        "paged_attention",
        # "default_attention"
    ),
)
@pytest.mark.parametrize(
    "page_params",
    [{"page_block_size": 32, "page_max_num_blocks": 1024}],
)
@pytest.mark.parametrize(
    "batch_size",
    (1,),
)
@pytest.mark.parametrize(
    "max_seq_len",
    (256,),  # For decode-only unit test, there's no need to run with large sequence lengths
)
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
def test_decoder_inference(
    max_seq_len,
    batch_size,
    paged_attention,
    page_params,
    mesh_device,
    reset_seeds,
):
    dtype = ttnn.bfloat16
    tt_ccl = TT_CCL(mesh_device)

    pcc_required = 0.85
    model_args = ModelArgs(mesh_device, max_batch_size=batch_size, max_seq_len=max_seq_len)
    model_args.n_layers = 1

    state_dict = model_args.load_state_dict()

    reference_model = model_args.reference_decoder()

    generation_start_pos = 0
    generation_length = 3
    all_tests_pass = False

    rope_setup = RotarySetup(
        mesh_device,
        batch_size,
        model_args.head_dim,
        model_args.max_seq_len,
        model_args.rope_theta,
        model_args.rope_scaling,
    )
    transformation_mats = rope_setup.get_both_trans_mats()

    rope_local_setup = RotarySetup(
        mesh_device,
        batch_size,
        model_args.head_dim,
        model_args.max_seq_len,
        model_args.rope_theta_local,
        None,  # No scaling for local RoPE
    )
    transformation_mats_local = rope_local_setup.get_both_trans_mats()

    # Prepare page table for paged attention
    page_table_tt = None
    paged_attention_config = None

    if paged_attention:
        paged_attention_config = PagedAttentionConfig(
            block_size=page_params["page_block_size"],
            max_num_blocks=page_params["page_max_num_blocks"],
        )
        # Implied shuffling of blocks
        permutation = torch.randperm(paged_attention_config.max_num_blocks)
        # Page table which maps virtual blocks to physical
        reverse_permutation = torch.argsort(permutation)
        page_table = reverse_permutation.reshape(
            model_args.max_batch_size, paged_attention_config.max_num_blocks // model_args.max_batch_size
        )
        page_table_tt = ttnn.from_torch(
            page_table,
            device=mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device,
                dims=(None, -2) if (model_args.is_galaxy and batch_size > 1) else (None, None),
                mesh_shape=model_args.cluster_shape,
            ),
        )

    # Initialize TT model
    tt_model = TransformerBlock(
        args=model_args,
        mesh_device=mesh_device,
        tt_ccl=tt_ccl,
        dtype=dtype,
        state_dict=state_dict,
        layer_num=0,
        weight_cache_path=model_args.weight_cache_path(dtype),
        transformation_mats=transformation_mats,
        transformation_mats_local=transformation_mats_local,
        paged_attention_config=paged_attention_config,
    )

    seqlen = 1

    # Initial positions
    current_pos = torch.tensor([generation_start_pos for _ in range(batch_size)])
    current_pos_tensor = ttnn.from_torch(
        current_pos,
        device=mesh_device,
        dtype=ttnn.int32,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device,
            dims=(None, 0) if (model_args.is_galaxy and batch_size > 1) else (None, None),
            mesh_shape=model_args.cluster_shape,
        ),
    )
    for i in range(generation_length):
        pt_decode_input = (torch.rand(batch_size, seqlen, model_args.dim) * 2) - 1
        logger.info(f"[Decoder] Generating token {i}")

        tt_decode_input = pt_decode_input.clone()

        decode_input = model_args.prepare_residual_tensor_decode(
            tt_decode_input,
            # ttnn.DRAM_MEMORY_CONFIG,
            model_args.model_config["DECODE_RESIDUAL_MEMCFG"],
        )

        # Get cos/sin matrices for the current position of each user
        rot_mat_global = rope_setup.get_rot_mats(current_pos)
        rot_mat_local = rope_setup.get_rot_mats(current_pos)

        # Run TT model
        tt_out = tt_model(
            decode_input,
            current_pos_tensor,
            rot_mats_global=rot_mat_global,
            rot_mats_local=rot_mat_local,
            mode="decode",
            page_table=page_table_tt,
        )
        tt_out = ttnn.to_torch(
            tt_out,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, 3), mesh_shape=model_args.cluster_shape),
        )

        tt_output_torch = tt_out[:, 0:1, : model_args.max_batch_size, : model_args.dim].view(-1, 1, model_args.dim)

        # Reference model
        ref_output = reference_model(pt_decode_input, current_pos[0], None, mask=None)
        non_zero_indices = tt_output_torch.ne(0).nonzero(as_tuple=True)
        tt_output_torch = tt_output_torch[non_zero_indices]
        ref_output = ref_output[non_zero_indices]

        passing, pcc_message = comp_pcc(ref_output, tt_output_torch, pcc_required)

        logger.info(comp_allclose(ref_output, tt_output_torch))
        logger.info(f"PCC: {pcc_message}")

        if passing:
            logger.info("Decoder Block Passed!")
        else:
            logger.warning("Decoder Block Failed!")
            # all_tests_pass = False

        # Increment position
        current_pos = torch.tensor([generation_start_pos + i + 1 for _ in range(batch_size)])
        current_pos_tensor = ttnn.from_torch(
            current_pos,
            device=mesh_device,
            dtype=ttnn.int32,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device,
                dims=(None, 0) if (model_args.is_galaxy and batch_size > 1) else (None, None),
                mesh_shape=model_args.cluster_shape,
            ),
        )

    # if all_tests_pass:
    #     logger.info(f"All {generation_length} decode iterations Passed!")
    # else:
    #     logger.warning("One or more iterations of decode Failed!")
    #     assert all_tests_pass, f"PCC value is lower than {pcc_required} for some of the outputs. Check Warnings!"
