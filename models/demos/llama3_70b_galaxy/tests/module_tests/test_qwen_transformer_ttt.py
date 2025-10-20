# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
import ttnn
from models.demos.llama3_70b_galaxy.tt.llama_common import (
    PagedAttentionConfig,
)
from models.demos.llama3_70b_galaxy.tt.qwen_model_config import TtQwenModelArgs
from models.demos.llama3_70b_galaxy.tt.llama_model import TtTransformer
from models.tt_transformers.tt.model_config import ModelArgs
from models.tt_transformers.tests.test_utils import get_ref_model_dype
from models.utility_functions import (
    comp_pcc,
    comp_allclose,
)
from models.utility_functions import skip_for_grayskull


@torch.no_grad()
@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "fabric_config": True,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        (8, 4),
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "paged_attention",
    (
        True,
        # False,
    ),
    ids=(
        "paged_attention",
        # "default_attention",
    ),
)
@pytest.mark.parametrize(
    "page_params",
    [{"page_block_size": 64, "page_max_num_blocks": 4096}],
)
@pytest.mark.parametrize(
    "batch_size",
    (32,),
)
@pytest.mark.parametrize(
    "max_seq_len",
    (256,),  # For decode-only unit test, there's no need to run with large sequence lengths
)
def test_qwen_transformer_ttt_inference(
    max_seq_len,
    batch_size,
    paged_attention,
    page_params,
    mesh_device,
    reset_seeds,
    ensure_gc,
):
    dtype = ttnn.bfloat8_b

    num_layers = 10

    # Load tt_transformers reference model args for reference transformer
    model_args_ref = ModelArgs(mesh_device, max_batch_size=batch_size, max_seq_len=max_seq_len, cache_hf=True)
    model_args_ref.n_layers = num_layers

    state_dict_ref = model_args_ref.load_state_dict()
    state_dict_prefix_ref = model_args_ref.get_state_dict_prefix("", None)
    reference_state_dict = {
        k[len(state_dict_prefix_ref) :]: v
        for k, v in state_dict_ref.items()
        if (
            any([f"{state_dict_prefix_ref}layers.{i}." in k for i in range(model_args_ref.n_layers)])
            or any(
                [
                    f"{state_dict_prefix_ref}{name}" in k
                    for name in ["tok_embeddings.weight", "norm.weight", "output.weight"]
                ]
            )
        )
    }

    # Use tt_transformers reference transformer
    reference_model = model_args_ref.reference_transformer()
    reference_model.load_state_dict(reference_state_dict)
    logger.info(f"tt_transformers Reference Transformer Model Loaded")

    # Load Qwen3 model using TtQwenModelArgs
    model_args = TtQwenModelArgs(mesh_device, max_batch_size=batch_size, max_seq_len=max_seq_len, dummy_weights=False)
    model_args.n_layers = num_layers

    state_dict = model_args.load_state_dict()
    logger.info(f"Qwen3 Transformer Model Loaded")

    generation_start_pos = 0
    generation_length = 5
    all_tests_pass = True

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
            model_args.batch_size_per_device_group,
            paged_attention_config.max_num_blocks // model_args.batch_size_per_device_group,
        )
        page_table_tt = ttnn.from_torch(
            page_table,
            device=mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device,
                dims=(None, None),
                mesh_shape=model_args.cluster_shape,
            ),
        )

    # Initialize TT Qwen model
    tt_model = TtTransformer(
        args=model_args,
        dtype=dtype,
        mesh_device=mesh_device,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
        paged_attention_config=paged_attention_config,
        mode="decode",
    )

    # # Embedding on host for reference model
    # embd_ref = model_args_ref.reference_embedding(reference_model)
    # embd_ref.load_state_dict({"emb.weight": state_dict_ref[f"{state_dict_prefix_ref}tok_embeddings.weight"]})

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
        logger.info(f"[Qwen Transformer TTT] Generating token {i}")

        # Create random input tensor with appropriate dtype for reference model
        pt_decode_input = (
            torch.rand(
                batch_size, seqlen, model_args.dim, dtype=get_ref_model_dype(reference_model, model_args_ref.model_name)
            )
            * 2
        ) - 1
        tt_decode_input = pt_decode_input.clone()

        decode_input = model_args.prepare_residual_tensor_decode(
            tt_decode_input,
            model_args.model_config["DECODE_RESIDUAL_MEMCFG"],
        )

        # Get cos/sin matrices for the current position of each user
        rot_mats = tt_model.rope_setup.get_rm_rot_mats(current_pos)

        # Run TT model
        tt_out = tt_model(
            decode_input,
            current_pos_tensor,
            rot_mats=rot_mats,
            mode="decode",
            page_table=page_table_tt,
        )

        # Convert ttnn tensor to torch tensor
        mesh_composer = ttnn.ConcatMesh2dToTensor(
            mesh_device, dims=(3, 1) if model_args.is_galaxy else (1, -1), mesh_shape=model_args.cluster_shape
        )

        outs = [ttnn.to_torch(out, mesh_composer=mesh_composer) for out in tt_out]
        outs = torch.concat(outs, dim=-1)
        tt_output_torch = outs.permute(2, 1, 0, 3).squeeze(2)[: model_args.max_batch_size, 0:1, : model_args.vocab_size]

        ref_output = reference_model(pt_decode_input.to(torch.bfloat16), current_pos[0])

        passing, pcc_message = comp_pcc(ref_output, tt_output_torch)

        logger.info(comp_allclose(ref_output, tt_output_torch))
        logger.info(f"PCC: {pcc_message}")

        if passing:
            logger.info("Qwen Transformer TTT Passed!")
        else:
            logger.warning("Qwen Transformer TTT Failed!")
            all_tests_pass = False

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

    tt_model.tt_ccl.close()

    if all_tests_pass:
        logger.info(f"All {generation_length} Qwen transformer TTT iterations Passed!")
    else:
        logger.warning("One or more iterations of Qwen transformer TTT Failed!")
        assert all_tests_pass, f"PCC value is lower than {0.99} for some of the outputs. Check Warnings!"
