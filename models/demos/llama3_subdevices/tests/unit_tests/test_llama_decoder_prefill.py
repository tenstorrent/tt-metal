# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
from loguru import logger
import os
import ttnn
from models.demos.llama3_subdevices.tt.llama_common import (
    get_prefill_rot_mat,
    get_rot_transformation_mat,
    PagedAttentionConfig,
)
from models.demos.llama3_subdevices.tt.llama_decoder import TtTransformerBlock
from models.demos.llama3_subdevices.tt.model_config import TtModelArgs
from models.demos.t3000.llama2_70b.reference.llama.llama31_8b.model import TransformerBlock, precompute_freqs_cis
from models.utility_functions import (
    comp_pcc,
    comp_allclose,
)
from models.utility_functions import skip_for_grayskull
from models.demos.llama3_subdevices.tt.prefetcher_common import TtLlamaPrefetcherSetup
from models.demos.llama3_subdevices.tt.llama_ccl import TT_CCL


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
    "max_seq_len",
    (
        4096,
        128,
    ),
)
@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
def test_llama_decoder_inference(
    max_seq_len,
    paged_attention,
    page_params,
    mesh_device,
    reset_seeds,
    ensure_gc,
):
    dtype = ttnn.bfloat8_b
    batch_size = 1  # For prefill we only support batch_size = 1

    model_args = TtModelArgs(mesh_device, max_batch_size=batch_size, max_seq_len=max_seq_len)
    model_args.use_prefetcher = False

    model_args.n_layers = 1

    state_dict = model_args.load_state_dict()

    # Ref model needs partial state dict, but our models use full state dict keys as cached weight names
    first_layer_prefix = model_args.get_state_dict_prefix("TtTransformerBlock", 0)
    partial_state_dict = {
        k[len(first_layer_prefix) :]: v for k, v in state_dict.items() if (k.startswith(first_layer_prefix))
    }

    reference_model = TransformerBlock(layer_id=0, args=model_args)
    reference_model.load_state_dict(partial_state_dict)

    generation_start_pos = 0
    generation_length = 1
    all_tests_pass = True

    # pre-compute the rotational embedding matrix and send to device
    rot_mats = get_prefill_rot_mat(
        model_args.head_dim,
        model_args.max_seq_len,
        mesh_device,
        seq_len=max_seq_len,
        scale_factor=model_args.rope_scaling_factor,
    )
    transformation_mat_torch = get_rot_transformation_mat(model_args.head_dim)
    transformation_mats_prefill = ttnn.as_tensor(
        transformation_mat_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    transformation_mats = {"prefill": transformation_mats_prefill}

    # Setup page table
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
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
    prefetcher_setup = TtLlamaPrefetcherSetup(mesh_device, n_tensors=0, n_layers=1, mode="prefill")
    mesh_device.set_sub_device_stall_group([prefetcher_setup.worker_sub_device_id])
    tt_ccl = TT_CCL(mesh_device, model_args, prefetcher_setup.worker_sub_device_id, mode="prefill")

    # Initialize TT model
    tt_model = TtTransformerBlock(
        mesh_device=mesh_device,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
        layer_num=0,
        n_layers=1,
        dtype=dtype,
        transformation_mats=transformation_mats,
        args=model_args,
        paged_attention_config=paged_attention_config,
        prefetcher_setup=prefetcher_setup,
        tt_ccl=tt_ccl,
    )

    for i in range(generation_length):
        logger.info(f"[Decoder] Generating token {i}")
        pt_decode_input = (torch.rand(batch_size, max_seq_len, model_args.dim) * 2) - 1
        tt_decode_input = pt_decode_input.clone()
        decode_input = model_args.prepare_residual_tensor_prefill(
            tt_decode_input,
        )
        positions = torch.LongTensor(range(max_seq_len))
        freqs_cis_i = precompute_freqs_cis(
            model_args.head_dim,
            model_args.max_seq_len * 2,
            model_args.rope_theta,
            model_args.use_scaled_rope,
            model_args.rope_scaling_factor,
        )[positions]

        # Reference model
        attn_mask = torch.full((max_seq_len, max_seq_len), torch.finfo(torch.float32).min)
        attn_mask_torch = torch.triu(attn_mask, diagonal=1)
        ref_output = reference_model(pt_decode_input, positions[0], freqs_cis_i, mask=attn_mask_torch)
        # Run TT model
        tt_out, _ = tt_model(decode_input, None, None, rot_mats, user_id=0, mode="prefill", page_table=page_table_tt)
        tt_out = ttnn.to_torch(
            tt_out,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, 3), mesh_shape=model_args.cluster_shape),
        )
        tt_output_torch = tt_out[:, 0:1, :, : model_args.dim].view(
            batch_size, max_seq_len, -1
        )  # [ batch_size, seq, hidden_dim]
        passing, pcc_message = comp_pcc(ref_output, tt_output_torch)

        logger.info(comp_allclose(ref_output, tt_output_torch))
        logger.info(f"PCC: {pcc_message}")
        if passing:
            logger.info("Llama Decoder Block Passed!")
        else:
            logger.warning("Llama Decoder Block Failed!")
            all_tests_pass = False
    tt_ccl.close()
    if all_tests_pass:
        logger.info(f"All Llama decode iterations Passed!")
    else:
        logger.warning("One or more iterations of Llama decode Failed!")
        assert all_tests_pass, f"PCC value is lower than {0.99} for some of the outputs. Check Warnings!"
