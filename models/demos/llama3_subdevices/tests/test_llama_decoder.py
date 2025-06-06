# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
from loguru import logger
import os
import ttnn
from models.demos.llama3_subdevices.tt.llama_common import (
    precompute_freqs,
    PagedAttentionConfig,
)
from models.demos.llama3_subdevices.tt.model_config import TtModelArgs
from models.demos.llama3_subdevices.tt.llama_decoder import TtTransformerBlock
from models.demos.llama3_subdevices.tt.llama_rope import TtLlamaRotarySetup
from models.demos.t3000.llama2_70b.reference.llama.llama31_8b.model import TransformerBlock
from models.utility_functions import (
    comp_pcc,
    comp_allclose,
)
from models.utility_functions import skip_for_grayskull
from models.demos.llama3_subdevices.tt.prefetcher_common import TtLlamaPrefetcherSetup
from models.demos.llama3_subdevices.tt.llama_ccl import TT_CCL

is_RING_6U = os.environ.get("RING_6U", "0") == "1"


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
    "batch_size",
    (32,),
)
@pytest.mark.parametrize(
    "max_seq_len",
    (256,),  # For decode-only unit test, there's no need to run with large sequence lengths
)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "trace_region_size": 165136000,
            "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING if is_RING_6U else ttnn.FabricConfig.FABRIC_1D,
        }
    ],
    indirect=True,
)
def test_llama_decoder_inference(
    max_seq_len,
    batch_size,
    paged_attention,
    page_params,
    mesh_device,
    use_program_cache,
    reset_seeds,
    ensure_gc,
):
    dtype = ttnn.bfloat8_b

    model_args = TtModelArgs(mesh_device, max_batch_size=batch_size, max_seq_len=max_seq_len, dummy_weights=False)
    model_args.n_layers = 1

    state_dict = model_args.load_state_dict()

    prefetcher_setup = TtLlamaPrefetcherSetup(
        mesh_device,
        n_tensors=5,
        n_layers=model_args.n_layers,
    )
    mesh_device.set_sub_device_stall_group(
        [prefetcher_setup.prefetcher_sub_device_id, prefetcher_setup.worker_sub_device_id]
    )

    tt_ccl = TT_CCL(mesh_device, model_args, prefetcher_setup.worker_sub_device_id)
    # Ref model needs partial state dict, but our models use full state dict keys as cached weight names
    first_layer_prefix = model_args.get_state_dict_prefix("TtTransformerBlock", 0)
    partial_state_dict = {
        k[len(first_layer_prefix) :]: v for k, v in state_dict.items() if (k.startswith(first_layer_prefix))
    }
    reference_model = TransformerBlock(layer_id=0, args=model_args)
    reference_model.load_state_dict(partial_state_dict)

    generation_start_pos = 127
    generation_length = 10
    all_tests_pass = True

    # Setup RoPE transformation matrices
    rope_setup = TtLlamaRotarySetup(
        mesh_device,
        model_args.max_batch_size,
        model_args.head_dim,
        model_args.max_seq_len,
        model_args.rope_theta,
        model_args.use_scaled_rope,
        model_args.rope_scaling_factor,
    )
    transformation_mats = rope_setup.get_both_trans_mats()

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

    # Initialize TT model
    tt_model = TtTransformerBlock(
        args=model_args,
        mesh_device=mesh_device,
        dtype=dtype,
        state_dict=state_dict,
        layer_num=0,
        n_layers=model_args.n_layers,
        weight_cache_path=model_args.weight_cache_path(dtype),
        transformation_mats=transformation_mats,
        paged_attention_config=paged_attention_config,
        prefetcher_setup=prefetcher_setup,
        tt_ccl=tt_ccl,
    )

    seqlen = 1

    cos, sin = precompute_freqs(
        model_args.head_dim,
        model_args.max_seq_len * 2,
        model_args.rope_theta,
        model_args.use_scaled_rope,
        model_args.rope_scaling_factor,
    )
    freqs_cis = torch.complex(cos, sin)

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
    # Explicitly allocate global CB to avoid memory fragmentation
    prefetcher_setup.create_global_cb()

    for i in range(generation_length):
        logger.info(f"[Decoder] Generating token {i}")

        # input = torch.randn(1, 32, 4096)
        pt_decode_input = (torch.rand(batch_size, seqlen, model_args.dim) * 2) - 1
        tt_decode_input = pt_decode_input.clone()

        decode_input = model_args.prepare_residual_tensor_decode(
            tt_decode_input,
            model_args.model_config["DECODE_RESIDUAL_MEMCFG"],
        )

        # Get cos/sin matrices for the current position of each user
        rot_mats = rope_setup.get_rm_rot_mats(current_pos)
        tt_pf = prefetcher_setup.get_input_tensors()
        ttnn.dram_prefetcher(
            tt_pf,
            num_layers=1,
            global_cb=prefetcher_setup.global_circular_buffer,
        )
        mesh_device.set_sub_device_stall_group([prefetcher_setup.worker_sub_device_id])

        # Run TT model
        res = None
        tt_out, res = tt_model(
            decode_input,
            res,
            current_pos_tensor,
            rot_mats=rot_mats,
            mode="decode",
            page_table=page_table_tt,
        )
        tt_output_torch = ttnn.to_torch(
            tt_out,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, 3), mesh_shape=model_args.cluster_shape),
        )[:, 0:1, : model_args.max_batch_size, : model_args.dim].view(-1, 1, model_args.dim)

        # In this test all users have the same position
        freqs_cis_i = freqs_cis[current_pos[0], :].unsqueeze(0)

        # Reference model
        ref_output = reference_model(pt_decode_input, current_pos[0], freqs_cis_i, mask=None)

        passing, pcc_message = comp_pcc(ref_output, tt_output_torch)

        logger.info(comp_allclose(ref_output, tt_output_torch))
        logger.info(f"PCC: {pcc_message}")

        if passing:
            logger.info("Llama Decoder Block Passed!")
        else:
            logger.warning("Llama Decoder Block Failed!")
            all_tests_pass = False

        # Increment position
        current_pos = torch.tensor([generation_start_pos + i for _ in range(batch_size)])
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

    tt_ccl.close()

    if all_tests_pass:
        logger.info(f"All {generation_length} Llama decode iterations Passed!")
    else:
        logger.warning("One or more iterations of Llama decode Failed!")
        assert all_tests_pass, f"PCC value is lower than {0.99} for some of the outputs. Check Warnings!"
