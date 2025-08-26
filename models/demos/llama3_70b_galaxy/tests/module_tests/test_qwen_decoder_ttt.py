# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
from loguru import logger
import ttnn
from models.demos.llama3_70b_galaxy.tt.llama_decoder import TtTransformerBlock
from models.demos.llama3_70b_galaxy.tt.llama_rope import TtLlamaRotarySetup
from models.demos.llama3_70b_galaxy.tt.qwen_model_config import TtQwenModelArgs
from models.tt_transformers.tt.model_config import ModelArgs
from models.tt_transformers.tests.test_utils import get_ref_model_dype
from models.demos.llama3_70b_galaxy.tt.llama_common import (
    precompute_freqs,
    PagedAttentionConfig,
)
from models.utility_functions import (
    comp_pcc,
    comp_allclose,
)
from models.utility_functions import skip_for_grayskull
from models.demos.llama3_70b_galaxy.tt.prefetcher_common import TtLlamaPrefetcherSetup
from models.demos.llama3_70b_galaxy.tt.llama_ccl import TT_CCL


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
def test_qwen_decoder_ttt_inference(
    max_seq_len,
    batch_size,
    paged_attention,
    page_params,
    mesh_device,
    reset_seeds,
):
    dtype = ttnn.bfloat8_b

    # Load tt_transformers reference model args for reference decoder
    model_args_ref = ModelArgs(mesh_device, max_batch_size=batch_size, max_seq_len=max_seq_len, cache_hf=True)
    model_args_ref.n_layers = 1  # For the unit test, just run a single layer

    state_dict_ref = model_args_ref.load_state_dict()

    # Ref model needs partial state dict, but our models use full state dict keys as cached weight names
    first_layer_prefix_ref = model_args_ref.get_state_dict_prefix("TransformerBlock", 0)
    partial_state_dict_ref = {
        k[len(first_layer_prefix_ref) :]: v for k, v in state_dict_ref.items() if (k.startswith(first_layer_prefix_ref))
    }

    # Use tt_transformers reference decoder
    reference_model = model_args_ref.reference_decoder()
    reference_model.load_state_dict(partial_state_dict_ref)
    logger.info(f"tt_transformers Reference Decoder Model Loaded")

    # Load Qwen3 model using TtQwenModelArgs
    model_args = TtQwenModelArgs(mesh_device, dummy_weights=False, max_batch_size=batch_size, max_seq_len=max_seq_len)
    model_args.n_layers = 1  # For the unit test, just run a single layer

    state_dict = model_args.load_state_dict()
    logger.info(f"Qwen3 Decoder Model Loaded")

    generation_start_pos = 0
    generation_length = 10
    all_tests_pass = True

    # Setup RoPE transformation matrices for Qwen
    rope_setup = TtLlamaRotarySetup(
        mesh_device,
        batch_size,
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

    prefetcher_setup = TtLlamaPrefetcherSetup(
        mesh_device,
        n_tensors=5,  # More tensors needed for decoder (attention + MLP)
        n_layers=1,
    )
    mesh_device.set_sub_device_stall_group(
        [prefetcher_setup.prefetcher_sub_device_id, prefetcher_setup.worker_sub_device_id]
    )

    tt_ccl = TT_CCL(mesh_device, model_args, prefetcher_setup.worker_sub_device_id, use_qwen_mlp=True)

    # Initialize TT Qwen decoder model
    tt_model = TtTransformerBlock(
        args=model_args,
        mesh_device=mesh_device,
        dtype=dtype,
        state_dict=state_dict,
        layer_num=0,
        n_layers=1,
        weight_cache_path=model_args.weight_cache_path(dtype),
        transformation_mats=transformation_mats,
        paged_attention_config=paged_attention_config,
        use_paged_kv_cache=paged_attention,
        prefetcher_setup=prefetcher_setup,
        tt_ccl=tt_ccl,
    )

    seqlen = 1

    # Setup freqs for Qwen
    cos, sin = precompute_freqs(
        model_args.head_dim,
        model_args.max_seq_len * 2,
        model_args.rope_theta,
        model_args.use_scaled_rope,
        model_args.rope_scaling_factor,
    )
    freqs_cis = torch.complex(cos, sin)

    # Setup freqs for reference model (tt_transformers)
    from models.tt_transformers.tt.common import precompute_freqs as tt_precompute_freqs

    cos_ref, sin_ref = tt_precompute_freqs(
        model_args_ref.head_dim,
        model_args_ref.max_seq_len * 2,
        model_args_ref.rope_theta,
        model_args_ref.rope_scaling.factor if model_args_ref.rope_scaling else None,
        model_args_ref.rope_scaling.original_max_position_embeddings if model_args_ref.rope_scaling else None,
    )
    freqs_cis_ref = torch.complex(cos_ref, sin_ref)

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

        # Create random input tensor
        pt_decode_input = (
            torch.rand(
                batch_size, seqlen, model_args.dim, dtype=get_ref_model_dype(reference_model, model_args_ref.model_name)
            )
            * 2
        ) - 1
        tt_decode_input = pt_decode_input.clone()
        current_pos_val = generation_start_pos + i

        decode_input = model_args.prepare_residual_tensor_decode(
            tt_decode_input,
            model_args.model_config["DECODE_RESIDUAL_MEMCFG"],
            force_replicated=False if model_args.is_galaxy else True,
        )

        # Get cos/sin matrices for the current position of each user
        rot_mats = rope_setup.get_rm_rot_mats(current_pos)

        ttnn.dram_prefetcher(
            prefetcher_setup.get_input_tensors(),
            num_layers=1,
            global_cb=prefetcher_setup.global_circular_buffer,
        )
        mesh_device.set_sub_device_stall_group([prefetcher_setup.worker_sub_device_id])

        # Run TT Qwen decoder model
        # For Qwen decoder, we need to handle the h tensor (residual connection state)
        if i == 0:
            # First iteration - h is None, will be created from x
            tt_out = tt_model(
                decode_input,
                None,  # h tensor
                current_pos_tensor,
                rot_mats=rot_mats,
                mode="decode",
                page_table=page_table_tt,
            )
        else:
            # Subsequent iterations - reuse h tensor (this would be more complex in a real scenario)
            tt_out = tt_model(
                decode_input,
                None,  # For simplicity, using None - in practice this would be the previous h
                current_pos_tensor,
                rot_mats=rot_mats,
                mode="decode",
                page_table=page_table_tt,
            )

        tt_out = ttnn.to_torch(
            tt_out,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, 3), mesh_shape=model_args.cluster_shape),
        )

        tt_output_torch = tt_out[:, 0:1, : model_args.max_batch_size, : model_args.dim].view(-1, 1, model_args.dim)

        # Prepare reference input - adjust dimensions if needed
        ref_input = pt_decode_input[:, :, : model_args_ref.dim]
        if ref_input.shape[-1] != model_args_ref.dim:
            # Pad or truncate to match reference model dimensions
            if ref_input.shape[-1] < model_args_ref.dim:
                padding = torch.zeros(ref_input.shape[:-1] + (model_args_ref.dim - ref_input.shape[-1],))
                ref_input = torch.cat([ref_input, padding], dim=-1)
            else:
                ref_input = ref_input[:, :, : model_args_ref.dim]

        # In this test all users have the same position
        freqs_cis_i_ref = freqs_cis_ref[current_pos_val, :].unsqueeze(0)

        # Reference model
        reference_output = reference_model(ref_input.to(torch.bfloat16), current_pos_val, freqs_cis_i_ref, mask=None)

        # Adjust reference output dimensions to match Qwen output if needed
        if reference_output.shape[-1] != model_args.dim:
            if reference_output.shape[-1] < model_args.dim:
                # Pad reference output to match Qwen dimensions
                padding = torch.zeros(reference_output.shape[:-1] + (model_args.dim - reference_output.shape[-1],))
                reference_output_padded = torch.cat([reference_output, padding], dim=-1)
                reference_output = reference_output_padded
            else:
                # Truncate reference output to match Qwen dimensions
                reference_output = reference_output[:, :, : model_args.dim]

        passing, pcc_message = comp_pcc(reference_output, tt_output_torch)

        logger.info(comp_allclose(reference_output, tt_output_torch))
        logger.info(f"PCC: {pcc_message}")

        if passing:
            logger.info("Qwen Decoder Block TTT Passed!")
        else:
            logger.warning("Qwen Decoder Block TTT Failed!")
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

    tt_ccl.close()
    if all_tests_pass:
        logger.info(f"All {generation_length} Qwen decode iterations TTT Passed!")
    else:
        logger.warning("One or more iterations of Qwen decode TTT Failed!")
        assert all_tests_pass, f"PCC value is lower than {0.99} for some of the outputs. Check Warnings!"
