# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
from loguru import logger
import ttnn
from models.demos.llama3_70b_galaxy.tt.llama_attention import TtLlamaAttention
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
def test_qwen_attention_ttt_inference(
    max_seq_len,
    batch_size,
    paged_attention,
    page_params,
    mesh_device,
    reset_seeds,
):
    dtype = ttnn.bfloat8_b
    pcc = 0.99

    # Load tt_transformers reference model args for reference attention
    model_args_ref = ModelArgs(mesh_device, max_batch_size=batch_size, max_seq_len=max_seq_len, cache_hf=True)
    model_args_ref.n_layers = 1  # For the unit test, just run a single layer

    state_dict_ref = model_args_ref.load_state_dict()

    first_layer_prefix_ref = model_args_ref.get_state_dict_prefix("Attention", 0) + "."
    # Ref model needs partial state dict, but our models use full state dict keys as cached weight names
    partial_state_dict_ref = {
        k[len(first_layer_prefix_ref) :]: v for k, v in state_dict_ref.items() if k.startswith(first_layer_prefix_ref)
    }

    # Use tt_transformers reference attention
    reference_model = model_args_ref.reference_attention()
    reference_model.load_state_dict(partial_state_dict_ref)
    logger.info(f"tt_transformers Reference Model Loaded")

    # Load Qwen3 model using TtQwenModelArgs
    model_args = TtQwenModelArgs(mesh_device, dummy_weights=False, max_batch_size=batch_size, max_seq_len=max_seq_len)
    model_args.n_layers = 1  # For the unit test, just run a single layer

    state_dict = model_args.load_state_dict()
    logger.info(f"Qwen3 Model Loaded")

    first_layer_prefix = model_args.get_state_dict_prefix("TtLlamaAttention", 0) + "."
    # Ref model needs partial state dict, but our models use full state dict keys as cached weight names
    partial_state_dict = {
        k[len(first_layer_prefix) :]: v for k, v in state_dict.items() if k.startswith(first_layer_prefix)
    }

    seq_len = 1

    generation_start_pos = 127
    generation_length = 1
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
        n_tensors=2,
        n_layers=1,
    )
    mesh_device.set_sub_device_stall_group(
        [prefetcher_setup.prefetcher_sub_device_id, prefetcher_setup.worker_sub_device_id]
    )

    tt_ccl = TT_CCL(mesh_device, model_args, prefetcher_setup.worker_sub_device_id, use_qwen_mlp=True)

    tt_model = TtLlamaAttention(
        mesh_device,
        state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
        layer_num=0,
        dtype=dtype,
        transformation_mats=transformation_mats,
        configuration=model_args,
        paged_attention_config=paged_attention_config,
        prefetcher_setup=prefetcher_setup,
        tt_ccl=tt_ccl,
    )

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
        # 70B attention block typically sees tensors with mean 0 and std 0.03 - 0.05 in layer 1
        pt_attention_input = (
            torch.randn(
                batch_size,
                seq_len,
                model_args.dim,
                dtype=get_ref_model_dype(reference_model, model_args_ref.model_name),
            )
            * 0.05
        )

        tt_attention_input = pt_attention_input.clone()

        attention_input = model_args.prepare_residual_tensor_decode(
            tt_attention_input,
            model_args.model_config["SHARDED_ATTN_INPUT_RING_MEMCFG"],
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

        logger.info("Starting attention computation")

        tt_out = tt_model(
            attention_input,
            current_pos_tensor,
            rot_mats=rot_mats,
            mode="decode",
            page_table=page_table_tt,
        )

        # multi-device attention module returns replicated output
        tt_out = ttnn.to_torch(
            tt_out,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, 3), mesh_shape=model_args.cluster_shape),
        )
        tt_output_torch = tt_out[:, 0:1, : model_args.max_batch_size, : model_args.dim].view(-1, 1, model_args.dim)

        # # Prepare reference input - adjust dimensions if needed
        # ref_input = pt_attention_input[:, :, : model_args_ref.dim]
        # if ref_input.shape[-1] != model_args_ref.dim:
        #     # Pad or truncate to match reference model dimensions
        #     if ref_input.shape[-1] < model_args_ref.dim:
        #         padding = torch.zeros(ref_input.shape[:-1] + (model_args_ref.dim - ref_input.shape[-1],))
        #         ref_input = torch.cat([ref_input, padding], dim=-1)
        #     else:
        #         ref_input = ref_input[:, :, : model_args_ref.dim]

        # In this test all users have the same position (if using batch > 1)
        freqs_cis_i_ref = freqs_cis_ref[current_pos[0], :].unsqueeze(0)

        reference_output = reference_model(
            pt_attention_input.to(torch.bfloat16), current_pos[0], freqs_cis_i_ref, mask=None
        )

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

        passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc)

        logger.info(comp_allclose(reference_output, tt_output_torch))
        logger.info(f"PCC: {pcc_message}")
        if passing:
            logger.info(f"[pos={current_pos[0]}] Qwen_Attention_TTT Passed!")
        else:
            logger.warning(f"[pos={current_pos[0]}] Qwen_Attention_TTT Failed!")
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

        check_kv_cache = True
        if check_kv_cache:
            # PyTorch output --------------------------------------------------------------------
            pytorch_layer_present = [
                reference_model.cache_k.clone().permute(0, 2, 1, 3),  # [batch_size, n_kv_heads, seq, head_dim]
                reference_model.cache_v.clone().permute(0, 2, 1, 3),  # [batch_size, n_kv_heads, seq, head_dim]
            ]
            # TT hardware execution -------------------------------------------------------------
            if paged_attention:
                tt_layer_present = [
                    (
                        ttnn.to_torch(
                            cache,
                            mesh_composer=ttnn.ConcatMesh2dToTensor(
                                mesh_device,
                                dims=(1, 0) if model_args.is_galaxy else (0, 1),
                                mesh_shape=model_args.cluster_shape,
                            ),
                        )
                        .reshape(
                            model_args.num_device_groups,
                            paged_attention_config.max_num_blocks,
                            model_args.n_kv_heads,
                            paged_attention_config.block_size,
                            model_args.head_dim,
                        )[
                            : 1 if batch_size == 1 else model_args.num_device_groups,
                            reverse_permutation,
                            : model_args.n_kv_heads,
                            :,
                            : model_args.head_dim,
                        ]
                        .reshape(
                            model_args.max_batch_size,
                            paged_attention_config.max_num_blocks // model_args.batch_size_per_device_group,
                            model_args.n_kv_heads,
                            paged_attention_config.block_size,
                            model_args.head_dim,
                        )
                        .transpose(1, 2)
                        .reshape(model_args.max_batch_size, model_args.n_kv_heads, -1, model_args.head_dim)[
                            :batch_size, ...
                        ]
                    )
                    for cache in tt_model.layer_past
                ]
            else:
                tt_layer_present = [
                    ttnn.to_torch(
                        cache,
                        mesh_composer=ttnn.ConcatMesh2dToTensor(
                            mesh_device,
                            dims=(1, 0) if model_args.is_galaxy else (0, 1),
                            mesh_shape=model_args.cluster_shape,
                        ),
                    )[:batch_size, :, :, :]
                    for cache in tt_model.layer_past
                ]

            for i, (cache_pt, cache_tt) in enumerate(zip(pytorch_layer_present, tt_layer_present)):
                cache_length_to_check = min(model_args.max_seq_len, generation_start_pos + generation_length + 1)

                # Adjust cache dimensions if needed
                if cache_pt.shape[1] != cache_tt.shape[1]:  # n_kv_heads mismatch
                    min_kv_heads = min(cache_pt.shape[1], cache_tt.shape[1])
                    cache_pt = cache_pt[:, :min_kv_heads, :, :]
                    cache_tt = cache_tt[:, :min_kv_heads, :, :]

                if cache_pt.shape[3] != cache_tt.shape[3]:  # head_dim mismatch
                    min_head_dim = min(cache_pt.shape[3], cache_tt.shape[3])
                    cache_pt = cache_pt[:, :, :, :min_head_dim]
                    cache_tt = cache_tt[:, :, :, :min_head_dim]

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
    tt_ccl.close()
    if all_tests_pass:
        logger.info("Qwen Attention TTT output Passed!")
    else:
        logger.warning("Qwen Attention TTT output Failed!")
        assert all_tests_pass, f"PCC value is lower than {pcc} for some of the outputs. Check Warnings!"
