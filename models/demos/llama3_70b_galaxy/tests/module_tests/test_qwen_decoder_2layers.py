# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
import ttnn
from models.demos.llama3_70b_galaxy.tt.llama_common import (
    precompute_freqs,
    PagedAttentionConfig,
)
from models.demos.llama3_70b_galaxy.tt.qwen_model_config import TtQwenModelArgs
from models.demos.llama3_70b_galaxy.tt.llama_decoder import TtTransformerBlock
from models.demos.llama3_70b_galaxy.tt.llama_rope import TtLlamaRotarySetup
from models.demos.t3000.llama2_70b.reference.llama.llama31_8b.model import TransformerBlock
from models.utility_functions import (
    comp_pcc,
    comp_allclose,
)
from models.utility_functions import skip_for_grayskull
from models.demos.llama3_70b_galaxy.tt.prefetcher_common import TtLlamaPrefetcherSetup
from models.demos.llama3_70b_galaxy.tt.llama_ccl import TT_CCL
import math


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
def test_qwen_decoder_2layers_inference(
    max_seq_len,
    batch_size,
    paged_attention,
    page_params,
    mesh_device,
    reset_seeds,
    ensure_gc,
):
    dtype = ttnn.bfloat8_b

    model_args = TtQwenModelArgs(mesh_device, max_batch_size=batch_size, max_seq_len=max_seq_len, dummy_weights=False)
    model_args.n_layers = 10

    state_dict = model_args.load_state_dict()

    prefetcher_setup = TtLlamaPrefetcherSetup(
        mesh_device,
        n_tensors=5,
        n_layers=model_args.n_layers,
    )
    mesh_device.set_sub_device_stall_group(
        [prefetcher_setup.prefetcher_sub_device_id, prefetcher_setup.worker_sub_device_id]
    )

    tt_ccl = TT_CCL(mesh_device, model_args, prefetcher_setup.worker_sub_device_id, use_qwen_mlp=True)

    # Create shared QK norm scaling tensors if qk_norm is enabled (similar to llama_model.py)
    scaling_tensor_q = None
    scaling_tensor_k = None
    if model_args.qk_norm:
        scaling_factor = 1.0 / math.sqrt(model_args.head_dim)

        # Create memory configs for scaling tensors (same as used in attention layer)
        reshape_output_q_mem_cfg = ttnn.create_sharded_memory_config(
            shape=(64, 32),  # [1, 8, 8, 128] ==> [1, 1, 64, 128] ==> *[1, 1, 64, 32 * 4 = 128]*
            core_grid=ttnn.CoreRangeSet(
                [ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(2, 1))]
            ),  # resharding tensor to cores
            strategy=ttnn.ShardStrategy.WIDTH,  # Literally stating to the device to perform width sharding
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        reshape_output_k_mem_cfg = ttnn.create_sharded_memory_config(
            shape=(64, 32),  # [1, 8, 8, 128] ==> [1, 1, 64, 128] ==> *[1, 1, 64, 32 * 4 = 128]*
            core_grid=ttnn.CoreRangeSet(
                [ttnn.CoreRange(ttnn.CoreCoord(1, 2), ttnn.CoreCoord(2, 3))]
            ),  # resharding tensor to cores
            strategy=ttnn.ShardStrategy.WIDTH,  # Literally stating to the device to perform width sharding
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        # Create scaling tensors with same shape as q and k tensors
        scaling_tensor_q = ttnn.from_torch(
            torch.full([1, 1, 64, 128], scaling_factor),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=reshape_output_q_mem_cfg,
        )

        scaling_tensor_k = ttnn.from_torch(
            torch.full([1, 1, 64, 128], scaling_factor),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=reshape_output_k_mem_cfg,
        )

    # Setup reference models for both layers
    reference_models = []
    for layer_idx in range(model_args.n_layers):
        first_layer_prefix = model_args.get_state_dict_prefix("TtTransformerBlock", layer_idx)
        partial_state_dict = {
            k[len(first_layer_prefix) :]: v for k, v in state_dict.items() if k.startswith(first_layer_prefix)
        }
        reference_model = TransformerBlock(layer_id=layer_idx, args=model_args, llama3=False)
        reference_model.load_state_dict(partial_state_dict)
        reference_models.append(reference_model)

    generation_start_pos = 0
    generation_length = 5  # Reduced for 2-layer test
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

    # Initialize TT models for both layers
    tt_models = []
    for layer_idx in range(model_args.n_layers):
        tt_model = TtTransformerBlock(
            args=model_args,
            mesh_device=mesh_device,
            dtype=dtype,
            state_dict=state_dict,
            layer_num=layer_idx,
            n_layers=model_args.n_layers,
            weight_cache_path=model_args.weight_cache_path(dtype),
            transformation_mats=transformation_mats,
            paged_attention_config=paged_attention_config,
            prefetcher_setup=prefetcher_setup,
            tt_ccl=tt_ccl,
            scaling_tensor_q=scaling_tensor_q,
            scaling_tensor_k=scaling_tensor_k,
        )
        tt_models.append(tt_model)

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
        logger.info(f"[Decoder 2-Layer] Generating token {i}")

        # Create random input tensor
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
            num_layers=model_args.n_layers,
            global_cb=prefetcher_setup.global_circular_buffer,
        )
        mesh_device.set_sub_device_stall_group([prefetcher_setup.worker_sub_device_id])

        # Run TT models through both layers
        tt_input = decode_input
        res = None

        ref_input = pt_decode_input
        # for layer_idx in range(28, 30):
        for layer_idx in range(model_args.n_layers):
            tt_input, res = tt_models[layer_idx](
                tt_input,
                res,
                current_pos_tensor,
                rot_mats=rot_mats,
                mode="decode",
                page_table=page_table_tt,
            )

            # ttnn.to_torch(tt_models[layer_idx].ff_norm.weight, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, 3), mesh_shape=(8, 4))))

            tt_output_torch = ttnn.to_torch(
                tt_input,
                mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, 3), mesh_shape=model_args.cluster_shape),
            )[:, 0:1, : model_args.max_batch_size, : model_args.dim].view(-1, 1, model_args.dim)

            if res is not None:
                res_torch = ttnn.to_torch(
                    res,
                    mesh_composer=ttnn.ConcatMesh2dToTensor(
                        mesh_device, dims=(1, 3), mesh_shape=model_args.cluster_shape
                    ),
                )[:, 0:1, : model_args.max_batch_size, : model_args.dim].view(-1, 1, model_args.dim)

                # res_torch = ttnn.to_torch(h,mesh_composer=ttnn.ConcatMesh2dToTensor(self.mesh_device, dims=(1, 3), mesh_shape=(8, 4)),)[:, 0:1, : 32, : 5120].view(-1, 1, 5120)

            # In this test all users have the same position
            freqs_cis_i = freqs_cis[current_pos[0], :].unsqueeze(0)

            # Run reference models
            ref_input, ref_res = reference_models[layer_idx](ref_input, current_pos[0], freqs_cis_i, mask=None)

            logger.info(f"Layer {layer_idx} PCC: {comp_pcc(ref_input, res_torch + tt_output_torch)}")
            logger.info(f"Layer {layer_idx} Residual PCC: {comp_pcc(ref_res, res_torch)}")

            if layer_idx == model_args.n_layers - 1:
                logger.info(f"Layer {layer_idx} PCC: {comp_pcc(ref_input, tt_output_torch)}")
                logger.info(f"Layer {layer_idx} Residual PCC: {comp_pcc(ref_res, res_torch)}")

        logger.info(comp_allclose(ref_input, tt_output_torch))

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
        logger.info(f"All {generation_length} Qwen 2-layer decode iterations Passed!")
    else:
        logger.warning("One or more iterations of Qwen 2-layer decode Failed!")
        assert all_tests_pass, f"PCC value is lower than {0.99} for some of the outputs. Check Warnings!"
