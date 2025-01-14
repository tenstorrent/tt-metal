# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
from loguru import logger
import os
import ttnn
from models.demos.llama3.tt.llama_attention import TtLlamaAttention
from models.demos.llama3.tt.llama_rope import TtLlamaRotarySetup
from models.demos.llama3.tt.model_config import TtModelArgs
from models.demos.llama3.tt.llama_common import (
    precompute_freqs,
    PagedAttentionConfig,
)
from models.demos.t3000.llama2_70b.reference.llama.llama31_8b.model import Attention
from models.utility_functions import (
    comp_pcc,
    comp_allclose,
    comp_equal,
)
from models.utility_functions import skip_for_grayskull

from tests.ttnn.unit_tests.operations.speculative_execution.sfd_common import (
    get_buffer_address,
    create_multi_device_tensors,
    read_multi_device_tensor,
)
from tests.ttnn.unit_tests.operations.ccl.test_reduce_scatter_async import (
    create_and_load_sub_device_manager_with_fabric_interface,
    teardown_fabric_interface,
)
from tests.ttnn.unit_tests.operations.speculative_execution.test_speculative_flash_decode import (
    nearest_n,
    nearest_pow_2,
    num_to_corerange,
    fa_rand,
    get_speculative_flash_decode_expected,
    prepare_test_config_and_data,
)

from tests.ttnn.unit_tests.operations.speculative_execution.test_speculative_flash_decode_ccl import (
    commit_priority_tensor,
    set_devices_speculation_state,
)


class TtSFDSetup(torch.nn.Module):
    def __init__(
        self,
        mesh_device,
        nh,
        d,
        b=1,
        grid_size=(8, 7),
        k_chunk_size=128,
        lambda_=1000.0,
        enable_async=True,
        full_setup=True,
    ):
        super().__init__()
        self.mesh_device = mesh_device
        self.nh = nh
        self.d = d
        self.b = b
        self.grid_size = grid_size
        self.k_chunk_size = k_chunk_size
        self.lambda_ = lambda_
        self.enable_async = enable_async

        if full_setup:
            self.setup()

    def setup(self):
        mesh_device = self.mesh_device
        nh = self.nh
        d = self.d
        b = self.b
        grid_size = self.grid_size
        k_chunk_size = self.k_chunk_size
        lambda_ = self.lambda_
        enable_async = self.enable_async

        ############################################################
        # Setup and Defines
        ############################################################
        num_devices = 2
        # Use Async mode based on test input config
        mesh_device.enable_async(enable_async)
        if enable_async:
            logger.info(f"Using Async Mode for All Gather Op Dispatch")
        enable_persistent_fabric = True

        ############################################################
        ### Persistent fabric and ccl setup ###
        ############################################################
        compute_grid_size = mesh_device.compute_with_storage_grid_size()
        self.ccl_sub_device_crs = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))}
        )
        worker_sub_device = ttnn.SubDevice([self.ccl_sub_device_crs])
        self.worker_sub_device_id = ttnn.SubDeviceId(0)
        self.sub_device_stall_group = [self.worker_sub_device_id]
        mesh_sub_device_manager_id = create_and_load_sub_device_manager_with_fabric_interface(
            mesh_device, [worker_sub_device], 0, 0, enable_persistent_fabric
        )
        mesh_device.set_sub_device_stall_group(self.sub_device_stall_group)
        ############################################################

        padded_num_heads = nearest_pow_2(nearest_n(nh, n=32))
        torch.manual_seed(1234)

        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

        self.dram_memcfg = ttnn.DRAM_MEMORY_CONFIG
        self.shard_grid = ttnn.CoreRangeSet({num_to_corerange(b)})
        self.shard_spec = ttnn.ShardSpec(self.shard_grid, (padded_num_heads, d), ttnn.ShardOrientation.ROW_MAJOR)
        self.height_sharded_memcfg = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, self.shard_spec
        )

        self.scale = d**-0.5

        # create global semaphore handles for speculative flash decode
        self.sfd_semaphore_handles = ttnn.create_global_semaphore_with_same_address(
            mesh_device, self.ccl_sub_device_crs, 0
        )
        self.swap_semaphore_handles = ttnn.create_global_semaphore_with_same_address(
            mesh_device, self.ccl_sub_device_crs, 0
        )
        self.k_cache_semaphore_handles = ttnn.create_global_semaphore_with_same_address(
            mesh_device, self.ccl_sub_device_crs, 0
        )
        self.v_cache_semaphore_handles = ttnn.create_global_semaphore_with_same_address(
            mesh_device, self.ccl_sub_device_crs, 0
        )
        addrs = ttnn.get_global_semaphore_address(self.sfd_semaphore_handles)
        logger.info(f"semaphore handle addresses: {addrs}")
        # assert all addresses are the same
        assert len(set(addrs)) == 1

        ##########################################
        #### Prepare test config and data
        ##########################################
        # Configure chunk size and program
        self.program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=grid_size,
            q_chunk_size=padded_num_heads,
            k_chunk_size=k_chunk_size,
            exp_approx_mode=False,
        )

        ##########################################
        #### Priority Tensor ####
        ##########################################
        priority_tensors = [
            torch.ones(1, 1, ttnn.TILE_SIZE, ttnn.TILE_SIZE),
            torch.zeros(1, 1, ttnn.TILE_SIZE, ttnn.TILE_SIZE),
        ]
        reset_priority_tensor = [
            torch.ones(1, 1, ttnn.TILE_SIZE, ttnn.TILE_SIZE),
        ] * num_devices
        self.tt_priority_tensors = create_multi_device_tensors(
            priority_tensors, mesh_device, self.dram_memcfg, ttnn.TILE_LAYOUT, ttnn.uint32
        )
        self.tt_gathered_priority_tensors = create_multi_device_tensors(
            read_multi_device_tensor(self.tt_priority_tensors)[::-1],
            self.mesh_device,
            self.dram_memcfg,
            ttnn.TILE_LAYOUT,
            ttnn.uint32,
        )
        self.tt_reset_priority_tensors = create_multi_device_tensors(
            reset_priority_tensor, mesh_device, self.dram_memcfg, ttnn.TILE_LAYOUT, ttnn.uint32
        )

        ##########################################
        #### Skip Tensor ####
        ##########################################
        skip_tensor = [torch.ones((64, 1, ttnn.TILE_SIZE, ttnn.TILE_SIZE))] * num_devices
        skip_tensor_mem_config = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, ttnn.TILE_SIZE),
            core_grid=self.ccl_sub_device_crs,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        self.tt_skip_tensor = create_multi_device_tensors(
            skip_tensor, mesh_device, skip_tensor_mem_config, ttnn.TILE_LAYOUT, ttnn.uint32
        )
        self.skip_tensor_address = get_buffer_address(self.tt_skip_tensor)
        logger.info(f"Skip tensor address: {self.skip_tensor_address}")

        # Commit the priority tensor once to get the ops program cached without SKIP_COMPUTE
        commit_priority_tensor(self.tt_reset_priority_tensors, self.tt_skip_tensor, mesh_device)
        # self.enable_speculation()
        self.done_first_run = False

    def run_speculative_flash_decode(
        self,
        Q,
        K,
        V,
        cur_pos_tensor,
    ):
        # logger.info(f"Starting speculative flash decode")
        tt_Q = ttnn.to_memory_config(Q, self.dram_memcfg)
        tt_K = ttnn.to_memory_config(K, self.dram_memcfg)
        tt_V = ttnn.to_memory_config(V, self.dram_memcfg)

        # self.reset_skip_tensor()

        # Run speculative flash decode
        outputs = ttnn.experimental.speculative_scaled_dot_product_attention_decode(
            tt_Q,
            tt_K,
            tt_V,
            lambda_=self.lambda_,
            cur_pos_tensor=cur_pos_tensor,
            scale=self.scale,
            program_config=self.program_config,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=self.dram_memcfg,
            priority_tensor=self.tt_priority_tensors,
            other_priority_tensor=self.tt_gathered_priority_tensors,
            ccl_enabled=True,
            multi_device_global_semaphore=self.sfd_semaphore_handles,
        )
        tt_back_gt_md, tt_back_spec_md, tt_back_spec_lp_distance_md, tt_back_lp_norm_x_md = outputs

        # Commit the priority tensor
        # set_devices_speculation_state(self.tt_skip_tensor, True)

        return tt_back_gt_md

    def consolidate_tensor(self, tt_tensor):
        tt_tensor_new = ttnn.experimental.consolidate_cache(
            tt_tensor,
            self.tt_priority_tensors,
            self.tt_gathered_priority_tensors,
            multi_device_global_semaphore=self.sfd_semaphore_handles,
            num_links=1,
        )
        return tt_tensor_new

    def consolidate_kv_cache(self, K_mesh, V_mesh):
        # K_new = read_multi_device_tensor(K_mesh)
        # V_new = read_multi_device_tensor(V_mesh)

        # priority = read_multi_device_tensor(self.tt_priority_tensors)
        # index = torch.argmax(torch.tensor([priority[0][0, 0, 0, 0], priority[1][0, 0, 0, 0]])).item()

        # K_new = [K_new[index], K_new[index]]
        # V_new = [V_new[index], V_new[index]]

        # K_new = create_multi_device_tensors(K_new, self.mesh_device, self.dram_memcfg, ttnn.TILE_LAYOUT, ttnn.bfloat16)
        # V_new = create_multi_device_tensors(V_new, self.mesh_device, self.dram_memcfg, ttnn.TILE_LAYOUT, ttnn.bfloat16)

        # return K_new, V_new

        K_new = ttnn.experimental.consolidate_cache(
            K_mesh,
            self.tt_priority_tensors,
            self.tt_gathered_priority_tensors,
            multi_device_global_semaphore=self.k_cache_semaphore_handles,
            num_links=1,
        )

        V_new = ttnn.experimental.consolidate_cache(
            V_mesh,
            self.tt_priority_tensors,
            self.tt_gathered_priority_tensors,
            multi_device_global_semaphore=self.v_cache_semaphore_handles,
            num_links=1,
        )

        return K_new, V_new

    def disable_speculation(self):
        set_devices_speculation_state(self.tt_skip_tensor, False)

    def enable_speculation(self):
        set_devices_speculation_state(self.tt_skip_tensor, True)

    def set_skip_tensor(self):
        # Set, so we skip ops after SFD
        commit_priority_tensor(self.tt_priority_tensors, self.tt_skip_tensor, self.mesh_device)

    def reset_skip_tensor(self):
        # Reset, so we don't skip SFD
        commit_priority_tensor(self.tt_reset_priority_tensors, self.tt_skip_tensor, self.mesh_device)

    def get_correct_tensor(self, multi_device_tensor_output_pt):
        priority = read_multi_device_tensor(self.tt_priority_tensors)
        priority_id = torch.tensor([priority[0][0, 0, 0, 0], priority[1][0, 0, 0, 0]])
        index = torch.argmax(priority_id)
        logger.debug(f"Read tensor index: {index} for priority_id: {priority_id}")

        shape = multi_device_tensor_output_pt.shape[-1] // 2
        return multi_device_tensor_output_pt[..., index * shape : (index + 1) * shape]

    def close_ccl(self):
        ############################################################
        ### Teardown persistent fabric ###
        ############################################################
        self.mesh_device.reset_sub_device_stall_group()
        teardown_fabric_interface(self.mesh_device)
        ############################################################


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
        False,
    ),
    ids=(
        "paged_attention",
        "default_attention",
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
    (16 * 1024,),  # For decode-only unit test, there's no need to run with large sequence lengths
)
def test_llama_attention_inference(
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
    pcc = 0.99

    mesh_device.enable_async(True)

    model_args = TtModelArgs(mesh_device, max_batch_size=batch_size, max_seq_len=max_seq_len)
    model_args.n_layers = 1  # For the unit test, just run a sigle layer

    state_dict = model_args.load_state_dict()

    first_layer_prefix = model_args.get_state_dict_prefix("TtLlamaAttention", 0) + "."
    # Ref model needs partial state dict, but our models use full state dict keys as cached weight names
    partial_state_dict = {
        k[len(first_layer_prefix) :]: v for k, v in state_dict.items() if (k.startswith(first_layer_prefix))
    }

    reference_model = Attention(args=model_args)
    reference_model.load_state_dict(partial_state_dict)

    seq_len = 1

    generation_start_pos = 1000
    generation_length = 3
    all_tests_pass = True

    # Setup RoPE transformation matrices
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

    sfd_setup = TtSFDSetup(mesh_device, model_args.n_heads, model_args.head_dim, batch_size)
    tt_model = TtLlamaAttention(
        mesh_device,
        state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
        layer_num=0,
        dtype=dtype,
        transformation_mats=transformation_mats,
        configuration=model_args,
        paged_attention_config=paged_attention_config,
        sfd_setup=sfd_setup,
    )

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
        mesh_mapper=model_args.fracture_scheme(
            mesh_device,
            dims=(None, 0) if (model_args.is_galaxy and batch_size > 1) else (None, None),
            mesh_shape=model_args.cluster_shape,
        ),
    )

    for i in range(generation_length):
        # 70B attention block typically sees tensors with mean 0 and std 0.03 - 0.05 in layer 1
        pt_attention_input = torch.randn(batch_size, seq_len, model_args.dim) * 0.05

        tt_attention_input = pt_attention_input.clone()

        attention_input = model_args.prepare_residual_tensor_decode(
            tt_attention_input,
            model_args.model_config["SHARDED_ATTN_INPUT_MEMCFG"],
            force_replicated=False if model_args.is_galaxy else True,
        )

        # Get cos/sin matrices for the current position of each user
        rot_mats = rope_setup.get_rot_mats(current_pos)

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

        # In this test all users have the same position (if using batch > 1)
        freqs_cis_i = freqs_cis[current_pos[0], :].unsqueeze(0)

        reference_output = reference_model(pt_attention_input, current_pos[0], freqs_cis_i, mask=None)

        passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc)

        logger.info(comp_allclose(reference_output, tt_output_torch))
        logger.info(f"PCC: {pcc_message}")
        if passing:
            logger.info(f"[pos={current_pos[0]}] Llama_Attention Passed!")
        else:
            logger.warning(f"[pos={current_pos[0]}] Llama_Attention Failed!")
            all_tests_pass = False

        # Increment position
        current_pos = torch.tensor([generation_start_pos + i for _ in range(batch_size)])
        current_pos_tensor = ttnn.from_torch(
            current_pos,
            device=mesh_device,
            dtype=ttnn.int32,
            mesh_mapper=model_args.fracture_scheme(
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
                                dims=(1, 3) if model_args.is_galaxy else (0, 1),
                                mesh_shape=model_args.cluster_shape,
                            ),
                        )[reverse_permutation][:, : model_args.n_kv_heads, :, : model_args.head_dim]
                        .reshape(
                            model_args.max_batch_size,
                            paged_attention_config.max_num_blocks // model_args.max_batch_size,
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
                    )[:batch_size, : model_args.n_kv_heads, :, :]
                    for cache in tt_model.layer_past
                ]

            for i, (cache_pt, cache_tt) in enumerate(zip(pytorch_layer_present, tt_layer_present)):
                cache_length_to_check = min(model_args.max_seq_len, generation_start_pos + generation_length + 1)
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
    # sfd_setup.close_ccl()

    if all_tests_pass:
        logger.info("Llama Attention output Passed!")
    else:
        logger.warning("Llama Attention output Failed!")
        assert all_tests_pass, f"PCC value is lower than {pcc} for some of the outputs. Check Warnings!"
