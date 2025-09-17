import torch

import ttnn
from models.demos.gpt_oss.utils.general_utils import get_cache_file_name
from models.experimental.stable_diffusion_35_large.tt.substate import substate
from models.utility_functions import nearest_y

from ..tt.sdpa import sdpa as tt_sdpa
from ..utils.general_utils import MAX_SEQ_LEN


class Attention:
    def __init__(self, mesh_device, hf_config, state_dict, layer_idx, ccl_manager, tensor_cache_path=None):
        self.layer_idx = layer_idx
        self.use_sliding_window = self.layer_idx % 2 == 0
        self.scaling = hf_config.head_dim**-0.5
        self.cache = None
        self.head_dim = hf_config.head_dim
        self.num_heads = hf_config.num_attention_heads
        self.num_kv_heads = hf_config.num_key_value_heads
        self.num_local_heads = self.num_heads // mesh_device.shape[1]
        self.num_local_kv_heads = self.num_kv_heads // mesh_device.shape[1]
        self.hidden_size = hf_config.hidden_size
        self.ccl_manager = ccl_manager
        self.mesh_device = mesh_device

        # (['sinks', 'q_proj.weight', 'q_proj.bias', 'k_proj.weight', 'k_proj.bias', 'v_proj.weight', 'v_proj.bias', 'o_proj.weight', 'o_proj.bias'])

        # TODO: Add mesh mapper
        q_proj = substate(state_dict, "q_proj")["weight"].transpose(-1, -2)
        q_proj_bias = substate(state_dict, "q_proj")["bias"]  # TODO: unsqueeze?

        k_proj = substate(state_dict, "k_proj")["weight"].transpose(-1, -2)
        k_proj_bias = substate(state_dict, "k_proj")["bias"]  # TODO: unsqueeze?

        v_proj = substate(state_dict, "v_proj")["weight"].transpose(-1, -2)
        v_proj_bias = substate(state_dict, "v_proj")["bias"]  # TODO: unsqueeze?

        o_proj = substate(state_dict, "o_proj")["weight"].transpose(-1, -2)
        o_proj_bias = substate(state_dict, "o_proj")["bias"]  # TODO: unsqueeze?

        sinks = state_dict["sinks"].reshape(1, hf_config.num_attention_heads, 1, 1)
        decode_sinks = torch.nn.functional.pad(
            sinks.view(-1, 1), (0, ttnn.TILE_SIZE - sinks.shape[-1]), "constant", value=0.0
        )
        decode_sinks /= self.scaling

        col_mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=-1)
        row_mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=-2)

        self.q_proj = ttnn.as_tensor(
            q_proj,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat8_b,
            mesh_mapper=col_mesh_mapper,
            cache_file_name=get_cache_file_name(tensor_cache_path, "q_proj"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.q_proj_bias = ttnn.as_tensor(
            q_proj_bias,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat8_b,
            mesh_mapper=col_mesh_mapper,
            cache_file_name=get_cache_file_name(tensor_cache_path, "q_proj_bias"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        self.k_proj = ttnn.as_tensor(
            k_proj,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat8_b,
            mesh_mapper=col_mesh_mapper,
            cache_file_name=get_cache_file_name(tensor_cache_path, "k_proj"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.k_proj_bias = ttnn.as_tensor(
            k_proj_bias,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat8_b,
            mesh_mapper=col_mesh_mapper,
            cache_file_name=get_cache_file_name(tensor_cache_path, "k_proj_bias"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        self.v_proj = ttnn.as_tensor(
            v_proj,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat8_b,
            mesh_mapper=col_mesh_mapper,
            cache_file_name=get_cache_file_name(tensor_cache_path, "v_proj"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.v_proj_bias = ttnn.as_tensor(
            v_proj_bias,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat8_b,
            mesh_mapper=col_mesh_mapper,
            cache_file_name=get_cache_file_name(tensor_cache_path, "v_proj_bias"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        if mesh_device.shape[1] > 1:
            o_proj_bias = torch.cat(
                [o_proj_bias] + [torch.zeros_like(o_proj_bias)] * (mesh_device.shape[1] - 1), dim=-1
            )

        self.o_proj = ttnn.as_tensor(
            o_proj,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat8_b,
            mesh_mapper=row_mesh_mapper,
            cache_file_name=get_cache_file_name(tensor_cache_path, "o_proj"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.o_proj_bias = ttnn.as_tensor(
            o_proj_bias,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat8_b,
            mesh_mapper=col_mesh_mapper,
            cache_file_name=get_cache_file_name(tensor_cache_path, "o_proj_bias"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        self.decode_sinks = ttnn.as_tensor(
            decode_sinks,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-2),
            cache_file_name=get_cache_file_name(tensor_cache_path, "decode_sinks"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.sinks = ttnn.as_tensor(
            sinks,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-3),
            cache_file_name=get_cache_file_name(tensor_cache_path, "sinks"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Set up kv cache
        cache_shape = [1, self.num_kv_heads, MAX_SEQ_LEN, self.head_dim]
        k_cache = ttnn.as_tensor(
            torch.zeros(cache_shape),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat8_b,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-3),
            cache_file_name=get_cache_file_name(tensor_cache_path, f"k_cache_{cache_shape}"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        v_cache = ttnn.as_tensor(
            torch.zeros(cache_shape),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat8_b,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-3),
            cache_file_name=get_cache_file_name(tensor_cache_path, f"v_cache_{cache_shape}"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.kv_cache = [k_cache, v_cache]

        grid_size = mesh_device.compute_with_storage_grid_size()
        kv_shape = (1, 1, self.num_local_kv_heads, self.head_dim)
        kv_shard_height = nearest_y(kv_shape[2], ttnn.TILE_SIZE)
        kv_shard_width = kv_shape[3]
        kv_num_cores = kv_shape[1]
        kv_core_grid = ttnn.num_cores_to_corerangeset(kv_num_cores, grid_size, row_wise=True)
        self.kv_mem_cfg = ttnn.create_sharded_memory_config(
            shape=(kv_shard_height, kv_shard_width),
            core_grid=kv_core_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            use_height_and_width_as_shard_shape=True,
        )

        # SDPA setup
        self.sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=mesh_device.compute_with_storage_grid_size(),
            q_chunk_size=0,
            k_chunk_size=128,
            exp_approx_mode=False,
        )

        self.sdpa_compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

    def __call__(self, x: ttnn.Tensor, mask, rope_stuff, position_idx=None):
        batch_size, seq_len, hidden_size = x.shape

        tt_q = ttnn.matmul(x, self.q_proj) + self.q_proj_bias
        tt_q = ttnn.reshape(tt_q, [1, seq_len * batch_size, -1, self.head_dim])

        tt_k = ttnn.matmul(x, self.k_proj) + self.k_proj_bias
        tt_k = ttnn.reshape(tt_k, [1, seq_len * batch_size, -1, self.head_dim])

        tt_v = ttnn.matmul(x, self.v_proj) + self.v_proj_bias
        tt_v = ttnn.reshape(tt_v, [1, seq_len * batch_size, -1, self.head_dim])

        apply_rope, tt_cos, tt_sin = rope_stuff
        tt_q = apply_rope(tt_q, tt_cos, tt_sin)
        tt_k = apply_rope(tt_k, tt_cos, tt_sin)
        # print("batch_size", batch_size)
        # print("seq_len", seq_len)
        if batch_size * seq_len == 1:
            # Update cache
            k_cache, v_cache = self.kv_cache

            tt_k = ttnn.to_memory_config(tt_k, self.kv_mem_cfg)
            tt_v = ttnn.to_memory_config(tt_v, self.kv_mem_cfg)

            ttnn.experimental.paged_update_cache(
                k_cache,
                tt_k,
                update_idxs_tensor=position_idx,
            )
            ttnn.experimental.paged_update_cache(
                v_cache,
                tt_v,
                update_idxs_tensor=position_idx,
            )

            tt_sdpa_out = ttnn.transformer.scaled_dot_product_attention_decode(
                tt_q,
                k_cache,
                v_cache,
                cur_pos_tensor=position_idx,
                is_causal=self.layer_idx % 2 != 0,
                attn_mask=mask if self.layer_idx % 2 == 0 else None,
                attention_sink=self.decode_sinks,
                scale=self.scaling,
                program_config=self.sdpa_program_config,
                compute_kernel_config=self.sdpa_compute_kernel_config,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

            tt_sdpa_out = ttnn.transpose(tt_sdpa_out, 1, 2)
            tt_sdpa_out = ttnn.experimental.nlp_concat_heads(
                tt_sdpa_out, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )  # [1, 1, num_tokens, dim * nh]
        else:
            # Fill cache
            assert batch_size == 1, f"Only batch 1 supported, but got {batch_size=}"

            k_cache, v_cache = self.kv_cache

            # FIXME: Should eventually remove
            tt_k = ttnn.transpose(tt_k, 1, 2)
            tt_k = ttnn.typecast(tt_k, k_cache.dtype)

            tt_v = ttnn.transpose(tt_v, 1, 2)
            tt_v = ttnn.typecast(tt_v, v_cache.dtype)

            ttnn.fill_cache(
                k_cache,
                tt_k,
                batch_idx=0,
            )
            ttnn.fill_cache(
                v_cache,
                tt_v,
                batch_idx=0,
            )

            # FIXME: Should eventually remove
            tt_k = ttnn.transpose(tt_k, 1, 2)
            tt_v = ttnn.transpose(tt_v, 1, 2)

            tt_q = ttnn.reshape(tt_q, [batch_size * seq_len, -1, self.num_local_heads, self.head_dim])
            tt_k = ttnn.reshape(tt_k, [batch_size * seq_len, -1, self.head_dim])
            tt_v = ttnn.reshape(tt_v, [batch_size * seq_len, -1, self.head_dim])

            tt_sdpa_out, _ = tt_sdpa(
                tt_q,
                tt_k,
                tt_v,
                self.sinks,
                sm_scale=self.scaling,
                tt_mask=mask,
                tt_cache=None,
                position_idx=None,
            )

        tt_out = ttnn.matmul(tt_sdpa_out, self.o_proj) + self.o_proj_bias
        tt_out = ttnn.reshape(tt_out, (batch_size, seq_len, self.hidden_size))

        if self.mesh_device.shape[1] > 1:
            # AllReduce
            tt_out = ttnn.unsqueeze(tt_out, 0)
            if tt_out.shape[-2] >= 32 and self.mesh_device.shape[1] == 8:
                tt_out = ttnn.pad(tt_out, [(0, 0), (0, 0), (0, 0), (0, 192)], 0)
            tt_out_scattered = ttnn.experimental.reduce_scatter_minimal_async(
                tt_out,
                dim=3,
                multi_device_global_semaphore=self.ccl_manager.get_rs_ping_pong_semaphore(),
                num_links=1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=self.ccl_manager.topology,
                cluster_axis=1,
                barrier_semaphore=self.ccl_manager.get_barrier_semaphore(),
            )
            tt_out = ttnn.experimental.all_gather_async(
                tt_out_scattered,
                dim=3,
                cluster_axis=1,
                mesh_device=self.ccl_manager.mesh_device,
                topology=self.ccl_manager.topology,
                multi_device_global_semaphore=self.ccl_manager.get_ag_ping_pong_semaphore(),
                num_links=1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                barrier_semaphore=self.ccl_manager.get_barrier_semaphore(),
            )
            if tt_out.shape[-2] >= 32 and self.mesh_device.shape[1] == 8:
                tt_out = tt_out[:, :, :, : self.hidden_size]
            tt_out = ttnn.reshape(tt_out, (batch_size, seq_len, self.hidden_size))

        return tt_out
