import torch

import ttnn
from models.experimental.stable_diffusion_35_large.tt.substate import substate

from ..tt.sdpa import sdpa as tt_sdpa


class Attention:
    def __init__(self, mesh_device, hf_config, state_dict, layer_idx, ccl_manager):
        self.layer_idx = layer_idx
        self.use_sliding_window = self.layer_idx % 2 == 0
        self.scaling = hf_config.head_dim**-0.5
        self.cache = None
        self.head_dim = hf_config.head_dim
        self.num_heads = hf_config.num_attention_heads
        self.num_local_heads = self.num_heads // mesh_device.shape[1]
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

        col_mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=-1)
        row_mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=-2)

        self.q_proj = ttnn.from_torch(
            q_proj, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=col_mesh_mapper
        )
        self.q_proj_bias = ttnn.from_torch(
            q_proj_bias, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=col_mesh_mapper
        )

        self.k_proj = ttnn.from_torch(
            k_proj, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=col_mesh_mapper
        )
        self.k_proj_bias = ttnn.from_torch(
            k_proj_bias, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=col_mesh_mapper
        )

        self.v_proj = ttnn.from_torch(
            v_proj, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=col_mesh_mapper
        )
        self.v_proj_bias = ttnn.from_torch(
            v_proj_bias, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=col_mesh_mapper
        )

        if mesh_device.shape[1] > 1:
            o_proj_bias = torch.cat(
                [o_proj_bias] + [torch.zeros_like(o_proj_bias)] * (mesh_device.shape[1] - 1), dim=-1
            )

        self.o_proj = ttnn.from_torch(
            o_proj, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=row_mesh_mapper
        )
        self.o_proj_bias = ttnn.from_torch(
            o_proj_bias, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=col_mesh_mapper
        )

        self.sinks = ttnn.from_torch(
            sinks,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-3),
        )

    def __call__(self, x: ttnn.Tensor, mask, rope_stuff):
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

        tt_q = ttnn.reshape(tt_q, [batch_size * seq_len, -1, self.num_local_heads, self.head_dim])
        tt_k = ttnn.reshape(tt_k, [batch_size * seq_len, -1, self.head_dim])
        tt_v = ttnn.reshape(tt_v, [batch_size * seq_len, -1, self.head_dim])

        tt_sdpa_out, self.cache = tt_sdpa(
            tt_q,
            tt_k,
            tt_v,
            self.sinks,
            sm_scale=self.scaling,
            tt_mask=mask,
            tt_cache=self.cache,
        )

        tt_out = ttnn.matmul(tt_sdpa_out, self.o_proj) + self.o_proj_bias
        print(tt_out.shape)
        tt_out = ttnn.reshape(tt_out, (batch_size, seq_len, self.hidden_size))

        if self.mesh_device.shape[1] > 1:
            # AllReduce
            tt_out = ttnn.unsqueeze(tt_out, 0)
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
            tt_out = ttnn.reshape(tt_out, (batch_size, seq_len, self.hidden_size))

        return tt_out
