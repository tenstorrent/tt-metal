# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.tt_transformers.tt.ccl import tt_all_reduce
from models.tt_transformers.tt.model_config import TensorGroup


class ExpertMLP(LightweightModule):
    def __init__(
        self, mesh_device, tt_ccl, state_dict, weight_cache_path, args, layer_num, dtypes, deallocate_torch=False
    ):
        super().__init__()

        self.mesh_device = mesh_device
        self.tt_ccl = tt_ccl
        self.dtypes = dtypes
        self.args = args
        self.layer_num = layer_num
        self.model_config = args.get_model_config()

        # Base name for expert weights in Grok
        base_name = lambda expert_num: f"model.layers.{layer_num}.block_sparse_moe.experts.{expert_num}"
        cache_name = (
            lambda name: args.weight_cache_path(dtypes[name])
            + f"model.layers.{layer_num}.feed_forward_multidevice_unsqueezed.experts.{name}"
        )

        # Concatenate weights from all 8 experts
        torch_weight = lambda name: torch.concat(
            [
                state_dict[f"{base_name(expert_num)}.{name}.weight"]
                .permute(1, 0)
                .unsqueeze(0)
                .unsqueeze(0)  # [1, 1, 8192, 16384]
                for expert_num in range(8)
            ],
            dim=1,  # [1, 8, 8192, 16384]
        )

        # Convert torch weights to ttnn tensors
        as_tensor = lambda name: ttnn.as_tensor(
            torch_weight(name),
            dtype=dtypes[name],
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                self.mesh_device, dims=(-1, -2) if name is not "w2" else (-2, -1), mesh_shape=(8, 4)
            ),
            layout=self.model_config["MLP_W_LAYOUT_TILE_EXPERTS"],
            memory_config=self.model_config["MLP_WEIGHTS_MEMCFG_EXPERTS"],
            cache_file_name=cache_name(name),
        )

        # Initialize weight tensors
        self.w1 = as_tensor("w1")  # gate_proj
        self.w2 = as_tensor("w2")  # down_proj
        self.w3 = as_tensor("w3")  # up_proj

        self.activation_type = ttnn.UnaryOpType.GELU

        if deallocate_torch:
            for expert_num in range(8):
                del state_dict[f"model.layers.{layer_num}.block_sparse_moe.experts.{expert_num}.w1.weight"]
                del state_dict[f"model.layers.{layer_num}.block_sparse_moe.experts.{expert_num}.w2.weight"]
                del state_dict[f"model.layers.{layer_num}.block_sparse_moe.experts.{expert_num}.w3.weight"]

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        # Simplified for mode="decode", TG=True, dim=8192 only

        x = ttnn.repeat(x, repeat_dims=(1, 8, 1, 1))

        layer_num = max(self.layer_num, 0)  # cross_block uses the configutation of the first decoder
        activation_dtype = self.model_config["DECODERS_OPTIMIZATIONS"].get_tensor_dtype(
            decoder_id=layer_num, tensor=TensorGroup.ACTIVATION
        )
        li_ff1_3_compute_kernel_cfg = self.args.compute_kernel_config_hifi2

        # TG decode mode config (dim=8192 >= 4096)
        pc_1 = self.model_config["FF1_3_TG_PROGCFG_SINGLE_EXPERT"]
        pc_2 = self.model_config["FF2_TG_PROGCFG_SINGLE_EXPERT"]
        pc_3 = self.model_config["FF1_3_TG_PROGCFG_SINGLE_EXPERT"]

        # Decode mode memory config
        memory_config = ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG
        w1_out = ttnn.matmul(x, self.w1)

        w3_out = ttnn.matmul(x, self.w3)
        # ttnn.deallocate(x)

        input_mem_cfg = w1_out.memory_config()
        w1_out = ttnn.experimental.reduce_scatter_minimal_async(
            w1_out,
            persistent_output_buffers=None,
            dim=3,
            multi_device_global_semaphore=self.tt_ccl.get_and_cycle_rs_semaphore_handles(1),
            barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(1),
            num_links=self.args.num_reduce_scatter_links,
            cluster_axis=1,
            memory_config=self.model_config["FF1_OUT_REDUCE_SCATTER_MEMCFG_SINGLE_EXPERT"],  # decode mode
            intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Linear,
            chunks_per_sync=10,
            num_workers_per_link=2,
            num_buffers_per_channel=2,
        )

        w3_out = ttnn.experimental.reduce_scatter_minimal_async(
            w3_out,
            persistent_output_buffers=None,
            dim=3,
            multi_device_global_semaphore=self.tt_ccl.get_and_cycle_rs_semaphore_handles(1),
            barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(1),
            num_links=1,
            cluster_axis=1,
            memory_config=self.model_config["FF1_OUT_REDUCE_SCATTER_MEMCFG_SINGLE_EXPERT"],  # decode mode
            intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Linear,
            chunks_per_sync=10,
            num_workers_per_link=2,
            num_buffers_per_channel=2,
        )

        w2_in = ttnn.mul(
            w1_out,
            w3_out,
            input_tensor_a_activations=[self.activation_type],
            dtype=activation_dtype or ttnn.bfloat8_b,
            memory_config=w1_out.memory_config(),
        )

        ttnn.deallocate(w3_out)
        ttnn.deallocate(w1_out)

        w2_in = ttnn.experimental.all_gather_async(
            w2_in,
            persistent_output_buffer=None,
            dim=3,
            multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(1),
            num_links=2,
            cluster_axis=1,
            topology=ttnn.Topology.Linear,
            memory_config=input_mem_cfg,
            barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(1),
            chunks_per_sync=10,
            num_workers_per_link=2,
            num_buffers_per_channel=2,
        )

        # Always decode mode
        w2_in = ttnn.to_memory_config(w2_in, ttnn.L1_MEMORY_CONFIG)

        w2_out = ttnn.matmul(w2_in, self.w2)
        ttnn.deallocate(w2_in)

        w2_out_reduced = tt_all_reduce(
            w2_out,
            self.mesh_device,
            self.tt_ccl,
            cluster_axis=0,
            dim=3,  # TG=True and dim=8192, so use dim=3
            num_reduce_scatter_links=self.args.num_reduce_scatter_links,
            num_all_gather_links=self.args.num_all_gather_links,
            sharded=True,  # decode mode
            memory_config=self.model_config["FF2_OUT_REDUCE_SCATTER_MEMCFG_SINGLE_EXPERT"],
            dtype=self.args.ccl_dtype,
            use_composite=True,  # dim=8192
            topology=self.args.ccl_topology(),
        )

        return w2_out_reduced
