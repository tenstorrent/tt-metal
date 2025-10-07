# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

# Simplified MLP for dim=8192, TG=True, mode="decode" only

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.tt_transformers.tt.ccl import tt_all_reduce
from models.tt_transformers.tt.model_config import TensorGroup


class MLP(LightweightModule):
    def __init__(
        self,
        mesh_device,
        tt_ccl,
        args,
        state_dict,
        weight_cache_path,
        layer_num,
        dtype,
        model_config,
        state_dict_prefix=None,
        deallocate_torch=False,
    ):
        super().__init__()

        self.mesh_device = mesh_device
        self.tt_ccl = tt_ccl
        self.args = args
        self.dim = args.dim
        self.model_config = model_config
        self.layer_num = layer_num
        state_dict_prefix = state_dict_prefix or args.get_state_dict_prefix(self.__class__.__name__, layer_num)
        torch_weight = lambda name: torch.transpose(state_dict[f"{state_dict_prefix}.{name}.weight"], -2, -1)
        # If pading was applied (e.g. via env var), add the unpadded hidden dim to the cache name to avoid loading incorrect weights
        hidden_dim_string = f".hidden_dim_{args.hidden_dim}" if args.hidden_dim != args.unpadded_hidden_dim else ""

        if args.dummy_weights:
            cache_name = lambda _: None
        else:
            cache_name = lambda name: weight_cache_path / f"{state_dict_prefix}.{name}{hidden_dim_string}"

        layer_num = max(layer_num, 0)  # cross_block uses the configutation of the first decoder

        ff1_3_dtype = self.model_config["DECODERS_OPTIMIZATIONS"].get_tensor_dtype(
            decoder_id=layer_num, tensor=TensorGroup.FF1_FF3
        )
        ff2_dtype = self.model_config["DECODERS_OPTIMIZATIONS"].get_tensor_dtype(
            decoder_id=layer_num, tensor=TensorGroup.FF2
        )

        self.w1 = ttnn.as_tensor(
            state_dict[f"{state_dict_prefix}.w1.weight"].transpose(-1, -2),
            dtype=ff1_3_dtype,
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=(-1, -2), mesh_shape=(8, 4)),
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            # cache_file_name=cache_name("w1_interleaved"),
        )
        self.w2 = ttnn.as_tensor(
            state_dict[f"{state_dict_prefix}.w2.weight"].transpose(-1, -2),
            dtype=ff2_dtype,
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=(-2, -1), mesh_shape=(8, 4)),
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            # cache_file_name=cache_name("w2_interleaved"),
        )
        self.w3 = ttnn.as_tensor(
            state_dict[f"{state_dict_prefix}.w3.weight"].transpose(-1, -2),
            dtype=ff1_3_dtype,
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=(-1, -2), mesh_shape=(8, 4)),
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            # cache_file_name=cache_name("w3_interleaved"),
        )

        if deallocate_torch:
            del state_dict[f"{state_dict_prefix}.w1.weight"]
            del state_dict[f"{state_dict_prefix}.w2.weight"]
            del state_dict[f"{state_dict_prefix}.w3.weight"]

        self.activation_type = ttnn.UnaryOpType.GELU

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        # Simplified for mode="decode", TG=True, dim=8192 only

        layer_num = max(self.layer_num, 0)  # cross_block uses the configutation of the first decoder
        activation_dtype = self.model_config["DECODERS_OPTIMIZATIONS"].get_tensor_dtype(
            decoder_id=layer_num, tensor=TensorGroup.ACTIVATION
        )
        li_ff1_3_compute_kernel_cfg = self.args.compute_kernel_config_hifi2

        # TG decode mode config (dim=8192 >= 4096)
        pc_1 = self.model_config["FF1_3_TG_PROGCFG"]
        pc_2 = self.model_config["FF2_TG_PROGCFG"]
        pc_3 = self.model_config["FF1_3_TG_PROGCFG"]

        # Decode mode memory config
        memory_config = ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG
        w1_out = ttnn.linear(
            x,
            self.w1,
            dtype=ttnn.bfloat8_b,  # TG=True
            core_grid=None,
            compute_kernel_config=li_ff1_3_compute_kernel_cfg,
            program_config=pc_1,
            memory_config=memory_config,
        )

        w3_out = ttnn.linear(
            x,
            self.w3,
            dtype=ttnn.bfloat8_b,
            core_grid=None,
            compute_kernel_config=li_ff1_3_compute_kernel_cfg,
            program_config=pc_3,
            memory_config=memory_config,
        )
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
            memory_config=self.model_config["FF1_OUT_REDUCE_SCATTER_MEMCFG"],  # decode mode
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
            memory_config=self.model_config["FF1_OUT_REDUCE_SCATTER_MEMCFG"],  # decode mode
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

        li_ff2_compute_kernel_cfg = self.args.compute_kernel_config_hifi2
        w2_out = ttnn.linear(
            w2_in,
            self.w2,
            compute_kernel_config=li_ff2_compute_kernel_cfg,
            dtype=self.args.ccl_dtype,  # TG=True
            program_config=pc_2,
            memory_config=memory_config,
            core_grid=None,
        )
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
            memory_config=self.model_config["FF2_OUT_REDUCE_SCATTER_MEMCFG"],
            dtype=self.args.ccl_dtype,
            use_composite=True,  # dim=8192
            topology=self.args.ccl_topology(),
        )

        # Ensure dim 0 and 1 are 1
        original_shape = w2_out_reduced.shape
        w2_out_reduced = ttnn.reshape(
            w2_out_reduced, (1, 1, original_shape[-4] * original_shape[-3] * original_shape[-2], original_shape[-1])
        )
        # Always decode mode
        w2_out_reduced = ttnn.to_memory_config(
            w2_out_reduced,
            self.model_config["SHARDED_ATTN_INPUT_MEMCFG"],
        )

        return w2_out_reduced
