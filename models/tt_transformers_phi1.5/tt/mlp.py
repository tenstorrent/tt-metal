# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.tt_transformers.tt.ccl import tt_all_reduce
from models.tt_transformers.tt.common import pad_to_size
from models.tt_transformers.tt.model_config import OpGroup, TensorGroup


class MLP(LightweightModule):
    def __init__(
        self, mesh_device, args, state_dict, weight_cache_path, layer_num, dtype, model_config, state_dict_prefix=None
    ):
        super().__init__()

        self.state_dict = state_dict
        self.mesh_device = mesh_device
        self.args = args
        self.dim = args.dim
        self.model_config = model_config
        self.layer_num = layer_num
        state_dict_prefix = state_dict_prefix or args.get_state_dict_prefix(self.__class__.__name__, layer_num)
        torch_weight = lambda name: torch.transpose(self.state_dict[f"{state_dict_prefix}.{name}.weight"], -2, -1)
        pad_hidden_dim = lambda tensor, dim: pad_to_size(tensor, dim=dim, size=args.hidden_dim)
        # If pading was applied (e.g. via env var), add the unpadded hidden dim to the cache name to avoid loading incorrect weights
        hidden_dim_string = f".hidden_dim_{args.hidden_dim}" if args.hidden_dim != args.unpadded_hidden_dim else ""

        if args.dummy_weights:
            cache_name = lambda _: None
        else:
            cache_name = lambda name: weight_cache_path / f"{state_dict_prefix}.{name}{hidden_dim_string}"

        w1_w3_mem_config = args.create_dram_sharded_mem_config(args.dim, args.hidden_dim // args.num_devices)
        w2_mem_config = args.create_dram_sharded_mem_config(args.hidden_dim // args.num_devices, args.dim)

        # TODO Clean up this code. With sharding, we load the normal weights and then shard them
        as_sharded_tensor = lambda name, type, dims: ttnn.as_tensor(
            pad_hidden_dim(
                torch_weight(name[:2]), dims[0] if args.is_galaxy else dims[-1]
            ),  # Grab only the wX part of the name
            dtype=type,
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=dims, mesh_shape=args.cluster_shape),
            layout=ttnn.TILE_LAYOUT,
            memory_config=(
                ttnn.DRAM_MEMORY_CONFIG if args.is_galaxy else w2_mem_config if "w2" in name else w1_w3_mem_config
            ),
            cache_file_name=cache_name(name),
        )

        # Sharded weights
        w1_dims = (-1, -2) if args.is_galaxy else (-2, -1)
        w2_dims = (-2, -1) if args.is_galaxy else (-1, -2)

        layer_num = max(layer_num, 0)  # cross_block uses the configutation of the first decoder

        ff1_3_dtype = self.model_config["DECODERS_OPTIMIZATIONS"].get_tensor_dtype(
            decoder_id=layer_num, tensor=TensorGroup.FF1_FF3
        )
        ff2_dtype = self.model_config["DECODERS_OPTIMIZATIONS"].get_tensor_dtype(
            decoder_id=layer_num, tensor=TensorGroup.FF2
        )

        self.w1 = as_sharded_tensor(
            "w1_sharded", ff1_3_dtype, dims=w1_dims
        )  # bfp4 normally ok here but sub .99 pcc for llama 3.1 weights
        self.w2 = as_sharded_tensor("w2_sharded", ff2_dtype, dims=w2_dims)
        self.w3 = as_sharded_tensor("w3_sharded", ff1_3_dtype, dims=w1_dims)

    def forward(self, x: ttnn.Tensor, mode) -> ttnn.Tensor:
        """
        w1 -> gate_proj
        w2 -> down_proj
        w3 -> up_proj
        HF reference: self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        """
        seq_len = x.shape[-2]
        TG = self.args.is_galaxy
        layer_num = max(self.layer_num, 0)  # cross_block uses the configutation of the first decoder
        activation_dtype = self.model_config["DECODERS_OPTIMIZATIONS"].get_tensor_dtype(
            decoder_id=layer_num, tensor=TensorGroup.ACTIVATION
        )
        li_ff1_3_compute_kernel_cfg = self.model_config["DECODERS_OPTIMIZATIONS"].get_math_fidelity(
            decoder_id=layer_num, op=OpGroup.LI_FF1_FF3, configuration=self.args
        )

        if mode == "decode":  # Sharded config
            if TG:  # TODO: Fix this when TG supports DRAM sharded matmuls
                pc_1 = self.model_config["FF1_3_TG_PROGCFG"] if self.dim >= 4096 else None
                pc_2 = self.model_config["FF2_TG_PROGCFG"] if self.dim >= 4096 else None
                pc_3 = self.model_config["FF1_3_TG_PROGCFG"] if self.dim >= 4096 else None
            else:
                pc_1 = self.model_config["DECODE_MLP_W1_W3_PRG_CONFIG"]
                pc_2 = self.model_config["DECODE_MLP_W2_PRG_CONFIG"]
                pc_3 = self.model_config["DECODE_MLP_W1_W3_PRG_CONFIG"]
        else:  # Update the program configs based for prefill
            if seq_len >= self.args.prefill_len_cutoff:  # 512 if Blackhole, 1024 if Wormhole
                # Reshape input to to fit on device and parallelize computation
                x = ttnn.reshape(x, [1, seq_len // self.args.prefill_len_cutoff, self.args.prefill_len_cutoff, -1])
            pc_1 = self.model_config["PREFILL_MLP_W1_W3_PRG_CONFIG"](seq_len)
            pc_2 = self.model_config["PREFILL_MLP_W2_PRG_CONFIG"](seq_len)
            pc_3 = self.model_config["PREFILL_MLP_W1_W3_PRG_CONFIG"](seq_len)

        # In decode mode (seqlen <= 32) do DRAM sharded matmuls
        # These use HiFi2; this drops 1 bit of the activations but would be FLOP-bound on 12 cores with HiFi4
        memory_config = ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG if mode == "decode" else ttnn.DRAM_MEMORY_CONFIG
        w1_out = ttnn.linear(
            x,
            self.w1,
            dtype=ttnn.bfloat8_b if TG else activation_dtype or ttnn.bfloat16,
            core_grid=None,  # FIXME: validate on TG ttnn.CoreGrid(y=8, x=8) if not pc_1 else None,
            compute_kernel_config=li_ff1_3_compute_kernel_cfg,
            program_config=pc_1,
            memory_config=memory_config,
        )

        w3_out = ttnn.linear(
            x,
            self.w3,
            dtype=ttnn.bfloat8_b if TG else activation_dtype or ttnn.bfloat16,
            core_grid=None,  # FIXME: validate on TG ttnn.CoreGrid(y=8, x=8) if not pc_3 else None,
            compute_kernel_config=li_ff1_3_compute_kernel_cfg,
            program_config=pc_3,
            memory_config=memory_config,
        )
        ttnn.deallocate(x)

        if TG:
            # if mode == "decode" and self.dim!=8192:
            #     w1_out = ttnn.to_memory_config(w1_out, ttnn.DRAM_MEMORY_CONFIG)
            #     w3_out = ttnn.to_memory_config(w3_out, ttnn.DRAM_MEMORY_CONFIG)
            if self.dim == 8192 or mode == "prefill":
                input_mem_cfg = w1_out.memory_config()
                w1_out = ttnn.reduce_scatter(
                    w1_out,
                    dim=3,
                    math_op=ttnn.ReduceType.Sum,
                    num_links=self.args.num_reduce_scatter_links,
                    cluster_axis=1,
                    mesh_device=self.mesh_device,
                    topology=ttnn.Topology.Linear,
                    memory_config=self.model_config["FF1_OUT_REDUCE_SCATTER_MEMCFG"] if mode == "decode" else None,
                )
                w3_out = ttnn.reduce_scatter(
                    w3_out,
                    dim=3,
                    math_op=ttnn.ReduceType.Sum,
                    num_links=1,
                    cluster_axis=1,
                    mesh_device=self.mesh_device,
                    topology=ttnn.Topology.Linear,
                    memory_config=self.model_config["FF1_OUT_REDUCE_SCATTER_MEMCFG"] if mode == "decode" else None,
                )
            else:
                w1_out = tt_all_reduce(
                    w1_out,
                    self.mesh_device,
                    cluster_axis=1,
                    num_all_gather_links=2,
                    sharded=True if mode == "decode" else False,
                    topology=self.args.ccl_topology(),
                    memory_config=self.model_config["FF1_OUT_GATHERED_MEMCFG"] if mode == "decode" else None,
                )
                w3_out = tt_all_reduce(
                    w3_out,
                    self.mesh_device,
                    cluster_axis=1,
                    num_all_gather_links=2,
                    sharded=True if mode == "decode" else False,
                    topology=self.args.ccl_topology(),
                    memory_config=self.model_config["FF1_OUT_GATHERED_MEMCFG"] if mode == "decode" else None,
                )

        w2_in = ttnn.mul(
            w1_out,
            w3_out,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            dtype=activation_dtype or ttnn.bfloat8_b,
            memory_config=w1_out.memory_config(),
        )

        if mode == "decode" and not TG:
            # w2 may use a different core grid, this is a no-op if they already match
            w2_in = ttnn.to_memory_config(w2_in, self.model_config["SHARDED_MLP2_INPUT_MEMCFG"])

        ttnn.deallocate(w3_out)
        ttnn.deallocate(w1_out)

        if TG and (self.dim == 8192 or mode == "prefill"):
            w2_in = ttnn.all_gather(
                w2_in,
                3,
                num_links=2,
                cluster_axis=1,
                mesh_device=self.mesh_device,
                topology=ttnn.Topology.Linear,
                memory_config=input_mem_cfg,
            )
            if mode == "decode":
                w2_in = ttnn.to_memory_config(w2_in, ttnn.L1_MEMORY_CONFIG)

        li_ff2_compute_kernel_cfg = self.model_config["DECODERS_OPTIMIZATIONS"].get_math_fidelity(
            decoder_id=layer_num, op=OpGroup.LI_FF2, configuration=self.args
        )
        w2_out = ttnn.linear(
            w2_in,
            self.w2,
            compute_kernel_config=li_ff2_compute_kernel_cfg,
            dtype=self.args.ccl_dtype if TG else activation_dtype or ttnn.bfloat16,
            program_config=pc_2,
            memory_config=memory_config,
            core_grid=None,  # FIXME: validate on TG ttnn.CoreGrid(y=8, x=8) if not pc_2 else None,
        )
        ttnn.deallocate(w2_in)
        # if mode == "decode" and not TG:
        #     w2_out = ttnn.sharded_to_interleaved(w2_out, ttnn.DRAM_MEMORY_CONFIG)
        w2_out_reduced = tt_all_reduce(
            w2_out,
            self.mesh_device,
            cluster_axis=0,
            dim=0 if (TG and self.dim < 8192) else 3,
            num_reduce_scatter_links=self.args.num_reduce_scatter_links,
            num_all_gather_links=self.args.num_all_gather_links,
            sharded=(mode == "decode"),
            memory_config=(
                (self.model_config["FF2_OUT_REDUCE_SCATTER_MEMCFG"] if TG else w2_out.memory_config())
                if mode == "decode"
                else ttnn.DRAM_MEMORY_CONFIG
            ),
            dtype=self.args.ccl_dtype,
            use_composite=True if self.dim == 8192 else False,
            topology=self.args.ccl_topology(),
        )

        # Ensure dim 0 and 1 are 1
        original_shape = w2_out_reduced.shape
        w2_out_reduced = ttnn.reshape(
            w2_out_reduced, (1, 1, original_shape[-4] * original_shape[-3] * original_shape[-2], original_shape[-1])
        )
        if mode == "decode":
            w2_out_reduced = ttnn.to_memory_config(
                w2_out_reduced,
                self.model_config["SHARDED_ATTN_INPUT_MEMCFG"] if TG else self.model_config["DECODE_RESIDUAL_MEMCFG"],
            )

        # ttnn.deallocate(w2_out)
        return w2_out_reduced
