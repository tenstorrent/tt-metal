# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.tt_transformers.tt.ccl import tt_all_reduce
from models.tt_transformers.tt.common import Mode, pad_to_size
from models.tt_transformers.tt.mlp import MLP
from models.tt_transformers.tt.model_config import OpGroup, TensorGroup


class Phi1MLP(MLP):
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
        prefetcher=None,
    ):
        LightweightModule.__init__(self)

        self.mesh_device = mesh_device
        self.tt_ccl = tt_ccl
        self.args = args
        self.dim = args.dim
        self.model_config = model_config
        self.layer_num = layer_num
        self.mlp_activation_name = getattr(args, "mlp_activation_name", "silu")
        self.prefetcher = prefetcher

        state_dict_prefix = state_dict_prefix or args.get_state_dict_prefix("MLP", layer_num)
        torch_weight = lambda name: torch.transpose(state_dict[f"{state_dict_prefix}.{name}.weight"], -2, -1)
        pad_hidden_dim = lambda tensor, dim: pad_to_size(tensor, dim=dim, size=args.hidden_dim)
        hidden_dim_string = f".hidden_dim_{args.hidden_dim}" if args.hidden_dim != args.unpadded_hidden_dim else ""

        if args.dummy_weights:
            cache_name = lambda _: None
        else:
            cache_name = lambda name: weight_cache_path / f"{state_dict_prefix}.{name}{hidden_dim_string}"

        w1_mem_config = args.create_dram_sharded_mem_config(args.dim, args.hidden_dim // args.num_devices)
        w2_mem_config = args.create_dram_sharded_mem_config(args.hidden_dim // args.num_devices, args.dim)

        def as_sharded_tensor(name, tensor_name, dims):
            raw_weight = torch_weight(tensor_name)
            padded_weight = pad_hidden_dim(raw_weight, dims[0] if args.is_galaxy else dims[-1])
            torch_tensor = padded_weight.unsqueeze(0).unsqueeze(0)

            return ttnn.as_tensor(
                torch_tensor,
                dtype=ff1_dtype if tensor_name == "w1" else ff2_dtype,
                device=self.mesh_device,
                mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=dims, mesh_shape=args.cluster_shape),
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG
                if args.is_galaxy
                else w2_mem_config
                if tensor_name == "w2"
                else w1_mem_config,
                cache_file_name=cache_name(name),
            )

        def as_sharded_bias(name, size):
            bias_key = f"{state_dict_prefix}.{name}.bias"
            if bias_key not in state_dict:
                return None

            bias = state_dict[bias_key]
            if size == args.hidden_dim:
                bias = pad_to_size(bias, dim=0, size=args.hidden_dim)
            bias = bias.view(1, 1, 1, -1)
            return ttnn.as_tensor(
                bias,
                dtype=ttnn.bfloat16,
                device=self.mesh_device,
                mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=-1),
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cache_file_name=cache_name(f"{name}_bias_sharded"),
            )

        w1_dims = (-1, -2) if args.is_galaxy else (-2, -1)
        w2_dims = (-2, -1) if args.is_galaxy else (-1, -2)

        layer_num = max(layer_num, 0)
        use_prefetcher = prefetcher is not None
        self.decoders_optimizations = self.args.decoders_optimizations

        ff1_dtype = self.decoders_optimizations.get_tensor_dtype(
            decoder_id=layer_num, tensor=TensorGroup.FF1_FF3, prefetcher=use_prefetcher
        )
        ff2_dtype = self.decoders_optimizations.get_tensor_dtype(
            decoder_id=layer_num, tensor=TensorGroup.FF2, prefetcher=use_prefetcher
        )

        self.w1 = as_sharded_tensor("w1_sharded", "w1", dims=w1_dims)
        self.w2 = as_sharded_tensor("w2_sharded", "w2", dims=w2_dims)
        self.w1_bias = as_sharded_bias("w1", args.hidden_dim)
        self.w2_bias = as_sharded_bias("w2", args.dim)

        if self.prefetcher is not None:

            def register_weights():
                self.prefetcher.insert_tensor(self.w1)
                self.prefetcher.insert_tensor(self.w2)
                if self.w1_bias is not None:
                    self.prefetcher.insert_tensor(self.w1_bias)
                if self.w2_bias is not None:
                    self.prefetcher.insert_tensor(self.w2_bias)

            self.prefetcher.register_callback(register_weights)

    def _apply_phi_activation(self, x: ttnn.Tensor) -> ttnn.Tensor:
        activation = self.mlp_activation_name
        if activation in ("gelu_new", "gelu_pytorch_tanh"):
            return ttnn.gelu(x, fast_and_approximate_mode=True)
        if activation == "gelu":
            return ttnn.gelu(x, fast_and_approximate_mode=False)
        if activation == "relu":
            return ttnn.relu(x)
        if activation in ("silu", "swish"):
            return ttnn.silu(x)
        raise NotImplementedError(f"Unsupported phi MLP activation '{activation}'")

    def forward(self, x: ttnn.Tensor, mode: Mode) -> ttnn.Tensor:
        seq_len = x.shape[-2]
        TG = self.args.is_galaxy
        layer_num = max(self.layer_num, 0)
        activation_dtype = self.decoders_optimizations.get_tensor_dtype(
            decoder_id=layer_num, tensor=TensorGroup.ACTIVATION
        )

        if mode == Mode.PREFILL and seq_len >= self.args.prefill_len_cutoff:
            x = ttnn.reshape(x, [1, seq_len // self.args.prefill_len_cutoff, self.args.prefill_len_cutoff, -1])

        pc_1 = self.args.get_mlp_ff1_3_prg_config(mode, seq_len, self.prefetcher)
        pc_2 = self.args.get_mlp_ff2_prg_config(mode, seq_len, self.prefetcher)

        li_ff1_compute_kernel_cfg = self.decoders_optimizations.get_math_fidelity(
            decoder_id=layer_num, op=OpGroup.LI_FF1_FF3, configuration=self.args
        )
        li_ff2_compute_kernel_cfg = self.decoders_optimizations.get_math_fidelity(
            decoder_id=layer_num, op=OpGroup.LI_FF2, configuration=self.args
        )

        w1_out = ttnn.linear(
            x,
            self.w1,
            dtype=ttnn.bfloat8_b if TG else activation_dtype or ttnn.bfloat16,
            compute_kernel_config=li_ff1_compute_kernel_cfg,
            program_config=pc_1,
            memory_config=self.args.get_mlp_ff1_3_mem_config(mode, self.prefetcher),
            global_cb=self.prefetcher.global_cb if self.prefetcher is not None and mode == Mode.DECODE else None,
            sub_device_id=self.prefetcher.worker_sub_device_id
            if self.prefetcher is not None and mode == Mode.DECODE
            else None,
        )
        ttnn.deallocate(x)

        if self.w1_bias is not None:
            w1_out = ttnn.add(
                w1_out,
                self.w1_bias,
                memory_config=w1_out.memory_config(),
                dtype=activation_dtype or ttnn.bfloat16,
            )

        w1_out = self._apply_phi_activation(w1_out)

        w2_out = ttnn.linear(
            w1_out,
            self.w2,
            compute_kernel_config=li_ff2_compute_kernel_cfg,
            dtype=self.args.ccl_dtype if TG else activation_dtype or ttnn.bfloat16,
            program_config=pc_2,
            memory_config=self.args.get_mlp_ff2_mem_config(mode, self.prefetcher),
            global_cb=self.prefetcher.global_cb if self.prefetcher is not None and mode == Mode.DECODE else None,
            sub_device_id=self.prefetcher.worker_sub_device_id
            if self.prefetcher is not None and mode == Mode.DECODE
            else None,
        )
        ttnn.deallocate(w1_out)

        w2_out_reduced = tt_all_reduce(
            w2_out,
            self.mesh_device,
            self.tt_ccl,
            cluster_axis=0,
            dim=0 if (TG and self.dim < 8192) else 3,
            sharded=(mode == Mode.DECODE),
            memory_config=self.args.get_mlp_ff2_all_reduce_mem_config(mode, w2_out),
            rs_memory_config=self.model_config["MLP_RS_CONFIG"]["rs_memory_config"]
            if mode == Mode.DECODE
            else ttnn.DRAM_MEMORY_CONFIG,
            dtype=self.args.ccl_dtype,
            use_composite=True if self.dim == 8192 else False,
            topology=self.args.ccl_topology(),
            chunks_per_sync=self.model_config["MLP_RS_CONFIG"]["chunks_per_sync"] if mode == Mode.DECODE else 10,
            num_workers_per_link=self.model_config["MLP_RS_CONFIG"]["num_workers_per_link"]
            if mode == Mode.DECODE
            else 2,
            subdevice_id=self.prefetcher.worker_sub_device_id
            if mode == Mode.DECODE and self.prefetcher is not None
            else None,
        )

        if self.w2_bias is not None:
            w2_out_reduced = ttnn.add(
                w2_out_reduced,
                self.w2_bias,
                memory_config=w2_out_reduced.memory_config(),
                dtype=activation_dtype or ttnn.bfloat16,
            )

        original_shape = w2_out_reduced.shape
        w2_out_reduced = ttnn.reshape(
            w2_out_reduced, (1, 1, original_shape[-4] * original_shape[-3] * original_shape[-2], original_shape[-1])
        )

        if mode == Mode.DECODE:
            w2_out_reduced = ttnn.to_memory_config(
                w2_out_reduced,
                self.args.get_mlp_output_mem_config(mode, self.prefetcher),
            )

        return w2_out_reduced
