# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Modular MLP implementation.

This is a refactored version of models/tt_transformers/tt/mlp.py with:
- Separated config computation from forward logic
- Encapsulated CCL strategies
- Cleaner mode-specific code paths
"""

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.modules.mlp.modular_attempt.ccl_strategies import create_ccl_strategy
from models.common.modules.mlp.modular_attempt.mlp_config import create_mlp_config_from_model_args
from models.tt_transformers.tt.common import pad_to_size


class ModularMLP(LightweightModule):
    """
    Modular MLP with separated concerns:
    - Config: Pre-computed at init, accessed via MLPConfig
    - CCL: Encapsulated in CCLStrategy
    - Compute: Clean forward pass with minimal branching
    """

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
    ):
        super().__init__()

        self.mesh_device = mesh_device
        self.tt_ccl = tt_ccl
        self.args = args
        self.layer_num = layer_num

        # Create config object (pre-computes all settings)
        self.config = create_mlp_config_from_model_args(args, model_config, max(layer_num, 0))

        # Create CCL strategy based on topology
        self.ccl_strategy = create_ccl_strategy(
            self.config.topology,
            mesh_device,
            tt_ccl,
            args,
        )

        # Load weights
        self._load_weights(args, state_dict, weight_cache_path, layer_num, dtype, state_dict_prefix)

    def _load_weights(self, args, state_dict, weight_cache_path, layer_num, dtype, state_dict_prefix):
        """Load and shard weight tensors"""
        state_dict_prefix = state_dict_prefix or args.get_state_dict_prefix(self.__class__.__name__, layer_num)

        def torch_weight(name):
            return torch.transpose(state_dict[f"{state_dict_prefix}.{name}.weight"], -2, -1)

        def pad_hidden_dim(tensor, dim):
            return pad_to_size(tensor, dim=dim, size=args.hidden_dim)

        # Cache name handling
        hidden_dim_string = f".hidden_dim_{args.hidden_dim}" if args.hidden_dim != args.unpadded_hidden_dim else ""

        if args.dummy_weights:
            cache_name = lambda _: None
        else:
            cache_name = lambda name: weight_cache_path / f"{state_dict_prefix}.{name}{hidden_dim_string}"

        # Memory configs for weights
        w1_w3_mem_config = args.create_dram_sharded_mem_config(args.dim, args.hidden_dim // args.num_devices)
        w2_mem_config = args.create_dram_sharded_mem_config(args.hidden_dim // args.num_devices, args.dim)

        # Sharding dimensions
        w1_dims = (-1, -2) if args.is_galaxy else (-2, -1)
        w2_dims = (-2, -1) if args.is_galaxy else (-1, -2)

        # Get dtypes from layer config
        layer_config = self.config.get_layer_config(max(layer_num, 0))
        ff1_3_dtype = layer_config.ff1_ff3_dtype
        ff2_dtype = layer_config.ff2_dtype

        def as_sharded_tensor(name, tensor_dtype, dims):
            return ttnn.as_tensor(
                pad_hidden_dim(torch_weight(name[:2]), dims[0] if args.is_galaxy else dims[-1]),
                dtype=tensor_dtype,
                device=self.mesh_device,
                mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=dims, mesh_shape=args.cluster_shape),
                layout=ttnn.TILE_LAYOUT,
                memory_config=(
                    ttnn.DRAM_MEMORY_CONFIG if args.is_galaxy else w2_mem_config if "w2" in name else w1_w3_mem_config
                ),
                cache_file_name=cache_name(name),
            )

        # Load weights
        self.w1 = as_sharded_tensor("w1_sharded", ff1_3_dtype, dims=w1_dims)
        self.w2 = as_sharded_tensor("w2_sharded", ff2_dtype, dims=w2_dims)
        self.w3 = as_sharded_tensor("w3_sharded", ff1_3_dtype, dims=w1_dims)

    def forward(self, x: ttnn.Tensor, mode: str) -> ttnn.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [1, 1, batch, dim] for decode, [1, batch, seq, dim] for prefill
            mode: "decode" or "prefill"

        Returns:
            Output tensor, same shape as input
        """
        if mode == "decode":
            return self._forward_decode(x)
        else:
            return self._forward_prefill(x)

    def _forward_decode(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Decode mode forward pass"""
        config = self.config
        layer_config = config.get_layer_config(max(self.layer_num, 0))

        # Get program configs
        pc_w1_w3, pc_w2 = config.get_program_config("decode")
        memory_config = config.get_memory_config("decode")

        # Activation dtype
        activation_dtype = layer_config.activation_dtype
        is_tg = config.is_galaxy

        # FF1: gate projection
        w1_out = ttnn.linear(
            x,
            self.w1,
            dtype=ttnn.bfloat8_b if is_tg else (activation_dtype or ttnn.bfloat16),
            compute_kernel_config=layer_config.ff1_ff3_compute_config,
            program_config=pc_w1_w3,
            memory_config=memory_config,
            core_grid=None,
        )

        # FF3: up projection
        w3_out = ttnn.linear(
            x,
            self.w3,
            dtype=ttnn.bfloat8_b if is_tg else (activation_dtype or ttnn.bfloat16),
            compute_kernel_config=layer_config.ff1_ff3_compute_config,
            program_config=pc_w1_w3,
            memory_config=memory_config,
            core_grid=None,
        )
        ttnn.deallocate(x)

        # CCL: reduce FF1/FF3 outputs if needed
        if is_tg:
            input_mem_cfg = w1_out.memory_config()
            w1_out, w3_out = self.ccl_strategy.reduce_after_ff1_ff3(w1_out, w3_out, "decode", config)

        # Activation: SiLU(w1) * w3
        w2_in = ttnn.mul(
            w1_out,
            w3_out,
            input_tensor_a_activations=[config.activation_type],
            dtype=activation_dtype or ttnn.bfloat8_b,
            memory_config=w1_out.memory_config(),
        )

        ttnn.deallocate(w3_out)
        ttnn.deallocate(w1_out)

        # Memory config transition for w2
        if not is_tg:
            w2_in = ttnn.to_memory_config(w2_in, config.memory_configs.sharded_mlp2_input)
        elif config.dim >= 8192:
            # All-gather before FF2 for large TG models
            w2_in = self.ccl_strategy.all_gather_before_ff2(w2_in, "decode", config, input_mem_cfg)

        # FF2: down projection
        w2_out = ttnn.linear(
            w2_in,
            self.w2,
            compute_kernel_config=layer_config.ff2_compute_config,
            dtype=self.args.ccl_dtype if is_tg else (activation_dtype or ttnn.bfloat16),
            program_config=pc_w2,
            memory_config=memory_config,
            core_grid=None,
        )
        ttnn.deallocate(w2_in)

        # CCL: reduce FF2 output
        w2_out_reduced = self.ccl_strategy.reduce_after_ff2(w2_out, "decode", config)

        # Reshape and finalize
        original_shape = w2_out_reduced.shape
        w2_out_reduced = ttnn.reshape(
            w2_out_reduced, (1, 1, original_shape[-4] * original_shape[-3] * original_shape[-2], original_shape[-1])
        )

        # Final memory config
        final_memcfg = config.memory_configs.sharded_attn_input if is_tg else config.memory_configs.decode_residual
        w2_out_reduced = ttnn.to_memory_config(w2_out_reduced, final_memcfg)

        return w2_out_reduced

    def _forward_prefill(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Prefill mode forward pass"""
        config = self.config
        layer_config = config.get_layer_config(max(self.layer_num, 0))

        seq_len = x.shape[-2]
        is_tg = config.is_galaxy

        # Reshape if needed for long sequences
        if seq_len >= config.prefill_len_cutoff:
            x = ttnn.reshape(x, [1, seq_len // config.prefill_len_cutoff, config.prefill_len_cutoff, -1])

        # Get program configs
        pc_w1_w3, pc_w2 = config.get_program_config("prefill", seq_len)
        memory_config = config.get_memory_config("prefill")

        activation_dtype = layer_config.activation_dtype

        # FF1: gate projection
        w1_out = ttnn.linear(
            x,
            self.w1,
            dtype=ttnn.bfloat8_b if is_tg else (activation_dtype or ttnn.bfloat16),
            compute_kernel_config=layer_config.ff1_ff3_compute_config,
            program_config=pc_w1_w3,
            memory_config=memory_config,
            core_grid=None,
        )

        # FF3: up projection
        w3_out = ttnn.linear(
            x,
            self.w3,
            dtype=ttnn.bfloat8_b if is_tg else (activation_dtype or ttnn.bfloat16),
            compute_kernel_config=layer_config.ff1_ff3_compute_config,
            program_config=pc_w1_w3,
            memory_config=memory_config,
            core_grid=None,
        )
        ttnn.deallocate(x)

        # CCL: reduce FF1/FF3 outputs if TG
        if is_tg:
            input_mem_cfg = w1_out.memory_config()
            w1_out, w3_out = self.ccl_strategy.reduce_after_ff1_ff3(w1_out, w3_out, "prefill", config)

        # Activation: SiLU(w1) * w3
        w2_in = ttnn.mul(
            w1_out,
            w3_out,
            input_tensor_a_activations=[config.activation_type],
            dtype=activation_dtype or ttnn.bfloat8_b,
            memory_config=w1_out.memory_config(),
        )

        ttnn.deallocate(w3_out)
        ttnn.deallocate(w1_out)

        # All-gather before FF2 for TG
        if is_tg:
            w2_in = self.ccl_strategy.all_gather_before_ff2(w2_in, "prefill", config, input_mem_cfg)

        # FF2: down projection
        w2_out = ttnn.linear(
            w2_in,
            self.w2,
            compute_kernel_config=layer_config.ff2_compute_config,
            dtype=self.args.ccl_dtype if is_tg else (activation_dtype or ttnn.bfloat16),
            program_config=pc_w2,
            memory_config=memory_config,
            core_grid=None,
        )
        ttnn.deallocate(w2_in)

        # CCL: reduce FF2 output
        w2_out_reduced = self.ccl_strategy.reduce_after_ff2(w2_out, "prefill", config)

        # Reshape
        original_shape = w2_out_reduced.shape
        w2_out_reduced = ttnn.reshape(
            w2_out_reduced, (1, 1, original_shape[-4] * original_shape[-3] * original_shape[-2], original_shape[-1])
        )

        return w2_out_reduced
