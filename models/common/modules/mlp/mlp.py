# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Modular MLP module - extracted from models/tt_transformers/tt/mlp.py

This module has all dependencies explicitly passed in, removing the coupling
to ModelArgs and model_config dict objects.
"""

from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule


def pad_to_size(x: torch.Tensor, dim: int, size: int) -> torch.Tensor:
    """
    Pads the specified dimension of the input tensor with zeros
    """
    if dim < 0:
        dim = x.dim() + dim
    current_size = x.size(dim)
    pad_size = size - current_size

    if pad_size == 0:
        return x

    pad = [0] * (2 * x.dim())
    pad_index = 2 * (x.dim() - dim - 1)
    pad[pad_index + 1] = pad_size

    return torch.nn.functional.pad(x, pad, mode="constant", value=0)


def tt_all_reduce(
    input_tensor,
    mesh_device,
    tt_ccl,
    cluster_axis=0,
    dim=0,
    num_reduce_scatter_links=1,
    num_all_gather_links=2,
    topology=ttnn.Topology.Linear,
    memory_config=None,
    sharded=False,
    dtype=ttnn.bfloat16,
    use_composite=False,
):
    """
    All-reduce operation copied from ccl.py for self-containment.
    """
    # N150
    if list(mesh_device.shape) == [1, 1] or (cluster_axis == 1 and 1 in list(mesh_device.shape)):
        return input_tensor

    # Ensure dim 0 and 1 are 1
    original_shape = input_tensor.shape
    if original_shape[0] != 1 or original_shape[1] != 1:
        input_tensor = ttnn.reshape(
            input_tensor, (1, 1, original_shape[-4] * original_shape[-3] * original_shape[-2], original_shape[-1])
        )

    # N300 and T3K: reduce_scatter
    if 1 in list(mesh_device.shape):
        if input_tensor.is_sharded() and not sharded:
            input_tensor_sharded = input_tensor
            input_tensor = ttnn.sharded_to_interleaved(input_tensor_sharded, ttnn.L1_MEMORY_CONFIG)
            input_tensor_sharded.deallocate(True)

        reduced = ttnn.experimental.reduce_scatter_minimal_async(
            input_tensor,
            persistent_output_buffers=None,
            dim=dim,
            multi_device_global_semaphore=tt_ccl.get_and_cycle_rs_semaphore_handles(),
            barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(),
            num_links=num_reduce_scatter_links,
            memory_config=memory_config,
            intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=topology,
            chunks_per_sync=10,
            num_workers_per_link=2,
            num_buffers_per_channel=2,
        )
        input_tensor.deallocate(True)
        return reduced

    # TG: all_reduce
    # Cast to CCL dtype
    if input_tensor.dtype != dtype:
        input_tensor = ttnn.to_memory_config(input_tensor, ttnn.L1_MEMORY_CONFIG, dtype)
        if sharded and memory_config is not None:
            input_tensor = ttnn.to_memory_config(input_tensor, memory_config, dtype)

    if not sharded:
        input_tensor = ttnn.to_memory_config(input_tensor, ttnn.DRAM_MEMORY_CONFIG)

    if not use_composite:
        gathered_tensor = ttnn.experimental.all_gather_async(
            input_tensor,
            persistent_output_buffer=None,
            dim=dim,
            multi_device_global_semaphore=tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis),
            num_links=num_all_gather_links,
            cluster_axis=cluster_axis,
            topology=topology,
            memory_config=ttnn.DRAM_MEMORY_CONFIG if not sharded else memory_config,
            barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis),
            chunks_per_sync=10,
            num_workers_per_link=2,
            num_buffers_per_channel=2,
        )

        if sharded:
            gathered_tensor = ttnn.to_memory_config(gathered_tensor, ttnn.L1_MEMORY_CONFIG)

        reduced_tensor = ttnn.experimental.fast_reduce_nc(
            gathered_tensor,
            dims=[dim],
            output=None,
            compute_kernel_config=None,
            memory_config=ttnn.L1_MEMORY_CONFIG if sharded else ttnn.DRAM_MEMORY_CONFIG,
        )

        gathered_tensor.deallocate(True)
    else:
        input_mem_cfg = input_tensor.memory_config()

        reduced_tensor = ttnn.experimental.reduce_scatter_minimal_async(
            input_tensor,
            persistent_output_buffers=None,
            dim=dim,
            multi_device_global_semaphore=tt_ccl.get_and_cycle_rs_semaphore_handles(cluster_axis),
            barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis),
            num_links=num_reduce_scatter_links,
            cluster_axis=cluster_axis,
            memory_config=ttnn.DRAM_MEMORY_CONFIG if not sharded else memory_config,
            intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=topology,
            chunks_per_sync=10,
            num_workers_per_link=2,
            num_buffers_per_channel=2,
        )

        reduced_tensor = ttnn.experimental.all_gather_async(
            reduced_tensor,
            persistent_output_buffer=None,
            dim=dim,
            multi_device_global_semaphore=tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis),
            num_links=num_all_gather_links,
            cluster_axis=cluster_axis,
            topology=topology,
            memory_config=input_mem_cfg,
            barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis),
            chunks_per_sync=10,
            num_workers_per_link=2,
            num_buffers_per_channel=2,
        )

    reduced_tensor = ttnn.reshape(reduced_tensor, original_shape)
    return reduced_tensor


@dataclass
class MLPConfig:
    """
    All configuration needed for MLP, explicitly defined instead of using ModelArgs/model_config.

    These are the "brought in" dependencies from the original args and model_config objects.
    """

    # Model dimensions
    dim: int
    hidden_dim: int

    # Device topology
    is_galaxy: bool  # num_devices == 32
    cluster_shape: list  # [rows, cols]
    num_devices: int

    # Prefill config
    prefill_len_cutoff: int  # 512 (BH) or 1024 (WH)

    # Weight loading
    dummy_weights: bool = False
    unpadded_hidden_dim: Optional[int] = None

    # CCL parameters
    ccl_dtype: Any = ttnn.bfloat8_b
    num_reduce_scatter_links: int = 1
    num_all_gather_links: int = 2

    # Activation type
    mlp_activation_type: Any = ttnn.UnaryOpType.SILU

    # Memory configs for w1/w3 and w2 weight sharding (must be provided)
    w1_w3_mem_config: Any = None
    w2_mem_config: Any = None

    def __post_init__(self):
        if self.unpadded_hidden_dim is None:
            self.unpadded_hidden_dim = self.hidden_dim


@dataclass
class MLPProgramConfigs:
    """
    Program configs for different execution modes.
    These are the extracted model_config["*"] entries.
    """

    # Decode mode configs (for non-TG)
    decode_mlp_w1_w3_prg_config: Any = None
    decode_mlp_w2_prg_config: Any = None

    # Decode mode configs (for TG, Galaxy)
    ff1_3_tg_progcfg: Any = None
    ff2_tg_progcfg: Any = None

    # Prefill mode configs - callables that take seq_len
    prefill_mlp_w1_w3_prg_config: Callable[[int], Any] = None
    prefill_mlp_w2_prg_config: Callable[[int], Any] = None

    # Memory configs for CCL (TG specific)
    ff1_out_reduce_scatter_memcfg: Any = None
    ff1_out_gathered_memcfg: Any = None

    # Memory configs for sharding
    sharded_mlp2_input_memcfg: Any = None
    ff2_out_reduce_scatter_memcfg: Any = None

    # Output memory configs
    sharded_attn_input_memcfg: Any = None
    decode_residual_memcfg: Any = None


@dataclass
class MLPOptimizationConfig:
    """
    Per-layer optimization settings from DECODERS_OPTIMIZATIONS.
    """

    # Weight dtypes
    ff1_3_dtype: Any = ttnn.bfloat8_b
    ff2_dtype: Any = ttnn.bfloat8_b

    # Activation dtype (None means use default)
    activation_dtype: Any = None

    # Compute kernel configs for math fidelity
    li_ff1_3_compute_kernel_cfg: Any = None
    li_ff2_compute_kernel_cfg: Any = None


# def ccl_topology_linear():
#     """Default CCL topology - Linear"""
#     return ttnn.Topology.Linear


def ccl_topology(num_devices):
    # Use ring on a T3K or 6U galaxy submesh
    if num_devices == 8 and ttnn.cluster.get_cluster_type() in [
        ttnn.cluster.ClusterType.T3K,
        ttnn.cluster.ClusterType.GALAXY,
    ]:
        return ttnn.Topology.Ring
    elif num_devices > 1:  # All other multi chip devices
        return ttnn.Topology.Linear
    return None


class MLP(LightweightModule):
    """
    MLP module with explicit dependencies.

    Instead of taking args (ModelArgs) and model_config (dict), this takes:
    - config: MLPConfig with model dimensions and device topology
    - program_configs: MLPProgramConfigs with matmul program configs
    - optimization_config: MLPOptimizationConfig with precision/fidelity settings
    - ccl_topology: Callable that returns the CCL topology

    For backward compatibility, you can still construct this from args/model_config
    using the from_model_args() class method.
    """

    def __init__(
        self,
        mesh_device,
        tt_ccl,
        config: MLPConfig,
        program_configs: MLPProgramConfigs,
        optimization_config: MLPOptimizationConfig,
        state_dict,
        weight_cache_path,
        layer_num,
        dtype,
        state_dict_prefix: Optional[str] = None,
        ccl_topology: Callable[[int], Any] | Any = ccl_topology,
    ):
        super().__init__()

        self.mesh_device = mesh_device
        self.tt_ccl = tt_ccl
        self.config = config
        self.program_configs = program_configs
        self.optimization_config = optimization_config
        self.layer_num = layer_num
        self.ccl_topology = ccl_topology(config.num_devices) if callable(ccl_topology) else ccl_topology

        # Convenience aliases
        self.dim = config.dim
        self.is_galaxy = config.is_galaxy

        # State dict prefix
        if state_dict_prefix is None:
            state_dict_prefix = f"layers.{layer_num}.feed_forward"

        torch_weight = lambda name: torch.transpose(state_dict[f"{state_dict_prefix}.{name}.weight"], -2, -1)
        pad_hidden_dim = lambda tensor, dim_idx: pad_to_size(tensor, dim=dim_idx, size=config.hidden_dim)

        # If padding was applied, add the unpadded hidden dim to the cache name to avoid loading incorrect weights
        hidden_dim_string = (
            f".hidden_dim_{config.hidden_dim}" if config.hidden_dim != config.unpadded_hidden_dim else ""
        )

        if config.dummy_weights:
            cache_name = lambda _: None
        else:
            cache_name = lambda name: weight_cache_path / f"{state_dict_prefix}.{name}{hidden_dim_string}"

        # Sharded weights
        w1_dims = (-1, -2) if config.is_galaxy else (-2, -1)
        w2_dims = (-2, -1) if config.is_galaxy else (-1, -2)

        as_sharded_tensor = lambda name, w_type, dims: ttnn.as_tensor(
            pad_hidden_dim(torch_weight(name[:2]), dims[0] if config.is_galaxy else dims[-1]),
            dtype=w_type,
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=dims, mesh_shape=config.cluster_shape),
            layout=ttnn.TILE_LAYOUT,
            memory_config=(
                ttnn.DRAM_MEMORY_CONFIG
                if config.is_galaxy
                else config.w2_mem_config
                if "w2" in name
                else config.w1_w3_mem_config
            ),
            cache_file_name=cache_name(name),
        )

        self.w1 = as_sharded_tensor("w1_sharded", optimization_config.ff1_3_dtype, dims=w1_dims)
        self.w2 = as_sharded_tensor("w2_sharded", optimization_config.ff2_dtype, dims=w2_dims)
        self.w3 = as_sharded_tensor("w3_sharded", optimization_config.ff1_3_dtype, dims=w1_dims)

        self.activation_type = config.mlp_activation_type

    @classmethod
    def from_model_args(
        cls,
        mesh_device,
        tt_ccl,
        args,  # ModelArgs
        model_config,  # dict
        state_dict,
        weight_cache_path,
        layer_num,
        dtype,
        state_dict_prefix=None,
    ):
        """
        Factory method for backward compatibility - creates MLP from the original
        args (ModelArgs) and model_config (dict) interface.
        """
        # Build MLPConfig from args
        config = MLPConfig(
            dim=args.dim,
            hidden_dim=args.hidden_dim,
            is_galaxy=args.is_galaxy,
            cluster_shape=args.cluster_shape,
            num_devices=args.num_devices,
            prefill_len_cutoff=args.prefill_len_cutoff,
            dummy_weights=args.dummy_weights,
            unpadded_hidden_dim=args.unpadded_hidden_dim,
            ccl_dtype=args.ccl_dtype,
            num_reduce_scatter_links=args.num_reduce_scatter_links,
            num_all_gather_links=args.num_all_gather_links,
            mlp_activation_type=getattr(args, "mlp_activation_type", ttnn.UnaryOpType.SILU),
            w1_w3_mem_config=args.create_dram_sharded_mem_config(args.dim, args.hidden_dim // args.num_devices),
            w2_mem_config=args.create_dram_sharded_mem_config(args.hidden_dim // args.num_devices, args.dim),
        )

        # Build MLPProgramConfigs from model_config
        program_configs = MLPProgramConfigs(
            decode_mlp_w1_w3_prg_config=model_config.get("DECODE_MLP_W1_W3_PRG_CONFIG"),
            decode_mlp_w2_prg_config=model_config.get("DECODE_MLP_W2_PRG_CONFIG"),
            ff1_3_tg_progcfg=model_config.get("FF1_3_TG_PROGCFG"),
            ff2_tg_progcfg=model_config.get("FF2_TG_PROGCFG"),
            prefill_mlp_w1_w3_prg_config=model_config.get("PREFILL_MLP_W1_W3_PRG_CONFIG"),
            prefill_mlp_w2_prg_config=model_config.get("PREFILL_MLP_W2_PRG_CONFIG"),
            ff1_out_reduce_scatter_memcfg=model_config.get("FF1_OUT_REDUCE_SCATTER_MEMCFG"),
            ff1_out_gathered_memcfg=model_config.get("FF1_OUT_GATHERED_MEMCFG"),
            sharded_mlp2_input_memcfg=model_config.get("SHARDED_MLP2_INPUT_MEMCFG"),
            ff2_out_reduce_scatter_memcfg=model_config.get("FF2_OUT_REDUCE_SCATTER_MEMCFG"),
            sharded_attn_input_memcfg=model_config.get("SHARDED_ATTN_INPUT_MEMCFG"),
            decode_residual_memcfg=model_config.get("DECODE_RESIDUAL_MEMCFG"),
        )

        # Build MLPOptimizationConfig from DECODERS_OPTIMIZATIONS
        decoders_opt = model_config.get("DECODERS_OPTIMIZATIONS")
        effective_layer_num = max(layer_num, 0)  # cross_block uses the configuration of the first decoder

        from models.tt_transformers.tt.model_config import OpGroup, TensorGroup

        optimization_config = MLPOptimizationConfig(
            ff1_3_dtype=decoders_opt.get_tensor_dtype(decoder_id=effective_layer_num, tensor=TensorGroup.FF1_FF3),
            ff2_dtype=decoders_opt.get_tensor_dtype(decoder_id=effective_layer_num, tensor=TensorGroup.FF2),
            activation_dtype=decoders_opt.get_tensor_dtype(
                decoder_id=effective_layer_num, tensor=TensorGroup.ACTIVATION
            ),
            li_ff1_3_compute_kernel_cfg=decoders_opt.get_math_fidelity(
                decoder_id=effective_layer_num, op=OpGroup.LI_FF1_FF3, configuration=args
            ),
            li_ff2_compute_kernel_cfg=decoders_opt.get_math_fidelity(
                decoder_id=effective_layer_num, op=OpGroup.LI_FF2, configuration=args
            ),
        )

        # State dict prefix
        if state_dict_prefix is None:
            state_dict_prefix = args.get_state_dict_prefix("MLP", layer_num)

        return cls(
            mesh_device=mesh_device,
            tt_ccl=tt_ccl,
            config=config,
            program_configs=program_configs,
            optimization_config=optimization_config,
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
            layer_num=layer_num,
            dtype=dtype,
            state_dict_prefix=state_dict_prefix,
            ccl_topology=args.ccl_topology(),
        )

    def forward(self, x: ttnn.Tensor, mode) -> ttnn.Tensor:
        """
        w1 -> gate_proj
        w2 -> down_proj
        w3 -> up_proj
        HF reference: self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        """
        seq_len = x.shape[-2]
        TG = self.is_galaxy

        # Get per-layer optimization settings
        activation_dtype = self.optimization_config.activation_dtype
        li_ff1_3_compute_kernel_cfg = self.optimization_config.li_ff1_3_compute_kernel_cfg
        li_ff2_compute_kernel_cfg = self.optimization_config.li_ff2_compute_kernel_cfg

        if mode == "decode":  # Sharded config
            if TG:  # TODO: Fix this when TG supports DRAM sharded matmuls
                pc_1 = self.program_configs.ff1_3_tg_progcfg if self.dim >= 4096 else None
                pc_2 = self.program_configs.ff2_tg_progcfg if self.dim >= 4096 else None
                pc_3 = self.program_configs.ff1_3_tg_progcfg if self.dim >= 4096 else None
            else:
                pc_1 = self.program_configs.decode_mlp_w1_w3_prg_config
                pc_2 = self.program_configs.decode_mlp_w2_prg_config
                pc_3 = self.program_configs.decode_mlp_w1_w3_prg_config
        else:  # Update the program configs based for prefill
            if seq_len >= self.config.prefill_len_cutoff:
                # Reshape input to fit on device and parallelize computation
                x = ttnn.reshape(x, [1, seq_len // self.config.prefill_len_cutoff, self.config.prefill_len_cutoff, -1])
            pc_1 = self.program_configs.prefill_mlp_w1_w3_prg_config(seq_len)
            pc_2 = self.program_configs.prefill_mlp_w2_prg_config(seq_len)
            pc_3 = self.program_configs.prefill_mlp_w1_w3_prg_config(seq_len)

        # In decode mode (seqlen <= 32) do DRAM sharded matmuls
        # These use HiFi2; this drops 1 bit of the activations but would be FLOP-bound on 12 cores with HiFi4
        memory_config = ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG if mode == "decode" else ttnn.DRAM_MEMORY_CONFIG
        w1_out = ttnn.linear(
            x,
            self.w1,
            dtype=ttnn.bfloat8_b if TG else activation_dtype or ttnn.bfloat16,
            core_grid=None,
            compute_kernel_config=li_ff1_3_compute_kernel_cfg,
            program_config=pc_1,
            memory_config=memory_config,
        )

        w3_out = ttnn.linear(
            x,
            self.w3,
            dtype=ttnn.bfloat8_b if TG else activation_dtype or ttnn.bfloat16,
            core_grid=None,
            compute_kernel_config=li_ff1_3_compute_kernel_cfg,
            program_config=pc_3,
            memory_config=memory_config,
        )
        ttnn.deallocate(x)

        if TG:
            if self.dim == 8192 or mode == "prefill":
                input_mem_cfg = w1_out.memory_config()

                cluster_axis = 1
                w1_out = ttnn.experimental.reduce_scatter_minimal_async(
                    w1_out,
                    persistent_output_buffers=None,
                    dim=3,
                    multi_device_global_semaphore=self.tt_ccl.get_and_cycle_rs_semaphore_handles(cluster_axis),
                    barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis),
                    num_links=self.config.num_reduce_scatter_links,
                    cluster_axis=cluster_axis,
                    memory_config=self.program_configs.ff1_out_reduce_scatter_memcfg if mode == "decode" else None,
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
                    multi_device_global_semaphore=self.tt_ccl.get_and_cycle_rs_semaphore_handles(cluster_axis),
                    barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis),
                    num_links=1,
                    cluster_axis=cluster_axis,
                    memory_config=self.program_configs.ff1_out_reduce_scatter_memcfg if mode == "decode" else None,
                    intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    topology=ttnn.Topology.Linear,
                    chunks_per_sync=10,
                    num_workers_per_link=2,
                    num_buffers_per_channel=2,
                )
            else:
                w1_out = tt_all_reduce(
                    w1_out,
                    self.mesh_device,
                    self.tt_ccl,
                    cluster_axis=1,
                    num_all_gather_links=2,
                    sharded=True if mode == "decode" else False,
                    topology=self.ccl_topology,
                    memory_config=self.program_configs.ff1_out_gathered_memcfg if mode == "decode" else None,
                )
                w3_out = tt_all_reduce(
                    w3_out,
                    self.mesh_device,
                    self.tt_ccl,
                    cluster_axis=1,
                    num_all_gather_links=2,
                    sharded=True if mode == "decode" else False,
                    topology=self.ccl_topology,
                    memory_config=self.program_configs.ff1_out_gathered_memcfg if mode == "decode" else None,
                )

        w2_in = ttnn.mul(
            w1_out,
            w3_out,
            input_tensor_a_activations=[self.activation_type],
            dtype=activation_dtype or ttnn.bfloat8_b,
            memory_config=w1_out.memory_config(),
        )

        if mode == "decode" and not TG:
            # w2 may use a different core grid, this is a no-op if they already match
            w2_in = ttnn.to_memory_config(w2_in, self.program_configs.sharded_mlp2_input_memcfg)

        ttnn.deallocate(w3_out)
        ttnn.deallocate(w1_out)

        if TG and (self.dim == 8192 or mode == "prefill"):
            cluster_axis = 1
            w2_in = ttnn.experimental.all_gather_async(
                w2_in,
                persistent_output_buffer=None,
                dim=3,
                multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis),
                num_links=2,
                cluster_axis=1,
                topology=ttnn.Topology.Linear,
                memory_config=input_mem_cfg,
                barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis),
                chunks_per_sync=10,
                num_workers_per_link=2,
                num_buffers_per_channel=2,
            )

            if mode == "decode":
                w2_in = ttnn.to_memory_config(w2_in, ttnn.L1_MEMORY_CONFIG)

        w2_out = ttnn.linear(
            w2_in,
            self.w2,
            compute_kernel_config=li_ff2_compute_kernel_cfg,
            dtype=self.config.ccl_dtype if TG else activation_dtype or ttnn.bfloat16,
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
            dim=0 if (TG and self.dim < 8192) else 3,
            num_reduce_scatter_links=self.config.num_reduce_scatter_links,
            num_all_gather_links=self.config.num_all_gather_links,
            sharded=(mode == "decode"),
            memory_config=(
                (self.program_configs.ff2_out_reduce_scatter_memcfg if TG else w2_out.memory_config())
                if mode == "decode"
                else ttnn.DRAM_MEMORY_CONFIG
            ),
            dtype=self.config.ccl_dtype,
            use_composite=True if self.dim == 8192 else False,
            topology=self.ccl_topology,
        )

        # Ensure dim 0 and 1 are 1
        original_shape = w2_out_reduced.shape
        w2_out_reduced = ttnn.reshape(
            w2_out_reduced, (1, 1, original_shape[-4] * original_shape[-3] * original_shape[-2], original_shape[-1])
        )
        if mode == "decode":
            w2_out_reduced = ttnn.to_memory_config(
                w2_out_reduced,
                self.program_configs.sharded_attn_input_memcfg if TG else self.program_configs.decode_residual_memcfg,
            )

        return w2_out_reduced

    # =========================================================================
    # TTTv2-style split forward functions (no TG if-else in hot path)
    # =========================================================================

    def forward_non_tg(self, x: ttnn.Tensor, mode: str) -> ttnn.Tensor:
        """
        MLP forward for non-TG (non-Galaxy) devices: N150, N300, T3K.

        Execution path:
          linear(w1) → linear(w3) → mul+silu → [reshard] → linear(w2) → all_reduce → [reshard]

        Remaining if-else: mode (decode/prefill) - can be split further.
        """
        seq_len = x.shape[-2]

        # Per-layer optimization settings
        activation_dtype = self.optimization_config.activation_dtype
        li_ff1_3_compute_kernel_cfg = self.optimization_config.li_ff1_3_compute_kernel_cfg
        li_ff2_compute_kernel_cfg = self.optimization_config.li_ff2_compute_kernel_cfg

        # --- Program config selection (mode-dependent) ---
        if mode == "decode":
            pc_1 = self.program_configs.decode_mlp_w1_w3_prg_config
            pc_2 = self.program_configs.decode_mlp_w2_prg_config
            pc_3 = self.program_configs.decode_mlp_w1_w3_prg_config
            memory_config = ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG
        else:  # prefill
            if seq_len >= self.config.prefill_len_cutoff:
                x = ttnn.reshape(x, [1, seq_len // self.config.prefill_len_cutoff, self.config.prefill_len_cutoff, -1])
            pc_1 = self.program_configs.prefill_mlp_w1_w3_prg_config(seq_len)
            pc_2 = self.program_configs.prefill_mlp_w2_prg_config(seq_len)
            pc_3 = self.program_configs.prefill_mlp_w1_w3_prg_config(seq_len)
            memory_config = ttnn.DRAM_MEMORY_CONFIG

        # --- STAGE 1: W1/W3 Linear ---
        w1_out = ttnn.linear(
            x,
            self.w1,
            dtype=activation_dtype or ttnn.bfloat16,
            core_grid=None,
            compute_kernel_config=li_ff1_3_compute_kernel_cfg,
            program_config=pc_1,
            memory_config=memory_config,
        )
        w3_out = ttnn.linear(
            x,
            self.w3,
            dtype=activation_dtype or ttnn.bfloat16,
            core_grid=None,
            compute_kernel_config=li_ff1_3_compute_kernel_cfg,
            program_config=pc_3,
            memory_config=memory_config,
        )
        ttnn.deallocate(x)

        # --- STAGE 2: No CCL for non-TG (skip) ---

        # --- STAGE 3: Activation + Multiply ---
        w2_in = ttnn.mul(
            w1_out,
            w3_out,
            input_tensor_a_activations=[self.activation_type],
            dtype=activation_dtype or ttnn.bfloat8_b,
            memory_config=w1_out.memory_config(),
        )

        # --- STAGE 3.5: Reshard for w2 (decode only) ---
        if mode == "decode":
            w2_in = ttnn.to_memory_config(w2_in, self.program_configs.sharded_mlp2_input_memcfg)

        ttnn.deallocate(w3_out)
        ttnn.deallocate(w1_out)

        # --- STAGE 4: No all_gather for non-TG (skip) ---

        # --- STAGE 5: W2 Linear ---
        w2_out = ttnn.linear(
            w2_in,
            self.w2,
            compute_kernel_config=li_ff2_compute_kernel_cfg,
            dtype=activation_dtype or ttnn.bfloat16,
            program_config=pc_2,
            memory_config=memory_config,
            core_grid=None,
        )
        ttnn.deallocate(w2_in)

        # --- STAGE 6: Final All-Reduce ---
        w2_out_reduced = tt_all_reduce(
            w2_out,
            self.mesh_device,
            self.tt_ccl,
            cluster_axis=0,
            dim=3,
            num_reduce_scatter_links=self.config.num_reduce_scatter_links,
            num_all_gather_links=self.config.num_all_gather_links,
            sharded=(mode == "decode"),
            memory_config=w2_out.memory_config() if mode == "decode" else ttnn.DRAM_MEMORY_CONFIG,
            dtype=self.config.ccl_dtype,
            use_composite=(self.dim == 8192),
            topology=self.ccl_topology,
        )

        # --- STAGE 7: Reshape + Final memory config ---
        original_shape = w2_out_reduced.shape
        w2_out_reduced = ttnn.reshape(
            w2_out_reduced, (1, 1, original_shape[-4] * original_shape[-3] * original_shape[-2], original_shape[-1])
        )
        if mode == "decode":
            w2_out_reduced = ttnn.to_memory_config(w2_out_reduced, self.program_configs.decode_residual_memcfg)

        return w2_out_reduced

    def forward_tg(self, x: ttnn.Tensor, mode: str) -> ttnn.Tensor:
        """
        MLP forward for TG (Galaxy) devices.

        Execution paths (still has dim-based branching):
          - dim < 8192 decode:  linear → linear → all_reduce(×2) → mul+silu → linear → all_reduce
          - dim == 8192 OR prefill: linear → linear → reduce_scatter(×2) → mul+silu → all_gather → linear → all_reduce

        Remaining if-else: mode (decode/prefill), dim (8192 vs others) - can be split further.
        """
        seq_len = x.shape[-2]

        # Per-layer optimization settings
        activation_dtype = self.optimization_config.activation_dtype
        li_ff1_3_compute_kernel_cfg = self.optimization_config.li_ff1_3_compute_kernel_cfg
        li_ff2_compute_kernel_cfg = self.optimization_config.li_ff2_compute_kernel_cfg

        # --- Program config selection (mode + dim dependent) ---
        if mode == "decode":
            pc_1 = self.program_configs.ff1_3_tg_progcfg if self.dim >= 4096 else None
            pc_2 = self.program_configs.ff2_tg_progcfg if self.dim >= 4096 else None
            pc_3 = self.program_configs.ff1_3_tg_progcfg if self.dim >= 4096 else None
            memory_config = ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG
        else:  # prefill
            if seq_len >= self.config.prefill_len_cutoff:
                x = ttnn.reshape(x, [1, seq_len // self.config.prefill_len_cutoff, self.config.prefill_len_cutoff, -1])
            pc_1 = self.program_configs.prefill_mlp_w1_w3_prg_config(seq_len)
            pc_2 = self.program_configs.prefill_mlp_w2_prg_config(seq_len)
            pc_3 = self.program_configs.prefill_mlp_w1_w3_prg_config(seq_len)
            memory_config = ttnn.DRAM_MEMORY_CONFIG

        # --- STAGE 1: W1/W3 Linear ---
        w1_out = ttnn.linear(
            x,
            self.w1,
            dtype=ttnn.bfloat8_b,
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
        ttnn.deallocate(x)

        # --- STAGE 2: CCL after W1/W3 (dim-dependent path) ---
        if self.dim == 8192 or mode == "prefill":
            # Path A: reduce_scatter (for large dim or prefill)
            input_mem_cfg = w1_out.memory_config()
            cluster_axis = 1

            w1_out = ttnn.experimental.reduce_scatter_minimal_async(
                w1_out,
                persistent_output_buffers=None,
                dim=3,
                multi_device_global_semaphore=self.tt_ccl.get_and_cycle_rs_semaphore_handles(cluster_axis),
                barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis),
                num_links=self.config.num_reduce_scatter_links,
                cluster_axis=cluster_axis,
                memory_config=self.program_configs.ff1_out_reduce_scatter_memcfg if mode == "decode" else None,
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
                multi_device_global_semaphore=self.tt_ccl.get_and_cycle_rs_semaphore_handles(cluster_axis),
                barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis),
                num_links=1,
                cluster_axis=cluster_axis,
                memory_config=self.program_configs.ff1_out_reduce_scatter_memcfg if mode == "decode" else None,
                intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=ttnn.Topology.Linear,
                chunks_per_sync=10,
                num_workers_per_link=2,
                num_buffers_per_channel=2,
            )
            use_all_gather = True
        else:
            # Path B: all_reduce (for smaller dim in decode)
            w1_out = tt_all_reduce(
                w1_out,
                self.mesh_device,
                self.tt_ccl,
                cluster_axis=1,
                num_all_gather_links=2,
                sharded=True,  # decode mode
                topology=self.ccl_topology,
                memory_config=self.program_configs.ff1_out_gathered_memcfg,
            )
            w3_out = tt_all_reduce(
                w3_out,
                self.mesh_device,
                self.tt_ccl,
                cluster_axis=1,
                num_all_gather_links=2,
                sharded=True,  # decode mode
                topology=self.ccl_topology,
                memory_config=self.program_configs.ff1_out_gathered_memcfg,
            )
            use_all_gather = False

        # --- STAGE 3: Activation + Multiply ---
        w2_in = ttnn.mul(
            w1_out,
            w3_out,
            input_tensor_a_activations=[self.activation_type],
            dtype=activation_dtype or ttnn.bfloat8_b,
            memory_config=w1_out.memory_config(),
        )

        # --- STAGE 3.5: No reshard for TG (skip) ---

        ttnn.deallocate(w3_out)
        ttnn.deallocate(w1_out)

        # --- STAGE 4: All-gather before W2 (if we used reduce_scatter) ---
        if use_all_gather:
            cluster_axis = 1
            w2_in = ttnn.experimental.all_gather_async(
                w2_in,
                persistent_output_buffer=None,
                dim=3,
                multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis),
                num_links=2,
                cluster_axis=1,
                topology=ttnn.Topology.Linear,
                memory_config=input_mem_cfg,
                barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis),
                chunks_per_sync=10,
                num_workers_per_link=2,
                num_buffers_per_channel=2,
            )
            if mode == "decode":
                w2_in = ttnn.to_memory_config(w2_in, ttnn.L1_MEMORY_CONFIG)

        # --- STAGE 5: W2 Linear ---
        w2_out = ttnn.linear(
            w2_in,
            self.w2,
            compute_kernel_config=li_ff2_compute_kernel_cfg,
            dtype=self.config.ccl_dtype,
            program_config=pc_2,
            memory_config=memory_config,
            core_grid=None,
        )
        ttnn.deallocate(w2_in)

        # --- STAGE 6: Final All-Reduce ---
        w2_out_reduced = tt_all_reduce(
            w2_out,
            self.mesh_device,
            self.tt_ccl,
            cluster_axis=0,
            dim=0 if self.dim < 8192 else 3,
            num_reduce_scatter_links=self.config.num_reduce_scatter_links,
            num_all_gather_links=self.config.num_all_gather_links,
            sharded=(mode == "decode"),
            memory_config=(
                self.program_configs.ff2_out_reduce_scatter_memcfg if mode == "decode" else ttnn.DRAM_MEMORY_CONFIG
            ),
            dtype=self.config.ccl_dtype,
            use_composite=(self.dim == 8192),
            topology=self.ccl_topology,
        )

        # --- STAGE 7: Reshape + Final memory config ---
        original_shape = w2_out_reduced.shape
        w2_out_reduced = ttnn.reshape(
            w2_out_reduced, (1, 1, original_shape[-4] * original_shape[-3] * original_shape[-2], original_shape[-1])
        )
        if mode == "decode":
            w2_out_reduced = ttnn.to_memory_config(w2_out_reduced, self.program_configs.sharded_attn_input_memcfg)

        return w2_out_reduced
