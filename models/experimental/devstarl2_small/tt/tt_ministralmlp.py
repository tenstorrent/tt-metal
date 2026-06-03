# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
# Ministral3 SwiGLU FFN (same w1/w3/w2 layout as tt_transformers MLP).

from __future__ import annotations

import os

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.experimental.devstarl2_small.devstral_utils.dram_sharded_matmul import width_sharded_l1_memcfg
from models.experimental.devstarl2_small.tt.tt_ministralrmsnorm import ministral_prefill_block_shard_mem_cfg
from models.tt_transformers.tt.ccl import tt_all_reduce
from models.tt_transformers.tt.common import Mode, pad_to_size
from models.tt_transformers.tt.model_config import OpGroup, TensorGroup


def _ff1_shard_n(args) -> int:
    return int(args.hidden_dim) // int(args.num_devices)


_TILE = 32
_FF1_1D_GRID_X = 8
_FF1_1D_GRID_Y = 4


def _padded_seq_len(seq_len: int) -> int:
    return ((int(seq_len) + _TILE - 1) // _TILE) * _TILE


def _ff1_input_block_sharding_enabled(args, seq_len: int, full_seq_len: int) -> bool:
    """Shard FF1/FF3 activations for short prefill (128×5120×8192)."""
    if int(full_seq_len) != int(seq_len):
        return False
    return int(seq_len) <= 128 and int(args.dim) == 5120


def _ff1_matmul_grid(args, seq_len: int, mesh_device) -> ttnn.CoreGrid:
    if _ff1_linear_sweep_fits_device(mesh_device) and int(seq_len) <= 128:
        return ttnn.CoreGrid(y=4, x=8)
    row_tiles = _padded_seq_len(seq_len) // _TILE
    col_tiles = int(args.dim) // _TILE
    rows, cols = args.find_prefill_grid(row_tiles, col_tiles)
    return ttnn.CoreGrid(y=rows, x=cols)


def _ff1_ws_input_mem_cfg(args, seq_len: int) -> ttnn.MemoryConfig:
    m_tiles = _padded_seq_len(seq_len) // _TILE
    k_tiles = int(args.dim) // _TILE
    return width_sharded_l1_memcfg(m_tiles, k_tiles, _FF1_1D_GRID_X, _FF1_1D_GRID_Y)


def _ff1_ws_output_mem_cfg(args, seq_len: int) -> ttnn.MemoryConfig:
    m_tiles = _padded_seq_len(seq_len) // _TILE
    n_tiles = _ff1_shard_n(args) // _TILE
    return width_sharded_l1_memcfg(m_tiles, n_tiles, _FF1_1D_GRID_X, _FF1_1D_GRID_Y)


def _ff2_ws_input_mem_cfg(args, seq_len: int) -> ttnn.MemoryConfig:
    m_tiles = _padded_seq_len(seq_len) // _TILE
    k_tiles = _ff1_shard_n(args) // _TILE
    return width_sharded_l1_memcfg(m_tiles, k_tiles, _FF1_1D_GRID_X, _FF1_1D_GRID_Y)


def _ff2_ws_output_mem_cfg(args, seq_len: int) -> ttnn.MemoryConfig:
    m_tiles = _padded_seq_len(seq_len) // _TILE
    n_tiles = int(args.dim) // _TILE
    return width_sharded_l1_memcfg(m_tiles, n_tiles, _FF1_1D_GRID_X, _FF1_1D_GRID_Y)


def _use_1d_mlp_dram_weights(args) -> bool:
    """Sweep 1D_ws/dram/ws uses interleaved DRAM weights (not DRAM width-sharded)."""
    return not args.is_galaxy and int(args.dim) == 5120 and int(args.hidden_dim) // int(args.num_devices) == 8192


def _prepare_ff1_ws_input(x: ttnn.Tensor, ws_mem_cfg: ttnn.MemoryConfig) -> ttnn.Tensor:
    mc = x.memory_config()
    if mc.is_sharded() and mc == ws_mem_cfg:
        return x
    if mc.is_sharded():
        x = ttnn.sharded_to_interleaved(x, ttnn.DRAM_MEMORY_CONFIG)
    return ttnn.interleaved_to_sharded(x, ws_mem_cfg)


def _ff1_linear_sweep_fits_device(mesh_device) -> bool:
    grid = mesh_device.compute_with_storage_grid_size()
    return int(grid.x) >= 8 and int(grid.y) >= 4


def _ff1_linear_sweep_enabled(args, seq_len: int, full_seq_len: int, mesh_device) -> bool:
    """Sweep winner for 128×5120×8192 prefill FF1/FF3 (test_linear_128x5120x8192_sweep)."""
    default = os.environ.get("TT_MINISTRAL3_SHORT_PREFILL_L1_WIDTH_MM", "1")
    if os.environ.get("TT_MINISTRAL3_FF1_LINEAR_SWEEP", default).strip().lower() in ("0", "false", "no"):
        return False
    # Sweep used [1,1,M,K]; batched reshape [1,B,128,H] (demo prefill L>128) needs a different grid.
    if int(full_seq_len) != int(seq_len):
        return False
    if not _ff1_linear_sweep_fits_device(mesh_device):
        return False
    return int(seq_len) <= 128 and int(args.dim) == 5120 and _ff1_shard_n(args) == 8192


def _ff1_linear_sweep_program_config() -> ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig:
    # 1D_ws/dram/ws 8x4 w5: ~147us vs Tracy 128×5120×8192 ~192us (test_matmul_128x5120x8192_sweep).
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(_FF1_1D_GRID_X, _FF1_1D_GRID_Y),
        in0_block_w=5,
        out_subblock_h=1,
        out_subblock_w=4,
        per_core_M=4,
        per_core_N=8,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )


def _ff2_linear_sweep_enabled(args, seq_len: int, full_seq_len: int, mesh_device) -> bool:
    """Sweep winner for 128×8192×5120 prefill FF2 (test_linear_128x8192x5120_sweep)."""
    default = os.environ.get("TT_MINISTRAL3_SHORT_PREFILL_L1_WIDTH_MM", "1")
    if os.environ.get("TT_MINISTRAL3_FF2_LINEAR_SWEEP", default).strip().lower() in ("0", "false", "no"):
        return False
    if int(full_seq_len) != int(seq_len):
        return False
    if not _ff1_linear_sweep_fits_device(mesh_device):
        return False
    return int(seq_len) <= 128 and int(args.dim) == 5120 and _ff1_shard_n(args) == 8192


def _ff2_linear_sweep_program_config() -> ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig:
    # 1D_ws/dram/ws 8x4 w4 (Kt=256, kt/core=8): 128×8192×5120 prefill FF2.
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(_FF1_1D_GRID_X, _FF1_1D_GRID_Y),
        in0_block_w=4,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=4,
        per_core_N=5,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )


class TtMinistralMLP(LightweightModule):
    _PREFILL_MLP_M_CAP = 128  # non-Galaxy prefill M chunk cap

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
        super().__init__()

        self.mesh_device = mesh_device
        self.tt_ccl = tt_ccl
        self.args = args
        self.dim = args.dim
        self.model_config = model_config
        self.layer_num = layer_num

        self.prefetcher = prefetcher

        state_dict_prefix = state_dict_prefix or args.get_state_dict_prefix("MLP", layer_num)

        def torch_weight(name):
            return torch.transpose(state_dict[f"{state_dict_prefix}.{name}.weight"], -2, -1)

        def pad_hidden_dim(tensor, dim):
            return pad_to_size(tensor, dim=dim, size=args.hidden_dim)

        hidden_dim_string = f".hidden_dim_{args.hidden_dim}" if args.hidden_dim != args.unpadded_hidden_dim else ""
        w1_w3_mem_config = args.create_dram_sharded_mem_config(args.dim, args.hidden_dim // args.num_devices)
        w2_mem_config = args.create_dram_sharded_mem_config(args.hidden_dim // args.num_devices, args.dim)
        use_1d_dram_weights = _use_1d_mlp_dram_weights(args)

        if args.dummy_weights:

            def cache_name(_name, _interleaved_dram=False):
                return None

        else:

            def _weight_cache_suffix(tensor_name: str, interleaved_dram: bool) -> str:
                if use_1d_dram_weights and interleaved_dram:
                    return f"{hidden_dim_string}_dram_il"
                return hidden_dim_string

            def cache_name(name, interleaved_dram=False):
                return weight_cache_path / f"{state_dict_prefix}.{name}{_weight_cache_suffix(name, interleaved_dram)}"

        def as_sharded_tensor(name, type, dims, *, interleaved_dram: bool | None = None):
            raw_weight = torch_weight(name[:2])
            padded_weight = pad_hidden_dim(raw_weight, dims[0] if args.is_galaxy else dims[-1])
            torch_tensor = padded_weight.unsqueeze(0).unsqueeze(0)
            if interleaved_dram is None:
                use_interleaved_dram = args.is_galaxy or (use_1d_dram_weights and "w2" not in name)
            else:
                use_interleaved_dram = interleaved_dram

            result = ttnn.as_tensor(
                torch_tensor,
                dtype=type,
                device=self.mesh_device,
                mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=dims, mesh_shape=args.cluster_shape),
                layout=ttnn.TILE_LAYOUT,
                memory_config=(
                    ttnn.DRAM_MEMORY_CONFIG
                    if use_interleaved_dram
                    else (w2_mem_config if "w2" in name else w1_w3_mem_config)
                ),
                cache_file_name=cache_name(name, use_interleaved_dram),
            )
            return result

        w1_dims = (-1, -2) if args.is_galaxy else (-2, -1)
        w2_dims = (-2, -1) if args.is_galaxy else (-1, -2)

        layer_num = max(layer_num, 0)

        use_prefetcher = prefetcher is not None

        self.decoders_optimizations = self.args.decoders_optimizations

        ff1_3_dtype = self.decoders_optimizations.get_tensor_dtype(
            decoder_id=layer_num, tensor=TensorGroup.FF1_FF3, prefetcher=use_prefetcher
        )
        ff2_dtype = self.decoders_optimizations.get_tensor_dtype(
            decoder_id=layer_num, tensor=TensorGroup.FF2, prefetcher=use_prefetcher
        )

        self.w1 = as_sharded_tensor("w1_sharded", ff1_3_dtype, dims=w1_dims)
        # Devstral 1D path: width-sharded W2 for decode linear + interleaved W2 for 128-token FF2 prefill sweep.
        if use_1d_dram_weights and not args.is_galaxy:
            self.w2 = as_sharded_tensor("w2_sharded", ff2_dtype, dims=w2_dims, interleaved_dram=False)
            self.w2_prefill_sweep = as_sharded_tensor("w2_sharded", ff2_dtype, dims=w2_dims, interleaved_dram=True)
        else:
            self.w2 = as_sharded_tensor("w2_sharded", ff2_dtype, dims=w2_dims)
            self.w2_prefill_sweep = self.w2
        self.w3 = as_sharded_tensor("w3_sharded", ff1_3_dtype, dims=w1_dims)

        self.activation_type = (
            args.mlp_activation_type if hasattr(args, "mlp_activation_type") else ttnn.UnaryOpType.SILU
        )

        if self.prefetcher is not None:

            def register_weights():
                self.prefetcher.insert_tensor(self.w1)
                self.prefetcher.insert_tensor(self.w3)
                self.prefetcher.insert_tensor(self.w2)

            self.prefetcher.register_callback(register_weights)

    def get_prefill_ff1_input_mem_config(self, full_seq_len: int) -> ttnn.MemoryConfig | None:
        cfg_seq = int(full_seq_len)
        max_chunk = min(int(self.args.prefill_len_cutoff), int(self._PREFILL_MLP_M_CAP))
        max_chunk = max(max_chunk, 1)
        chunk = max_chunk
        while chunk > 1 and cfg_seq % chunk != 0:
            chunk -= 1
        if cfg_seq > chunk:
            cfg_seq = chunk
        if not _ff1_input_block_sharding_enabled(self.args, cfg_seq, full_seq_len):
            return None
        if not _ff1_linear_sweep_enabled(self.args, cfg_seq, full_seq_len, self.mesh_device):
            return None
        # RMSNorm stays BLOCK-sharded; MLP converts to WIDTH-sharded L1 for 1D matmul.
        return ministral_prefill_block_shard_mem_cfg(self.args, cfg_seq)

    def forward(self, x: ttnn.Tensor, mode: Mode) -> ttnn.Tensor:
        full_seq_len = int(x.shape[-2])
        TG = self.args.is_galaxy
        activation_dtype = self.decoders_optimizations.get_tensor_dtype(
            decoder_id=self.layer_num, tensor=TensorGroup.ACTIVATION
        )

        li_ff1_3_compute_kernel_cfg = self.decoders_optimizations.get_math_fidelity(
            decoder_id=self.layer_num, op=OpGroup.LI_FF1_FF3, configuration=self.args
        )

        cfg_seq = full_seq_len
        if mode == Mode.PREFILL:
            max_chunk = min(int(self.args.prefill_len_cutoff), int(self._PREFILL_MLP_M_CAP))
            max_chunk = max(max_chunk, 1)
            chunk = max_chunk
            while chunk > 1 and full_seq_len % chunk != 0:
                chunk -= 1
            if full_seq_len > chunk:
                x = ttnn.reshape(x, [1, full_seq_len // chunk, chunk, -1])
                cfg_seq = chunk

        ff1_x = x
        ff1_input_sharded = False

        use_ff2_sweep = (
            mode == Mode.PREFILL
            and not TG
            and cfg_seq <= 128
            and _ff2_linear_sweep_enabled(self.args, cfg_seq, full_seq_len, self.mesh_device)
        )
        pc_2 = (
            _ff2_linear_sweep_program_config()
            if use_ff2_sweep
            else self.args.get_mlp_ff2_prg_config(mode, cfg_seq, self.prefetcher)
        )

        if (
            mode == Mode.PREFILL
            and not TG
            and _ff1_input_block_sharding_enabled(self.args, cfg_seq, full_seq_len)
            and _ff1_linear_sweep_enabled(self.args, cfg_seq, full_seq_len, self.mesh_device)
        ):
            ff1_in_mem = _ff1_ws_input_mem_cfg(self.args, cfg_seq)
            ff1_x = _prepare_ff1_ws_input(x, ff1_in_mem)
            ff1_input_sharded = ff1_x is not x

        if mode == Mode.PREFILL and not TG:
            if _ff1_linear_sweep_enabled(self.args, cfg_seq, full_seq_len, self.mesh_device):
                pc_ff13 = _ff1_linear_sweep_program_config()
                mem_ff13 = _ff1_ws_output_mem_cfg(self.args, cfg_seq)
                w1_out = ttnn.matmul(
                    ff1_x,
                    self.w1,
                    program_config=pc_ff13,
                    memory_config=mem_ff13,
                    compute_kernel_config=li_ff1_3_compute_kernel_cfg,
                    dtype=ttnn.bfloat8_b,
                )
                w3_out = ttnn.matmul(
                    ff1_x,
                    self.w3,
                    program_config=pc_ff13,
                    memory_config=mem_ff13,
                    compute_kernel_config=li_ff1_3_compute_kernel_cfg,
                    dtype=ttnn.bfloat8_b,
                )
            else:
                grid = self.args.mlp1_3_grid(cfg_seq)
                mmc_ff13 = ttnn.MinimalMatmulConfig(
                    M_block_size=8,
                    K_block_size=8,
                    N_block_size=8,
                    compute_with_storage_grid_size=ttnn.CoreCoord(grid[0], grid[1]),
                )
                w1_out = ttnn.experimental.minimal_matmul(
                    x,
                    self.w1,
                    compute_kernel_config=li_ff1_3_compute_kernel_cfg,
                    config=mmc_ff13,
                    dtype=ttnn.bfloat8_b,
                )
                w3_out = ttnn.experimental.minimal_matmul(
                    x,
                    self.w3,
                    compute_kernel_config=li_ff1_3_compute_kernel_cfg,
                    config=mmc_ff13,
                    dtype=ttnn.bfloat8_b,
                )
        elif mode == Mode.DECODE and not TG and self.prefetcher is None:
            # decode FF1/FF3 via minimal_matmul (DRAM-sharded weights).
            # Config tuned by tests/matmul/test_decode_ff13_matmul_sweep.py: at
            # decode M=32 (1 tile) the M axis maps to grid.y, so the old 8x8 grid
            # padded M across 8 core-rows and wasted 7/8 of them. grid 8x2 +
            # M_block_size=1 + a larger K chunk recover ~1.12x (382us -> ~342us).
            x_dram = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
            mmc_ff13 = ttnn.MinimalMatmulConfig(
                M_block_size=1,
                K_block_size=16,
                N_block_size=8,
                subblock_h=1,
                subblock_w=4,
                compute_with_storage_grid_size=ttnn.CoreCoord(8, 2),
            )
            w1_out = ttnn.experimental.minimal_matmul(
                x_dram,
                self.w1,
                compute_kernel_config=li_ff1_3_compute_kernel_cfg,
                config=mmc_ff13,
                dtype=ttnn.bfloat8_b,
            )
            w3_out = ttnn.experimental.minimal_matmul(
                x_dram,
                self.w3,
                compute_kernel_config=li_ff1_3_compute_kernel_cfg,
                config=mmc_ff13,
                dtype=ttnn.bfloat8_b,
            )
            ttnn.deallocate(x_dram)
        else:
            pc_1 = self.args.get_mlp_ff1_3_prg_config(mode, cfg_seq, self.prefetcher)
            pc_3 = self.args.get_mlp_ff1_3_prg_config(mode, cfg_seq, self.prefetcher)
            w1_out = ttnn.linear(
                ff1_x,
                self.w1,
                dtype=ttnn.bfloat8_b if TG else activation_dtype or ttnn.bfloat16,
                core_grid=None,
                compute_kernel_config=li_ff1_3_compute_kernel_cfg,
                program_config=pc_1,
                memory_config=self.args.get_mlp_ff1_3_mem_config(mode, self.prefetcher),
                global_cb=self.prefetcher.global_cb if self.prefetcher is not None and mode == Mode.DECODE else None,
                sub_device_id=self.prefetcher.worker_sub_device_id
                if self.prefetcher is not None and mode == Mode.DECODE
                else None,
            )
            w3_out = ttnn.linear(
                ff1_x,
                self.w3,
                dtype=ttnn.bfloat8_b if TG else activation_dtype or ttnn.bfloat16,
                core_grid=None,
                compute_kernel_config=li_ff1_3_compute_kernel_cfg,
                program_config=pc_3,
                memory_config=self.args.get_mlp_ff1_3_mem_config(mode, self.prefetcher),
                global_cb=self.prefetcher.global_cb if self.prefetcher is not None and mode == Mode.DECODE else None,
                sub_device_id=self.prefetcher.worker_sub_device_id
                if self.prefetcher is not None and mode == Mode.DECODE
                else None,
            )
        if ff1_input_sharded:
            ttnn.deallocate(ff1_x)
        ttnn.deallocate(x)

        if TG:
            if self.dim == 8192 or mode == Mode.PREFILL:
                input_mem_cfg = w1_out.memory_config()

                cluster_axis = 1
                w1_out = ttnn.experimental.reduce_scatter_minimal_async(
                    w1_out,
                    persistent_output_buffers=None,
                    dim=3,
                    multi_device_global_semaphore=self.tt_ccl.get_and_cycle_rs_semaphore_handles(cluster_axis),
                    barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis),
                    num_links=self.tt_ccl.get_num_links(cluster_axis),
                    cluster_axis=cluster_axis,
                    memory_config=self.model_config["FF1_OUT_REDUCE_SCATTER_MEMCFG"] if mode == Mode.DECODE else None,
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
                    memory_config=self.model_config["FF1_OUT_REDUCE_SCATTER_MEMCFG"] if mode == Mode.DECODE else None,
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
                    sharded=True if mode == Mode.DECODE else False,
                    topology=self.args.ccl_topology(),
                    memory_config=self.model_config["FF1_OUT_GATHERED_MEMCFG"] if mode == Mode.DECODE else None,
                )
                w3_out = tt_all_reduce(
                    w3_out,
                    self.mesh_device,
                    self.tt_ccl,
                    cluster_axis=1,
                    num_all_gather_links=2,
                    sharded=True if mode == Mode.DECODE else False,
                    topology=self.args.ccl_topology(),
                    memory_config=self.model_config["FF1_OUT_GATHERED_MEMCFG"] if mode == Mode.DECODE else None,
                )

        w2_in = ttnn.mul(
            w1_out,
            w3_out,
            input_tensor_a_activations=[self.activation_type],
            dtype=activation_dtype or ttnn.bfloat8_b,
            memory_config=w1_out.memory_config(),
        )

        if mode == Mode.DECODE and not TG and self.prefetcher is None:
            w2_in = ttnn.to_memory_config(w2_in, self.args.get_mlp_binary_mult_mem_config(mode))

        ttnn.deallocate(w3_out)
        ttnn.deallocate(w1_out)

        if TG and (self.dim == 8192 or mode == Mode.PREFILL):
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

            if mode == Mode.DECODE:
                w2_in = ttnn.to_memory_config(w2_in, ttnn.L1_MEMORY_CONFIG)

        li_ff2_compute_kernel_cfg = self.decoders_optimizations.get_math_fidelity(
            decoder_id=self.layer_num, op=OpGroup.LI_FF2, configuration=self.args
        )

        if cfg_seq > 128 and mode != Mode.DECODE and not use_ff2_sweep:
            w2_out = ttnn.experimental.minimal_matmul(
                w2_in,
                self.w2,
                compute_kernel_config=li_ff2_compute_kernel_cfg,
                config=pc_2,
            )
        else:
            if use_ff2_sweep:
                w2_in = _prepare_ff1_ws_input(w2_in, _ff2_ws_input_mem_cfg(self.args, cfg_seq))
                mem_ff2 = _ff2_ws_output_mem_cfg(self.args, cfg_seq)
                w2_out = ttnn.matmul(
                    w2_in,
                    self.w2_prefill_sweep,
                    program_config=pc_2,
                    memory_config=mem_ff2,
                    compute_kernel_config=li_ff2_compute_kernel_cfg,
                    dtype=ttnn.bfloat8_b,
                )
                w2_out = ttnn.sharded_to_interleaved(w2_out, ttnn.DRAM_MEMORY_CONFIG)
            else:
                w2_out = ttnn.linear(
                    w2_in,
                    self.w2,
                    compute_kernel_config=li_ff2_compute_kernel_cfg,
                    dtype=self.args.ccl_dtype if TG else activation_dtype or ttnn.bfloat16,
                    program_config=pc_2,
                    memory_config=self.args.get_mlp_ff2_mem_config(mode, self.prefetcher),
                    core_grid=None,
                    global_cb=self.prefetcher.global_cb
                    if self.prefetcher is not None and mode == Mode.DECODE
                    else None,
                    sub_device_id=self.prefetcher.worker_sub_device_id
                    if self.prefetcher is not None and mode == Mode.DECODE
                    else None,
                )
        ttnn.deallocate(w2_in)

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


__all__ = ["TtMinistralMLP"]
