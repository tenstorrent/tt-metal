# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
GLM KV Cache Branch fused operation (adapted from DSv3 KVCacheBranch).

Fuses: DKV Matmul + Gather + RMSNorm + RoPE into a single dispatch.
All intermediate CBs use TILE_1x32 format (1 row x 32 cols).
RMSNorm uses REDUCE_ROW (not REDUCE_SCALAR) — natural scalar on 1-row tiles.

Kernel reads input x, cos, sin directly from DRAM via NOC.
Kernel writes nope+rope output directly to a DRAM tensor via NOC.
Only the matmul weight reshard remains as a Python-side op.
"""

import struct

import ttnn
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import (
    PerCoreCompileTimeDescriptor,
    UnifiedCompileTimeCoreDescriptor,
    UnifiedKernelDescriptor,
)


def _bf16_to_uint16(value: float) -> int:
    """Convert a Python float to bf16, return as uint16."""
    f32 = struct.pack("f", value)
    # BF16 = upper 16 bits of float32
    return struct.unpack("H", f32[2:4])[0]


class GLMKVCacheBranch:
    """GLM KV Cache Branch: DKV matmul + gather + RMSNorm + RoPE in one dispatch."""

    @staticmethod
    def op(
        input_tensor,
        dkv_matmul_weights_tensor,
        gamma_tensor,
        cos_tensor,
        sin_tensor,
        trans_mat_tensor,
        kvpe_output_tensor,
        rope_core_grid,
        epsilon: float = 1e-5,
    ):
        data_format = input_tensor.dtype
        device = input_tensor.device()

        # DKV Matmul weight grid = full 18-core grid
        dkv_matmul_weights_mc = dkv_matmul_weights_tensor.memory_config()
        dkv_matmul_weights_core_grid = dkv_matmul_weights_mc.shard_spec.grid
        dkv_matmul_weights_tile = dkv_matmul_weights_tensor.get_tile()
        dkv_matmul_weights_shard_width = dkv_matmul_weights_mc.shard_spec.shape[1]
        dkv_matmul_out_w = dkv_matmul_weights_shard_width // dkv_matmul_weights_tile.tile_shape[1]

        # Core grids derived from weight grid and rope_core_grid param
        input_core_grid = dkv_matmul_weights_core_grid
        krope_core_grid = rope_core_grid

        gather_noc0_receiver_semaphore_id = 2
        gather_noc1_receiver_semaphore_id = 3

        hidden_size = int(input_tensor.shape[-1])
        dkv_matmul_k_num_tiles = hidden_size // 32

        TILE_1x32 = ttnn.Tile((1, 32))
        tile_1x32_size = TILE_1x32.get_tile_size(data_format)

        # CB indices
        cos_cb = 0
        sin_cb = 1
        trans_mat_cb = 2
        rotated_input_interm_cb = 3
        cos_interm_cb = 4
        sin_interm_cb = 5
        dkv_matmul_input_cb = 6
        dkv_matmul_output_cb = 7
        dkv_matmul_weights_cb = 8
        kv_rmsnorm_input_cb = 9  # gather destination / rmsnorm input (intermediate)
        kv_rmsnorm_gamma_cb = 10  # gamma weights (sharded)
        kv_rmsnorm_output_cb = 11  # rmsnorm output (intermediate, NCRISC writes to DRAM)
        k_rope_output_cb = 12
        kv_rmsnorm_x2_cb = 13  # intermediate: x^2
        kv_rmsnorm_var_cb = 14  # intermediate: variance
        kv_rmsnorm_scaler_cb = 15  # intermediate: reduce scaler (1/N)
        kv_rmsnorm_eps_cb = 16  # intermediate: epsilon

        nope_num_tiles = 512 // 32  # 16 tiles of TILE_1x32
        rope_dim = int(cos_tensor.shape[-1])
        num_rope_cores = krope_core_grid.num_cores()

        # ================================================================
        # Compile-time args
        # ================================================================
        dkv_matmul_ncrisc_args = [
            ("dkv_matmul_in0", dkv_matmul_input_cb),
            ("dkv_matmul_in1", dkv_matmul_weights_cb),
            ("dkv_matmul_k_num_tiles", dkv_matmul_k_num_tiles),
            ("dkv_matmul_out_w_per_core", dkv_matmul_out_w),
        ]
        dkv_matmul_trisc_args = [
            ("dkv_matmul_in0", dkv_matmul_input_cb),
            ("dkv_matmul_in1", dkv_matmul_weights_cb),
            ("dkv_matmul_out", dkv_matmul_output_cb),
            ("dkv_matmul_k_num_tiles", dkv_matmul_k_num_tiles),
            ("dkv_matmul_out_w_per_core", dkv_matmul_out_w),
        ]

        # ================================================================
        # Gather: knope matmul cores (senders) -> rmsnorm core (receiver)
        # ================================================================
        rms_core_grid = gamma_tensor.memory_config().shard_spec.grid
        rms_core = rms_core_grid.ranges()[0].start
        dkv_gather_sender_grid = dkv_matmul_weights_core_grid.subtract(krope_core_grid)
        dkv_gather_dest_noc_core = device.worker_core_from_logical_core(rms_core)

        dkv_gather_sender_cores_list = ttnn.corerange_to_cores(dkv_gather_sender_grid, row_wise=True)
        dkv_gather_num_senders = len(dkv_gather_sender_cores_list)

        dkv_gather_src_num_pages = dkv_matmul_out_w
        dkv_gather_data_size_bytes = dkv_gather_src_num_pages * tile_1x32_size
        dkv_gather_dst_total_pages = dkv_gather_num_senders * dkv_gather_src_num_pages

        dkv_gather_sender_idx_descriptor = PerCoreCompileTimeDescriptor(
            named_compile_time_arg="dkv_gather_sender_idx",
            core_values=[(core, idx) for idx, core in enumerate(dkv_gather_sender_cores_list)],
            other_value=0,
        )

        dkv_gather_sender_args = [
            ("dkv_gather_dest_noc_x", dkv_gather_dest_noc_core.x),
            ("dkv_gather_dest_noc_y", dkv_gather_dest_noc_core.y),
            ("dkv_gather_data_size_bytes", dkv_gather_data_size_bytes),
            ("dkv_gather_receiver_semaphore_id", gather_noc0_receiver_semaphore_id),
            ("dkv_gather_src_cb", dkv_matmul_output_cb),
            ("dkv_gather_src_num_pages", dkv_gather_src_num_pages),
            ("dkv_gather_sender_grid_start_x", 0),
            ("dkv_gather_sender_grid_start_y", 0),
            ("dkv_gather_sender_grid_end_x", 0),
            ("dkv_gather_sender_grid_end_y", 0),
            ("dkv_gather_row_major", 1),
            ("dkv_gather_dst_cb", kv_rmsnorm_input_cb),
        ]
        dkv_gather_receiver_args = [
            ("dkv_gather_noc0_num_senders", dkv_gather_num_senders),
            ("dkv_gather_noc1_num_senders", 0),
            ("dkv_gather_noc0_receiver_semaphore_id", gather_noc0_receiver_semaphore_id),
            ("dkv_gather_noc1_receiver_semaphore_id", gather_noc1_receiver_semaphore_id),
            ("dkv_gather_dst_cb", kv_rmsnorm_input_cb),
            ("dkv_gather_dst_num_pages", dkv_gather_dst_total_pages),
        ]

        # RMSNorm compile-time args
        kv_rmsnorm_common_args = [
            ("kv_rmsnorm_gamma_cb", kv_rmsnorm_gamma_cb),
            ("kv_rmsnorm_output_cb", kv_rmsnorm_output_cb),
            ("kv_rmsnorm_num_tiles", nope_num_tiles),
        ]

        # RoPE compile-time args
        krope_brisc_args = [("k_rope_output_cb", k_rope_output_cb), ("Wt", 1), ("Ht", 1)]
        krope_ncrisc_args = [
            ("in_cb", dkv_matmul_output_cb),
            ("cos_cb", cos_cb),
            ("sin_cb", sin_cb),
            ("trans_mat_cb", trans_mat_cb),
            ("k_rope_output_cb", k_rope_output_cb),
            ("Wt", 1),
            ("Ht", 1),
        ]
        krope_trisc_args = [
            ("in_cb", dkv_matmul_output_cb),
            ("cos_cb", cos_cb),
            ("sin_cb", sin_cb),
            ("trans_mat_cb", trans_mat_cb),
            ("rotated_in_interm_cb", rotated_input_interm_cb),
            ("cos_interm_cb", cos_interm_cb),
            ("sin_interm_cb", sin_interm_cb),
            ("out_cb", k_rope_output_cb),
            ("Wt", 1),
            ("Ht", 1),
        ]

        # DRAM I/O compile-time args for NCRISC
        kvpe_dim = int(kvpe_output_tensor.shape[-1])
        nope_bytes = nope_num_tiles * tile_1x32_size
        cos_sin_dram_page_size = rope_dim * 2  # bf16
        kvpe_out_dram_page_size = kvpe_dim * 2  # bf16

        dram_io_ncrisc_args = [
            ("cos_sin_dram_page_size", cos_sin_dram_page_size),
            ("kvpe_out_dram_page_size", kvpe_out_dram_page_size),
            ("kvpe_out_nope_bytes", nope_bytes),
        ]

        # Per-core tile offset for rope cores (0 for core 0, 1 for core 1)
        rope_cores_list = ttnn.corerange_to_cores(krope_core_grid, row_wise=True)
        krope_tile_offset_descriptor = PerCoreCompileTimeDescriptor(
            named_compile_time_arg="krope_core_tile_offset",
            core_values=[(core, idx) for idx, core in enumerate(rope_cores_list)],
            other_value=0,
        )

        # ================================================================
        # Circular Buffer descriptors
        # ================================================================
        # Input CB: intermediate (kernel reads x from DRAM into this CB)
        input_tiles_per_core = hidden_size // 32
        dkv_matmul_input_cb_format = ttnn.CBFormatDescriptor(
            buffer_index=dkv_matmul_input_cb,
            data_format=data_format,
            page_size=tile_1x32_size,
            tile=ttnn.TileDescriptor(TILE_1x32),
        )
        dkv_matmul_input_cb_descriptor = ttnn.CBDescriptor(
            total_size=input_tiles_per_core * tile_1x32_size,
            core_ranges=input_core_grid,
            format_descriptors=[dkv_matmul_input_cb_format],
        )

        dkv_matmul_output_page_size = tile_1x32_size
        dkv_matmul_output_cb_format = ttnn.CBFormatDescriptor(
            buffer_index=dkv_matmul_output_cb,
            data_format=data_format,
            page_size=dkv_matmul_output_page_size,
            tile=ttnn.TileDescriptor(TILE_1x32),
        )
        dkv_matmul_output_cb_descriptor = ttnn.CBDescriptor(
            total_size=dkv_matmul_output_page_size,
            core_ranges=dkv_matmul_weights_core_grid,
            format_descriptors=[dkv_matmul_output_cb_format],
        )

        dkv_matmul_weights_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
            dkv_matmul_weights_cb, dkv_matmul_weights_tensor
        )

        # ---- RMSNorm input CB (gather destination, intermediate on rmsnorm core) ----
        rmsnorm_input_cb_format = ttnn.CBFormatDescriptor(
            buffer_index=kv_rmsnorm_input_cb,
            data_format=data_format,
            page_size=tile_1x32_size,
            tile=ttnn.TileDescriptor(TILE_1x32),
        )
        rmsnorm_input_cb_on_rms = ttnn.CBDescriptor(
            total_size=nope_num_tiles * tile_1x32_size,
            core_ranges=rms_core_grid,
            format_descriptors=[rmsnorm_input_cb_format],
        )
        other_cores = input_core_grid.subtract(rms_core_grid)
        rmsnorm_input_cb_dummy = None
        if other_cores.num_cores() > 0:
            rmsnorm_input_cb_dummy = ttnn.CBDescriptor(
                total_size=nope_num_tiles * tile_1x32_size,
                core_ranges=other_cores,
                format_descriptors=[rmsnorm_input_cb_format],
            )

        # ---- RMSNorm gamma CB (sharded on rmsnorm core, backed by gamma_tensor) ----
        rmsnorm_gamma_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(kv_rmsnorm_gamma_cb, gamma_tensor)

        # ---- RMSNorm output CB (intermediate, NCRISC writes to DRAM) ----
        rmsnorm_output_cb_format = ttnn.CBFormatDescriptor(
            buffer_index=kv_rmsnorm_output_cb,
            data_format=data_format,
            page_size=tile_1x32_size,
            tile=ttnn.TileDescriptor(TILE_1x32),
        )
        rmsnorm_output_cb_descriptor = ttnn.CBDescriptor(
            total_size=nope_num_tiles * tile_1x32_size,
            core_ranges=rms_core_grid,
            format_descriptors=[rmsnorm_output_cb_format],
        )

        # ---- RMSNorm intermediate CBs ----
        def _make_rms_interm_cb(idx, num_tiles_alloc):
            fmt = ttnn.CBFormatDescriptor(
                buffer_index=idx,
                data_format=data_format,
                page_size=tile_1x32_size,
                tile=ttnn.TileDescriptor(TILE_1x32),
            )
            return ttnn.CBDescriptor(
                total_size=num_tiles_alloc * tile_1x32_size,
                core_ranges=rms_core_grid,
                format_descriptors=[fmt],
            )

        rmsnorm_x2_cb_descriptor = _make_rms_interm_cb(kv_rmsnorm_x2_cb, nope_num_tiles)
        rmsnorm_var_cb_descriptor = _make_rms_interm_cb(kv_rmsnorm_var_cb, 1)
        rmsnorm_scaler_cb_descriptor = _make_rms_interm_cb(kv_rmsnorm_scaler_cb, 1)
        rmsnorm_eps_cb_descriptor = _make_rms_interm_cb(kv_rmsnorm_eps_cb, 1)

        # ---- RoPE CBs (intermediate, kernel reads cos/sin from DRAM into these) ----
        cos_sin_tiles_per_core = 1  # Each rope core gets rope_dim / num_rope_cores / 32 = 1 tile
        cos_cb_format = ttnn.CBFormatDescriptor(
            buffer_index=cos_cb,
            data_format=data_format,
            page_size=tile_1x32_size,
            tile=ttnn.TileDescriptor(TILE_1x32),
        )
        cos_cb_descriptor = ttnn.CBDescriptor(
            total_size=cos_sin_tiles_per_core * tile_1x32_size,
            core_ranges=krope_core_grid,
            format_descriptors=[cos_cb_format],
        )
        sin_cb_format = ttnn.CBFormatDescriptor(
            buffer_index=sin_cb,
            data_format=data_format,
            page_size=tile_1x32_size,
            tile=ttnn.TileDescriptor(TILE_1x32),
        )
        sin_cb_descriptor = ttnn.CBDescriptor(
            total_size=cos_sin_tiles_per_core * tile_1x32_size,
            core_ranges=krope_core_grid,
            format_descriptors=[sin_cb_format],
        )
        trans_mat_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(trans_mat_cb, trans_mat_tensor)

        def _make_rope_interm_cb(idx, grid):
            fmt = ttnn.CBFormatDescriptor(
                buffer_index=idx,
                data_format=data_format,
                page_size=tile_1x32_size,
                tile=ttnn.TileDescriptor(TILE_1x32),
            )
            return ttnn.CBDescriptor(total_size=tile_1x32_size, core_ranges=grid, format_descriptors=[fmt])

        rotated_interm_cb_descriptor = _make_rope_interm_cb(rotated_input_interm_cb, krope_core_grid)
        cos_interm_cb_descriptor = _make_rope_interm_cb(cos_interm_cb, krope_core_grid)
        sin_interm_cb_descriptor = _make_rope_interm_cb(sin_interm_cb, krope_core_grid)

        # ---- RoPE output CB (intermediate, NCRISC writes to DRAM) ----
        k_rope_output_cb_format = ttnn.CBFormatDescriptor(
            buffer_index=k_rope_output_cb,
            data_format=data_format,
            page_size=tile_1x32_size,
            tile=ttnn.TileDescriptor(TILE_1x32),
        )
        k_rope_output_cb_descriptor = ttnn.CBDescriptor(
            total_size=cos_sin_tiles_per_core * tile_1x32_size,
            core_ranges=krope_core_grid,
            format_descriptors=[k_rope_output_cb_format],
        )

        # ================================================================
        # Semaphores
        # ================================================================
        gather_noc0_sem = ttnn.SemaphoreDescriptor(
            id=gather_noc0_receiver_semaphore_id,
            core_ranges=input_core_grid,
            initial_value=0,
        )
        gather_noc1_sem = ttnn.SemaphoreDescriptor(
            id=gather_noc1_receiver_semaphore_id,
            core_ranges=input_core_grid,
            initial_value=0,
        )

        # ================================================================
        # Runtime args
        # ================================================================
        scaler_bf16 = _bf16_to_uint16(1.0 / 512.0)  # 1/N where N=512 (kv_lora_rank)
        eps_bf16 = _bf16_to_uint16(epsilon)

        # NCRISC common runtime args: RMSNorm params (0-3) + DRAM addresses (4-7)
        ncrisc_rmsnorm_rt_args = [
            kv_rmsnorm_scaler_cb,  # arg 0: cb_scaler
            kv_rmsnorm_eps_cb,  # arg 1: cb_eps
            scaler_bf16,  # arg 2: scaler_packed
            eps_bf16,  # arg 3: eps_packed
            input_tensor.buffer_address(),  # arg 4: x DRAM address
            cos_tensor.buffer_address(),  # arg 5: cos DRAM address
            sin_tensor.buffer_address(),  # arg 6: sin DRAM address
            kvpe_output_tensor.buffer_address(),  # arg 7: kvpe output DRAM address
        ]
        trisc_rmsnorm_rt_args = [
            kv_rmsnorm_input_cb,  # arg 0: input_cb
            kv_rmsnorm_gamma_cb,  # arg 1: gamma_cb
            kv_rmsnorm_output_cb,  # arg 2: output_cb
            kv_rmsnorm_x2_cb,  # arg 3: cb_x2
            kv_rmsnorm_var_cb,  # arg 4: cb_var
            kv_rmsnorm_scaler_cb,  # arg 5: cb_scaler
            kv_rmsnorm_eps_cb,  # arg 6: cb_eps
        ]

        # ================================================================
        # Unified kernel descriptor
        # ================================================================
        unified_kernel = UnifiedKernelDescriptor(
            kernel_source="models/demos/glm4_moe_lite/fused_ops/kv_cache_branch/kernels/kv_cache_branch_kernel.cpp",
            core_ranges=input_core_grid,
            ncrisc_named_compile_time_args=(
                dkv_matmul_ncrisc_args
                + dkv_gather_sender_args
                + kv_rmsnorm_common_args
                + krope_ncrisc_args
                + dram_io_ncrisc_args
            ),
            brisc_named_compile_time_args=(dkv_gather_receiver_args + kv_rmsnorm_common_args + krope_brisc_args),
            trisc_named_compile_time_args=(dkv_matmul_trisc_args + kv_rmsnorm_common_args + krope_trisc_args),
            trisc_compute_config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.LoFi,
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                dst_full_sync_en=False,
            ),
            unified_compile_time_core_descriptors=[
                UnifiedCompileTimeCoreDescriptor(
                    named_compile_time_arg="is_dkv_matmul_core",
                    core_range=dkv_matmul_weights_core_grid,
                    value=1,
                    other_value=0,
                ),
                UnifiedCompileTimeCoreDescriptor(
                    named_compile_time_arg="is_kv_rmsnorm_core",
                    core_range=rms_core_grid,
                    value=1,
                    other_value=0,
                ),
                UnifiedCompileTimeCoreDescriptor(
                    named_compile_time_arg="is_knope_core",
                    core_range=dkv_gather_sender_grid,
                    value=1,
                    other_value=0,
                ),
                UnifiedCompileTimeCoreDescriptor(
                    named_compile_time_arg="is_krope_core",
                    core_range=krope_core_grid,
                    value=1,
                    other_value=0,
                ),
            ],
            per_core_compile_time_descriptors=[
                dkv_gather_sender_idx_descriptor,
                krope_tile_offset_descriptor,
            ],
            ncrisc_common_runtime_args=ncrisc_rmsnorm_rt_args,
            trisc_common_runtime_args=trisc_rmsnorm_rt_args,
        )

        # ================================================================
        # Program descriptor
        # ================================================================
        cb_list = [
            dkv_matmul_input_cb_descriptor,
            dkv_matmul_output_cb_descriptor,
            dkv_matmul_weights_cb_descriptor,
            rmsnorm_input_cb_on_rms,
            rmsnorm_gamma_cb_descriptor,
            rmsnorm_output_cb_descriptor,
            rmsnorm_x2_cb_descriptor,
            rmsnorm_var_cb_descriptor,
            rmsnorm_scaler_cb_descriptor,
            rmsnorm_eps_cb_descriptor,
            cos_cb_descriptor,
            sin_cb_descriptor,
            trans_mat_cb_descriptor,
            rotated_interm_cb_descriptor,
            cos_interm_cb_descriptor,
            sin_interm_cb_descriptor,
            k_rope_output_cb_descriptor,
        ]
        if rmsnorm_input_cb_dummy is not None:
            cb_list.append(rmsnorm_input_cb_dummy)

        program_descriptor = ttnn.ProgramDescriptor(
            kernels=unified_kernel.get_kernel_descriptors().kernels,
            cbs=cb_list,
            semaphores=[gather_noc0_sem, gather_noc1_sem],
        )

        # io_tensors: sharded tensors for CB backing + DRAM tensors for lifetime
        io_tensors = [
            input_tensor,
            dkv_matmul_weights_tensor,
            gamma_tensor,
            cos_tensor,
            sin_tensor,
            trans_mat_tensor,
            kvpe_output_tensor,
        ]
        ttnn.generic_op(io_tensors, program_descriptor)
        return kvpe_output_tensor
