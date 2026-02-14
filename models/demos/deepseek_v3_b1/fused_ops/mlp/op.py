# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Fused MLP operation: Dense MLP + Shared Expert (no routing).

Same pipeline as MoE but without routing logic (gate MM, gate gather, gate/TopK,
index mcast, expert scale mcast).

Architecture:
  MlpRoutedExpertOp  — context setup & build methods for dense MLP path
  MlpOp              — top-level orchestrator (MlpRoutedExpertOp + MoeSharedExpertOp)

Pipeline (dense MLP):
  0. Residual Mcast → 0b. RMSNorm → 1. Input Mcast
  6. gate_proj+SiLU → 7. up_proj → 8. mul (no scalar)
  9. down_proj gather → 10. down_proj mcast → 11. down_proj MM
  → shared expert path → 12. Eltwise add
"""

import math
from dataclasses import dataclass
from typing import Any

import ttnn
from models.demos.deepseek_v3_b1.fused_ops.moe.op import MoeOp, MoeRoutedExpertOp, MoeSharedExpertOp
from models.demos.deepseek_v3_b1.fused_ops.shared_expert.op import SharedExpertOp
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import (
    PerCoreCompileTimeDescriptor,
    UnifiedCompileTimeCoreDescriptor,
    UnifiedKernelDescriptor,
)
from models.demos.deepseek_v3_b1.utils import float_to_uint32


@dataclass
class _MlpRoutedExpertContext:
    """Holds all computed values needed by MlpRoutedExpertOp helper methods.

    Same as _MoeRoutedExpertContext but without routing-specific fields:
    - No gate_mm_* CBs, gate_input/bias/indices/output CBs
    - No gate_proj_cb_index (expert index CB)
    - No mul_cb_scalar_src, mul_cb_scalar (scalar CBs)
    - No gate_mm_params, gate_mm_gather_params, gate_params
    - No index_mcast_*, expert_scale_mcast_* fields
    """

    # Device & mesh
    device: Any
    full_device_grid: Any
    device_grid_size: Any
    mesh_rows: int
    mesh_cols: int

    # Core grids
    sender_core: Any
    input_core_grid: Any
    mcast_grid: Any
    gate_proj_core_ranges: Any
    num_gate_proj_cores: int

    # Data format & tiles
    data_format: Any
    tile_1x32_size: int
    num_tiles_k: int

    # CB indices (no gate_mm CBs 2-8, no gate_proj_cb_index 10, no mul_scalar CBs 20-21)
    rmsnorm_output_cb: int
    gate_mm_input_cb: int  # CB 1 still used as mcast destination for gate_proj/up_proj input
    gate_proj_cb_in1: int
    gate_proj_cb_out: int
    up_proj_cb_in1: int
    up_proj_cb_mm_out: int
    mul_cb_in0: int
    mul_cb_in1: int
    mul_cb_out: int
    down_proj_gather_dst_cb: int
    down_proj_mcast_dst_cb: int
    down_proj_cb_in1: int
    down_proj_cb_out: int
    add_cb_in0: int
    add_cb_in1: int
    add_cb_out: int

    # Semaphore IDs (no expert_scale_mcast semaphores)
    mcast_data_sender_semaphore_id: int
    mcast_data_receiver_semaphore_id: int
    gather_noc0_receiver_semaphore_id: int
    gather_noc1_receiver_semaphore_id: int

    # Setup result dicts (no gate_mm_params, gate_mm_gather_params, gate_params)
    rmsnorm_mcast_params: dict
    gate_proj_params: dict
    up_proj_params: dict
    mul_params: dict
    down_proj_gather_params: dict
    down_proj_mcast_params: dict
    down_proj_params: dict
    add_params: dict

    # Derived values
    mul_num_tiles: int

    # Pre-built CB descriptors (no gate_proj_cb_index_descriptor)
    rmsnorm_output_cb_descriptor: Any
    gate_mm_input_cb_descriptor: Any

    # Residual mcast (input → shared expert matmul cores)
    residual_mcast_src_cb: int
    residual_mcast_dst_cb: int
    residual_mcast_receiver_semaphore_id: int
    residual_mcast_src_cb_descriptor: Any
    residual_mcast_params: dict

    # RMSNorm (sender core: raw input → normalized input before input mcast)
    rmsnorm_gamma_cb: int
    rmsnorm_gamma_cb_descriptor: Any
    rmsnorm_epsilon_packed: int
    rmsnorm_scalar_packed: int
    rmsnorm_num_tiles: int
    rmsnorm_gamma_num_pages: int

    # Per-core values (for core descriptors)
    bank_id_core_values: list
    vc_core_values: list
    sender_idx_core_values: list


class MlpRoutedExpertOp:
    """
    MLP dense expert fused operation (no routing).

    Same structure as MoeRoutedExpertOp but without routing-related logic.
    Reuses setup_dram_matmul, setup_eltwise_add from MoeRoutedExpertOp.
    """

    @staticmethod
    def setup_eltwise_mul_no_scalar(
        in0_tensor,
        in1_tensor,
        out_tensor,
        cb_in0_index,
        cb_in1_index,
        cb_out_index,
        per_core_n,
    ):
        """
        Set up parameters and CB descriptors for element-wise multiply without scalar (MLP mode).
        Same as MoeRoutedExpertOp.setup_eltwise_mul but without scalar CBs.
        """
        M = 1
        tile_width = 32
        total_elements = M * per_core_n * tile_width
        mul_num_tiles = math.ceil(total_elements / 256)

        TILE_16x16 = ttnn.Tile((16, 16))
        tile_16x16_size = TILE_16x16.get_tile_size(ttnn.bfloat16)
        tile_16x16_desc = ttnn.TileDescriptor(TILE_16x16)

        # CB for in0: alias of in0_tensor with 16x16 tile format
        cb_in0_descriptor = ttnn.cb_descriptor_from_sharded_tensor(cb_in0_index, in0_tensor)
        cb_in0_descriptor.total_size = mul_num_tiles * tile_16x16_size
        cb_in0_descriptor.format_descriptors[0].tile = tile_16x16_desc
        cb_in0_descriptor.format_descriptors[0].page_size = tile_16x16_size

        # CB for in1: alias of in1_tensor with 16x16 tile format
        cb_in1_descriptor = ttnn.cb_descriptor_from_sharded_tensor(cb_in1_index, in1_tensor)
        cb_in1_descriptor.total_size = mul_num_tiles * tile_16x16_size
        cb_in1_descriptor.format_descriptors[0].tile = tile_16x16_desc
        cb_in1_descriptor.format_descriptors[0].page_size = tile_16x16_size

        # CB for out: output with 16x16 tile format (tensor-backed)
        cb_out_descriptor = ttnn.cb_descriptor_from_sharded_tensor(cb_out_index, out_tensor)
        cb_out_descriptor.total_size = mul_num_tiles * tile_16x16_size
        cb_out_descriptor.format_descriptors[0].tile = tile_16x16_desc
        cb_out_descriptor.format_descriptors[0].page_size = tile_16x16_size

        return {
            "mul_num_tiles": mul_num_tiles,
            "cb_in0_descriptor": cb_in0_descriptor,
            "cb_in1_descriptor": cb_in1_descriptor,
            "cb_out_descriptor": cb_out_descriptor,
        }

    # ========================================================================
    # Setup & build methods
    # ========================================================================

    @staticmethod
    def _setup_dimensions(
        input_tensor,
        mcast_output_tensor,
        gate_proj_weights_tensor,
        gate_proj_output_tensor,
        up_proj_weights_tensor,
        up_proj_mm_out_tensor,
        fused_output_tensor,
        down_proj_gather_output_tensor,
        down_proj_mcast_output_tensor,
        down_proj_weights_tensor,
        down_proj_output_tensor,
        fused_add_tensor,
        final_output_tensor,
        gate_proj_in1_buf_tensor,
        down_proj_in1_buf_tensor,
        shared_residual_mcast_src_tensor,
        shared_residual_mcast_dst_tensor,
        rmsnorm_gamma_tensor,
        epsilon=1e-6,
    ):
        """Compute all dimensions, grids, setup params, CB descriptors, and per-core values."""

        # ==================================================================
        # Tensor properties
        # ==================================================================
        data_format = input_tensor.dtype
        TILE_1x32 = ttnn.Tile((1, 32))
        tile_1x32_size = TILE_1x32.get_tile_size(data_format)
        K = input_tensor.shape[1]
        num_tiles_k = K // TILE_1x32.tile_shape[1]

        # Sender core (from input tensor's shard spec)
        input_core_grid = input_tensor.memory_config().shard_spec.grid
        sender_core = list(input_core_grid.ranges())[0].start

        # Device & mesh
        mesh_device = input_tensor.device()
        device_grid_size = mesh_device.compute_with_storage_grid_size()
        mesh_shape = mesh_device.shape
        mesh_rows, mesh_cols = mesh_shape[0], mesh_shape[1]
        device = ttnn.get_device_tensors(input_tensor)[0].device()

        # Core grids
        gate_proj_core_ranges = gate_proj_output_tensor.memory_config().shard_spec.grid
        mcast_grid = mcast_output_tensor.memory_config().shard_spec.grid
        num_gate_proj_cores = gate_proj_core_ranges.num_cores()

        # Full device grid
        full_device_grid = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(device_grid_size.x - 1, device_grid_size.y - 1))]
        )

        # ==================================================================
        # Semaphore IDs (no expert_scale_mcast — fewer needed)
        # ==================================================================
        mcast_data_sender_semaphore_id = 0
        mcast_data_receiver_semaphore_id = 1
        gather_noc0_receiver_semaphore_id = 2
        gather_noc1_receiver_semaphore_id = 3
        residual_mcast_receiver_semaphore_id = 5

        # ==================================================================
        # CB indices (skip gate_mm CBs 2-8, gate_proj_cb_index 10, mul_scalar CBs 20-21)
        # ==================================================================
        rmsnorm_output_cb = 0
        gate_mm_input_cb = 1  # Still used as mcast destination for gate_proj/up_proj input
        gate_proj_cb_in1 = 9
        gate_proj_cb_out = 11
        up_proj_cb_in1 = gate_proj_cb_in1  # Shared CB
        up_proj_cb_mm_out = 12
        mul_cb_in0 = 13
        mul_cb_in1 = 14
        mul_cb_out = 15
        down_proj_gather_dst_cb = 16
        down_proj_mcast_dst_cb = 17
        down_proj_cb_in1 = 18
        down_proj_cb_out = 19
        add_cb_in0 = 22
        add_cb_in1 = 23
        add_cb_out = 24
        # Shared expert CBs (defined in MoeSharedExpertOp)
        residual_mcast_src_cb = MoeSharedExpertOp.RESIDUAL_MCAST_SRC_CB
        residual_mcast_dst_cb = MoeSharedExpertOp.RESIDUAL_MCAST_DST_CB
        rmsnorm_gamma_cb = MoeSharedExpertOp.RMSNORM_GAMMA_CB

        # ==================================================================
        # RMSNorm tile reinterpretation
        # ==================================================================
        FULL_32x32_TILE = ttnn.Tile((32, 32))
        HALF_16x32_TILE = ttnn.Tile((16, 32))
        is_16x32_tile = (K // FULL_32x32_TILE.tile_shape[1]) % FULL_32x32_TILE.tile_shape[0] != 0
        rmsnorm_interpreted_tile = HALF_16x32_TILE if is_16x32_tile else FULL_32x32_TILE
        rmsnorm_tile_descriptor = ttnn.TileDescriptor(rmsnorm_interpreted_tile)
        rmsnorm_cb_page_size = rmsnorm_interpreted_tile.get_tile_size(data_format)
        rmsnorm_num_tiles = K // (rmsnorm_interpreted_tile.tile_shape[0] * rmsnorm_interpreted_tile.tile_shape[1])

        # ==================================================================
        # Residual Mcast
        # ==================================================================
        residual_mcast_src_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
            residual_mcast_src_cb, shared_residual_mcast_src_tensor
        )
        residual_mcast_src_cb_descriptor.format_descriptors[0].tile = rmsnorm_tile_descriptor
        residual_mcast_src_cb_descriptor.format_descriptors[0].page_size = rmsnorm_cb_page_size

        residual_mcast_data_size_bytes = num_tiles_k * tile_1x32_size
        residual_mcast_params = MoeOp.setup_mcast(
            device=device,
            sender_core=sender_core,
            mcast_grid=mcast_grid,
            src_cb=residual_mcast_src_cb,
            src_tensor=shared_residual_mcast_src_tensor,
            dst_cb=residual_mcast_dst_cb,
            dst_tensor=shared_residual_mcast_dst_tensor,
            sender_semaphore_id=mcast_data_sender_semaphore_id,
            receiver_semaphore_id=residual_mcast_receiver_semaphore_id,
            data_size_bytes=residual_mcast_data_size_bytes,
        )
        residual_mcast_params["src_num_pages"] = rmsnorm_num_tiles

        # ==================================================================
        # RMSNorm
        # ==================================================================
        rmsnorm_output_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(rmsnorm_output_cb, input_tensor)
        rmsnorm_output_cb_descriptor.format_descriptors[0].tile = rmsnorm_tile_descriptor
        rmsnorm_output_cb_descriptor.format_descriptors[0].page_size = rmsnorm_cb_page_size

        rmsnorm_gamma_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(rmsnorm_gamma_cb, rmsnorm_gamma_tensor)
        rmsnorm_gamma_cb_descriptor.format_descriptors[0].tile = rmsnorm_tile_descriptor
        rmsnorm_gamma_cb_descriptor.format_descriptors[0].page_size = rmsnorm_cb_page_size

        rmsnorm_epsilon_packed = float_to_uint32(epsilon)
        rmsnorm_scalar_packed = float_to_uint32(1.0 / math.sqrt(float(K)))
        rmsnorm_gamma_num_pages = rmsnorm_num_tiles

        # ==================================================================
        # RMSNorm Mcast
        # ==================================================================
        rmsnorm_mcast_data_size_bytes = num_tiles_k * tile_1x32_size
        rmsnorm_mcast_params = MoeOp.setup_mcast(
            device=device,
            sender_core=sender_core,
            mcast_grid=mcast_grid,
            src_cb=rmsnorm_output_cb,
            src_tensor=input_tensor,
            dst_cb=gate_mm_input_cb,
            dst_tensor=mcast_output_tensor,
            sender_semaphore_id=mcast_data_sender_semaphore_id,
            receiver_semaphore_id=mcast_data_receiver_semaphore_id,
            data_size_bytes=rmsnorm_mcast_data_size_bytes,
        )
        rmsnorm_mcast_params["src_num_pages"] = rmsnorm_num_tiles

        # Pre-built CB descriptors
        gate_mm_input_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(gate_mm_input_cb, mcast_output_tensor)

        # ==================================================================
        # No Gate MM, Gate MM Gather, Gate, Index Mcast, Expert Scale Mcast
        # ==================================================================

        # ==================================================================
        # DRAM Streaming Matmul: gate_proj
        # ==================================================================
        gate_proj_params = MoeRoutedExpertOp.setup_dram_matmul(
            device=device,
            weights_tensor=gate_proj_weights_tensor,
            output_tensor=gate_proj_output_tensor,
            working_buf_tensor=gate_proj_in1_buf_tensor,
            core_ranges=gate_proj_core_ranges,
            cb_in1_index=gate_proj_cb_in1,
            cb_out_index=gate_proj_cb_out,
            fp32_dest_acc_en=True,
            num_subblocks_k=4,
        )

        # ==================================================================
        # DRAM Streaming Matmul: up_proj
        # ==================================================================
        up_proj_params = MoeRoutedExpertOp.setup_dram_matmul(
            device=device,
            weights_tensor=up_proj_weights_tensor,
            output_tensor=up_proj_mm_out_tensor,
            working_buf_tensor=gate_proj_in1_buf_tensor,
            core_ranges=gate_proj_core_ranges,
            cb_in1_index=up_proj_cb_in1,
            cb_out_index=up_proj_cb_mm_out,
            fp32_dest_acc_en=True,
            num_subblocks_k=4,
        )

        # ==================================================================
        # Eltwise Mul: silu(gate_proj) * up_proj (no expert_scale)
        # ==================================================================
        mul_params = MlpRoutedExpertOp.setup_eltwise_mul_no_scalar(
            in0_tensor=up_proj_mm_out_tensor,
            in1_tensor=gate_proj_output_tensor,
            out_tensor=fused_output_tensor,
            cb_in0_index=mul_cb_in0,
            cb_in1_index=mul_cb_in1,
            cb_out_index=mul_cb_out,
            per_core_n=gate_proj_params["per_core_n"],
        )
        mul_num_tiles = mul_params["mul_num_tiles"]

        # ==================================================================
        # down_proj Gather
        # ==================================================================
        down_proj_gather_data_size_bytes = gate_proj_params["per_core_n"] * tile_1x32_size
        down_proj_gather_params = MoeOp.setup_gather(
            device=device,
            receiver_core=sender_core,
            sender_core_ranges=gate_proj_core_ranges,
            num_senders=num_gate_proj_cores,
            data_size_bytes_per_sender=down_proj_gather_data_size_bytes,
            src_cb=mul_cb_out,
            src_num_pages=mul_num_tiles,
            dst_cb=down_proj_gather_dst_cb,
            dst_tensor=down_proj_gather_output_tensor,
            noc0_receiver_semaphore_id=gather_noc0_receiver_semaphore_id,
            noc1_receiver_semaphore_id=gather_noc1_receiver_semaphore_id,
            row_major=True,
            use_explicit_sender_index=True,
        )

        # ==================================================================
        # down_proj Mcast
        # ==================================================================
        fused_output_shard = down_proj_gather_output_tensor.memory_config().shard_spec.shape
        down_proj_mcast_num_tiles = (fused_output_shard[0] * fused_output_shard[1]) // (
            TILE_1x32.tile_shape[0] * TILE_1x32.tile_shape[1]
        )
        down_proj_mcast_data_size_bytes = down_proj_mcast_num_tiles * tile_1x32_size
        down_proj_mcast_params = MoeOp.setup_mcast(
            device=device,
            sender_core=sender_core,
            mcast_grid=mcast_grid,
            src_cb=down_proj_gather_dst_cb,
            src_tensor=down_proj_gather_output_tensor,
            dst_cb=down_proj_mcast_dst_cb,
            dst_tensor=down_proj_mcast_output_tensor,
            sender_semaphore_id=mcast_data_sender_semaphore_id,
            receiver_semaphore_id=mcast_data_receiver_semaphore_id,
            data_size_bytes=down_proj_mcast_data_size_bytes,
        )

        # ==================================================================
        # DRAM Streaming Matmul: down_proj
        # ==================================================================
        down_proj_params = MoeRoutedExpertOp.setup_dram_matmul(
            device=device,
            weights_tensor=down_proj_weights_tensor,
            output_tensor=down_proj_output_tensor,
            working_buf_tensor=down_proj_in1_buf_tensor,
            core_ranges=gate_proj_core_ranges,
            cb_in1_index=down_proj_cb_in1,
            cb_out_index=down_proj_cb_out,
            fp32_dest_acc_en=True,
            num_subblocks_k=2,
        )

        # ==================================================================
        # Eltwise Add: down_proj + fused_add
        # ==================================================================
        add_params = MoeRoutedExpertOp.setup_eltwise_add(
            in0_tensor=down_proj_output_tensor,
            in1_tensor=fused_add_tensor,
            out_tensor=final_output_tensor,
            cb_in0_index=add_cb_in0,
            cb_in1_index=add_cb_in1,
            cb_out_index=add_cb_out,
        )

        # ==================================================================
        # Per-core bank_id, vc, sender_idx
        # ==================================================================
        gate_proj_optimal_workers = device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
        core_to_bank_id = {}
        for bank_id, core in enumerate(gate_proj_optimal_workers):
            core_to_bank_id[(core.x, core.y)] = bank_id

        bank_id_core_values = []
        vc_core_values = []
        sender_idx_core_values = []
        bank_ids = []
        for idx, core in enumerate(gate_proj_optimal_workers):
            bank_id = core_to_bank_id[(core.x, core.y)]
            vc = bank_id & 0x3
            for j in range(idx):
                prev_core = gate_proj_optimal_workers[j]
                if prev_core.y == core.y and (bank_ids[j] & 0x3) == (bank_id & 0x3):
                    vc = (vc + 1) & 0x3
                    break
            bank_ids.append(bank_id)
            bank_id_core_values.append((core, bank_id))
            vc_core_values.append((core, vc))
            sender_idx_core_values.append((core, idx))

        # ==================================================================
        # Return context
        # ==================================================================
        return _MlpRoutedExpertContext(
            # Device & mesh
            device=device,
            full_device_grid=full_device_grid,
            device_grid_size=device_grid_size,
            mesh_rows=mesh_rows,
            mesh_cols=mesh_cols,
            # Core grids
            sender_core=sender_core,
            input_core_grid=input_core_grid,
            mcast_grid=mcast_grid,
            gate_proj_core_ranges=gate_proj_core_ranges,
            num_gate_proj_cores=num_gate_proj_cores,
            # Data format & tiles
            data_format=data_format,
            tile_1x32_size=tile_1x32_size,
            num_tiles_k=num_tiles_k,
            # CB indices
            rmsnorm_output_cb=rmsnorm_output_cb,
            gate_mm_input_cb=gate_mm_input_cb,
            gate_proj_cb_in1=gate_proj_cb_in1,
            gate_proj_cb_out=gate_proj_cb_out,
            up_proj_cb_in1=up_proj_cb_in1,
            up_proj_cb_mm_out=up_proj_cb_mm_out,
            mul_cb_in0=mul_cb_in0,
            mul_cb_in1=mul_cb_in1,
            mul_cb_out=mul_cb_out,
            down_proj_gather_dst_cb=down_proj_gather_dst_cb,
            down_proj_mcast_dst_cb=down_proj_mcast_dst_cb,
            down_proj_cb_in1=down_proj_cb_in1,
            down_proj_cb_out=down_proj_cb_out,
            add_cb_in0=add_cb_in0,
            add_cb_in1=add_cb_in1,
            add_cb_out=add_cb_out,
            # Semaphore IDs
            mcast_data_sender_semaphore_id=mcast_data_sender_semaphore_id,
            mcast_data_receiver_semaphore_id=mcast_data_receiver_semaphore_id,
            gather_noc0_receiver_semaphore_id=gather_noc0_receiver_semaphore_id,
            gather_noc1_receiver_semaphore_id=gather_noc1_receiver_semaphore_id,
            # Setup result dicts
            rmsnorm_mcast_params=rmsnorm_mcast_params,
            gate_proj_params=gate_proj_params,
            up_proj_params=up_proj_params,
            mul_params=mul_params,
            down_proj_gather_params=down_proj_gather_params,
            down_proj_mcast_params=down_proj_mcast_params,
            down_proj_params=down_proj_params,
            add_params=add_params,
            # Derived
            mul_num_tiles=mul_num_tiles,
            # Pre-built CB descriptors
            rmsnorm_output_cb_descriptor=rmsnorm_output_cb_descriptor,
            gate_mm_input_cb_descriptor=gate_mm_input_cb_descriptor,
            # Residual mcast
            residual_mcast_src_cb=residual_mcast_src_cb,
            residual_mcast_dst_cb=residual_mcast_dst_cb,
            residual_mcast_receiver_semaphore_id=residual_mcast_receiver_semaphore_id,
            residual_mcast_src_cb_descriptor=residual_mcast_src_cb_descriptor,
            residual_mcast_params=residual_mcast_params,
            # RMSNorm
            rmsnorm_gamma_cb=rmsnorm_gamma_cb,
            rmsnorm_gamma_cb_descriptor=rmsnorm_gamma_cb_descriptor,
            rmsnorm_epsilon_packed=rmsnorm_epsilon_packed,
            rmsnorm_scalar_packed=rmsnorm_scalar_packed,
            rmsnorm_num_tiles=rmsnorm_num_tiles,
            rmsnorm_gamma_num_pages=rmsnorm_gamma_num_pages,
            # Per-core values
            bank_id_core_values=bank_id_core_values,
            vc_core_values=vc_core_values,
            sender_idx_core_values=sender_idx_core_values,
        )

    @staticmethod
    def _build_compile_time_args(ctx, mesh_chip_id):
        """Build NCRISC, BRISC, and TRISC compile-time arg lists for MLP (no routing)."""

        ncrisc_named_compile_time_args = [
            # Input mcast (sender sharded buffer + receiver)
            ("mcast_src_cb", ctx.rmsnorm_mcast_params["src_cb"]),
            ("mcast_src_num_pages", ctx.rmsnorm_mcast_params["src_num_pages"]),
            ("mcast_data_receiver_semaphore", ctx.rmsnorm_mcast_params["receiver_semaphore_id"]),
            ("mcast_dst_cb", ctx.rmsnorm_mcast_params["dst_cb"]),
            ("mcast_dst_num_pages", ctx.rmsnorm_mcast_params["dst_num_pages"]),
            # Residual mcast source (setup_sharded_buffer on sender core)
            ("shared_residual_mcast_src_cb", ctx.residual_mcast_params["src_cb"]),
            ("shared_residual_mcast_src_num_pages", ctx.residual_mcast_params["src_num_pages"]),
            # Residual mcast receiver
            ("shared_residual_mcast_data_receiver_semaphore", ctx.residual_mcast_receiver_semaphore_id),
            ("shared_residual_cb", ctx.residual_mcast_dst_cb),
            ("shared_residual_num_pages", ctx.residual_mcast_params["dst_num_pages"]),
            # RMSNorm (setup_sharded_buffer for gamma on sender core)
            ("rmsnorm_gamma_cb", ctx.rmsnorm_gamma_cb),
            ("rmsnorm_gamma_num_pages", ctx.rmsnorm_gamma_num_pages),
            # No gate_mm, gate_gather, gate, index_mcast, expert_scale_mcast args
            # Mul reader (setup mul_in1 buffer)
            ("mul_cb_in1", ctx.mul_cb_in1),
            ("mul_num_tiles", ctx.mul_num_tiles),
            # down_proj gather receiver
            ("down_proj_gather_noc0_num_senders", ctx.down_proj_gather_params["noc0_num_senders"]),
            ("down_proj_gather_noc1_num_senders", ctx.down_proj_gather_params["noc1_num_senders"]),
            ("down_proj_gather_noc0_receiver_semaphore_id", ctx.down_proj_gather_params["noc0_receiver_semaphore_id"]),
            ("down_proj_gather_noc1_receiver_semaphore_id", ctx.down_proj_gather_params["noc1_receiver_semaphore_id"]),
            ("down_proj_gather_dst_cb", ctx.down_proj_gather_params["dst_cb"]),
            ("down_proj_gather_dst_num_pages", ctx.down_proj_gather_params["dst_num_pages"]),
            # down_proj mcast receiver
            ("down_proj_mcast_receiver_semaphore", ctx.down_proj_mcast_params["receiver_semaphore_id"]),
            ("down_proj_mcast_dst_cb", ctx.down_proj_mcast_params["dst_cb"]),
            ("down_proj_mcast_dst_num_pages", ctx.down_proj_mcast_params["dst_num_pages"]),
            # Eltwise add
            ("add_cb_in0", ctx.add_cb_in0),
            ("add_cb_in1", ctx.add_cb_in1),
            ("add_cb_in0_wait_tiles", ctx.add_params["cb_in0_wait_tiles"]),
            ("add_cb_in1_wait_tiles", ctx.add_params["cb_in1_wait_tiles"]),
            # gate_proj DRAM matmul reader (no indexing params)
            ("gate_proj_cb_in1", ctx.gate_proj_cb_in1),
            ("gate_proj_cb_out", ctx.gate_proj_cb_out),
            ("gate_proj_in1_tensor_addr", ctx.gate_proj_params["in1_tensor_addr"]),
            ("gate_proj_in1_page_size", ctx.gate_proj_params["in1_page_size"]),
            ("gate_proj_in1_num_pages", ctx.gate_proj_params["in1_num_pages"]),
            ("gate_proj_subblock_k", ctx.gate_proj_params["subblock_k"]),
            ("gate_proj_per_core_n", ctx.gate_proj_params["per_core_n"]),
            ("gate_proj_in1_block_size_bytes", ctx.gate_proj_params["in1_block_size_bytes"]),
            ("gate_proj_out_num_tiles", ctx.gate_proj_params["out_num_tiles"]),
            ("gate_proj_num_subblocks_k", ctx.gate_proj_params["num_subblocks_k"]),
            # up_proj DRAM matmul reader (no indexing params)
            ("up_proj_cb_in1", ctx.up_proj_cb_in1),
            ("up_proj_in1_tensor_addr", ctx.up_proj_params["in1_tensor_addr"]),
            ("up_proj_in1_page_size", ctx.up_proj_params["in1_page_size"]),
            ("up_proj_in1_num_pages", ctx.up_proj_params["in1_num_pages"]),
            ("up_proj_subblock_k", ctx.up_proj_params["subblock_k"]),
            ("up_proj_per_core_n", ctx.up_proj_params["per_core_n"]),
            ("up_proj_in1_block_size_bytes", ctx.up_proj_params["in1_block_size_bytes"]),
            ("up_proj_out_num_tiles", ctx.up_proj_params["out_num_tiles"]),
            ("up_proj_num_subblocks_k", ctx.up_proj_params["num_subblocks_k"]),
            ("gate_proj_in1_buf_addr", ctx.gate_proj_params["in1_buf_addr"]),
            ("up_proj_cb_mm_out", ctx.up_proj_cb_mm_out),
            # down_proj DRAM matmul reader (no indexing params)
            ("down_proj_cb_in1", ctx.down_proj_cb_in1),
            ("down_proj_cb_out", ctx.down_proj_cb_out),
            ("down_proj_in1_tensor_addr", ctx.down_proj_params["in1_tensor_addr"]),
            ("down_proj_in1_page_size", ctx.down_proj_params["in1_page_size"]),
            ("down_proj_in1_num_pages", ctx.down_proj_params["in1_num_pages"]),
            ("down_proj_subblock_k", ctx.down_proj_params["subblock_k"]),
            ("down_proj_per_core_n", ctx.down_proj_params["per_core_n"]),
            ("down_proj_in1_block_size_bytes", ctx.down_proj_params["in1_block_size_bytes"]),
            ("down_proj_out_num_tiles", ctx.down_proj_params["out_num_tiles"]),
            ("down_proj_num_subblocks_k", ctx.down_proj_params["num_subblocks_k"]),
            ("down_proj_in1_buf_addr", ctx.down_proj_params["in1_buf_addr"]),
        ]

        brisc_named_compile_time_args = [
            # Input mcast sender
            ("mcast_dest_noc_start_x", ctx.rmsnorm_mcast_params["dest_noc_start_x"]),
            ("mcast_dest_noc_start_y", ctx.rmsnorm_mcast_params["dest_noc_start_y"]),
            ("mcast_dest_noc_end_x", ctx.rmsnorm_mcast_params["dest_noc_end_x"]),
            ("mcast_dest_noc_end_y", ctx.rmsnorm_mcast_params["dest_noc_end_y"]),
            ("mcast_num_cores", ctx.rmsnorm_mcast_params["num_cores"]),
            ("mcast_data_sender_semaphore", ctx.rmsnorm_mcast_params["sender_semaphore_id"]),
            ("mcast_data_receiver_semaphore", ctx.rmsnorm_mcast_params["receiver_semaphore_id"]),
            ("mcast_data_size_bytes", ctx.rmsnorm_mcast_params["data_size_bytes"]),
            ("mcast_src_cb", ctx.rmsnorm_mcast_params["src_cb"]),
            ("mcast_dst_cb", ctx.rmsnorm_mcast_params["dst_cb"]),
            ("mcast_src_num_pages", ctx.rmsnorm_mcast_params["src_num_pages"]),
            ("mcast_is_part_of_receiver_grid", ctx.rmsnorm_mcast_params["is_sender_part_of_receiver_grid"]),
            # Residual mcast sender
            ("shared_residual_mcast_data_sender_semaphore", ctx.mcast_data_sender_semaphore_id),
            ("shared_residual_mcast_data_receiver_semaphore", ctx.residual_mcast_receiver_semaphore_id),
            ("shared_residual_mcast_data_size_bytes", ctx.residual_mcast_params["data_size_bytes"]),
            ("shared_residual_mcast_src_cb", ctx.residual_mcast_params["src_cb"]),
            ("shared_residual_mcast_src_num_pages", ctx.residual_mcast_params["src_num_pages"]),
            ("shared_residual_mcast_dst_cb", ctx.residual_mcast_dst_cb),
            # No gate_gather, gate, index_mcast, expert_scale_mcast args
            # Mul writer (no scalar args)
            ("mul_cb_out", ctx.mul_cb_out),
            ("mul_num_tiles", ctx.mul_num_tiles),
            # down_proj gather sender
            ("down_proj_gather_dest_noc_x", ctx.down_proj_gather_params["dest_noc_x"]),
            ("down_proj_gather_dest_noc_y", ctx.down_proj_gather_params["dest_noc_y"]),
            ("down_proj_gather_data_size_bytes", ctx.down_proj_gather_params["data_size_bytes"]),
            ("down_proj_gather_receiver_semaphore_id", ctx.down_proj_gather_params["receiver_semaphore_id"]),
            ("down_proj_gather_src_cb", ctx.down_proj_gather_params["src_cb"]),
            ("down_proj_gather_src_num_pages", ctx.down_proj_gather_params["src_num_pages"]),
            ("down_proj_gather_sender_grid_start_x", ctx.down_proj_gather_params["sender_grid_start_x"]),
            ("down_proj_gather_sender_grid_start_y", ctx.down_proj_gather_params["sender_grid_start_y"]),
            ("down_proj_gather_sender_grid_end_x", ctx.down_proj_gather_params["sender_grid_end_x"]),
            ("down_proj_gather_sender_grid_end_y", ctx.down_proj_gather_params["sender_grid_end_y"]),
            ("down_proj_gather_row_major", ctx.down_proj_gather_params["row_major"]),
            ("down_proj_gather_receiver_data_addr", ctx.down_proj_gather_params["receiver_data_addr"]),
            # down_proj mcast sender
            ("down_proj_mcast_sender_semaphore", ctx.down_proj_mcast_params["sender_semaphore_id"]),
            ("down_proj_mcast_receiver_semaphore", ctx.down_proj_mcast_params["receiver_semaphore_id"]),
            ("down_proj_mcast_data_size_bytes", ctx.down_proj_mcast_params["data_size_bytes"]),
            ("down_proj_mcast_src_cb", ctx.down_proj_mcast_params["src_cb"]),
            ("down_proj_mcast_dst_cb", ctx.down_proj_mcast_params["dst_cb"]),
            ("down_proj_mcast_src_num_pages", ctx.down_proj_mcast_params["src_num_pages"]),
            # CB reset addresses
            ("gate_proj_in1_buf_addr", ctx.gate_proj_params["in1_buf_addr"]),
            ("down_proj_in1_buf_addr", ctx.down_proj_params["in1_buf_addr"]),
            # Eltwise add CB
            ("add_cb_in1", ctx.add_cb_in1),
        ]

        trisc_named_compile_time_args = [
            # RMSNorm compute
            ("rmsnorm_input_cb", ctx.residual_mcast_src_cb),
            ("rmsnorm_gamma_cb", ctx.rmsnorm_gamma_cb),
            ("rmsnorm_output_cb", ctx.rmsnorm_output_cb),
            ("rmsnorm_fp32_acc", 0),
            ("rmsnorm_num_tiles", ctx.rmsnorm_num_tiles),
            ("rmsnorm_rsqrt_fast_approx", 0),
            # No gate_mm, gate compute args
            # gate_proj compute
            ("gate_proj_cb_in0", ctx.gate_mm_input_cb),
            ("gate_proj_cb_in1", ctx.gate_proj_cb_in1),
            ("gate_proj_cb_out", ctx.gate_proj_cb_out),
            ("gate_proj_subblock_k", ctx.gate_proj_params["subblock_k"]),
            ("gate_proj_per_core_n", ctx.gate_proj_params["per_core_n"]),
            ("gate_proj_subblock_w", ctx.gate_proj_params["subblock_w"]),
            ("gate_proj_num_subblocks_k", ctx.gate_proj_params["num_subblocks_k"]),
            ("gate_proj_tile_r_dim", ctx.gate_proj_params["tile_r_dim"]),
            ("gate_proj_fuse_silu", 1),
            ("gate_proj_fp32_dest_acc_en", 1),
            # up_proj compute
            ("up_proj_cb_in0", ctx.gate_mm_input_cb),
            ("up_proj_cb_in1", ctx.up_proj_cb_in1),
            ("up_proj_subblock_k", ctx.up_proj_params["subblock_k"]),
            ("up_proj_per_core_n", ctx.up_proj_params["per_core_n"]),
            ("up_proj_subblock_w", ctx.up_proj_params["subblock_w"]),
            ("up_proj_num_subblocks_k", ctx.up_proj_params["num_subblocks_k"]),
            ("up_proj_tile_r_dim", ctx.up_proj_params["tile_r_dim"]),
            ("up_proj_fuse_silu", 0),
            ("up_proj_fp32_dest_acc_en", 1),
            ("up_proj_cb_mm_out", ctx.up_proj_cb_mm_out),
            # Mul compute (no scalar)
            ("mul_cb_in0", ctx.mul_cb_in0),
            ("mul_cb_in1", ctx.mul_cb_in1),
            ("mul_cb_out", ctx.mul_cb_out),
            ("mul_num_tiles", ctx.mul_num_tiles),
            ("mul_fp32_dest_acc_en", 1),
            ("up_proj_per_core_n", ctx.up_proj_params["per_core_n"]),
            # down_proj compute
            ("down_proj_cb_in0", ctx.down_proj_mcast_dst_cb),
            ("down_proj_cb_in1", ctx.down_proj_cb_in1),
            ("down_proj_cb_out", ctx.down_proj_cb_out),
            ("down_proj_subblock_k", ctx.down_proj_params["subblock_k"]),
            ("down_proj_per_core_n", ctx.down_proj_params["per_core_n"]),
            ("down_proj_subblock_w", ctx.down_proj_params["subblock_w"]),
            ("down_proj_num_subblocks_k", ctx.down_proj_params["num_subblocks_k"]),
            ("down_proj_tile_r_dim", ctx.down_proj_params["tile_r_dim"]),
            ("down_proj_fuse_silu", 0),
            ("down_proj_fp32_dest_acc_en", 1),
            # Eltwise add compute
            ("add_cb_in0", ctx.add_cb_in0),
            ("add_cb_in1", ctx.add_cb_in1),
            ("add_cb_out", ctx.add_cb_out),
            ("add_num_tiles", ctx.add_params["num_tiles"]),
            ("add_cb_in0_wait_tiles", ctx.add_params["cb_in0_wait_tiles"]),
            ("add_cb_in1_wait_tiles", ctx.add_params["cb_in1_wait_tiles"]),
            ("add_slice_size_bytes", ctx.add_params["slice_size_bytes"]),
            # CB reset addresses
            ("gate_proj_in1_buf_addr", ctx.gate_proj_params["in1_buf_addr"]),
            ("down_proj_in1_buf_addr", ctx.down_proj_params["in1_buf_addr"]),
        ]

        return ncrisc_named_compile_time_args, brisc_named_compile_time_args, trisc_named_compile_time_args

    @staticmethod
    def _build_cb_descriptors(ctx):
        """Build circular buffer descriptors for MLP (no gate_mm CBs, no gate_proj_cb_index, no mul_scalar CBs)."""
        return [
            ctx.rmsnorm_output_cb_descriptor,
            ctx.gate_mm_input_cb_descriptor,
            # No gate_mm weights/output CB descriptors (CBs 2-8)
            ctx.gate_proj_params["cb_in1_descriptor"],  # Shared by gate_proj and up_proj
            # No gate_proj_cb_index_descriptor (CB 10)
            ctx.gate_proj_params["cb_out_descriptor"],
            ctx.up_proj_params["cb_out_descriptor"],
            ctx.mul_params["cb_in0_descriptor"],
            ctx.mul_params["cb_in1_descriptor"],
            ctx.mul_params["cb_out_descriptor"],
            # No mul_scalar CBs (CB 20-21)
            ctx.down_proj_gather_params["dst_cb_descriptor"],
            ctx.down_proj_mcast_params["dst_cb_descriptor"],
            ctx.down_proj_params["cb_in1_descriptor"],
            ctx.down_proj_params["cb_out_descriptor"],
            ctx.add_params["cb_in0_descriptor"],
            ctx.add_params["cb_in1_descriptor"],
            ctx.add_params["cb_out_descriptor"],
            ctx.residual_mcast_src_cb_descriptor,
            ctx.residual_mcast_params["dst_cb_descriptor"],
            ctx.rmsnorm_gamma_cb_descriptor,
        ]

    @staticmethod
    def _build_core_descriptors(ctx):
        """Build unified and per-core compile-time core descriptors for MLP."""
        unified_compile_time_core_descriptors = [
            UnifiedCompileTimeCoreDescriptor(
                named_compile_time_arg="is_sender_core",
                core_range=ctx.sender_core,
                value=1,
                other_value=0,
            ),
            UnifiedCompileTimeCoreDescriptor(
                named_compile_time_arg="is_mcast_grid_core",
                core_range=ctx.mcast_grid,
                value=1,
                other_value=0,
            ),
            # No is_gate_mm_core — MLP has no routing matmul
            UnifiedCompileTimeCoreDescriptor(
                named_compile_time_arg="is_gate_proj_core",
                core_range=ctx.gate_proj_core_ranges,
                value=1,
                other_value=0,
            ),
        ]

        per_core_compile_time_descriptors = [
            PerCoreCompileTimeDescriptor(
                named_compile_time_arg="gate_proj_bank_id",
                core_values=ctx.bank_id_core_values,
                other_value=0,
            ),
            PerCoreCompileTimeDescriptor(
                named_compile_time_arg="gate_proj_vc",
                core_values=ctx.vc_core_values,
                other_value=0,
            ),
            PerCoreCompileTimeDescriptor(
                named_compile_time_arg="up_proj_bank_id",
                core_values=ctx.bank_id_core_values,
                other_value=0,
            ),
            PerCoreCompileTimeDescriptor(
                named_compile_time_arg="up_proj_vc",
                core_values=ctx.vc_core_values,
                other_value=0,
            ),
            PerCoreCompileTimeDescriptor(
                named_compile_time_arg="down_proj_bank_id",
                core_values=ctx.bank_id_core_values,
                other_value=0,
            ),
            PerCoreCompileTimeDescriptor(
                named_compile_time_arg="down_proj_vc",
                core_values=ctx.vc_core_values,
                other_value=0,
            ),
            PerCoreCompileTimeDescriptor(
                named_compile_time_arg="down_proj_gather_sender_idx",
                core_values=ctx.sender_idx_core_values,
                other_value=0,
            ),
            PerCoreCompileTimeDescriptor(
                named_compile_time_arg="add_sender_index",
                core_values=ctx.add_params["sender_index_core_values"],
                other_value=0,
            ),
        ]

        return unified_compile_time_core_descriptors, per_core_compile_time_descriptors


class MlpOp:
    """
    Top-level fused MLP operation.

    Composes MlpRoutedExpertOp and MoeSharedExpertOp (reused from moe.op),
    merging their CB descriptors, compile-time args, and core descriptors
    into a single unified kernel invocation.
    """

    @staticmethod
    def golden(
        input_tensor,
        shared_gate_weights,
        shared_up_weights,
        shared_down_weights,
        gate_proj_weights,
        up_proj_weights,
        down_proj_weights,
        rmsnorm_gamma=None,
        rmsnorm_epsilon=1e-6,
    ):
        """
        PyTorch reference for the full fused MLP (dense MLP + shared expert + eltwise add).

        No routing: single expert (expert 0), no expert scale.

        Args:
            input_tensor: [1, K] — raw input (pre-norm)
            shared_gate_weights: [K, K_down] shared expert gate weights
            shared_up_weights: [K, K_down] shared expert up weights
            shared_down_weights: [K_down, N] shared expert down weights
            gate_proj_weights: [1, 1, K, N_expert] single expert gate weights
            up_proj_weights: [1, 1, K, N_expert] single expert up weights
            down_proj_weights: [1, 1, N_expert, K] single expert down weights
            rmsnorm_gamma: [1, K] RMSNorm gamma weights
            rmsnorm_epsilon: RMSNorm epsilon

        Returns:
            final_output tensor
        """
        import torch

        # Apply RMSNorm: raw input → normalized input (truncate to bfloat16 to match device)
        x = input_tensor.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        normalized_input = ((x * torch.rsqrt(variance + rmsnorm_epsilon)) * rmsnorm_gamma.float()).bfloat16().float()

        # Shared expert: normalized input for compute, raw input for residual
        shared_output = SharedExpertOp.golden(
            normalized_input.float(),
            shared_gate_weights.float(),
            shared_up_weights.float(),
            shared_down_weights.float(),
            input_tensor.float(),  # residual is the raw (pre-norm) input
        ).bfloat16()

        # Dense MLP (no routing, no expert scale)
        input_for_expert = normalized_input.reshape(1, 1, 1, -1).float()

        # gate_proj + SiLU
        gate_proj_output = input_for_expert @ gate_proj_weights.float()
        gate_proj_output = torch.nn.functional.silu(gate_proj_output)

        # up_proj (no activation)
        up_proj_output = input_for_expert @ up_proj_weights.float()

        # Fused output: silu(gate_proj) * up_proj (no expert_scale)
        fused_output = gate_proj_output * up_proj_output

        # down_proj
        down_proj_output = fused_output @ down_proj_weights.float()

        # Eltwise add: down_proj + shared_expert_output
        shared_for_add = shared_output.float().reshape(1, 1, 1, -1)
        final_output = down_proj_output + shared_for_add

        return final_output

    @staticmethod
    def op(
        input_tensor,
        mcast_output_tensor,
        gate_proj_weights_tensor,
        gate_proj_output_tensor,
        up_proj_weights_tensor,
        up_proj_mm_out_tensor,
        fused_output_tensor,
        down_proj_gather_output_tensor,
        down_proj_mcast_output_tensor,
        down_proj_weights_tensor,
        down_proj_output_tensor,
        fused_add_tensor,
        final_output_tensor,
        gate_proj_in1_buf_tensor,
        down_proj_in1_buf_tensor,
        # RMSNorm gamma weights (sender core)
        rmsnorm_gamma_tensor,
        # Shared expert tensors
        shared_residual_mcast_src_tensor,
        shared_gate_up_weights_tensor,
        shared_residual_mcast_dst_tensor,
        shared_down_mcast_dst_tensor,
        shared_down_weights_tensor,
        shared_output_tensor,
        # Shared expert tensor-backed CB tensors
        shared_ag_gather_dst_tensor,
        shared_bg_gather_dst_tensor,
        shared_gu_out_tensor,
        shared_intermed_tensor,
        shared_down_mcast_src_tensor,
        shared_down_matmul_out_tensor,
        shared_residual_add_out_tensor,
        shared_k_parallel,
        shared_n_parallel,
        epsilon=1e-6,
    ):
        """
        Execute the full fused MLP operation (dense MLP + shared expert).

        No routing tensors needed (no gate_mm_weights, gate_bias, gate_indices,
        gate_output_scores, gate_output_indices, expert_index, expert_scale, mul_scalar_buf).

        Returns:
            final_output_tensor
        """
        # ==================================================================
        # Setup routed expert context (MLP — no routing)
        # ==================================================================
        routed_ctx = MlpRoutedExpertOp._setup_dimensions(
            input_tensor,
            mcast_output_tensor,
            gate_proj_weights_tensor,
            gate_proj_output_tensor,
            up_proj_weights_tensor,
            up_proj_mm_out_tensor,
            fused_output_tensor,
            down_proj_gather_output_tensor,
            down_proj_mcast_output_tensor,
            down_proj_weights_tensor,
            down_proj_output_tensor,
            fused_add_tensor,
            final_output_tensor,
            gate_proj_in1_buf_tensor,
            down_proj_in1_buf_tensor,
            shared_residual_mcast_src_tensor=shared_residual_mcast_src_tensor,
            shared_residual_mcast_dst_tensor=shared_residual_mcast_dst_tensor,
            rmsnorm_gamma_tensor=rmsnorm_gamma_tensor,
            epsilon=epsilon,
        )

        # ==================================================================
        # Setup shared expert context (reused from MoE)
        # ==================================================================
        device_tensor = ttnn.get_device_tensors(input_tensor)[0]
        input_tile = device_tensor.get_tile()
        input_tile_size = input_tile.get_tile_size(routed_ctx.data_format)

        shared_ctx = MoeSharedExpertOp._setup_dimensions(
            device=routed_ctx.device,
            shared_gate_up_weights_tensor=shared_gate_up_weights_tensor,
            shared_down_mcast_dst_tensor=shared_down_mcast_dst_tensor,
            shared_down_weights_tensor=shared_down_weights_tensor,
            shared_output_tensor=shared_output_tensor,
            shared_ag_gather_dst_tensor=shared_ag_gather_dst_tensor,
            shared_bg_gather_dst_tensor=shared_bg_gather_dst_tensor,
            shared_gu_out_tensor=shared_gu_out_tensor,
            shared_intermed_tensor=shared_intermed_tensor,
            shared_down_mcast_src_tensor=shared_down_mcast_src_tensor,
            shared_down_matmul_out_tensor=shared_down_matmul_out_tensor,
            shared_residual_add_out_tensor=shared_residual_add_out_tensor,
            num_tiles_k=routed_ctx.num_tiles_k,
            tile_1x32_size=routed_ctx.tile_1x32_size,
            data_format=routed_ctx.data_format,
            input_tile=input_tile,
            input_tile_size=input_tile_size,
            sender_core=routed_ctx.sender_core,
            sender_core_grid=routed_ctx.input_core_grid,
            mcast_grid=routed_ctx.mcast_grid,
            k_parallel=shared_k_parallel,
            n_parallel=shared_n_parallel,
        )

        # ==================================================================
        # Build CB descriptors (routed + shared)
        # ==================================================================
        cb_descriptors = MlpRoutedExpertOp._build_cb_descriptors(routed_ctx)
        cb_descriptors += MoeSharedExpertOp._build_cb_descriptors(shared_ctx)

        # ==================================================================
        # Build core descriptors (routed + shared)
        # ==================================================================
        unified_core_descs, per_core_descs = MlpRoutedExpertOp._build_core_descriptors(routed_ctx)
        shared_unified, shared_per_core = MoeSharedExpertOp._build_core_descriptors(
            shared_ctx, sender_core_grid=routed_ctx.input_core_grid
        )
        unified_core_descs += shared_unified
        per_core_descs += shared_per_core

        # ==================================================================
        # Semaphore descriptors (no expert_scale_mcast semaphore)
        # ==================================================================
        semaphore_descriptors = [
            ttnn.SemaphoreDescriptor(
                id=routed_ctx.mcast_data_sender_semaphore_id,
                core_ranges=routed_ctx.full_device_grid,
                initial_value=0,
            ),
            ttnn.SemaphoreDescriptor(
                id=routed_ctx.mcast_data_receiver_semaphore_id,
                core_ranges=routed_ctx.full_device_grid,
                initial_value=0,
            ),
            ttnn.SemaphoreDescriptor(
                id=routed_ctx.gather_noc0_receiver_semaphore_id,
                core_ranges=routed_ctx.full_device_grid,
                initial_value=0,
            ),
            ttnn.SemaphoreDescriptor(
                id=routed_ctx.gather_noc1_receiver_semaphore_id,
                core_ranges=routed_ctx.full_device_grid,
                initial_value=0,
            ),
            # No expert_scale_mcast_receiver_semaphore (was semaphore 4 in MoE)
        ]
        # Shared expert gather semaphores
        semaphore_descriptors += [
            ttnn.SemaphoreDescriptor(
                id=shared_ctx.ag_receiver_semaphore_id,
                core_ranges=routed_ctx.full_device_grid,
                initial_value=0,
            ),
            ttnn.SemaphoreDescriptor(
                id=shared_ctx.bg_receiver_semaphore_id,
                core_ranges=routed_ctx.full_device_grid,
                initial_value=0,
            ),
            ttnn.SemaphoreDescriptor(
                id=shared_ctx.ag_noc1_receiver_semaphore_id,
                core_ranges=routed_ctx.full_device_grid,
                initial_value=0,
            ),
            ttnn.SemaphoreDescriptor(
                id=shared_ctx.bg_noc1_receiver_semaphore_id,
                core_ranges=routed_ctx.full_device_grid,
                initial_value=0,
            ),
            ttnn.SemaphoreDescriptor(
                id=shared_ctx.shared_mcast_receiver_semaphore_id,
                core_ranges=routed_ctx.full_device_grid,
                initial_value=0,
            ),
            ttnn.SemaphoreDescriptor(
                id=shared_ctx.output_gather_noc0_receiver_semaphore_id,
                core_ranges=routed_ctx.full_device_grid,
                initial_value=0,
            ),
            ttnn.SemaphoreDescriptor(
                id=shared_ctx.output_gather_noc1_receiver_semaphore_id,
                core_ranges=routed_ctx.full_device_grid,
                initial_value=0,
            ),
            ttnn.SemaphoreDescriptor(
                id=shared_ctx.output_mcast_receiver_semaphore_id,
                core_ranges=routed_ctx.full_device_grid,
                initial_value=0,
            ),
            ttnn.SemaphoreDescriptor(
                id=routed_ctx.residual_mcast_receiver_semaphore_id,
                core_ranges=routed_ctx.full_device_grid,
                initial_value=0,
            ),
        ]

        # ==================================================================
        # IO tensors (no routing tensors)
        # ==================================================================
        io_tensors = [
            input_tensor,
            mcast_output_tensor,
            gate_proj_weights_tensor,
            gate_proj_output_tensor,
            up_proj_weights_tensor,
            up_proj_mm_out_tensor,
            fused_output_tensor,
            down_proj_gather_output_tensor,
            down_proj_mcast_output_tensor,
            down_proj_weights_tensor,
            down_proj_output_tensor,
            fused_add_tensor,
            final_output_tensor,
            # Tensor-backed working buffers
            gate_proj_in1_buf_tensor,
            down_proj_in1_buf_tensor,
            # RMSNorm gamma weights (sender core)
            rmsnorm_gamma_tensor,
            # Shared expert tensors
            shared_residual_mcast_src_tensor,
            shared_gate_up_weights_tensor,
            shared_ag_gather_dst_tensor,
            shared_bg_gather_dst_tensor,
            shared_residual_mcast_dst_tensor,
            shared_down_mcast_dst_tensor,
            shared_down_weights_tensor,
            shared_output_tensor,
            # Shared expert tensor-backed CB tensors
            shared_gu_out_tensor,
            shared_intermed_tensor,
            shared_down_mcast_src_tensor,
            shared_down_matmul_out_tensor,
            shared_residual_add_out_tensor,
        ]

        # ==================================================================
        # Create per-device programs (mesh loop)
        # ==================================================================
        mesh_program_descriptor = ttnn.MeshProgramDescriptor()
        rmsnorm_mcast_dst_cb = routed_ctx.rmsnorm_mcast_params["dst_cb"]

        for row in range(routed_ctx.mesh_rows):
            for col in range(routed_ctx.mesh_cols):
                coord = ttnn.MeshCoordinate(row, col)
                chip_id = row * routed_ctx.mesh_cols + col

                # Build compile-time args: routed + shared
                ncrisc_args, brisc_args, trisc_args = MlpRoutedExpertOp._build_compile_time_args(routed_ctx, chip_id)
                shared_ncrisc, shared_brisc, shared_trisc = MoeSharedExpertOp._build_compile_time_args(
                    shared_ctx, rmsnorm_mcast_dst_cb, routed_ctx.rmsnorm_mcast_params
                )
                ncrisc_args += shared_ncrisc
                brisc_args += shared_brisc
                trisc_args += shared_trisc

                # Create unified kernel
                unified_kernel = UnifiedKernelDescriptor(
                    kernel_source="models/demos/deepseek_v3_b1/fused_ops/mlp/mlp_kernel.cpp",
                    core_ranges=routed_ctx.full_device_grid,
                    ncrisc_named_compile_time_args=ncrisc_args,
                    brisc_named_compile_time_args=brisc_args,
                    trisc_named_compile_time_args=trisc_args,
                    trisc_common_runtime_args=[
                        routed_ctx.rmsnorm_epsilon_packed,
                        routed_ctx.rmsnorm_scalar_packed,
                    ],
                    trisc_compute_config=ttnn.ComputeConfigDescriptor(
                        math_fidelity=ttnn.MathFidelity.LoFi,
                        math_approx_mode=False,
                        fp32_dest_acc_en=False,
                        dst_full_sync_en=False,
                    ),
                    unified_compile_time_core_descriptors=unified_core_descs,
                    per_core_compile_time_descriptors=per_core_descs,
                )

                # Create program
                program = ttnn.ProgramDescriptor(
                    kernels=unified_kernel.get_kernel_descriptors().kernels,
                    cbs=cb_descriptors,
                    semaphores=semaphore_descriptors,
                )

                mesh_program_descriptor[ttnn.MeshCoordinateRange(coord, coord)] = program

        # Execute
        ttnn.generic_op(io_tensors, mesh_program_descriptor)

        return final_output_tensor
