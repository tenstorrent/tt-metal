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
from typing import Any, Optional

import ttnn
from models.demos.deepseek_v3_b1.fused_ops.moe.op import MoeOp, MoeRoutedExpertOp, MoeSharedExpertOp
from models.demos.deepseek_v3_b1.fused_ops.moe_routed_expert.op import (
    MESH_LEAF,
    MESH_ROOT1,
    MESH_ROOT2,
    MESH_ROOT3,
    get_reduce_device_role,
)
from models.demos.deepseek_v3_b1.fused_ops.shared_expert.op import SharedExpertOp
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import (
    PerCoreCompileTimeDescriptor,
    PerCoreRuntimeArgsDescriptor,
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

    # ReduceToOne
    enable_reduce_to_one: bool = False
    reduce_local_cb: int = 0
    reduce_received_cb_r1: int = 0
    reduce_received_cb_r2: int = 0
    reduce_received_cb_r3: int = 0
    reduce_output_cb: int = 0
    reduce_scratch_cb: int = 0
    reduce_packet_cb: int = 0
    reduce_packet_header_cb: int = 0
    reduce_params: dict = None


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
        reduce_intermediate_tensors=None,
        reduce_output_tensor=None,
        reduce_semaphores=None,
        reduce_root_coord=None,
        # Semaphore IDs (caller-provided, see MlpOp for top-level definitions)
        mcast_data_sender_semaphore_id=0,
        mcast_data_receiver_semaphore_id=1,
        gather_noc0_receiver_semaphore_id=2,
        gather_noc1_receiver_semaphore_id=3,
        residual_mcast_receiver_semaphore_id=5,
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
        # ReduceToOne CBs (for multi-device reduce)
        # Must be after shared expert CBs (25-38) to avoid conflicts
        reduce_local_cb = add_cb_out  # Local data CB (same as add_cb_out for fusion)
        reduce_received_cb_r1 = 39  # Round 1: receives LEAF data
        reduce_received_cb_r2 = 40  # Round 2: receives ROOT3 data
        reduce_received_cb_r3 = 41  # Round 3: receives ROOT2 data
        reduce_output_cb = 42  # Final reduced output
        reduce_scratch_cb = 43  # Scratch for compute
        reduce_packet_cb = 44  # Scratch for sending packets
        reduce_packet_header_cb = 45  # Packet header (persistent)
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
        # ReduceToOne Setup
        # ==================================================================
        enable_reduce_to_one = (
            reduce_intermediate_tensors is not None
            and reduce_output_tensor is not None
            and reduce_semaphores is not None
            and reduce_root_coord is not None
            and mesh_rows == 4
            and mesh_cols == 2
        )

        reduce_params = {}
        if enable_reduce_to_one:
            # Get semaphore addresses
            reduce_sem_round1_addr = ttnn.get_global_semaphore_address(reduce_semaphores[0])
            reduce_sem_round2_addr = ttnn.get_global_semaphore_address(reduce_semaphores[1])
            reduce_sem_round3_addr = ttnn.get_global_semaphore_address(reduce_semaphores[2])
            reduce_sem_exit_addr = ttnn.get_global_semaphore_address(reduce_semaphores[3])

            # Get per-device tensors for reduce
            reduce_intermediate_r1_per_device = ttnn.get_device_tensors(reduce_intermediate_tensors[0])
            reduce_intermediate_r2_per_device = ttnn.get_device_tensors(reduce_intermediate_tensors[1])
            reduce_intermediate_r3_per_device = ttnn.get_device_tensors(reduce_intermediate_tensors[2])
            reduce_output_per_device = ttnn.get_device_tensors(reduce_output_tensor)

            # Calculate reduce tensor properties
            reduce_sample = reduce_intermediate_r1_per_device[0]
            reduce_element_size = 2  # bfloat16

            reduce_shard_spec = reduce_sample.memory_config().shard_spec
            reduce_shard_shape = reduce_shard_spec.shape

            # Compute tiles use 32x32 format
            reduce_compute_tile_h = 32
            reduce_compute_tile_w = 32
            reduce_compute_tile_size = reduce_compute_tile_h * reduce_compute_tile_w * reduce_element_size
            reduce_shard_elements = reduce_shard_shape[0] * reduce_shard_shape[1]
            reduce_num_tiles = (reduce_shard_elements + (reduce_compute_tile_h * reduce_compute_tile_w) - 1) // (
                reduce_compute_tile_h * reduce_compute_tile_w
            )

            reduce_payload_size_bytes = reduce_shard_elements * reduce_element_size
            reduce_packet_header_size = 96
            reduce_slot_size_bytes = reduce_packet_header_size + reduce_payload_size_bytes

            # Worker cores from final_output_tensor shard grid (same as gate_proj cores)
            reduce_worker_grid = final_output_tensor.memory_config().shard_spec.grid
            reduce_worker_cores_list = ttnn.corerange_to_cores(reduce_worker_grid, row_wise=True)
            reduce_num_workers = len(reduce_worker_cores_list)

            # Build core -> column mapping for fabric core assignment
            reduce_column_to_cores = {}
            for core in reduce_worker_cores_list:
                x = core.x
                if x not in reduce_column_to_cores:
                    reduce_column_to_cores[x] = []
                reduce_column_to_cores[x].append(core)

            reduce_sorted_columns = sorted(reduce_column_to_cores.keys())
            for x in reduce_sorted_columns:
                reduce_column_to_cores[x].sort(key=lambda c: c.y)

            reduce_num_columns = len(reduce_sorted_columns)
            reduce_num_workers_per_column = len(reduce_column_to_cores[reduce_sorted_columns[0]])

            # Fabric cores: one per column, placed to the right of bottom core
            reduce_fabric_cores = []
            reduce_column_to_fabric_core = {}
            for x in reduce_sorted_columns:
                bottom_core = max(reduce_column_to_cores[x], key=lambda c: c.y)
                fabric_core = ttnn.CoreCoord(bottom_core.x + 1, bottom_core.y)
                reduce_fabric_cores.append(fabric_core)
                reduce_column_to_fabric_core[x] = fabric_core

            # Core to slot index within column
            reduce_core_to_slot_idx = {}
            for x in reduce_sorted_columns:
                for slot_idx, core in enumerate(reduce_column_to_cores[x]):
                    reduce_core_to_slot_idx[(core.x, core.y)] = slot_idx

            # Core to shard index
            reduce_core_to_shard_idx = {}
            for shard_idx, core in enumerate(reduce_worker_cores_list):
                reduce_core_to_shard_idx[(core.x, core.y)] = shard_idx

            # Get output core from reduce_output_tensor
            reduce_output_sample = reduce_output_per_device[0]
            reduce_output_shard_spec = reduce_output_sample.memory_config().shard_spec
            reduce_output_core = reduce_output_shard_spec.grid.ranges()[0].start

            # Get per-device final_output_tensor for reduce_local_cb
            final_output_per_device = ttnn.get_device_tensors(final_output_tensor)

            reduce_params = {
                "sem_round1_addr": reduce_sem_round1_addr,
                "sem_round2_addr": reduce_sem_round2_addr,
                "sem_round3_addr": reduce_sem_round3_addr,
                "sem_exit_addr": reduce_sem_exit_addr,
                "intermediate_r1_per_device": reduce_intermediate_r1_per_device,
                "intermediate_r2_per_device": reduce_intermediate_r2_per_device,
                "intermediate_r3_per_device": reduce_intermediate_r3_per_device,
                "output_per_device": reduce_output_per_device,
                "final_output_per_device": final_output_per_device,
                "num_tiles": reduce_num_tiles,
                "payload_size_bytes": reduce_payload_size_bytes,
                "slot_size_bytes": reduce_slot_size_bytes,
                "compute_tile_size": reduce_compute_tile_size,
                "worker_cores_list": reduce_worker_cores_list,
                "num_workers": reduce_num_workers,
                "num_workers_per_column": reduce_num_workers_per_column,
                "num_columns": reduce_num_columns,
                "fabric_cores": reduce_fabric_cores,
                "column_to_fabric_core": reduce_column_to_fabric_core,
                "core_to_slot_idx": reduce_core_to_slot_idx,
                "core_to_shard_idx": reduce_core_to_shard_idx,
                "output_core": reduce_output_core,
            }

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
            # ReduceToOne
            enable_reduce_to_one=enable_reduce_to_one,
            reduce_local_cb=reduce_local_cb,
            reduce_received_cb_r1=reduce_received_cb_r1,
            reduce_received_cb_r2=reduce_received_cb_r2,
            reduce_received_cb_r3=reduce_received_cb_r3,
            reduce_output_cb=reduce_output_cb,
            reduce_scratch_cb=reduce_scratch_cb,
            reduce_packet_cb=reduce_packet_cb,
            reduce_packet_header_cb=reduce_packet_header_cb,
            reduce_params=reduce_params,
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
            # ReduceToOne reader args (CB indices + common RT arg base)
            ("reduce_local_cb", ctx.reduce_local_cb),
            ("reduce_received_cb_r1", ctx.reduce_received_cb_r1),
            ("reduce_received_cb_r2", ctx.reduce_received_cb_r2),
            ("reduce_received_cb_r3", ctx.reduce_received_cb_r3),
            ("reduce_ncrisc_common_rt_arg_base", 0),
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
            # ReduceToOne writer args (CB indices + RT arg bases)
            ("reduce_local_cb", ctx.reduce_local_cb),
            ("reduce_scratch_cb", ctx.reduce_scratch_cb),
            ("reduce_brisc_rt_arg_base", 0),
            ("reduce_brisc_fabric_rt_arg_base", 0),
        ]

        trisc_named_compile_time_args = [
            # RMSNorm compute
            ("rmsnorm_input_cb", ctx.residual_mcast_src_cb),
            ("rmsnorm_gamma_cb", ctx.rmsnorm_gamma_cb),
            ("rmsnorm_output_cb", ctx.rmsnorm_output_cb),
            ("rmsnorm_fp32_acc", 0),
            ("rmsnorm_num_tiles", ctx.rmsnorm_num_tiles),
            ("rmsnorm_rsqrt_fast_approx", 0),
            ("rmsnorm_trisc_common_rt_arg_base", 0),
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
            # ReduceToOne compute args (CB indices)
            ("reduce_local_cb", ctx.reduce_local_cb),
            ("reduce_received_cb_r1", ctx.reduce_received_cb_r1),
            ("reduce_received_cb_r2", ctx.reduce_received_cb_r2),
            ("reduce_received_cb_r3", ctx.reduce_received_cb_r3),
            ("reduce_output_cb", ctx.reduce_output_cb),
            ("reduce_scratch_cb", ctx.reduce_scratch_cb),
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
            # ReduceToOne core descriptors — will be updated per-device if enabled
            UnifiedCompileTimeCoreDescriptor(
                named_compile_time_arg="is_reduce_worker_core",
                core_range=ctx.gate_proj_core_ranges if ctx.enable_reduce_to_one else ttnn.CoreRangeSet([]),
                value=1,
                other_value=0,
            ),
            UnifiedCompileTimeCoreDescriptor(
                named_compile_time_arg="is_reduce_fabric_core",
                core_range=ttnn.CoreRangeSet([]),  # Set per-device
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

    # Semaphore IDs (top-level definition)
    # Gather sems overlap 1:1 with mcast receiver sems (different physical cores).
    # noc1_num_senders=0 for all gathers, so only noc0 sem is used.
    MCAST_SENDER_SEM = 0
    MCAST_DATA_RECEIVER_SEM = 1  # mcast grid; overlaps with DOWN_PROJ_GATHER_SEM on sender core
    DOWN_PROJ_GATHER_SEM = 1  # sender core
    RESIDUAL_MCAST_RECEIVER_SEM = 2  # mcast grid; overlaps with AG_GATHER_SEM on sender core
    AG_GATHER_SEM = 2  # sender core
    BG_GATHER_SEM = 3  # sender core
    SHARED_DOWN_MCAST_RECEIVER_SEM = 3  # mcast grid; overlaps with BG_GATHER_SEM on sender core
    OUTPUT_GATHER_SEM = 4  # sender core
    SHARED_OUTPUT_MCAST_RECEIVER_SEM = 4  # mcast grid; overlaps with OUTPUT_GATHER_SEM on sender core
    REDUCE_WORKER_FABRIC_SEM_BASE = 5  # on fabric cores only (5, 6, 7, 8 per worker slot)

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
        shared_gate_weights_overlapped,
        shared_up_weights_overlapped,
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
        num_iterations=1,
        # ReduceToOne parameters
        reduce_intermediate_tensors: Optional[list] = None,
        reduce_output_tensor: Optional[ttnn.Tensor] = None,
        reduce_semaphores: Optional[list] = None,
        reduce_root_coord: Optional[ttnn.MeshCoordinate] = None,
    ):
        """
        Execute the full fused MLP operation (dense MLP + shared expert).

        No routing tensors needed (no gate_mm_weights, gate_bias, gate_indices,
        gate_output_scores, gate_output_indices, expert_index, expert_scale, mul_scalar_buf).

        Args:
            num_iterations: Number of iterations to loop inside the kernel (default 1).
            reduce_intermediate_tensors: (Optional) List of 3 intermediate tensors for reduce rounds.
            reduce_output_tensor: (Optional) Final reduced output tensor on ROOT1 device.
            reduce_semaphores: (Optional) List of 4 global semaphores for reduce synchronization.
            reduce_root_coord: (Optional) MeshCoordinate of ROOT1 device.

        Returns:
            final_output_tensor, or reduce_output_tensor if reduce enabled.
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
            reduce_intermediate_tensors=reduce_intermediate_tensors,
            reduce_output_tensor=reduce_output_tensor,
            reduce_semaphores=reduce_semaphores,
            reduce_root_coord=reduce_root_coord,
            # Semaphore IDs from top-level
            mcast_data_sender_semaphore_id=MlpOp.MCAST_SENDER_SEM,
            mcast_data_receiver_semaphore_id=MlpOp.MCAST_DATA_RECEIVER_SEM,
            gather_noc0_receiver_semaphore_id=MlpOp.DOWN_PROJ_GATHER_SEM,
            gather_noc1_receiver_semaphore_id=MlpOp.DOWN_PROJ_GATHER_SEM,
            residual_mcast_receiver_semaphore_id=MlpOp.RESIDUAL_MCAST_RECEIVER_SEM,
        )

        # ==================================================================
        # Setup shared expert context (reused from MoE)
        # ==================================================================
        device_tensor = ttnn.get_device_tensors(input_tensor)[0]
        input_tile = device_tensor.get_tile()
        input_tile_size = input_tile.get_tile_size(routed_ctx.data_format)

        shared_ctx = MoeSharedExpertOp._setup_dimensions(
            device=routed_ctx.device,
            shared_gate_weights_overlapped=shared_gate_weights_overlapped,
            shared_up_weights_overlapped=shared_up_weights_overlapped,
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
            # Semaphore IDs from top-level
            ag_receiver_semaphore_id=MlpOp.AG_GATHER_SEM,
            bg_receiver_semaphore_id=MlpOp.BG_GATHER_SEM,
            ag_noc1_receiver_semaphore_id=MlpOp.AG_GATHER_SEM,
            bg_noc1_receiver_semaphore_id=MlpOp.BG_GATHER_SEM,
            shared_mcast_sender_semaphore_id=MlpOp.MCAST_SENDER_SEM,
            shared_mcast_receiver_semaphore_id=MlpOp.SHARED_DOWN_MCAST_RECEIVER_SEM,
            output_gather_noc0_receiver_semaphore_id=MlpOp.OUTPUT_GATHER_SEM,
            output_gather_noc1_receiver_semaphore_id=MlpOp.OUTPUT_GATHER_SEM,
            output_mcast_sender_semaphore_id=MlpOp.MCAST_SENDER_SEM,
            output_mcast_receiver_semaphore_id=MlpOp.SHARED_OUTPUT_MCAST_RECEIVER_SEM,
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
        # Semaphore descriptors (5 unique IDs: 0-4)
        # Gather sems overlap with mcast receiver sems (different physical cores).
        # Reduce fabric sems (5+) are added later when enable_reduce_to_one.
        # ==================================================================
        semaphore_descriptors = [
            ttnn.SemaphoreDescriptor(
                id=MlpOp.MCAST_SENDER_SEM,
                core_ranges=routed_ctx.full_device_grid,
                initial_value=0,
            ),
            ttnn.SemaphoreDescriptor(
                id=MlpOp.MCAST_DATA_RECEIVER_SEM,
                core_ranges=routed_ctx.full_device_grid,
                initial_value=0,
            ),
            ttnn.SemaphoreDescriptor(
                id=MlpOp.RESIDUAL_MCAST_RECEIVER_SEM,
                core_ranges=routed_ctx.full_device_grid,
                initial_value=0,
            ),
            ttnn.SemaphoreDescriptor(
                id=MlpOp.SHARED_DOWN_MCAST_RECEIVER_SEM,
                core_ranges=routed_ctx.full_device_grid,
                initial_value=0,
            ),
            ttnn.SemaphoreDescriptor(
                id=MlpOp.SHARED_OUTPUT_MCAST_RECEIVER_SEM,
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
            shared_gate_weights_overlapped.fused_tensor,
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
        enable_reduce_to_one = routed_ctx.enable_reduce_to_one
        reduce_params = routed_ctx.reduce_params
        mesh_device = input_tensor.device()

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

                # Loop iteration count (available to all RISCs in common section)
                ncrisc_args += [("num_iterations", num_iterations)]
                brisc_args += [("num_iterations", num_iterations)]
                trisc_args += [("num_iterations", num_iterations)]

                # Per-device copies for reduce modification
                device_unified_ct_core_descs = list(unified_core_descs)
                device_per_core_ct_descs = list(per_core_descs)
                device_cb_descriptors = list(cb_descriptors)
                device_semaphore_descriptors = list(semaphore_descriptors)
                device_runtime_args_descriptor = None

                # ReduceToOne per-device setup
                if enable_reduce_to_one:
                    # Get device role
                    device_role = get_reduce_device_role(coord, reduce_root_coord)

                    # Determine destination coordinate based on role
                    if device_role == MESH_LEAF:
                        if row == 0:
                            dest_coord = ttnn.MeshCoordinate(row + 1, col)
                        else:  # row == 3
                            dest_coord = ttnn.MeshCoordinate(row - 1, col)
                    elif device_role == MESH_ROOT3:
                        dest_coord = ttnn.MeshCoordinate(reduce_root_coord[0], col)
                    elif device_role == MESH_ROOT2:
                        dest_coord = reduce_root_coord
                    else:  # MESH_ROOT1
                        dest_coord = reduce_root_coord  # No actual send

                    # Get fabric node IDs
                    dest_fabric_node_id = mesh_device.get_fabric_node_id(dest_coord)

                    # Get per-device tensors for this device
                    r1_tensor = reduce_params["intermediate_r1_per_device"][chip_id]
                    r2_tensor = reduce_params["intermediate_r2_per_device"][chip_id]
                    r3_tensor = reduce_params["intermediate_r3_per_device"][chip_id]
                    out_tensor = reduce_params["output_per_device"][chip_id]

                    # Create CB descriptors for reduce operation
                    reduce_all_cores_set = ttnn.CoreRangeSet(
                        [ttnn.CoreRange(c, c) for c in reduce_params["worker_cores_list"]]
                        + [ttnn.CoreRange(c, c) for c in reduce_params["fabric_cores"]]
                    )
                    reduce_tile_desc = ttnn.TileDescriptor(32, 32)  # 32x32 compute tiles
                    reduce_payload = reduce_params["payload_size_bytes"]
                    reduce_dtype = ttnn.bfloat16

                    # reduce_received_cb_r1: backed by intermediate r1 tensor
                    reduce_cb_r1_desc = ttnn.cb_descriptor_from_sharded_tensor(
                        routed_ctx.reduce_received_cb_r1, r1_tensor
                    )
                    reduce_cb_r1_desc.core_ranges = reduce_all_cores_set
                    reduce_cb_r1_desc.total_size = reduce_payload
                    reduce_cb_r1_desc.format_descriptors = [
                        ttnn.CBFormatDescriptor(
                            buffer_index=routed_ctx.reduce_received_cb_r1,
                            data_format=reduce_dtype,
                            page_size=reduce_payload,
                            tile=reduce_tile_desc,
                        )
                    ]
                    device_cb_descriptors.append(reduce_cb_r1_desc)

                    # reduce_received_cb_r2: backed by intermediate r2 tensor
                    reduce_cb_r2_desc = ttnn.cb_descriptor_from_sharded_tensor(
                        routed_ctx.reduce_received_cb_r2, r2_tensor
                    )
                    reduce_cb_r2_desc.core_ranges = reduce_all_cores_set
                    reduce_cb_r2_desc.total_size = reduce_payload
                    reduce_cb_r2_desc.format_descriptors = [
                        ttnn.CBFormatDescriptor(
                            buffer_index=routed_ctx.reduce_received_cb_r2,
                            data_format=reduce_dtype,
                            page_size=reduce_payload,
                            tile=reduce_tile_desc,
                        )
                    ]
                    device_cb_descriptors.append(reduce_cb_r2_desc)

                    # reduce_received_cb_r3: backed by intermediate r3 tensor
                    reduce_cb_r3_desc = ttnn.cb_descriptor_from_sharded_tensor(
                        routed_ctx.reduce_received_cb_r3, r3_tensor
                    )
                    reduce_cb_r3_desc.core_ranges = reduce_all_cores_set
                    reduce_cb_r3_desc.total_size = reduce_payload
                    reduce_cb_r3_desc.format_descriptors = [
                        ttnn.CBFormatDescriptor(
                            buffer_index=routed_ctx.reduce_received_cb_r3,
                            data_format=reduce_dtype,
                            page_size=reduce_payload,
                            tile=reduce_tile_desc,
                        )
                    ]
                    device_cb_descriptors.append(reduce_cb_r3_desc)

                    # reduce_output_cb: backed by reduce output tensor
                    reduce_cb_out_desc = ttnn.cb_descriptor_from_sharded_tensor(routed_ctx.reduce_output_cb, out_tensor)
                    reduce_cb_out_desc.core_ranges = reduce_all_cores_set
                    reduce_cb_out_desc.total_size = reduce_payload
                    reduce_cb_out_desc.format_descriptors = [
                        ttnn.CBFormatDescriptor(
                            buffer_index=routed_ctx.reduce_output_cb,
                            data_format=reduce_dtype,
                            page_size=reduce_payload,
                            tile=reduce_tile_desc,
                        )
                    ]
                    device_cb_descriptors.append(reduce_cb_out_desc)

                    # reduce_scratch_cb: scratch buffer for compute (non-tensor-backed)
                    reduce_scratch_size = reduce_params["compute_tile_size"] * reduce_params["num_tiles"]
                    reduce_cb_scratch_desc = ttnn.CBDescriptor(
                        total_size=reduce_scratch_size,
                        core_ranges=reduce_all_cores_set,
                        format_descriptors=[
                            ttnn.CBFormatDescriptor(
                                buffer_index=routed_ctx.reduce_scratch_cb,
                                data_format=ttnn.bfloat16,
                                page_size=reduce_params["compute_tile_size"],
                                tile=reduce_tile_desc,
                            )
                        ],
                    )
                    device_cb_descriptors.append(reduce_cb_scratch_desc)

                    # reduce_packet_cb
                    reduce_packet_size = reduce_params["slot_size_bytes"] * reduce_params["num_workers_per_column"]
                    reduce_cb_packet_desc = ttnn.CBDescriptor(
                        total_size=reduce_packet_size,
                        core_ranges=reduce_all_cores_set,
                        format_descriptors=[
                            ttnn.CBFormatDescriptor(
                                buffer_index=routed_ctx.reduce_packet_cb,
                                data_format=ttnn.bfloat16,
                                page_size=reduce_params["slot_size_bytes"],
                            )
                        ],
                    )
                    device_cb_descriptors.append(reduce_cb_packet_desc)

                    # reduce_packet_header_cb: persistent packet header storage
                    reduce_packet_header_size = 96  # Standard packet header size
                    reduce_cb_packet_header_desc = ttnn.CBDescriptor(
                        total_size=reduce_packet_header_size,
                        core_ranges=reduce_all_cores_set,
                        format_descriptors=[
                            ttnn.CBFormatDescriptor(
                                buffer_index=routed_ctx.reduce_packet_header_cb,
                                data_format=ttnn.bfloat16,
                                page_size=reduce_packet_header_size,
                            )
                        ],
                    )
                    device_cb_descriptors.append(reduce_cb_packet_header_desc)

                    # Destination L1 address depends on role
                    if device_role == MESH_LEAF:
                        dst_l1_addr = r1_tensor.buffer_address()
                        dst_sem_addr = reduce_params["sem_round1_addr"]
                    elif device_role == MESH_ROOT3:
                        dst_l1_addr = r2_tensor.buffer_address()
                        dst_sem_addr = reduce_params["sem_round2_addr"]
                    elif device_role == MESH_ROOT2:
                        dst_l1_addr = r3_tensor.buffer_address()
                        dst_sem_addr = reduce_params["sem_round3_addr"]
                    else:  # MESH_ROOT1
                        dst_l1_addr = 0
                        dst_sem_addr = reduce_params["sem_exit_addr"]

                    # Get physical coords for output core
                    output_core_phys = routed_ctx.device.worker_core_from_logical_core(reduce_params["output_core"])

                    # Add reduce-specific compile-time args (per-device values)
                    ncrisc_args.extend(
                        [
                            ("reduce_device_role", device_role),
                            ("reduce_num_tiles", reduce_params["num_tiles"]),
                        ]
                    )

                    brisc_args.extend(
                        [
                            ("reduce_device_role", device_role),
                            ("reduce_num_tiles", reduce_params["num_tiles"]),
                            ("reduce_payload_size_bytes", reduce_params["payload_size_bytes"]),
                            ("reduce_num_hops", 1),
                            ("reduce_dst_fabric_node_chip_id", dest_fabric_node_id.chip_id),
                            ("reduce_dst_fabric_node_mesh_id", int(dest_fabric_node_id.mesh_id)),
                            ("reduce_output_core_noc_x", output_core_phys.x),
                            ("reduce_output_core_noc_y", output_core_phys.y),
                            ("reduce_num_workers", reduce_params["num_workers_per_column"]),
                            ("reduce_slot_size_bytes", reduce_params["slot_size_bytes"]),
                            ("reduce_packet_cb", routed_ctx.reduce_packet_cb),
                            ("reduce_packet_header_cb", routed_ctx.reduce_packet_header_cb),
                        ]
                    )

                    trisc_args.extend(
                        [
                            ("reduce_device_role", device_role),
                            ("reduce_num_tiles", reduce_params["num_tiles"]),
                        ]
                    )

                    # Update fabric core descriptor for this device
                    fabric_core_set = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in reduce_params["fabric_cores"]])
                    for i, desc in enumerate(device_unified_ct_core_descs):
                        if desc.named_compile_time_arg == "is_reduce_fabric_core":
                            device_unified_ct_core_descs[i] = UnifiedCompileTimeCoreDescriptor(
                                named_compile_time_arg="is_reduce_fabric_core",
                                core_range=fabric_core_set,
                                value=1,
                                other_value=0,
                            )
                            break

                    # Build per-core BRISC args for reduce worker cores
                    reduce_worker_fabric_sem_base = MlpOp.REDUCE_WORKER_FABRIC_SEM_BASE
                    reduce_brisc_per_core_args = []
                    for core in reduce_params["worker_cores_list"]:
                        fabric_core = reduce_params["column_to_fabric_core"][core.x]
                        fabric_core_phys = routed_ctx.device.worker_core_from_logical_core(fabric_core)
                        slot_idx = reduce_params["core_to_slot_idx"][(core.x, core.y)]
                        shard_idx = reduce_params["core_to_shard_idx"][(core.x, core.y)]

                        worker_args = [
                            fabric_core_phys.x,  # fabric_core_noc_x
                            fabric_core_phys.y,  # fabric_core_noc_y
                            slot_idx,  # my_slot_idx
                            reduce_worker_fabric_sem_base + slot_idx,  # worker_sem_id
                            dst_l1_addr,  # dst_l1_addr
                            dst_sem_addr,  # dst_sem_addr
                            out_tensor.buffer_address(),  # output_base_addr
                            shard_idx,  # shard_idx
                        ]
                        reduce_brisc_per_core_args.append((core, worker_args))

                    # Fabric cores BRISC args: worker semaphore IDs
                    for fc in reduce_params["fabric_cores"]:
                        fabric_args = [
                            reduce_worker_fabric_sem_base + i for i in range(reduce_params["num_workers_per_column"])
                        ]
                        reduce_brisc_per_core_args.append((fc, fabric_args))

                    device_runtime_args_descriptor = PerCoreRuntimeArgsDescriptor(
                        brisc_args=reduce_brisc_per_core_args,
                    )

                # Build NCRISC common runtime args for reduce (semaphore addresses)
                ncrisc_common_rt_args = []
                if enable_reduce_to_one:
                    ncrisc_common_rt_args = [
                        reduce_params["sem_round1_addr"],
                        reduce_params["sem_round2_addr"],
                        reduce_params["sem_round3_addr"],
                    ]

                # Build defines list
                kernel_defines = []
                if enable_reduce_to_one:
                    kernel_defines.append(("ENABLE_REDUCE_TO_ONE", "1"))

                # Create unified kernel
                unified_kernel = UnifiedKernelDescriptor(
                    kernel_source="models/demos/deepseek_v3_b1/fused_ops/mlp/mlp_kernel.cpp",
                    core_ranges=routed_ctx.full_device_grid,
                    ncrisc_named_compile_time_args=ncrisc_args,
                    brisc_named_compile_time_args=brisc_args,
                    trisc_named_compile_time_args=trisc_args,
                    ncrisc_common_runtime_args=ncrisc_common_rt_args,
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
                    unified_compile_time_core_descriptors=device_unified_ct_core_descs,
                    per_core_compile_time_descriptors=device_per_core_ct_descs,
                    per_core_runtime_args_descriptor=device_runtime_args_descriptor,
                    defines=kernel_defines,
                )

                kernel_result = unified_kernel.get_kernel_descriptors()

                # Add worker→fabric semaphores on fabric cores for reduce signaling
                if enable_reduce_to_one:
                    fabric_core_set = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in reduce_params["fabric_cores"]])
                    for worker_idx in range(reduce_params["num_workers_per_column"]):
                        sem_desc = ttnn.SemaphoreDescriptor(
                            id=reduce_worker_fabric_sem_base + worker_idx,
                            core_type=ttnn.CoreType.WORKER,
                            core_ranges=fabric_core_set,
                            initial_value=0,
                        )
                        device_semaphore_descriptors.append(sem_desc)

                # Create program
                program = ttnn.ProgramDescriptor(
                    kernels=kernel_result.kernels,
                    cbs=device_cb_descriptors,
                    semaphores=device_semaphore_descriptors,
                )

                # Setup fabric connection for reduce fabric cores
                # Note: fabric cores may span multiple kernel groups due to shared expert
                # core descriptors (is_shared_gate_compute_core, etc.), so we find the
                # correct BRISC kernel index per fabric core rather than assuming one group.
                if enable_reduce_to_one and device_role != MESH_ROOT1:
                    fabric_node_id = mesh_device.get_fabric_node_id(coord)
                    num_columns = reduce_params["num_columns"]
                    for fc_idx, fc in enumerate(reduce_params["fabric_cores"]):
                        col_idx = fc_idx
                        link_idx = 0 if col_idx < num_columns // 2 else 1
                        # Find the kernel group containing this specific fabric core
                        fc_kernel_idx = None
                        for group in kernel_result.groups:
                            if group.compile_time_arg_values.get(
                                "is_reduce_fabric_core"
                            ) == 1 and group.core_range_set.contains(fc):
                                fc_kernel_idx = group.brisc_kernel_index
                                break
                        fabric_rt_args_ref = program.kernels[fc_kernel_idx].runtime_args[fc.x][fc.y]
                        fabric_conn_args = ttnn.setup_fabric_connection(
                            fabric_node_id,
                            dest_fabric_node_id,
                            link_idx,
                            program,
                            fc,
                        )
                        fabric_rt_args_ref.extend(fabric_conn_args)

                # Assign to mesh coordinate
                mesh_program_descriptor[ttnn.MeshCoordinateRange(coord, coord)] = program

        # Add reduce tensors to io_tensors if enabled
        if enable_reduce_to_one:
            io_tensors.extend(
                [
                    reduce_intermediate_tensors[0],
                    reduce_intermediate_tensors[1],
                    reduce_intermediate_tensors[2],
                    reduce_output_tensor,
                ]
            )

        # Execute
        ttnn.generic_op(io_tensors, mesh_program_descriptor)

        # Return appropriate output based on reduce mode
        if enable_reduce_to_one:
            return reduce_output_tensor
        return final_output_tensor
