# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Expert Kernel: SRAM matmul with compressed weights.

Unified single-device and multi-device implementation. Single device is the
degenerate case of a 1×1 mesh — no special handling needed. For each device in
the mesh, a separate ProgramDescriptor is built with per-device per-core runtime
metadata tensors derived from that device's compressed format assignment.

Callers are responsible for creating fmt tensors via create_expert_fmt_tensors()
and passing them to ExpertKernel.op().
"""

import numpy as np
import torch

import ttnn
from models.demos.deepseek_v3_b1.micro_ops.matmul_custom_compressed.op import _CB_ADDR_SHIFT, pack_tile_pairs
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import PerCoreCompileTimeDescriptor, UnifiedKernelDescriptor

_KERNEL_SOURCE = "models/demos/deepseek_v3_b1/micro_ops/matmul_expert/kernels/matmul_expert_kernel.cpp"


def _align(n: int, alignment: int) -> int:
    return ((n + alignment - 1) // alignment) * alignment


def create_expert_fmt_tensors(cts: list, mesh_device, core_grid, num_tiles: int) -> dict:
    """Create per-device per-core format metadata tensors for ExpertKernel.

    Builds a 2D table [num_experts, num_tiles] of packed (addr:24|fmt:8) uint32 entries
    per core. The kernel indexes into this table via expert_idx to select the active expert.

    Must be called before ExpertKernel.op() and the returned tensors must be
    kept alive for the duration of the op call.

    Args:
        cts: List of CompressedTensor, one per expert (all with per-core allocation,
             same K×N tile dimensions but potentially different compressed sizes).
        mesh_device: The mesh device.
        core_grid: CoreRangeSet used for B and output tensors.
        num_tiles: Total tiles per core per expert (K // 32 * N_per_device // 32).

    Returns:
        dict {MeshCoordinate: {core_idx: ttnn.Tensor}}
    """
    num_experts = len(cts)
    mesh_shape = mesh_device.shape
    all_cores = ttnn.corerange_to_cores(core_grid)
    raw_size = num_experts * num_tiles * 4
    dram_alignment = ttnn._ttnn.bfp_utils.get_dram_alignment()
    aligned_size = _align(max(raw_size, dram_alignment), dram_alignment)

    result = {}
    for row in range(mesh_shape[0]):
        for col in range(mesh_shape[1]):
            coord = ttnn.MeshCoordinate(row, col)
            tensors = {}
            for core_idx, core_coord in enumerate(all_cores):
                # Pack all experts consecutively: [expert0_tile0..tileN, expert1_tile0..tileN, ...]
                all_tiles = []
                for ct in cts:
                    base_addr_shifted = (
                        ct.get_data_l1_address_per_core(core_coord, device_coord=coord) >> _CB_ADDR_SHIFT
                    ) - 1
                    shard_assignment = ct.get_assignment_per_shard(core_coord, device_coord=coord)
                    all_tiles.extend(pack_tile_pairs(shard_assignment, base_addr_shifted))

                data_np = np.array(all_tiles, dtype=np.uint32).view(np.uint8)
                pad = aligned_size - raw_size
                if pad > 0:
                    data_np = np.concatenate([data_np, np.zeros(pad, dtype=np.uint8)])
                core_torch = torch.from_numpy(data_np.copy()).reshape(1, aligned_size)
                core_shard_spec = ttnn.ShardSpec(
                    ttnn.CoreRangeSet([ttnn.CoreRange(core_coord, core_coord)]),
                    [1, aligned_size],
                    ttnn.ShardOrientation.ROW_MAJOR,
                )
                core_mem_config = ttnn.per_core_allocation.MemoryConfig(
                    ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                    ttnn.BufferType.L1,
                    core_shard_spec,
                )
                host_tensor = ttnn.from_torch(core_torch, dtype=ttnn.uint8, layout=ttnn.ROW_MAJOR_LAYOUT)
                tensors[core_idx] = ttnn._ttnn.per_core_allocation.to_single_device(
                    host_tensor,
                    mesh_device,
                    coord,
                    core_mem_config,
                )
            result[coord] = tensors
    return result


def _build_program_for_device(
    cts: list, a_device, out_device, index_device, fmt_tensors: dict, coord
) -> ttnn.ProgramDescriptor:
    """Build a ProgramDescriptor for one device in the mesh.

    Args:
        cts: List of CompressedTensors (one per expert). The CT with the largest
             per-core buffer is used for cb_in1 setup.
        index_device: Per-device index tensor ([1, 16] HEIGHT_SHARDED, uint16). The first
                      element is the expert index; NCRISC sets up its CB so TRISC can read it.
    """
    core_grid = a_device.memory_config().shard_spec.grid
    a_shard_shape = a_device.memory_config().shard_spec.shape
    out_shard_shape = out_device.memory_config().shard_spec.shape
    num_tiles_k = a_shard_shape[1] // 32
    out_w = out_shard_shape[1] // 32

    assert (num_tiles_k * out_w) % 2 == 0, f"total tiles K*N={num_tiles_k * out_w} must be even"
    assert out_w == 1 or out_w % 2 == 0, f"out_w must be 1 or even, got {out_w}"

    cb_in0, cb_in1, cb_out, cb_index = 0, 1, 2, 3

    cb0_desc = ttnn.cb_descriptor_from_sharded_tensor(cb_in0, a_device)
    # cb_in1 is used for initial format config only — actual per-tile addresses
    # come from the fmt table. Any CT works; cts[0] is simplest.
    cb1_descs = cts[0].cb_descriptor_from_compressed_tensor(cb_in1, device_coord=coord)
    cb2_desc = ttnn.cb_descriptor_from_sharded_tensor(cb_out, out_device)
    cb3_desc = ttnn.cb_descriptor_from_sharded_tensor(cb_index, index_device)

    all_cores = ttnn.corerange_to_cores(core_grid)

    named_ct_args = [
        ("cb_in0", cb_in0),
        ("cb_in1", cb_in1),
        ("cb_out", cb_out),
        ("cb_index", cb_index),
        ("num_tiles_k", num_tiles_k),
        ("out_w", out_w),
        ("cb_in0_num_pages", num_tiles_k),
        ("cb_in1_num_pages", 1),
    ]
    unified_kernel = UnifiedKernelDescriptor(
        kernel_source=_KERNEL_SOURCE,
        core_ranges=core_grid,
        ncrisc_named_compile_time_args=named_ct_args,
        brisc_named_compile_time_args=named_ct_args,
        trisc_named_compile_time_args=named_ct_args,
        trisc_compile_time_args=[],
        per_core_compile_time_descriptors=[
            PerCoreCompileTimeDescriptor(
                named_compile_time_arg="fmt_l1_addr",
                core_values=[
                    (all_cores[i], ttnn.per_core_allocation.per_core_buffer_address(fmt_tensors[i], all_cores[i]))
                    for i in range(len(all_cores))
                ],
                other_value=0,
            )
        ],
        trisc_compute_config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            dst_full_sync_en=False,
        ),
    )

    return ttnn.ProgramDescriptor(
        kernels=unified_kernel.get_kernel_descriptors().kernels,
        cbs=[cb0_desc, *cb1_descs, cb2_desc, cb3_desc],
        semaphores=[],
    )


class ExpertKernel:
    """SRAM matmul with compressed weights — single-device and multi-device unified."""

    @staticmethod
    def op(
        a_tensor: ttnn.Tensor,
        cts: list,
        output_tensor: ttnn.Tensor,
        fmt_tensors_per_device: dict,
        index_tensor: ttnn.Tensor = None,
    ) -> ttnn.Tensor:
        """
        Execute expert kernel across all devices in the mesh.

        Args:
            a_tensor: A [M, K] mesh tensor (HEIGHT_SHARDED, one core per device).
            cts: List of CompressedTensor for B [K, N], per-core allocation, multi-device.
                 Pass a single-element list for single-expert use (index_tensor with value 0).
            output_tensor: Pre-allocated output [M, N] mesh tensor, WIDTH_SHARDED.
            fmt_tensors_per_device: {MeshCoordinate: {core_idx: ttnn.Tensor}} from
                create_expert_fmt_tensors(). Tensors must stay alive through this call.
            index_tensor: HEIGHT_SHARDED mesh tensor, [1, 16] per core, uint16. The first
                          element is the expert index to run (selects row in the fmt table).
                          Same format as DRAMStreamingExpertsMatmul index_tensor.

        Returns:
            output_tensor (modified in-place by generic_op).
        """
        mesh_device = a_tensor.device()
        mesh_shape = mesh_device.shape
        mesh_rows, mesh_cols = mesh_shape[0], mesh_shape[1]

        a_per_device = ttnn.get_device_tensors(a_tensor)
        out_per_device = ttnn.get_device_tensors(output_tensor)
        index_per_device = ttnn.get_device_tensors(index_tensor)

        mesh_program = ttnn.MeshProgramDescriptor()

        for row in range(mesh_rows):
            for col in range(mesh_cols):
                coord = ttnn.MeshCoordinate(row, col)
                device_idx = row * mesh_cols + col

                a_device = a_per_device[device_idx]
                out_device = out_per_device[device_idx]
                index_device = index_per_device[device_idx]

                program = _build_program_for_device(
                    cts, a_device, out_device, index_device, fmt_tensors_per_device[coord], coord
                )
                mesh_program[ttnn.MeshCoordinateRange(coord, coord)] = program

        all_fmt_tensors = [t for per_device in fmt_tensors_per_device.values() for t in per_device.values()]
        all_ct_data = [t for ct in cts for t in ct.get_data_tensors()]
        io_tensors = [a_tensor, *all_ct_data, output_tensor, index_tensor, *all_fmt_tensors]
        ttnn.generic_op(io_tensors, mesh_program)
        return output_tensor
