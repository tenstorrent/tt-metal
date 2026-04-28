# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Fused Gate + Up DRAM expert matmul reproducer.

Runs MatmulExpertCompressedDRAM::Op twice in the same kernel invocation, once
with gate_proj's CTArgs and once with up_proj's. Both ops share cb_in0
(activation), cb_index, and cb_in1 L1 region (gate's backing). gate has
pop_in0=false/pop_index=false, up has pop_in0=true/pop_index=true. ResetCBIn1
is true for both with cb_in1_buf_addr = gate's in1_backing addr.

Used to isolate the K-split gate→up back-to-back hang in MoE.
"""

from loguru import logger

import ttnn
from models.demos.deepseek_v3_b1.micro_ops.host_io.utils import dtype_size
from models.demos.deepseek_v3_b1.micro_ops.matmul_expert.op import (
    _TILE_SIZES,
    _align,
    _meta_words_for_tiles,
    _pad_to_face_r_dim,
)
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import PerCoreCompileTimeDescriptor, UnifiedKernelDescriptor

_KERNEL_SOURCE = "models/demos/deepseek_v3_b1/fused_ops/fused_gate_up/kernels/fused_gate_up_kernel.cpp"


class FusedGateUp:
    """Run gate_proj then up_proj DRAM matmul back-to-back in a single kernel."""

    @staticmethod
    def op(
        a_tensor,
        gate_dram_cts,
        up_dram_cts,
        gate_output_tensor,
        up_output_tensor,
        index_tensor,
        gate_dram_meta_tensors,
        up_dram_meta_tensors,
        num_active_experts,
        subblock_k,
        subblock_n,
        dram_per_core_n,
        dram_core_grid,
        cores_per_dram_bank=1,
        k_parallel_per_bank=1,
        primary_at_last_offset=False,
        dram_fuse_silu=False,
    ):
        """
        Args mirror ExpertKernel.op but with separate inputs for gate and up
        (weights, dram_meta_tensors, output tensors). Activation and index are
        shared. cb_in1 L1 region is shared across both ops via gate's backing.
        """
        mesh_device = a_tensor.device()
        mesh_shape = mesh_device.shape
        mesh_rows, mesh_cols = mesh_shape[0], mesh_shape[1]

        a_per_device = ttnn.get_device_tensors(a_tensor)
        idx_per_device = ttnn.get_device_tensors(index_tensor)
        gate_out_per_device = ttnn.get_device_tensors(gate_output_tensor)
        up_out_per_device = ttnn.get_device_tensors(up_output_tensor)

        mesh_program = ttnn.MeshProgramDescriptor()
        for row in range(mesh_rows):
            for col in range(mesh_cols):
                coord = ttnn.MeshCoordinate(row, col)
                dev_idx = row * mesh_cols + col

                program = _build_program(
                    a_dev=a_per_device[dev_idx],
                    idx_dev=idx_per_device[dev_idx],
                    gate_out_dev=gate_out_per_device[dev_idx],
                    up_out_dev=up_out_per_device[dev_idx],
                    gate_meta=gate_dram_meta_tensors[coord],
                    up_meta=up_dram_meta_tensors[coord],
                    coord=coord,
                    num_active_experts=num_active_experts,
                    subblock_k=subblock_k,
                    subblock_n=subblock_n,
                    dram_per_core_n=dram_per_core_n,
                    dram_core_grid=dram_core_grid,
                    cores_per_dram_bank=cores_per_dram_bank,
                    k_parallel_per_bank=k_parallel_per_bank,
                    primary_at_last_offset=primary_at_last_offset,
                    dram_fuse_silu=dram_fuse_silu,
                )
                mesh_program[ttnn.MeshCoordinateRange(coord, coord)] = program

        # Collect live tensors so the ttnn runtime keeps them alive.
        all_ct_data = [t for ct in (gate_dram_cts + up_dram_cts) for t in ct.get_data_tensors()]
        per_device_dram = []
        for meta in list(gate_dram_meta_tensors.values()) + list(up_dram_meta_tensors.values()):
            in1_backing, (offset_t, bsize_t), fmt_info, *_ = meta
            per_device_dram.extend([in1_backing, *offset_t.values(), *bsize_t.values(), fmt_info["fmt_dram_tensor"]])

        io_tensors = [
            a_tensor,
            *all_ct_data,
            gate_output_tensor,
            up_output_tensor,
            index_tensor,
            *per_device_dram,
        ]

        logger.info("FusedGateUp: running kernel...")
        ttnn.generic_op(io_tensors, mesh_program)
        return gate_output_tensor, up_output_tensor


def _build_program(
    a_dev,
    idx_dev,
    gate_out_dev,
    up_out_dev,
    gate_meta,
    up_meta,
    coord,
    num_active_experts,
    subblock_k,
    subblock_n,
    dram_per_core_n,
    dram_core_grid,
    cores_per_dram_bank,
    k_parallel_per_bank,
    primary_at_last_offset,
    dram_fuse_silu,
):
    """Build the per-device program with all gate_X and up_X CT args."""
    # Unpack dram_meta tuples (17 elements each — see _assemble_dram_results).
    (
        gate_in1_backing,
        (gate_offset_tensors, gate_bsize_tensors),
        gate_fmt_info,
        (gate_offsets_l1_cv, gate_bsize_l1_cv),
        gate_pcv,
        _gate_num_in1_buf,
        gate_fmt_cb_l1_addr,
        gate_fmt_sem_addr_0,
        gate_fmt_sem_addr_1,
        _gs0,
        _gs1,
        gate_partial_sem_addr,
        gate_pipeline_sem_addr,
        _gp,
        _gpi,
        gate_gather_sync_sem_addr,
        _ggs,
    ) = gate_meta
    (
        up_in1_backing,
        (up_offset_tensors, up_bsize_tensors),
        up_fmt_info,
        (up_offsets_l1_cv, up_bsize_l1_cv),
        up_pcv,
        _up_num_in1_buf,
        up_fmt_cb_l1_addr,
        up_fmt_sem_addr_0,
        up_fmt_sem_addr_1,
        _us0,
        _us1,
        up_partial_sem_addr,
        up_pipeline_sem_addr,
        _up_,
        _upi,
        up_gather_sync_sem_addr,
        _ugs,
    ) = up_meta

    # Activation params.
    K_total = a_dev.memory_config().shard_spec.shape[1]
    Kt = K_total // 32
    tile_h, tile_w = a_dev.get_tile().tile_shape
    in0_page_size = tile_h * tile_w * dtype_size(a_dev.dtype)

    num_subblocks_k = Kt // subblock_k
    assert num_subblocks_k % k_parallel_per_bank == 0
    num_subblocks_k_local = num_subblocks_k // k_parallel_per_bank

    # NOC max page size by arch.
    arch = a_dev.device().arch()
    if arch == ttnn.device.Arch.BLACKHOLE:
        noc_max_page_size = 16384
    elif arch == ttnn.device.Arch.WORMHOLE_B0:
        noc_max_page_size = 8192
    else:
        raise ValueError(f"Unsupported architecture: {arch}")

    # CB sizing — 3 buffers of subblock_k×subblock_n bf4 tiles.
    max_tile_size = _TILE_SIZES[1]
    num_in1_buffers = 3
    in1_region_bytes = subblock_k * subblock_n * num_in1_buffers * max_tile_size
    dram_alignment = ttnn._ttnn.bfp_utils.get_dram_alignment()

    # CB ids.
    cb_in0 = 0
    cb_in1 = 1  # shared between gate and up (same id, same L1)
    gate_cb_out = 2
    cb_index = 3
    gate_cb_fmt = 4
    up_cb_fmt = 5
    up_cb_out = 6
    gate_cb_out_silu = 7  # alias of gate_cb_out

    cbs = []

    cbs.append(ttnn.cb_descriptor_from_sharded_tensor(cb_in0, a_dev))
    cbs.append(ttnn.cb_descriptor_from_sharded_tensor(cb_index, idx_dev))

    # cb_in1 — point at gate's backing tensor's in1 region.
    cb_in1_desc = ttnn.cb_descriptor_from_sharded_tensor(
        cb_in1, gate_in1_backing, address_offset=0, total_size=in1_region_bytes
    )
    cb_in1_desc.format_descriptors = [
        ttnn.CBFormatDescriptor(
            buffer_index=cb_in1,
            data_format=ttnn.bfloat4_b,
            page_size=_TILE_SIZES[1],
        ),
    ]
    cbs.append(cb_in1_desc)

    cbs.append(ttnn.cb_descriptor_from_sharded_tensor(gate_cb_out, gate_out_dev))
    cbs.append(ttnn.cb_descriptor_from_sharded_tensor(up_cb_out, up_out_dev))

    # cb_fmt for gate (overlaid on gate's backing at offset in1_region_bytes).
    gate_fmt_per_expert = gate_fmt_info["fmt_per_expert_bytes"]
    gate_cb_fmt_page_size = _align(max(gate_fmt_per_expert, dram_alignment), dram_alignment)
    cb_fmt_gate_desc = ttnn.cb_descriptor_from_sharded_tensor(
        gate_cb_fmt, gate_in1_backing, address_offset=in1_region_bytes, total_size=gate_cb_fmt_page_size
    )
    cb_fmt_gate_desc.format_descriptors = [
        ttnn.CBFormatDescriptor(buffer_index=gate_cb_fmt, data_format=ttnn.uint8, page_size=gate_cb_fmt_page_size),
    ]
    cbs.append(cb_fmt_gate_desc)

    # cb_fmt for up (overlaid on up's backing — separate L1 for up's fmt).
    up_fmt_per_expert = up_fmt_info["fmt_per_expert_bytes"]
    up_cb_fmt_page_size = _align(max(up_fmt_per_expert, dram_alignment), dram_alignment)
    cb_fmt_up_desc = ttnn.cb_descriptor_from_sharded_tensor(
        up_cb_fmt, up_in1_backing, address_offset=in1_region_bytes, total_size=up_cb_fmt_page_size
    )
    cb_fmt_up_desc.format_descriptors = [
        ttnn.CBFormatDescriptor(buffer_index=up_cb_fmt, data_format=ttnn.uint8, page_size=up_cb_fmt_page_size),
    ]
    cbs.append(cb_fmt_up_desc)

    # silu fast-path alias (only when fuse_silu).
    silu_tile_h = 0
    if dram_fuse_silu:
        raw_silu_tile_h = num_active_experts * dram_per_core_n * tile_h
        silu_tile_h = _pad_to_face_r_dim(raw_silu_tile_h)
        silu_tile = ttnn.Tile([silu_tile_h, tile_w])
        silu_tile_bytes = silu_tile.get_tile_size(gate_out_dev.dtype)
        cb_silu_desc = ttnn.cb_descriptor_from_sharded_tensor(gate_cb_out_silu, gate_out_dev)
        cb_silu_desc.format_descriptors[0].tile = ttnn.TileDescriptor(silu_tile)
        cb_silu_desc.format_descriptors[0].page_size = silu_tile_bytes
        cbs.append(cb_silu_desc)

    gate_meta_words_per_block = _meta_words_for_tiles(subblock_k * subblock_n)
    up_meta_words_per_block = _meta_words_for_tiles(subblock_k * subblock_n)

    # Named CT args (shared across all RISCs).
    named_ct_args = [
        # ── Shared ─────────────────────────────────────────────
        ("cb_in0", cb_in0),
        ("cb_in1", cb_in1),
        ("cb_index", cb_index),
        ("num_tiles_k", Kt),
        ("subblock_k", subblock_k),
        ("subblock_n", subblock_n),
        ("num_subblocks_k", num_subblocks_k),
        ("per_core_n", dram_per_core_n),
        ("cb_in1_size_bytes", in1_region_bytes),
        ("cb_in1_buf_addr", gate_in1_backing.buffer_address()),
        ("noc_max_page_size", noc_max_page_size),
        ("num_active_experts", num_active_experts),
        ("index_l1_addr", idx_dev.buffer_address()),
        ("cores_per_dram_bank", cores_per_dram_bank),
        ("k_parallel_per_bank", k_parallel_per_bank),
        ("num_subblocks_k_local", num_subblocks_k_local),
        ("primary_at_last_offset", 1 if primary_at_last_offset else 0),
        ("in0_page_size", in0_page_size),
        # ── Gate ───────────────────────────────────────────────
        ("gate_cb_out", gate_cb_out),
        ("gate_cb_fmt", gate_cb_fmt),
        ("gate_fmt_dram_addr", gate_fmt_info["fmt_dram_addr"]),
        ("gate_fmt_per_expert_bytes", gate_fmt_per_expert),
        ("gate_fmt_per_core_bytes", gate_fmt_info["fmt_per_core_bytes"]),
        ("gate_fmt_cb_l1_addr", gate_fmt_cb_l1_addr),
        ("gate_fmt_cb_page_size", gate_cb_fmt_page_size),
        ("gate_fmt_sem_addr_0", gate_fmt_sem_addr_0),
        ("gate_fmt_sem_addr_1", gate_fmt_sem_addr_1),
        ("gate_partial_sem_addr", gate_partial_sem_addr),
        ("gate_pipeline_sem_addr", gate_pipeline_sem_addr),
        ("gate_gather_sync_sem_addr", gate_gather_sync_sem_addr),
        ("gate_dram_fmt_l1_addr", 0),  # legacy, replaced by per-expert fmt CB
        ("gate_dram_meta_words_per_block", gate_meta_words_per_block),
        ("gate_dram_fuse_silu", 1 if dram_fuse_silu else 0),
        ("gate_cb_out_silu", gate_cb_out_silu),
        ("gate_silu_tile_h", silu_tile_h),
        # ── Up ─────────────────────────────────────────────────
        ("up_cb_out", up_cb_out),
        ("up_cb_fmt", up_cb_fmt),
        ("up_fmt_dram_addr", up_fmt_info["fmt_dram_addr"]),
        ("up_fmt_per_expert_bytes", up_fmt_per_expert),
        ("up_fmt_per_core_bytes", up_fmt_info["fmt_per_core_bytes"]),
        ("up_fmt_cb_l1_addr", up_fmt_cb_l1_addr),
        ("up_fmt_cb_page_size", up_cb_fmt_page_size),
        ("up_fmt_sem_addr_0", up_fmt_sem_addr_0),
        ("up_fmt_sem_addr_1", up_fmt_sem_addr_1),
        ("up_partial_sem_addr", up_partial_sem_addr),
        ("up_pipeline_sem_addr", up_pipeline_sem_addr),
        ("up_gather_sync_sem_addr", up_gather_sync_sem_addr),
        ("up_dram_fmt_l1_addr", 0),
        ("up_dram_meta_words_per_block", up_meta_words_per_block),
    ]

    # Per-core descriptors. gate_X / up_X expert_offsets and block_sizes are
    # different per-op; bank_id/vc/core_in_bank_idx/k_slice_idx/next_core_X are
    # shared (gate and up use the same physical core layout).
    per_core_descriptors = [
        PerCoreCompileTimeDescriptor(
            named_compile_time_arg="gate_expert_offsets_l1_addr",
            core_values=gate_offsets_l1_cv,
            other_value=0,
        ),
        PerCoreCompileTimeDescriptor(
            named_compile_time_arg="gate_block_sizes_l1_addr",
            core_values=gate_bsize_l1_cv,
            other_value=0,
        ),
        PerCoreCompileTimeDescriptor(
            named_compile_time_arg="up_expert_offsets_l1_addr",
            core_values=up_offsets_l1_cv,
            other_value=0,
        ),
        PerCoreCompileTimeDescriptor(
            named_compile_time_arg="up_block_sizes_l1_addr",
            core_values=up_bsize_l1_cv,
            other_value=0,
        ),
    ]
    # Shared per-core values from gate's setup (gate and up have identical core layout).
    for name in ("bank_id", "vc", "core_in_bank_idx", "next_core_noc_x", "next_core_noc_y", "k_slice_idx"):
        per_core_descriptors.append(
            PerCoreCompileTimeDescriptor(
                named_compile_time_arg=name,
                core_values=gate_pcv.get(name, []),
                other_value=0,
            )
        )

    unified_kernel = UnifiedKernelDescriptor(
        kernel_source=_KERNEL_SOURCE,
        core_ranges=dram_core_grid,
        ncrisc_named_compile_time_args=named_ct_args,
        brisc_named_compile_time_args=named_ct_args,
        trisc_named_compile_time_args=named_ct_args,
        trisc_compile_time_args=[],
        per_core_compile_time_descriptors=per_core_descriptors,
        trisc_compute_config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            dst_full_sync_en=False,
        ),
    )

    return ttnn.ProgramDescriptor(
        kernels=unified_kernel.get_kernel_descriptors().kernels,
        cbs=cbs,
        semaphores=[],
    )
