# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Local-only token-dispatch combine as one generic_op on every core (Laguna; see kernels/lc_*.cpp)."""

from pathlib import Path

import ttnn

_KDIR = Path(__file__).resolve().parent / "kernels"


def combine_local(expert_outputs, metadata, counts, region_offsets, global_expert_idx, seq_len, top_k, out=None):
    """expert_outputs: TILE bf8 [R, H] routed-expert output; metadata: dispatch metadata [.., R, 5] int32 row-major;
    counts / region_offsets: per-global-expert uint32 (one page); global_expert_idx: local -> global ids (one page).
    Returns [1, 1, T, K, H] bf16 row-major with every LOCAL (token, slot) row written (others left unwritten)."""
    device = expert_outputs.device()
    H = expert_outputs.shape[-1]
    Ht = H // 32
    BLK = 8
    while Ht % BLK:
        BLK -= 1
    E = global_expert_idx.shape[-1]
    if out is None:
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, seq_len, top_k, H]), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
        )
    gs = device.compute_with_storage_grid_size()
    cores = gs.x * gs.y
    grid = ttnn.num_cores_to_corerangeset(cores, gs, True)
    tile_bytes = 1088
    meta_page = 64  # L1 slot per metadata row (>= its aligned page)
    row_bytes = H * 2
    vec_page = 4096  # L1 slot per vector; each read is that tensor's own (one) page
    page_bytes = [t.shape[-1] * 4 for t in (counts, global_expert_idx, region_offsets)]
    assert max(page_bytes) <= vec_page, page_bytes

    def cb(i, fmt, page, pages):
        return ttnn.CBDescriptor(
            total_size=page * pages,
            core_ranges=grid,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=fmt, page_size=page)],
        )

    cbs = [
        cb(0, ttnn.bfloat8_b, tile_bytes, 2 * BLK),
        cb(1, ttnn.uint32, 16, 1),
        cb(2, ttnn.uint32, 128 + 32 * meta_page, 2),
        cb(3, ttnn.uint32, 3 * vec_page, 1),
        cb(16, ttnn.bfloat16, row_bytes, 64),
    ]
    acc = []
    for t in (expert_outputs, metadata, counts, global_expert_idx, region_offsets):
        acc.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    reader = ttnn.KernelDescriptor(
        kernel_source=str(_KDIR / "lc_reader.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=grid,
        compile_time_args=[E, Ht, BLK, tile_bytes, meta_page, gs.x, cores, vec_page] + page_bytes + acc,
        common_runtime_args=[
            expert_outputs.buffer_address(),
            metadata.buffer_address(),
            counts.buffer_address(),
            global_expert_idx.buffer_address(),
            region_offsets.buffer_address(),
        ],
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=str(_KDIR / "lc_writer.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=grid,
        compile_time_args=[top_k, row_bytes, meta_page] + list(ttnn.TensorAccessorArgs(out).get_compile_time_args()),
        common_runtime_args=[out.buffer_address()],
        config=ttnn.WriterConfigDescriptor(),
    )
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.fp32_dest_acc_en = False  # 16-bit DEST: an 8-tile pack_untilize block fits half-sync DEST
    compute = ttnn.KernelDescriptor(
        kernel_source=str(_KDIR / "lc_compute.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=grid,
        compile_time_args=[Ht, BLK],
        config=cfg,
    )
    ttnn.generic_op(
        [expert_outputs, metadata, counts, global_expert_idx, region_offsets, out],
        ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs),
    )
    return out
