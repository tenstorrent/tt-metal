# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Greedy token pick over a vocabulary-sharded LM head (Laguna; kernels/argmax_*.cpp).

The batch sampler's greedy path runs a top-k over each chip's [32, 25088] logits shard (one tile row on few cores,
~120 us) plus sampling and seed ops. Here:
  1. every core takes a run of vocabulary tiles and keeps, per row, the best packed key (order-preserving bf16 value
     << 16 | 0xFFFF - column: ties keep the lowest column): the tile engine reduces each tile's rows to their max and
     only the winning tile's row is scanned (TT_LAGUNA_GREEDY_V2=0: both data-movement RISCs scan every value)
  2. one core reduces the runs per row and adds this chip's first vocabulary id
  3. the chips' (key, id) candidates are all-gathered
  4. one core picks per row the largest key (ties: the lowest chip, i.e. the lowest id) into the token buffer
Bit-exact with argmax over the full vocabulary (first index on ties)."""

from pathlib import Path

import torch
import ttnn

_KDIR = Path(__file__).resolve().parent / "kernels"
TILE = 32


def chip_offsets(mesh, v_local):
    """Mesh-sharded uint32 [1, 8] per chip: chip d's first vocabulary id (d * v_local) in word 0."""
    D = mesh.get_num_devices()
    t = torch.zeros((D, 8), dtype=torch.int32)
    t[:, 0] = torch.arange(D, dtype=torch.int32) * int(v_local)
    return ttnn.from_torch(t, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh,
                           memory_config=ttnn.L1_MEMORY_CONFIG, mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0))  # fmt: skip


def _cb(grid, i, page, pages=1, fmt=ttnn.uint32):
    return ttnn.CBDescriptor(total_size=page * pages, core_ranges=grid,
                             format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=fmt, page_size=page)])  # fmt: skip


_V2 = __import__("os").environ.get("TT_LAGUNA_GREEDY_V2", "1") == "1"


def _stage1_v2(shards, parts, gs, Vt, Rt, device):
    """Stage 1 on the tile engine (kernels/argmax2_*.cpp): one unit per core; compute takes each vocabulary tile's
    per-row max (reduce MAX), the writer picks per row the first tile holding the row's largest value and scans only
    that tile's row (32 values) instead of every value of the unit's tiles (the scalar stage 1: ~20 cycles per value).
    Returns W (units)."""
    S = max(1, (gs.x * gs.y) // Rt)
    per = -(-Vt // S)
    S = -(-Vt // per)
    W = Rt * S
    grid = ttnn.num_cores_to_corerangeset(W, gs, True)
    ct = [gs.x, Vt, S, per, W]
    k_in = list(ttnn.TensorAccessorArgs(shards).get_compile_time_args())
    k_out = list(ttnn.TensorAccessorArgs(parts).get_compile_time_args())
    reader = ttnn.KernelDescriptor(
        kernel_source=str(_KDIR / "argmax2_reader.cpp"), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=grid, compile_time_args=ct + k_in, common_runtime_args=[shards.buffer_address()],
        config=ttnn.ReaderConfigDescriptor(),
    )  # fmt: skip
    writer = ttnn.KernelDescriptor(
        kernel_source=str(_KDIR / "argmax2_writer.cpp"), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=grid, compile_time_args=ct + k_in + k_out,
        common_runtime_args=[shards.buffer_address(), parts.buffer_address()], config=ttnn.WriterConfigDescriptor(),
    )  # fmt: skip
    cfg = ttnn.ComputeConfigDescriptor()
    compute = ttnn.KernelDescriptor(
        kernel_source=str(_KDIR / "argmax2_compute.cpp"), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=grid, compile_time_args=ct, config=cfg,
    )  # fmt: skip
    cbs = [_cb(grid, 0, 2048, 2, ttnn.bfloat16), _cb(grid, 1, 2048, 1, ttnn.bfloat16),
           _cb(grid, 2, 32 * 128 + 128, 1, ttnn.bfloat16), _cb(grid, 16, 2048, per, ttnn.bfloat16)]  # fmt: skip
    ttnn.generic_op([shards, parts], ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs))
    return S, W


def greedy_pick(shards, out_tok, offsets, rows_out=None, cluster_axis=1, topology=ttnn.Topology.Ring, num_links=2):
    """shards: [1, 1, R, V] bf16 TILE logits shard per chip (R padded to tiles: rows past rows_out are ignored);
    out_tok: [1, 1, 1, rows_out] uint32 row-major (replicated) receiving the global argmax ids; offsets: chip_offsets."""
    device = shards.device()
    R = -(-int(shards.shape[-2]) // TILE) * TILE
    V = int(shards.shape[-1])
    rows_out = int(out_tok.shape[-1]) if rows_out is None else int(rows_out)
    assert shards.dtype == ttnn.bfloat16 and shards.layout == ttnn.TILE_LAYOUT and V % TILE == 0, (shards.dtype, V)
    Vt, Rt = V // TILE, R // TILE
    gs = device.compute_with_storage_grid_size()
    D = device.get_num_devices()
    mem = ttnn.L1_MEMORY_CONFIG
    if _V2:
        S = max(1, (gs.x * gs.y) // Rt)
        S = -(-Vt // -(-Vt // S))
    else:
        S = max(1, 2 * gs.x * gs.y // Rt)
        per = -(-Vt // S)
        S = -(-Vt // per)
    W = Rt * S
    parts = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, W, 32]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, device, mem)
    cand = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, R, 8]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, device, mem)
    if _V2:
        _stage1_v2(shards, parts, gs, Vt, Rt, device)
    else:
        cores = -(-W // 2)
        grid = ttnn.num_cores_to_corerangeset(cores, gs, True)
        acc = list(ttnn.TensorAccessorArgs(shards).get_compile_time_args()) + list(
            ttnn.TensorAccessorArgs(parts).get_compile_time_args()
        )
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(_KDIR / "argmax_part.cpp"),
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=grid,
                compile_time_args=[gs.x, Vt, S, per, W, risc] + acc,
                common_runtime_args=[shards.buffer_address(), parts.buffer_address()],
                config=ttnn.ReaderConfigDescriptor() if risc == 0 else ttnn.WriterConfigDescriptor(),
            )
            for risc in range(2)
        ]
        ttnn.generic_op([shards, parts], ttnn.ProgramDescriptor(
            kernels=kernels, semaphores=[], cbs=[_cb(grid, 0, 4096 + 128, 1, ttnn.bfloat16), _cb(grid, 1, 4096 + 128, 1, ttnn.bfloat16)]))  # fmt: skip
    one = ttnn.num_cores_to_corerangeset(1, gs, True)
    n_final = min(R, gs.x * gs.y)  # stage-2 cores (each reduces rows c, c + n_final, ...)
    rows_grid = ttnn.num_cores_to_corerangeset(n_final, gs, True)
    k2 = ttnn.KernelDescriptor(
        kernel_source=str(_KDIR / "argmax_final.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=rows_grid,
        compile_time_args=[R, S, W, gs.x, n_final] + list(ttnn.TensorAccessorArgs(parts).get_compile_time_args())
        + list(ttnn.TensorAccessorArgs(cand).get_compile_time_args())
        + list(ttnn.TensorAccessorArgs(offsets).get_compile_time_args()),
        common_runtime_args=[parts.buffer_address(), cand.buffer_address(), offsets.buffer_address()],
        config=ttnn.ReaderConfigDescriptor(),
    )
    ttnn.generic_op([parts, offsets, cand], ttnn.ProgramDescriptor(kernels=[k2], semaphores=[],
                                                                  cbs=[_cb(rows_grid, 0, S * 128 + 128)]))  # fmt: skip
    g = cand if D == 1 else ttnn.all_gather(cand, dim=3, cluster_axis=cluster_axis, topology=topology,
                                            num_links=num_links, memory_config=mem)  # fmt: skip  [R, 8 D]
    k3 = ttnn.KernelDescriptor(
        kernel_source=str(_KDIR / "argmax_pick.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=one,
        compile_time_args=[R, D, rows_out] + list(ttnn.TensorAccessorArgs(g).get_compile_time_args())
        + list(ttnn.TensorAccessorArgs(out_tok).get_compile_time_args()),
        common_runtime_args=[g.buffer_address(), out_tok.buffer_address()],
        config=ttnn.ReaderConfigDescriptor(),
    )
    ttnn.generic_op([g, out_tok], ttnn.ProgramDescriptor(kernels=[k3], semaphores=[],
                                                        cbs=[_cb(one, 0, R * D * 32 + R * 4)]))  # fmt: skip
    return out_tok
