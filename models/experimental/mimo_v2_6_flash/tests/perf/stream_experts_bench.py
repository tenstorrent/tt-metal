# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DRAM bandwidth of DeepSeek-B1's DRAMStreamingExpertsMatmul at MiMo's per-chip expert shapes.

Runs on Galaxy row 0 (a plain ttnn.generic_op ProgramDescriptor runs on all 8 chips of the row mesh):
gate|up (K=4096, N=512, BFP4) and down (K=256, N=4096, BFP4), 8 selected experts per launch, the 4
decode users as one [4, 32] tiny-tile in0. One core per DRAM bank streams its own bank. Prints
microseconds and GB/s per launch, and the PCC against an fp32 reference on the BFP4-dequantized weights.

Measured (Blackhole Galaxy): gate|up 23.5 us = 402 GB/s, PCC 0.99999; down 27.8 us = 170 GB/s (one
4.6 KB read per output column with ~2 blocks in flight).

    python models/experimental/mimo_v2_6_flash/tests/perf/stream_experts_bench.py
"""

import time

import torch

import ttnn
from models.experimental.mimo_v2_6_flash.tests.perf import moe_decode_common as common
from models.demos.deepseek_v3_b1.micro_ops.dram_streaming_experts_matmul.op import DRAMStreamingExpertsMatmul
from models.demos.deepseek_v3_b1.tests.unit_tests.test_dram_streaming_matmul import shuffle_tensor_tiles

EXPERTS = 16  # resident experts (the bandwidth does not depend on how many exist)
SELECTED = 8
USERS = 4
LAUNCHES = 40
TILE = 32


def unshuffle(t, banks):
    """Inverse of shuffle_tensor_tiles for one [K, N] matrix (per-bank column-major tiles -> row-major)."""
    K, N = t.shape
    per_n = N // banks
    kt, nt = K // TILE, per_n // TILE
    out = torch.empty_like(t)
    for b in range(banks):
        tiles = (
            t[:, b * per_n : (b + 1) * per_n].reshape(kt, TILE, nt, TILE).permute(0, 2, 1, 3).reshape(-1, TILE, TILE)
        )
        src = torch.empty_like(tiles)
        for i in range(kt * nt):  # stored position i holds source tile (i % kt, i // kt)
            src[(i % kt) * nt + i // kt] = tiles[i]
        out[:, b * per_n : (b + 1) * per_n] = src.reshape(kt, nt, TILE, TILE).permute(0, 2, 1, 3).reshape(K, per_n)
    return out


def sharded(grid, shape, layout, buffer):
    return ttnn.MemoryConfig(layout, buffer, ttnn.ShardSpec(grid, shape, ttnn.ShardOrientation.ROW_MAJOR))


def bench(row, k, n, subblock_k=None):
    cores = row.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
    banks = row.dram_grid_size().x
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
    dram_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))})
    per_n = n // banks
    w = torch.randn(EXPERTS, k, n) * 0.02
    w_mc = sharded(dram_grid, [k, per_n], ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM)
    # One tensor per expert, allocated back to back: the kernel reads expert e at base + e * expert bytes.
    experts = [
        ttnn.from_torch(
            shuffle_tensor_tiles(w[e : e + 1], TILE, banks).reshape(1, 1, k, n),
            dtype=ttnn.bfloat4_b,
            layout=ttnn.TILE_LAYOUT,
            device=row,
            memory_config=w_mc,
        )
        for e in range(EXPERTS)
    ]
    addrs = [t.buffer_address() for t in experts]
    assert all(addrs[i + 1] - addrs[i] == addrs[1] - addrs[0] for i in range(EXPERTS - 1)), "experts not contiguous"
    order = list(range(EXPERTS)) if addrs[1] > addrs[0] else list(reversed(range(EXPERTS)))
    x = torch.randn(1, 1, USERS, k)
    in0 = ttnn.from_torch(
        x.repeat(1, 1, len(cores), 1),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=row,
        memory_config=sharded(grid, [USERS, k], ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1),
        tile=ttnn.Tile([USERS, TILE]),
    )
    picked = torch.randperm(EXPERTS)[:SELECTED]
    ids = torch.zeros(1, 16, dtype=torch.int32)
    ids[0, :SELECTED] = torch.tensor([order.index(int(e)) for e in picked])  # index from the lowest address
    index = ttnn.from_torch(
        ids.to(torch.uint16).repeat(len(cores), 1),
        dtype=ttnn.uint16,
        layout=ttnn.TILE_LAYOUT,
        device=row,
        memory_config=sharded(grid, (1, 16), ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1),
        tile=ttnn.Tile([1, 16]),
    )
    out = ttnn.from_torch(
        torch.zeros(1, 1, USERS, SELECTED * n),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=row,
        memory_config=sharded(
            grid, (USERS, SELECTED * per_n), ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1
        ),
        tile=ttnn.Tile([USERS, TILE]),
    )

    def launch():
        DRAMStreamingExpertsMatmul.op(
            in0,
            experts[order[0]],
            out,
            index_tensor=index,
            fp32_dest_acc_en=True,
            math_fidelity=ttnn.MathFidelity.LoFi,
            subblock_k=subblock_k,
            selected_experts_k=SELECTED,
        )

    launch()
    got = ttnn.to_torch(ttnn.get_device_tensors(out)[0]).float()  # core c: [e0 cols c | e1 cols c | ...]
    ref = torch.zeros_like(got)
    xq = x.bfloat16().float()
    for j, e in enumerate(picked.tolist()):
        wq = unshuffle(ttnn.to_torch(ttnn.get_device_tensors(experts[e])[0]).float().reshape(k, n), banks)
        for c in range(banks):
            ref[..., (c * SELECTED + j) * per_n : (c * SELECTED + j + 1) * per_n] = (
                xq @ wq[:, c * per_n : (c + 1) * per_n]
            )
    pcc = torch.corrcoef(torch.stack([got.flatten(), ref.flatten()]))[0, 1].item()

    ttnn.synchronize_device(row)
    trace = ttnn.begin_trace_capture(row, cq_id=0)
    for _ in range(LAUNCHES):
        launch()
    ttnn.end_trace_capture(row, trace, cq_id=0)
    times = []
    for _ in range(5):
        ttnn.synchronize_device(row)
        t0 = time.perf_counter()
        ttnn.execute_trace(row, trace, cq_id=0, blocking=False)
        ttnn.synchronize_device(row)
        times.append((time.perf_counter() - t0) / LAUNCHES)
    ttnn.release_trace(row, trace)
    t = sorted(times)[len(times) // 2]
    nbytes = SELECTED * k * n * 0.5625
    print(
        f"K={k} N={n}: {t * 1e6:.1f} us per launch of {SELECTED} experts, {nbytes / t / 1e9:.0f} GB/s per chip, "
        f"PCC vs BFP4-dequantized fp32 reference {pcc:.6f}",
        flush=True,
    )
    for tensor in (*experts, in0, index, out):
        tensor.deallocate(True)


def main():
    mesh, row = common.open_row(trace_region_size=32 << 20)
    try:
        bench(row, 4096, 512, subblock_k=32)  # gate|up: 18 KB contiguous blocks
        bench(row, 256, 4096)  # down: one 4.6 KB block per output column
    finally:
        common.close(mesh, row)


if __name__ == "__main__":
    main()
