# SPDX-License-Identifier: Apache-2.0
"""
Benchmark the deepseek_v3_b1 DRAM-streaming matmul micro-op at gemma2-9B decode shapes.

Compares achieved DRAM bandwidth against the stock dram_sharded matmul ceiling
(~308 GB/s for bfp4, ~490 GB/s for bfp8 measured previously on this P150).

Weights bfp4_b / bfp8_b, activations bf16, LoFi -- matching the production decode config.
Timing uses the kernel-internal num_loop_iters so dispatch overhead is amortised away.
"""
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_b1.micro_ops.dram_streaming_matmul.op import DRAMStreamingMatmul
from models.demos.deepseek_v3_b1.tests.unit_tests.test_dram_streaming_matmul import (
    pad_to_dram_banks,
    shuffle_tensor_tiles,
)

# gemma2-9B, single device, batch-1 decode
SHAPES = [
    ("QKV", 3584, 8192, None),
    ("WO", 4096, 3584, None),
    ("FF1_FF3", 3584, 14336, None),
    ("FF1_silu", 3584, 14336, "silu"),
    ("FF2", 14336, 3584, None),
]

# bytes per 32x32 tile: data + per-face exponents
TILE_BYTES = {ttnn.bfloat4_b: 512 + 64, ttnn.bfloat8_b: 1024 + 64}
PEAK_BW_GBS = 550.0
# Kernel-internal loop count. Host-side generic_op program construction costs ~8ms per
# call, so wall-clock timing is useless here; the device profiler reports the kernel
# duration for the single op, which covers all ITERS matmuls.
ITERS = 100


@pytest.mark.parametrize("m", [1, 32], ids=["m1", "m32"])
@pytest.mark.parametrize("wdtype", [ttnn.bfloat4_b], ids=["bf4"])
@pytest.mark.parametrize("name, k, n, act", SHAPES, ids=[s[0] for s in SHAPES])
def test_bench(device, name, k, n, act, wdtype, m):
    # m=1 uses tiny tiles (what deepseek_b1 does); m=32 is what tt_transformers decode
    # actually produces, since batch-1 activations are padded up to a full tile.
    tile_w = 32
    in0_tile = ttnn.Tile([m, tile_w])
    out_tile = ttnn.Tile([m, tile_w])

    compute_cores = device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
    num_cores = len(compute_cores)
    num_banks = device.dram_grid_size().x
    assert num_cores == num_banks

    n_padded = pad_to_dram_banks(n, tile_w, tile_w * num_banks)

    in0 = torch.randn([1, 1, m, k]).bfloat16().float()
    in1 = torch.randn([1, 1, k, n_padded]).bfloat16().float()
    in1_shuffled = shuffle_tensor_tiles(in1, tile_w, num_banks)

    compute_core_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(c.x, c.y), ttnn.CoreCoord(c.x, c.y)) for c in compute_cores]
    )
    # in0 is REPLICATED: every compute core needs the full [m, k] activation row, so the
    # torch tensor is repeated num_cores times and HEIGHT_SHARDED one copy per core.
    in0_t = ttnn.from_torch(
        in0.repeat(1, 1, num_cores, 1),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(compute_core_grid, [m, k], ttnn.ShardOrientation.ROW_MAJOR),
        ),
        tile=in0_tile,
    )

    in1_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(device.dram_grid_size().x - 1, device.dram_grid_size().y - 1),
            )
        }
    )
    in1_t = ttnn.from_torch(
        in1_shuffled,
        dtype=wdtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(in1_grid, [k, n_padded // num_banks], ttnn.ShardOrientation.ROW_MAJOR),
        ),
    )

    out_t = ttnn.from_torch(
        torch.zeros([1, 1, m, n_padded]).bfloat16().float(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(compute_core_grid, (m, n_padded // num_banks), ttnn.ShardOrientation.ROW_MAJOR),
        ),
        tile=out_tile,
    )

    subblock_k = k // tile_w // 4 if (k // tile_w) % 4 == 0 else k // tile_w // 2
    num_in1_buffers = 3
    in1_CB_tiles = subblock_k * num_in1_buffers
    working_buf_t = ttnn.from_torch(
        torch.zeros([1, 1, tile_w, in1_CB_tiles * tile_w * num_cores]).bfloat16().float(),
        dtype=wdtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(compute_core_grid, (tile_w, in1_CB_tiles * tile_w), ttnn.ShardOrientation.ROW_MAJOR),
        ),
        tile=ttnn.Tile([tile_w, tile_w]),
    )

    def run(iters):
        return DRAMStreamingMatmul.op(
            in0_t,
            in1_t,
            out_t,
            fp32_dest_acc_en=False,
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            subblock_k=subblock_k,
            fused_activation=act,
            num_loop_iters=iters,
            working_buf_tensor=working_buf_t,
        )

    result = run(1)  # warm up / compile
    ttnn.synchronize_device(device)

    # Correctness gate: the bandwidth number is only interesting if the op is numerically
    # usable at gemma2's shapes, which are larger than the deepseek shapes it ships with.
    pt_out = DRAMStreamingMatmul.golden(in0, in1, act)
    _, pcc_msg = comp_pcc(pt_out, ttnn.to_torch(result), 0.0)

    # Each op() call costs a constant host-side program-construction time C plus N times
    # the on-device matmul time d. Timing at two loop counts and taking the slope
    # cancels C, which otherwise dwarfs the kernel by ~100x.
    def timed(iters):
        best = None
        for _ in range(3):
            t0 = time.perf_counter()
            run(iters)
            ttnn.synchronize_device(device)
            el = time.perf_counter() - t0
            best = el if best is None else min(best, el)
        return best

    n_lo, n_hi = 50, 250
    t_lo, t_hi = timed(n_lo), timed(n_hi)
    d = (t_hi - t_lo) / (n_hi - n_lo)
    host_c = t_lo - n_lo * d

    wbytes = (k // 32) * (n_padded // 32) * TILE_BYTES[wdtype]
    bw = wbytes / d / 1e9
    dt = "bf4" if wdtype == ttnn.bfloat4_b else "bf8"

    logger.info(
        f"RESULT {name:9s} {dt} K={k:5d} N={n_padded:5d} cores={num_cores} "
        f"per_mm={d*1e6:7.2f}us  {wbytes/1e6:7.2f}MB  {bw:6.1f} GB/s  "
        f"{bw/PEAK_BW_GBS*100:5.1f}% MBU  | {pcc_msg}"
    )
