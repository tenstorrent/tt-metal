# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Single-chip trace-replay microbench of the Qwen3.8-27B TP=8 decode projections (M = 32 rows, B=8 padded): today's
1D (mcast_in0) decode matmul on an interleaved weight vs QWEN36_DSHARD_MC's multi-core DRAM-sharded pipeline
(num_workers_per_dram_bank W, in0_block_w ibw), BFP8 and BFP4 weights, LoFi and HiFi2.

Each arm captures `QWEN36_SWEEP_ITERS` (default 200) back-to-back matmuls in one trace, replays it 5 times and
reports the best per-matmul wall time (us; trace replay, so device-bound) and the weight-read bandwidth (GB/s of the
stored weight, DRAM padding included). Every arm's output is checked against a float64 matmul of the operands as
stored (the quantized weight), so the bandwidth numbers are for correct results. The go/no-go per target
(bfp4-matmul design, step 1): the best dsmc BFP4 arm at least 3 us faster than 1D BFP4.

Run (one chip of row 0):
    pytest models/demos/blackhole/qwen36/tests/test_bfp4_decode_matmul_sweep.py -s
    pytest ... -k "gateup and bfp4"          # one target / dtype
Prints one `SWEEP` line per arm; grep them into a table.
"""
import math
import os
import time

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc, is_blackhole
from models.demos.blackhole.qwen36.tt import tp_common as tpc

TILE = 32
M = 32
ITERS = int(os.environ.get("QWEN36_SWEEP_ITERS", "200"))
TILE_BYTES = {ttnn.bfloat8_b: 1088, ttnn.bfloat4_b: 576}

# target -> (K, N, 1D num_cores (model_config, TP=8), 1D max in0_block_w)
TARGETS = {
    "gateup": (5120, 2176, 44, 16),
    "down": (2176, 5120, 84, 8),
    "gdn_in": (5120, 2060, 84, 8),
    "gdn_out": (768, 5120, 84, 8),
}
DTYPES = {"bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}
FIDELITIES = {"lofi": ttnn.MathFidelity.LoFi, "hifi2": ttnn.MathFidelity.HiFi2}
ARMS = [("1d", 0, 0)] + [(f"dsmc_w{w}_ibw{b}", w, b) for w in (2, 3, 4) for b in (2, 4, 8, 16)]


def _compute(device, fidelity):
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=FIDELITIES[fidelity],
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


def _weight(device, k, n, dtype, memcfg, seed=0):
    torch.manual_seed(seed)
    w = torch.randn(1, 1, k, n) / math.sqrt(k)
    return ttnn.from_torch(w, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memcfg)


def _reference(x, w, n):
    xt = ttnn.to_torch(x).double().reshape(M, -1)
    wt = ttnn.to_torch(w).double().reshape(xt.shape[1], -1)[:, :n]
    return xt @ wt


def _time(device, run):
    out = run()  # compile
    ttnn.deallocate(out)
    ttnn.synchronize_device(device)
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    out = None
    for _ in range(ITERS):
        if out is not None:
            ttnn.deallocate(out)
        out = run()
    ttnn.end_trace_capture(device, tid, cq_id=0)
    ttnn.synchronize_device(device)
    ttnn.execute_trace(device, tid, cq_id=0, blocking=False)  # warm replay
    ttnn.synchronize_device(device)
    best = float("inf")
    for _ in range(5):
        t0 = time.perf_counter()
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(device)
        best = min(best, time.perf_counter() - t0)
    ttnn.release_trace(device, tid)
    ttnn.deallocate(out)
    return best / ITERS * 1e6


@pytest.mark.skipif(not is_blackhole(), reason="Blackhole (8 DRAM banks) decode shapes")
@pytest.mark.parametrize("device_params", [{"trace_region_size": 64 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize("arm, workers, ibw", ARMS, ids=[a[0] for a in ARMS])
@pytest.mark.parametrize("fidelity", sorted(FIDELITIES))
@pytest.mark.parametrize("wdt", sorted(DTYPES))
@pytest.mark.parametrize("target", sorted(TARGETS))
def test_decode_matmul_sweep(device, target, wdt, fidelity, arm, workers, ibw):
    k, n, cores_1d, ibw_1d = TARGETS[target]
    dtype = DTYPES[wdt]
    cfg = _compute(device, fidelity)
    n_tiles = math.ceil(n / TILE)
    torch.manual_seed(1)
    x_t = torch.randn(1, 1, M, k)
    if workers == 0:
        x = ttnn.from_torch(
            x_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        w = _weight(device, k, n, dtype, ttnn.DRAM_MEMORY_CONFIG)
        pc = tpc.create_matmul_1d_decode_progcfg(
            M,
            k,
            n,
            num_cores=cores_1d,
            grid_w=device.compute_with_storage_grid_size().x,
            max_in0_block_w=ibw_1d,
        )
        out_mc = ttnn.L1_MEMORY_CONFIG
        weight_bytes = math.ceil(k / TILE) * n_tiles * TILE_BYTES[dtype]
    else:
        bank_tiles = tpc.dshard_mc_bank_tiles(n)
        if not tpc.dshard_mc_workers_ok(bank_tiles, workers):
            pytest.skip(f"W={workers} is not a valid layout for {bank_tiles} tiles/bank")
        x = ttnn.from_torch(
            x_t,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=tpc.create_activation_shard_config(k),
        )
        w = _weight(device, k, n, dtype, tpc.create_dram_sharded_mem_config(k, n))
        pc = tpc.create_dram_sharded_mc_progcfg(k, n, tpc.dshard_mc_out_per_core_n(n_tiles), workers, ibw)
        out_mc = ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG
        weight_bytes = math.ceil(k / TILE) * bank_tiles * tpc.DRAM_CORES * TILE_BYTES[dtype]

    def run():
        return ttnn.linear(x, w, program_config=pc, compute_kernel_config=cfg, memory_config=out_mc)

    out = run()
    got = ttnn.to_torch(out).double().reshape(M, -1)[:, :n]
    ttnn.deallocate(out)
    ref = _reference(x, w, n)
    passed, pcc = comp_pcc(ref, got, 0.9999)
    us = _time(device, run)
    print(
        f"SWEEP target={target} wdt={wdt} fid={fidelity} arm={arm} W={workers} ibw={ibw or ibw_1d} "
        f"us={us:.2f} GBps={weight_bytes / us / 1e3:.1f} pcc={pcc}"
    )
    ttnn.deallocate(x)
    ttnn.deallocate(w)
    assert passed, pcc


@pytest.mark.skipif(not is_blackhole(), reason="Blackhole (8 DRAM banks) decode shapes")
@pytest.mark.parametrize("target", sorted(TARGETS))
def test_bfp4_lofi_matches_hifi2(device, target):
    """QWEN36_MLP_FIDELITY=lofi gate: on bfloat4_b weights (in1 -> srcA, 3-bit mantissa) the HiFi2 second phase
    should add nothing, so LoFi and HiFi2 outputs are expected bit-identical on the dsmc path (default W / ibw).
    Reports the max abs difference when they are not; the MLP LoFi arm is used only if this passes."""
    k, n, *_ = TARGETS[target]
    n_tiles = math.ceil(n / TILE)
    torch.manual_seed(1)
    x = ttnn.from_torch(
        torch.randn(1, 1, M, k),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=tpc.create_activation_shard_config(k),
    )
    w = _weight(device, k, n, ttnn.bfloat4_b, tpc.create_dram_sharded_mem_config(k, n))
    bank_tiles = tpc.dshard_mc_bank_tiles(n)
    pc = tpc.create_dram_sharded_mc_progcfg(
        k, n, tpc.dshard_mc_out_per_core_n(n_tiles), tpc.dram_multicore_workers(bank_tiles), tpc.DSHARD_MC_DEFAULT_IBW
    )
    outs = {}
    for fid in ("lofi", "hifi2"):
        o = ttnn.linear(
            x,
            w,
            program_config=pc,
            compute_kernel_config=_compute(device, fid),
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        )
        outs[fid] = ttnn.to_torch(o).reshape(M, -1)[:, :n]
        ttnn.deallocate(o)
    diff = (outs["lofi"].float() - outs["hifi2"].float()).abs().max().item()
    print(f"LOFI_VS_HIFI2 target={target} max_abs_diff={diff}")
    assert torch.equal(outs["lofi"], outs["hifi2"]), diff
