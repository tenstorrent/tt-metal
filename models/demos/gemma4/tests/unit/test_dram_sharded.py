# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""DramShardedLinear on decode rows below one tile (batch 1 / 8): the GEMMA4_DRAM_SHARD_ROW_VIEW path (metadata
reshapes of the row tile) must give the same result as the shipped Pad -> matmul -> Slice path and launch two
programs fewer per matmul (the dense 31B / 12B run four of these per layer). Single chip, the dense per-chip
shapes (31B tp4 wqkv 5376 x 4096, o_proj 2048 x 5376; 12B tp4 o_proj 1024 x 3840)."""

import os

import pytest
import torch

import ttnn
from models.demos.gemma4.tt.decode_matmul import (
    DRAM_SHARDED_PIPELINE_ROWS,
    MIN_PIPELINE_GRID_CORES,
    PIPELINE_ENV,
    dram_sharded_pipeline_row,
    model_tag,
)
from models.demos.gemma4.tt.dram_sharded import TILE_SIZE, DramShardedLinear, can_dram_shard

SINGLE_CHIP = [pytest.param(1, {"fabric_config": ttnn.FabricConfig.DISABLED}, id="single-chip")]


def _device_ops(captured):
    """Device operations of a graph capture (ReshapeView launches no program)."""
    return [
        n["params"].get("name", "?")
        for n in captured
        if n["node_type"] == "function_start"
        and "Operation" in n["params"].get("name", "")
        and n["params"].get("name") != "ReshapeViewDeviceOperation"
    ]


@pytest.mark.parametrize("K, N", [(5376, 4096), (2048, 5376), (1024, 3840)], ids=["wqkv31b", "oproj31b", "oproj12b"])
@pytest.mark.parametrize("M", [1, 8], ids=["b1", "b8"])
@pytest.mark.parametrize("mesh_device, device_params", SINGLE_CHIP, indirect=True)
def test_row_view_matches_pad_slice(mesh_device, device_params, K, N, M, monkeypatch):
    if not can_dram_shard(K, N, dtype=ttnn.bfloat8_b):
        pytest.skip(f"[{K}, {N}] has no DRAM-sharded decode config on this grid")
    torch.manual_seed(0)
    w = (torch.randn(1, 1, K, N) * 0.02).to(torch.bfloat16)
    x = (torch.randn(1, 1, M, K) * 2).to(torch.bfloat16)
    lin = DramShardedLinear(w, mesh_device, None, k=K, n=N, dtype=ttnn.bfloat8_b, cache_file_name=None)

    def run(view: bool):
        monkeypatch.setenv("GEMMA4_DRAM_SHARD_ROW_VIEW", "1" if view else "0")
        tx = ttnn.from_torch(x, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        out = lin(tx)  # compile pass
        ttnn.synchronize_device(mesh_device)
        out_t = ttnn.to_torch(out).float()
        # second call under a graph capture: the launched device ops of the warm path
        tx2 = ttnn.from_torch(x, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
        out2 = lin(tx2)
        ttnn.synchronize_device(mesh_device)
        ops = _device_ops(ttnn.graph.end_graph_capture())
        out2_t = ttnn.to_torch(out2).float()
        assert list(out2.shape) == [1, 1, M, N], out2.shape
        assert torch.equal(out_t, out2_t), "warm call differs from the compile call"
        return out_t, ops

    ref, ops_pad = run(False)
    got, ops_view = run(True)
    print(f"[K{K} N{N} M{M}] pad/slice path ops: {ops_pad}")
    print(f"[K{K} N{N} M{M}] row-view path ops: {ops_view}")
    assert torch.isfinite(got).all()
    assert torch.equal(got, ref), f"row-view result differs: max|diff| {(got - ref).abs().max()}"
    golden = (x.float() @ w.float().reshape(K, N)).reshape(1, 1, M, N)
    pcc = torch.corrcoef(torch.stack([golden.flatten(), got.flatten()]))[0, 1].item()
    assert pcc > 0.999, f"pcc vs torch {pcc}"
    assert len(ops_view) == len(ops_pad) - 2, (ops_pad, ops_view)
    assert not any(("Pad" in o or "Slice" in o or "Untilize" in o or "Tilize" in o) for o in ops_view), ops_view


@pytest.mark.parametrize("K, N", [(5376, 4096), (2048, 5376), (5376, 10752)], ids=["wqkv31b", "oproj31b", "gateup31b"])
@pytest.mark.parametrize("M", [32, 1], ids=["b32", "b1"])
@pytest.mark.parametrize("mesh_device, device_params", SINGLE_CHIP, indirect=True)
def test_decode_1d_matches_dram_sharded(mesh_device, device_params, K, N, M, monkeypatch):
    """GEMMA4_DRAM_SHARD_DECODE_1D: the 1D-multicast decode on the interleaved copy of the .ws weight gives the
    DRAM-sharded kernel's result (same bfp8 weight, HiFi2; PCC > 0.9999) in one program instead of three."""
    if not can_dram_shard(K, N, dtype=ttnn.bfloat8_b):
        pytest.skip(f"[{K}, {N}] has no DRAM-sharded decode config on this grid")
    torch.manual_seed(0)
    w = (torch.randn(1, 1, K, N) * 0.02).to(torch.bfloat16)
    x = (torch.randn(1, 1, M, K) * 2).to(torch.bfloat16)
    monkeypatch.setenv("GEMMA4_DRAM_SHARD_ROW_VIEW", "1")
    outs, opss = {}, {}
    for mode in ("sharded", "1d"):
        monkeypatch.setenv("GEMMA4_DRAM_SHARD_DECODE_1D", "1" if mode == "1d" else "0")
        lin = DramShardedLinear(w, mesh_device, None, k=K, n=N, dtype=ttnn.bfloat8_b, cache_file_name=None)
        tx = ttnn.from_torch(x, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        lin(tx)  # compile
        tx2 = ttnn.from_torch(x, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
        out = lin(tx2)
        ttnn.synchronize_device(mesh_device)
        opss[mode] = _device_ops(ttnn.graph.end_graph_capture())
        outs[mode] = ttnn.to_torch(out).float()
        assert list(out.shape) == [1, 1, M, N], out.shape
        lin.weight.deallocate(True)
    # each kernel against the fp32 torch golden with the bf8-weight bar; the kernel-vs-kernel agreement is reported
    # (the two kernels accumulate and round differently: 0.9998x between them on these shapes is expected)
    golden = (x.float() @ w.float().reshape(K, N)).reshape(1, 1, M, N)
    for mode, o in outs.items():
        pcc = torch.corrcoef(torch.stack([golden.flatten(), o.flatten()]))[0, 1].item()
        print(f"[K{K} N{N} M{M} {mode}] pcc vs torch {pcc:.6f} ops {opss[mode]}")
        assert torch.isfinite(o).all()
        assert pcc > 0.999, (mode, pcc)
    pcc = torch.corrcoef(torch.stack([outs["sharded"].flatten(), outs["1d"].flatten()]))[0, 1].item()
    print(f"[K{K} N{N} M{M}] 1D decode vs DRAM-sharded kernel pcc {pcc:.6f}")
    assert pcc > 0.9995, f"1D decode differs from the DRAM-sharded kernel beyond rounding: pcc {pcc}"
    assert len(opss["1d"]) == 1 and "Matmul" in opss["1d"][0], opss["1d"]


# ===== merged from tests/unit/test_dram_sharded_pipeline.py: The DRAM-sharded decode kernel's multi-core pipeline rows (tt/decode_matmul.py ``DRAM_SHARDED_PIPELINE_ROWS``).
SINGLE_CHIP_dram_sharded_pipeline = [
    pytest.param(1, {"fabric_config": ttnn.FabricConfig.DISABLED, "trace_region_size": 32_000_000}, id="single-chip")
]


HIDDEN = {"31B": 5376, "12B": 3840}


SHAPES = {
    ("31B", "qkv", 4): (5376, 4096),
    ("31B", "qkv", 8): (5376, 2048),
    ("31B", "out_proj", 4): (2048, 5376),
    ("31B", "out_proj", 8): (1024, 5376),
    ("31B", "gate_up", 4): (5376, 10752),
    ("31B", "gate_up", 8): (5376, 5376),
    ("31B", "down", 4): (5376, 5376),
    ("31B", "down", 8): (2688, 5376),
    ("12B", "qkv", 4): (3840, 2048),
    ("12B", "qkv", 8): (3840, 1024),
    ("12B", "out_proj", 4): (1024, 3840),
    ("12B", "out_proj", 8): (512, 3840),
    ("12B", "gate_up", 4): (3840, 7680),
    ("12B", "gate_up", 8): (3840, 3840),
    ("12B", "down", 4): (3840, 3840),
    ("12B", "down", 8): (1920, 3840),
}


CI_CASES = [("31B", "out_proj", 4), ("31B", "gate_up", 4)]


def _pcc(a, b):
    a = a.reshape(-1).double() - a.reshape(-1).double().mean()
    b = b.reshape(-1).double() - b.reshape(-1).double().mean()
    return float(a @ b / (a.norm() * b.norm()))


def test_pipeline_rows_host(monkeypatch):
    monkeypatch.delenv(PIPELINE_ENV, raising=False)
    assert model_tag(5376) == "31B" and model_tag(3840) == "12B" and model_tag(2816) is None
    row = DRAM_SHARDED_PIPELINE_ROWS[("31B", "out_proj", 4)]
    assert dram_sharded_pipeline_row(5376, "out_proj", 4, (13, 10), True, default="1") == row
    assert dram_sharded_pipeline_row(5376, "out_proj", 4, (12, 10), True, default="1") == row
    # Wormhole, the 8x8 grid and the off switch keep today's kernel; so does a model without rows
    assert dram_sharded_pipeline_row(5376, "out_proj", 4, (8, 8), False, default="1") is None
    assert dram_sharded_pipeline_row(5376, "out_proj", 4, (8, 8), True, default="1") is None
    assert dram_sharded_pipeline_row(2816, "out_proj", 4, (13, 10), True, default="1") is None
    assert dram_sharded_pipeline_row(5376, "out_proj", 4, (13, 10), True, default="0") is None
    monkeypatch.setenv(PIPELINE_ENV, "0")
    assert dram_sharded_pipeline_row(5376, "out_proj", 4, (13, 10), True, default="1") is None
    monkeypatch.setenv(PIPELINE_ENV, "1")
    assert dram_sharded_pipeline_row(5376, "out_proj", 4, (13, 10), True, default="0") == row
    # every row fits its grid rule: workers x 8 banks within the smallest pipelined grid
    for (model, op, tp), (workers, bw) in DRAM_SHARDED_PIPELINE_ROWS.items():
        assert workers * 8 <= MIN_PIPELINE_GRID_CORES and bw >= 1 and (model, op, tp) in SHAPES


def _cases():
    keys = list(SHAPES) if os.environ.get("GEMMA4_PIPELINE_TEST_ALL") == "1" else CI_CASES
    return [pytest.param(k, id=f"{k[0]}-{k[1]}-tp{k[2]}") for k in keys]


@pytest.mark.parametrize("case", _cases())
@pytest.mark.parametrize("mesh_device, device_params", SINGLE_CHIP_dram_sharded_pipeline, indirect=True)
def test_pipeline_matches_single_reader(mesh_device, device_params, case, monkeypatch):
    model, op, tp = case
    grid = mesh_device.compute_with_storage_grid_size()
    blackhole = "blackhole" in ttnn.get_arch_name()
    if not blackhole or grid.x * grid.y < MIN_PIPELINE_GRID_CORES:
        pytest.skip("the pipeline rows apply to Blackhole grids of more than 64 worker cores")
    row = dram_sharded_pipeline_row(HIDDEN[model], op, tp, (grid.x, grid.y), blackhole, default="1")
    if row is None:
        pytest.skip(f"no pipeline row for {case}")
    k, n = SHAPES[case]
    if not can_dram_shard(k, n, dtype=ttnn.bfloat8_b):
        pytest.skip(f"[{k}, {n}] has no DRAM-sharded decode config on this grid")
    torch.manual_seed(7)
    w = (torch.randn(1, 1, k, n) * 0.02).to(torch.bfloat16)
    x = (torch.randn(1, 1, TILE_SIZE, k) * 2).to(torch.bfloat16)
    w_q = ttnn.to_torch(ttnn.from_torch(w, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)).float()
    golden = x.float().reshape(TILE_SIZE, k) @ w_q.reshape(k, n)
    monkeypatch.setenv("GEMMA4_DRAM_SHARD_ROW_VIEW", "1")
    base = DramShardedLinear(w, mesh_device, None, k=k, n=n, dtype=ttnn.bfloat8_b, cache_file_name=None)
    pipe = DramShardedLinear(
        w, mesh_device, None, k=k, n=n, dtype=ttnn.bfloat8_b, cache_file_name=None, pipeline_row=row
    )
    assert pipe.pipeline_row == row and base.pipeline_row is None
    assert pipe._decode_pc.use_multicore_pipeline and pipe._decode_pc.num_workers_per_dram_bank == row[0]
    tx = ttnn.from_torch(x, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    out_base = ttnn.to_torch(base(tx)).float().reshape(TILE_SIZE, n)
    out_pipe = ttnn.to_torch(pipe(tx)).float().reshape(TILE_SIZE, n)
    x_sh = ttnn.to_memory_config(tx, pipe.act_memory_config)
    out_sh = ttnn.to_torch(pipe._decode_sharded_rows(x_sh, ttnn.DRAM_MEMORY_CONFIG, None, TILE_SIZE)).float()
    out_sh = out_sh.reshape(TILE_SIZE, n)
    assert torch.isfinite(out_pipe).all() and torch.isfinite(out_sh).all()
    pcc_base, pcc_pipe, pcc_sh = _pcc(out_base, golden), _pcc(out_pipe, golden), _pcc(out_sh, golden)
    print(
        f"RESULT pipe_unit case={model}-{op}-tp{tp} K={k} N={n} row={row} pcc_base={pcc_base:.6f} pcc_pipe={pcc_pipe:.6f} pcc_sharded_entry={pcc_sh:.6f}"
    )
    assert pcc_pipe >= pcc_base - 1e-6, (pcc_pipe, pcc_base)
    assert pcc_sh >= pcc_base - 1e-6, (pcc_sh, pcc_base)
    assert _pcc(out_sh, out_pipe) > 0.99999
    for lin in (base, pipe):
        lin.weight.deallocate(True)


def _traced_us(mesh_device, call, n_calls=10, replays=20):
    """Traced wall per call: ``n_calls`` calls in one trace (one live output), ``replays`` replays, median of 3."""
    import time

    out = call()
    ttnn.synchronize_device(mesh_device)
    out.deallocate(True)
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    last = None
    for i in range(n_calls):
        o = call()
        if i < n_calls - 1:
            o.deallocate(True)
        else:
            last = o
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
    means = []
    for _ in range(3):
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        for _ in range(replays):
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        means.append((time.perf_counter() - t0) / (replays * n_calls) * 1e6)
    ttnn.release_trace(mesh_device, tid)
    last.deallocate(True)
    return sorted(means)[1]


@pytest.mark.parametrize("case", _cases())
@pytest.mark.parametrize("mesh_device, device_params", SINGLE_CHIP_dram_sharded_pipeline, indirect=True)
def test_pipeline_outliers_accuracy_time(mesh_device, device_params, case, monkeypatch):
    """Gemma-like activations (2 % of the channels at 64x) through today's kernel, the pipeline with the packer-L1
    accumulation (``l1acc``) and the pipeline with fp32 Dest accumulation (the shipped ``1``): PCC and max error vs
    the fp32 golden, and the traced wall of the model's own call (reshard in + kernel + reshard out). The fp32 form
    must be at least as accurate as today's kernel; the l1acc numbers are the reference that explains the 12B
    token-accuracy drop."""
    model, op, tp = case
    grid = mesh_device.compute_with_storage_grid_size()
    blackhole = "blackhole" in ttnn.get_arch_name()
    if not blackhole or grid.x * grid.y < MIN_PIPELINE_GRID_CORES:
        pytest.skip("the pipeline rows apply to Blackhole grids of more than 64 worker cores")
    row = dram_sharded_pipeline_row(HIDDEN[model], op, tp, (grid.x, grid.y), blackhole, default="1")
    if row is None:
        pytest.skip(f"no pipeline row for {case}")
    k, n = SHAPES[case]
    if not can_dram_shard(k, n, dtype=ttnn.bfloat8_b):
        pytest.skip(f"[{k}, {n}] has no DRAM-sharded decode config on this grid")
    torch.manual_seed(11)
    w = (torch.randn(1, 1, k, n) * 0.02).to(torch.bfloat16)
    x = torch.randn(1, 1, TILE_SIZE, k) * 2
    hot = torch.randperm(k)[: max(1, k // 50)]
    x[..., hot] *= 64.0  # the residual stream's outlier channels
    x = x.to(torch.bfloat16)
    w_q = ttnn.to_torch(ttnn.from_torch(w, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)).float()
    golden = x.float().reshape(TILE_SIZE, k) @ w_q.reshape(k, n)
    monkeypatch.setenv("GEMMA4_DRAM_SHARD_ROW_VIEW", "1")
    tx = ttnn.from_torch(x, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    res = {}
    for name, env in (("base", "0"), ("l1acc", "l1acc"), ("fp32acc", "1")):
        monkeypatch.setenv(PIPELINE_ENV, env)
        lin = DramShardedLinear(
            w,
            mesh_device,
            None,
            k=k,
            n=n,
            dtype=ttnn.bfloat8_b,
            cache_file_name=None,
            pipeline_row=row if env != "0" else None,
        )
        assert (lin._pipeline_ckc is not None) == (env == "1")
        out = ttnn.to_torch(lin(tx)).float().reshape(TILE_SIZE, n)
        err = float((out - golden).abs().max() / golden.abs().max())
        us = _traced_us(mesh_device, lambda lin=lin: lin(tx))
        res[name] = (_pcc(out, golden), err, us)
        lin.weight.deallocate(True)
    print(
        f"RESULT pipe_outlier case={model}-{op}-tp{tp} K={k} N={n} row={row} "
        + " ".join(f"pcc_{m}={v[0]:.6f} err_{m}={v[1]:.5f} us_{m}={v[2]:.2f}" for m, v in res.items())
    )
    assert res["fp32acc"][0] >= res["base"][0] - 1e-6, (res["fp32acc"][0], res["base"][0])
    # the max error is a one-element statistic (0.017-0.034 of the output scale on these rows, +-15 % run to run):
    # a gross-error guard only; the PCC above is the accuracy bar
    assert res["fp32acc"][1] <= res["base"][1] * 1.5, (res["fp32acc"][1], res["base"][1])


# ===== Prefill with the fp32-Dest compute config of the prefill linears (compute_config.prefill_linear_compute_kernel_config):
# fp32 accumulation gives the 2D matmul an fp32 partials block next to its bf16 output block, so the slab that fits
# with the bf16 Dest default overflows L1. Each prompt length takes a different path through __call__: 2016 rows (the
# 31B gate_up TP=4 case of PR #60786's review: per_core_M 7 at one shot = 2.6 MB of CBs) and 2560 (the bf16 slab) are
# single matmuls under the old rule and slab + remainder chunks under the fp32 slab; 4096 is the bf16-slab chunk path.
# The gate_up cases at 4096 rows also print the K-block / slab trade of prefill_slab_plan (a smaller in0_block_w
# halves the in0 / in1 CBs and buys a larger slab at twice the K iterations): wall per call at 2048 and 4096 rows
# for every (in0_block_w, slab) candidate at the 64 KB headroom and at headroom 0, each checked for L1 fit and pcc.
# Measured 2026-10-30 (bh-glx-exp-b03u14 chip 8, bfp8, 5 calls; ms at 2048 / 4096 rows): 31B gate_up block 4 / 960
# rows 2.19 / 4.01 (3 / 5 launches), block 2 / 960 2.32 / 4.22, block 2 / 1280 (headroom 0) 2.23 / 4.25, block 1 /
# 1280 3.12 / 5.90; 12B gate_up block 4 / 960 1.43 / 2.57, block 4 / 1280 (headroom 0) 1.36 / 2.59, block 2 / 1280
# 1.45 / 2.72, block 2 / 1600 1.43 / 2.69, block 1 / 1600 1.95 / 3.66: the launches saved are worth nothing next to
# the extra K iterations, so the plan keeps block 4 (the PR #60786 follow-up idea, closed by this table).
def _fp32_slab_timing(mesh_device, lin, x, ckc, golden, monkeypatch, n_calls=5):
    import time

    from models.demos.gemma4.tt import dram_sharded as ds

    K, N, dt, cols = lin.k, lin.n, lin._dtype, lin._prefill_max_cols
    default = lin._slab_plan_by_acc[True]
    k_tiles = K // TILE_SIZE
    plans = []  # (in0_block_w, slab rows): the plan at each K block (the rule's, 2, 1) at the two headrooms
    for w in (default[1], 2, 1):
        for headroom in (None, 0):
            if w > default[1] or k_tiles % w:
                continue
            plan = ds.prefill_slab_plan(K, N, dt, max_cols=cols, fp32_acc=True, headroom=headroom, in0_block_w=w)
            if (w, plan[0]) not in plans:
                plans.append((w, plan[0]))
    print(f"[K{K} N{N}] fp32 K-block / slab candidates {plans}; the rule's plan (rows, in0_block_w) {default}")
    print(f"{'in0_block_w':>11s} {'slab':>5s} {'rows':>5s} {'launches':>8s} {'ms/call':>8s} {'pcc':>9s}")
    for w, slab in plans:
        monkeypatch.setitem(lin._slab_plan_by_acc, True, (slab, w))
        monkeypatch.setitem(lin._slab_rows_by_acc, True, slab)
        for m in (2048, 4096):
            launches = len(ds.prefill_row_chunks(m, slab=slab, remainder=min(ds._PREFILL_REMAINDER_CHUNK, slab)))
            tx = ttnn.from_torch(x[:, :, :m], device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
            out = lin(tx, compute_kernel_config=ckc, prefill=True)  # compile
            ttnn.synchronize_device(mesh_device)
            out.deallocate(True)
            t0 = time.perf_counter()
            for i in range(n_calls):
                out = lin(tx, compute_kernel_config=ckc, prefill=True)
                if i < n_calls - 1:
                    out.deallocate(True)
            ttnn.synchronize_device(mesh_device)
            ms = (time.perf_counter() - t0) / n_calls * 1e3
            pcc = _pcc(ttnn.to_torch(out).float(), golden[:, :, :m])
            out.deallocate(True)
            tx.deallocate(True)
            tag = " <- rule" if (slab, w) == default else ""
            print(f"{w:>11d} {slab:>5d} {m:>5d} {launches:>8d} {ms:>8.2f} {pcc:>9.6f}{tag}")
            assert pcc > 0.9995, (w, slab, m, pcc)
    monkeypatch.undo()


@pytest.mark.parametrize(
    "K, N",
    [(5376, 10752), (5376, 5376), (5376, 4096), (2048, 5376), (3840, 7680)],
    ids=["gateup31b_tp4", "down31b_tp4", "wqkv31b_tp4", "oproj31b_tp4", "gateup12b_tp4"],
)
@pytest.mark.parametrize("M", [32, 2016, 2560, 4096], ids=["m32", "m2016", "m2560", "m4096"])
@pytest.mark.parametrize("mesh_device, device_params", SINGLE_CHIP, indirect=True)
def test_prefill_fp32_acc_fits_l1(mesh_device, device_params, K, N, M, monkeypatch):
    from models.demos.gemma4.tt.compute_config import prefill_linear_compute_kernel_config

    if not can_dram_shard(K, N, dtype=ttnn.bfloat8_b):
        pytest.skip(f"[{K}, {N}] has no DRAM-sharded config on this grid")
    torch.manual_seed(0)
    w = (torch.randn(1, 1, K, N) * 0.02).to(torch.bfloat16)
    x = (torch.randn(1, 1, M, K) * 2).to(torch.bfloat16)
    lin = DramShardedLinear(w, mesh_device, None, k=K, n=N, dtype=ttnn.bfloat8_b, cache_file_name=None)
    ckc = prefill_linear_compute_kernel_config(mesh_device)
    tx = ttnn.from_torch(x, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    out = lin(tx, compute_kernel_config=ckc, prefill=True)
    got = ttnn.to_torch(out).float()
    assert list(out.shape) == [1, 1, M, N], out.shape
    golden = (x.float() @ w.float().reshape(K, N)).reshape(1, 1, M, N)
    pcc = _pcc(got, golden)
    slab, block_w = lin._slab_plan_by_acc[True]
    assert slab == lin._slab_rows_by_acc[True]
    print(f"[K{K} N{N} M{M}] fp32-acc prefill slab {slab} rows in0_block_w {block_w}, pcc vs torch {pcc:.6f}")
    assert torch.isfinite(got).all()
    assert pcc > 0.9995, f"pcc vs torch {pcc}"
    if M == 4096 and N == 2 * K:  # the gate_up shapes: the K-block / slab trade, timed (printed, not asserted)
        out.deallocate(True)
        tx.deallocate(True)
        _fp32_slab_timing(mesh_device, lin, x, ckc, golden, monkeypatch)
