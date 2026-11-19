# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""P.81 probe (not a gate): the attention projections at 160 rows a chip (ATTN_SP32_PROJ) on the mesh, the shipped
config vs DRAM-sharded weights (the stock DRAM-sharded matmul's multi-core pipeline: N cores next to each DRAM bank,
each streaming its own columns from that bank) and wider K blocks. Device time per call from the real-time profiler
(every chip runs it); output vs the shipped config's (bit-identical or the largest difference) and vs float64.

    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all --no-precompile \\
        models/demos/deepseek_v41_flash_d_p/tests/perf/test_attn_proj_probe.py
P81_SITES=wq_b,wo_b,wo_a selects; P81_ITERS (default 20).
"""

from __future__ import annotations

import os

import torch

import ttnn
from models.demos.common.bringup.testing import profiler
from models.demos.common.bringup.testing.harness import mesh_parametrize
from models.demos.deepseek_v41_flash_d_p.tt import mm_configs as mm

ITERS = int(os.environ.get("P81_ITERS", "20"))
DRAM = ttnn.DRAM_MEMORY_CONFIG


def _rep(mesh, t, dtype, mc=DRAM):
    return ttnn.from_torch(
        t.contiguous(),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=mc,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def _chip0(t) -> torch.Tensor:
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).double()


def _rect(x, y):
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(x - 1, y - 1))})


def _ws_l1(grid, rows, width_tiles):
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, (rows, width_tiles * 32), ttnn.ShardOrientation.ROW_MAJOR),
    )


def _ws_dram(mesh, k, n):
    banks = mesh.dram_grid_size().x
    assert n % (32 * banks) == 0, (n, banks)
    g = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))})
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(g, (k, n // banks), ttnn.ShardOrientation.ROW_MAJOR),
    )


def _grid_for(n_shards):
    """A rows x cols rectangle (<= 11 x 10) of exactly n_shards cores."""
    for y in range(1, 11):
        if n_shards % y == 0 and n_shards // y <= 11:
            return _rect(n_shards // y, y)
    raise ValueError(n_shards)


def _ckc(mesh):
    return ttnn.init_device_compute_kernel_config(
        mesh.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )


def _ds_cfg(kb, M, cpb, per_core_n):
    return ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=kb,
        per_core_M=M // 32,
        per_core_N=per_core_n,
        fused_activation=None,
        num_workers_per_dram_bank=cpb,
        use_multicore_pipeline=True,
    )


def _time(mesh, cands):
    """cands: [(tag, [fn, ...])]: each fn one program (or a few); each tag's fns run ITERS times, each fn in its own
    section ``tag/i``. Returns {tag: [ms per call of fn i]}."""
    profiler.enable(mesh, rt=True)
    try:
        for tag, fns in cands:
            for _ in range(ITERS):
                keep = []
                for i, fn in enumerate(fns):
                    profiler.signpost(f"{tag}/{i}")
                    keep.append(fn(keep[-1] if keep else None))
                profiler.signpost("gap")
                for t in keep:
                    if t is not None:
                        ttnn.deallocate(t)
        profiler.signpost("end")
        profiler.finish_rt()
        prof = profiler.result()
    finally:
        profiler.disable()
    ns = prof["kernel_ns"]
    return {tag: [ns.get(f"{tag}/{i}", float("nan")) / ITERS / 1e6 for i in range(len(fns))] for tag, fns in cands}


def _report(name, base_out, truth, results, times):
    print(f"[p81] == {name}")
    for tag, out in results:
        o = _chip0(out)
        d = (o - base_out).abs()
        rel = float((o - truth).norm() / truth.norm())
        t = times[tag]
        print(
            f"[p81]    {sum(t):.4f} ms ({' + '.join(f'{v:.4f}' for v in t)})  {tag:40s} "
            f"max|d| vs shipped {float(d.max()):.3e} ({int((d > 0).sum())} differ)  rel vs fp64 {rel:.3e}"
        )


def _plain(mesh, x, w, odt, ckc, spec):
    pc = mm.build(spec)
    kw = {} if pc is None else {"program_config": pc[1]}
    return lambda _=None: ttnn.linear(x, w, dtype=odt, compute_kernel_config=ckc, memory_config=DRAM, **kw)


def _one_site(mesh, name, M, K, N, odt, shipped, extra_1d, cpbs, x_shards, out_shards):
    torch.manual_seed(0)
    ckc = _ckc(mesh)
    xt = torch.randn(1, 1, M, K)
    wt = torch.randn(1, 1, K, N) / K**0.5
    x = _rep(mesh, xt, ttnn.bfloat8_b)
    w = _rep(mesh, wt, ttnn.bfloat8_b)
    w_ds = _rep(mesh, wt, ttnn.bfloat8_b, _ws_dram(mesh, K, N))
    truth = _chip0(x) @ _chip0(w)
    cands, outs = [], []
    f = _plain(mesh, x, w, odt, ckc, shipped)
    cands.append(("shipped", [f]))
    outs.append(("shipped", f()))
    for tag, spec in extra_1d:
        f = _plain(mesh, x, w, odt, ckc, spec)
        try:
            outs.append((tag, f()))
            cands.append((tag, [f]))
        except Exception as ex:
            print(f"[p81]    {tag}: FAILED {str(ex).splitlines()[0][:160]}")
    Kt, Nt = K // 32, N // 32
    x_mc = _ws_l1(_grid_for(x_shards), M, Kt // x_shards)
    o_mc = _ws_l1(_grid_for(out_shards), M, Nt // out_shards)
    for cpb in cpbs:
        for kb in (4, 8):
            tag = f"dram_sharded cpb{cpb} kb{kb}"
            cfg = _ds_cfg(kb, M, cpb, Nt // out_shards)

            def f_in(_=None):
                return ttnn.to_memory_config(x, x_mc)

            def f_mm(xs, cfg=cfg):
                return ttnn.linear(
                    xs, w_ds, program_config=cfg, memory_config=o_mc, dtype=odt, compute_kernel_config=ckc
                )

            def f_out(y):
                return ttnn.to_memory_config(y, DRAM)

            try:
                xs = f_in()
                y = f_mm(xs)
                outs.append((tag, f_out(y)))
                ttnn.deallocate(xs)
                ttnn.deallocate(y)
                cands.append((tag, [f_in, f_mm, f_out]))
            except Exception as ex:
                print(f"[p81]    {tag}: FAILED {str(ex).splitlines()[0][:200]}")
    # determinism: the second run of each candidate equals its first
    times = _time(mesh, cands)
    _report(f"{name} {M}x{K}x{N}", _chip0(outs[0][1]), truth, outs, times)
    for _, o in outs:
        ttnn.deallocate(o)


def _1d(pn, ob_w, bw, sw, grid=(11, 10)):
    return (
        "1d",
        dict(
            grid=grid,
            mcast_in0=True,
            in0_block_w=bw,
            out_subblock_h=1,
            out_subblock_w=sw,
            out_block_h=5,
            out_block_w=ob_w,
            per_core_M=5,
            per_core_N=pn,
        ),
    )


def _wo_a(mesh):
    """wo_a [1, 8, 160, 4096] x [1, 8, 4096, 1024] -> bfp8: the shipped 2D (fuse_batch False) vs wider in0 blocks."""
    torch.manual_seed(0)
    ckc = _ckc(mesh)
    x = _rep(mesh, torch.randn(1, 8, 160, 4096), ttnn.bfloat8_b)
    w = _rep(mesh, torch.randn(1, 8, 4096, 1024) / 64, ttnn.bfloat8_b)
    truth = _chip0(x) @ _chip0(w)
    base = mm.GALAXY[((1, 8, 160, 4096), (1, 8, 4096, 1024), "BFLOAT8_B", "BFLOAT8_B", "BFLOAT8_B", True, 11, 10)]
    cands, outs = [], []
    for tag, bw, pn, ob, gx in (
        ("shipped", 32, 3, 3, 11),
        ("2d bw64", 64, 3, 3, 11),
        ("2d bw128", 128, 3, 3, 11),
        ("2d bw64 ob1", 64, 3, 1, 11),
        ("2d bw32 pn4 g8", 32, 4, 4, 8),
    ):
        p = dict(base[1])
        p.update(in0_block_w=bw, per_core_N=pn, out_block_w=ob, out_subblock_w=ob, grid=(gx, 5))
        pc = mm.build(("2d", p))
        f = lambda _=None, pc=pc: ttnn.matmul(
            x, w, dtype=ttnn.bfloat8_b, compute_kernel_config=ckc, memory_config=DRAM, program_config=pc[1]
        )
        try:
            outs.append((tag, f()))
            cands.append((tag, [f]))
        except Exception as ex:
            print(f"[p81]    {tag}: FAILED {str(ex).splitlines()[0][:160]}")
    times = _time(mesh, cands)
    _report("wo_a 8x160x4096x1024", _chip0(outs[0][1]), truth, outs, times)


@mesh_parametrize
def test_attn_proj_probe(mesh_device):
    mesh = mesh_device
    print(f"[p81] dram banks {mesh.dram_grid_size().x}, grid {mesh.compute_with_storage_grid_size()}")
    sites = set(os.environ.get("P81_SITES", "wq_b,wo_b,wo_a").split(","))
    g = mm.GALAXY
    if "wq_b" in sites:
        shipped = g[((1, 1, 160, 1280), (1, 1, 1280, 32768), "BFLOAT8_B", "BFLOAT8_B", "BFLOAT16", True, 11, 10)]
        _one_site(
            mesh,
            "wq_b",
            160,
            1280,
            32768,
            ttnn.bfloat16,
            shipped,
            [
                ("1d ob5x2", _1d(10, 2, 8, 2)),
                ("1d ob5x2 bw4", _1d(10, 2, 4, 2)),
                ("1d ob5x2 bw5", _1d(10, 2, 5, 2)),
                ("1d ob5x2 bw10", _1d(10, 2, 10, 2)),
                ("1d ob5x2 bw20", _1d(10, 2, 20, 2)),
                ("1d ob5x2 bw40", _1d(10, 2, 40, 2)),
                ("1d ob5x5 sw1", _1d(10, 5, 8, 1)),
                ("1d ob5x1 sw1", _1d(10, 1, 8, 1)),
            ],
            tuple(int(c) for c in os.environ.get("P81_CPB", "").split(",") if c),
            8,
            64,
        )
    if "wo_b" in sites:
        shipped = g[((1, 1, 160, 8192), (1, 1, 8192, 5120), "BFLOAT8_B", "BFLOAT8_B", "BFLOAT16", True, 11, 10)]
        _one_site(
            mesh,
            "wo_b",
            160,
            8192,
            5120,
            ttnn.bfloat16,
            shipped,
            [("1d bw16", _1d(2, 2, 16, 2)), ("1d ob5x1", _1d(2, 1, 8, 1))],
            tuple(int(c) for c in os.environ.get("P81_CPB", "").split(",") if c),
            64,
            40,
        )
    if "wo_a" in sites:
        _wo_a(mesh)
