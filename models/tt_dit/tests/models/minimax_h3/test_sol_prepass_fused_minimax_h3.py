# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Fused Sol-Attn pre-pass (sol_prepass_fused.SolPrepassFused): pooling and routing vs the torch mirror
(sol_prepass.torch_reference) and vs the ttnn-op pre-pass (SolPrepass), plus a timing comparison.

Run (Galaxy):
  pytest models/tt_dit/tests/models/minimax_h3/test_sol_prepass_fused_minimax_h3.py -k 4x8 -s --tb=short
Env: SOL_FUSED_REPS (bench reps, default 5), SOL_FUSED_RESULTS (jsonl path for the bench rows).
"""

from __future__ import annotations

import json
import os
import time

import pytest
import torch
from loguru import logger

import ttnn

from ....models.transformers.minimax_h3.sol_prepass import (
    NEG,
    SolGeometry,
    SolPrepass,
    build_constants,
    torch_reference,
)
from ....models.transformers.minimax_h3.sol_prepass_fused import SolPrepassFused, torch_reference_chunk
from ....parallel.manager import CCLManager
from ....utils.tensor import from_torch
from .common import GALAXY_RING

H = 14
D = 128


def _structured_qkv(S: int, rows: int, seed: int = 1):
    """Q/K that share a few 'topics' so the routing is non-trivial; bf16-representable fp32."""
    torch.manual_seed(seed)
    topics = torch.randn(8, D)
    tid = torch.randint(0, 8, (S * rows,))
    q = (topics[tid] + 0.7 * torch.randn(S * rows, D)).view(S, 1, rows, D).expand(S, H, rows, D).clone()
    q += 0.3 * torch.randn(S, H, rows, D)
    k = (topics[tid] + 0.7 * torch.randn(S * rows, D)).view(S, 1, rows, D).expand(S, H, rows, D).clone()
    k += 0.3 * torch.randn(S, H, rows, D)
    v = torch.randn(S, H, rows, D)
    return tuple(x.bfloat16().float() for x in (q, k, v))


def _upload(mesh_device, sp_axis, x):  # [S,H,rows,D] -> per device [1,H,rows,D], sharded on SP, replicated on TP
    return from_torch(
        x, device=mesh_device, dtype=ttnn.bfloat16, mesh_axes=[sp_axis, None, None, None], layout=ttnn.TILE_LAYOUT
    )


def _to_host(mesh_device, sp_axis, tp_axis, t):
    dims = [None, None]
    dims[sp_axis] = 0
    dims[tp_axis] = 1
    return ttnn.to_torch(
        t, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=dims, mesh_shape=tuple(mesh_device.shape))
    )


def _check_fused(
    mesh_device, sp_axis, tp_axis, ccl_manager, *, geo, tau, mode, sink_tokens, q, k, v, compare_ttnn=True
):
    S = geo.shards
    scale = D**-0.5
    consts = build_constants(
        geo, band=1, sink_tokens=sink_tokens, force_first_chunk=True, scale=scale
    )  # all modes force block 0 per shard
    ref = torch_reference(q, k, v, geo, consts, tau=tau, scale=scale)
    compare_ttnn = compare_ttnn and mode != "compact"  # the ttnn SolPrepass has no compact mode
    if mode == "chunk":
        refc = torch_reference_chunk(ref, consts)
        ref_mask, ref_bias = refc["block_mask"], refc["bias"]
        cols = torch.cat([torch.arange(s * geo.nkc, s * geo.nkc + geo.nkc) for s in range(S)])
    else:
        ref_mask, ref_bias = ref["block_mask"], ref["bias"]
        cols = torch.cat([torch.arange(s * geo.nb_pad, s * geo.nb_pad + geo.nb) for s in range(S)])

    tq, tk, tv = (_upload(mesh_device, sp_axis, x) for x in (q, k, v))
    pre = SolPrepassFused(mesh_device=mesh_device, ccl_manager=ccl_manager, sp_axis=sp_axis, tau=tau, mode=mode)
    pre.prepare(geo, sink_tokens, scale)

    # ---- pooling alone
    pooled = pre.pool(tq, tk, tv)
    ttnn.synchronize_device(mesh_device)
    qb = _to_host(mesh_device, sp_axis, tp_axis, pooled["q_bar"])[:, :H].float()  # [S,H,nbp,D]
    torch.testing.assert_close(qb, ref["q_bar"].bfloat16().float(), atol=3e-2, rtol=3e-2)
    kcl = _to_host(mesh_device, sp_axis, tp_axis, pooled["kc_local"])[:, :H].float()
    kc_ref = ref["pooled_k"].view(H, S, geo.nb_pad, D).permute(1, 0, 2, 3)
    torch.testing.assert_close(kcl, kc_ref.bfloat16().float(), atol=3e-2, rtol=3e-2)
    for t in pooled.values():
        ttnn.deallocate(t)
    logger.info("pool OK")

    # ---- full pre-pass
    out = pre(tq, tk, tv, scale=scale, report_density=True)
    ttnn.synchronize_device(mesh_device)
    pk = _to_host(mesh_device, sp_axis, tp_axis, out["pooled_k"])[0, :H].float()
    torch.testing.assert_close(pk, ref["pooled_k"].bfloat16().float(), atol=3e-2, rtol=3e-2)
    pv = _to_host(mesh_device, sp_axis, tp_axis, out["pooled_v"])[0, :H].float()
    torch.testing.assert_close(pv, ref["pooled_v"].bfloat16().float(), atol=3e-2, rtol=3e-2)

    dev_mask = _to_host(mesh_device, sp_axis, tp_axis, out["block_mask"])[:, :H].float()  # [S,H,nqc_pad,W]
    assert dev_mask.shape == ref_mask.shape, (dev_mask.shape, ref_mask.shape)
    agree_all = (dev_mask == ref_mask).float().mean().item()
    agree_real = (dev_mask[:, :, : geo.nqc][..., cols] == ref_mask[:, :, : geo.nqc][..., cols]).float().mean().item()
    dens_dev = dev_mask[:, :, : geo.nqc][..., cols].mean().item()
    dens_ref = ref_mask[:, :, : geo.nqc][..., cols].mean().item()
    logger.info(
        f"mask vs mirror: agreement all {agree_all:.5f} real {agree_real:.5f}; density dev {dens_dev:.4f} ref {dens_ref:.4f} (reported {pre.last_density:.4f})"
    )
    dump = os.environ.get("SOL_FUSED_DUMP")
    if dump:  # offline analysis: device vs reference routing
        os.makedirs(dump, exist_ok=True)
        torch.save(
            {
                "mode": mode,
                "tau": tau,
                "geo": geo.__dict__,
                "dev_mask": dev_mask,
                "ref_mask": ref_mask,
                "ref_bias": ref_bias,
                "q_bar": ref["q_bar"],
                "pooled_k": ref["pooled_k"],
                "threshold": ref["threshold"],
                "cols": cols,
                "dev_bias_row0": _to_host(mesh_device, sp_axis, tp_axis, out["bias"])[:, :H]
                .float()
                .view(S, H, geo.nqc_pad, 32, geo.n_blocks_pad)[:, :, :, 0],
            },
            os.path.join(dump, f"fused_{mode}_{geo.rows}_{geo.logical_n}_{geo.k_chunk}_tau{tau}.pt"),
        )
    assert set(dev_mask.unique().tolist()) <= {0.0, 1.0}
    if mode == "chunk":
        assert torch.all(dev_mask[..., S * geo.nkc :] == 0), "mask padding words must be zero"

    if out["bias"] is None:  # TT_SOL_EMIT_BIAS=0: the pre-pass emits no bias tiles, mask-only check
        logger.info("bias not emitted (TT_SOL_EMIT_BIAS=0): mask-only check")
        marker_agree, bias_err = 1.0, 0.0
    else:
        bias = (
            _to_host(mesh_device, sp_axis, tp_axis, out["bias"])[:, :H]
            .float()
            .view(S, H, geo.nqc_pad, 32, geo.n_blocks_pad)
        )
        real = bias[:, :, : geo.nqc]
        assert torch.equal(real[:, :, :, 0], real[:, :, :, 17]), "bias tile rows must be identical"
        b0 = real[:, :, :, 0]
        kept_dev = b0 < NEG / 2
        kept_ref = ref_bias[:, :, : geo.nqc] < NEG / 2
        marker_agree = (kept_dev == kept_ref).float().mean().item()
        live = ~kept_ref & ~kept_dev
        bias_err = (b0[live] - ref_bias[:, :, : geo.nqc][live].bfloat16().float()).abs().max().item() if live.any() else 0.0
        logger.info(f"bias: kept-marker agreement {marker_agree:.5f}, max abs err on approximated blocks {bias_err:.3f}")
        assert marker_agree > 0.985

    result = {"agree_all": agree_all, "agree_real": agree_real, "marker_agree": marker_agree, "bias_err": bias_err}
    if compare_ttnn:
        pre_ttnn = SolPrepass(mesh_device=mesh_device, ccl_manager=ccl_manager, sp_axis=sp_axis, tau=tau, mode=mode)
        pre_ttnn.prepare(geo, sink_tokens, scale)
        out2 = pre_ttnn(tq, tk, tv, scale=scale, report_density=True)
        ttnn.synchronize_device(mesh_device)
        m2 = _to_host(mesh_device, sp_axis, tp_axis, out2["block_mask"])[:, :H].float()
        agree_ttnn = (m2[:, :, : geo.nqc][..., cols] == dev_mask[:, :, : geo.nqc][..., cols]).float().mean().item()
        agree_ttnn_ref = (m2[:, :, : geo.nqc][..., cols] == ref_mask[:, :, : geo.nqc][..., cols]).float().mean().item()
        logger.info(
            f"mask fused vs ttnn pre-pass: agreement {agree_ttnn:.5f} (ttnn vs mirror {agree_ttnn_ref:.5f}; ttnn density {pre_ttnn.last_density:.4f})"
        )
        result["agree_ttnn"] = agree_ttnn
        result["agree_ttnn_ref"] = agree_ttnn_ref
    for t in out.values():
        if t is not None:
            ttnn.deallocate(t)
    return result


@GALAXY_RING
@pytest.mark.parametrize(
    ("rows", "logical_n", "q_chunk", "k_chunk"),
    [(1184, 9000, 256, 512), (1184, 9472, 256, 256), (1184, 7000, 256, 512)],
    ids=["mid_shard_cut", "k256", "two_padding_shards"],
)
@pytest.mark.parametrize("mode", ["chunk", "gather", "compact"])
def test_sol_prepass_fused(
    mesh_device, sp_axis, tp_axis, num_links, topology, is_fsdp, rows, logical_n, q_chunk, k_chunk, mode, reset_seeds
):
    S = tuple(mesh_device.shape)[sp_axis]
    geo = SolGeometry(shards=S, rows=rows, logical_n=logical_n, q_chunk=q_chunk, k_chunk=k_chunk)
    q, k, v = _structured_qkv(S, rows)
    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)
    r = _check_fused(
        mesh_device, sp_axis, tp_axis, ccl_manager, geo=geo, tau=1.0, mode=mode, sink_tokens=300, q=q, k=k, v=v
    )
    # Bar: at least as close to the torch mirror as the production ttnn pre-pass is on the same inputs (its bf16 scores
    # flip blocks near the threshold too), and never below 98.5 %.
    bar = max(0.985, r["agree_ttnn_ref"] - 0.002) if "agree_ttnn_ref" in r else 0.985
    assert r["agree_real"] >= bar, (r, bar)


@GALAXY_RING
@pytest.mark.parametrize("tau", [-50.0, 50.0], ids=["all_dense", "constants_only"])
def test_sol_prepass_fused_extremes(mesh_device, sp_axis, tp_axis, num_links, topology, is_fsdp, tau, reset_seeds):
    """tau -> -inf keeps every full block (dense); tau -> +inf keeps only band/sink/prefix/forced blocks."""
    S = tuple(mesh_device.shape)[sp_axis]
    geo = SolGeometry(shards=S, rows=1184, logical_n=9000, q_chunk=256, k_chunk=512)
    q, k, v = _structured_qkv(S, 1184, seed=3)
    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)
    r = _check_fused(
        mesh_device,
        sp_axis,
        tp_axis,
        ccl_manager,
        geo=geo,
        tau=tau,
        mode="chunk",
        sink_tokens=300,
        q=q,
        k=k,
        v=v,
        compare_ttnn=False,
    )
    assert r["agree_all"] == 1.0 and r["marker_agree"] == 1.0, r


@GALAXY_RING
@pytest.mark.parametrize(
    ("rows", "logical_n", "q_chunk", "k_chunk"),
    [
        (4736, 37800, 320, 384),
        (9184, 73425, 256, 512),
        (13664, 109101, 256, 512),
    ],  # the module's chunking per working point
    ids=["5s", "10s", "15s"],
)
def test_sol_prepass_fused_bench(
    mesh_device, sp_axis, tp_axis, num_links, topology, is_fsdp, rows, logical_n, q_chunk, k_chunk, reset_seeds
):
    """Production geometries, random data: correctness vs mirror, then wall time of SolPrepass vs SolPrepassFused."""
    S = tuple(mesh_device.shape)[sp_axis]
    geo = SolGeometry(shards=S, rows=rows, logical_n=logical_n, q_chunk=q_chunk, k_chunk=k_chunk)
    scale = D**-0.5
    sink_tokens = {13664: 1245, 9184: 849, 4736: 453}[rows]  # real t2va prefix lengths (text + audio) per working point
    torch.manual_seed(0)
    q, k, v = (torch.randn(S, H, rows, D).bfloat16().float() for _ in range(3))
    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)
    r = _check_fused(
        mesh_device,
        sp_axis,
        tp_axis,
        ccl_manager,
        geo=geo,
        tau=1.0,
        mode="chunk",
        sink_tokens=sink_tokens,
        q=q,
        k=k,
        v=v,
    )
    assert r["agree_real"] >= max(0.985, r.get("agree_ttnn_ref", 1.0) - 0.002), r

    tq, tk, tv = (_upload(mesh_device, sp_axis, x) for x in (q, k, v))
    reps = int(os.environ.get("SOL_FUSED_REPS", "5"))
    rows_out = []
    for name, cls in (("ttnn", SolPrepass), ("fused", SolPrepassFused)):
        pre = cls(mesh_device=mesh_device, ccl_manager=ccl_manager, sp_axis=sp_axis, tau=1.0, mode="chunk")
        pre.prepare(geo, sink_tokens, scale)
        times = []
        for i in range(reps + 1):
            ttnn.synchronize_device(mesh_device)
            t0 = time.perf_counter()
            out = pre(tq, tk, tv, scale=scale)
            t_enq = (time.perf_counter() - t0) * 1e3
            ttnn.synchronize_device(mesh_device)
            t_all = (time.perf_counter() - t0) * 1e3
            for t in out.values():
                if t is not None:
                    ttnn.deallocate(t)
            if i > 0:
                times.append((t_enq, t_all))
        enq = sorted(t[0] for t in times)[len(times) // 2]
        tot = sorted(t[1] for t in times)[len(times) // 2]
        logger.info(
            f"pre-pass {name:5s} rows {rows} chunks {q_chunk}x{k_chunk}: median wall {tot:.2f} ms (host enqueue {enq:.2f} ms) over {reps} reps"
        )
        rows_out.append(
            {
                "prepass": name,
                "rows": rows,
                "logical_n": logical_n,
                "q_chunk": q_chunk,
                "k_chunk": k_chunk,
                "mode": "chunk",
                "wall_ms": tot,
                "enqueue_ms": enq,
                "reps": reps,
                "load": "uncontended (device handshake)",
            }
        )
    # Breakdown of the fused call: pool program, the two all-gathers, route program (host enqueue / wall, median).
    pre = SolPrepassFused(mesh_device=mesh_device, ccl_manager=ccl_manager, sp_axis=sp_axis, tau=1.0, mode="chunk")
    pre.prepare(geo, sink_tokens, scale)
    pre(tq, tk, tv, scale=scale)  # warm
    parts = {"pool": [], "gather_k": [], "gather_v": [], "route": []}
    for _ in range(reps):
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        outs = pre._run_pool(tq, tk, tv)
        t1 = time.perf_counter()
        ttnn.synchronize_device(mesh_device)
        t2 = time.perf_counter()
        parts["pool"].append(((t1 - t0) * 1e3, (t2 - t0) * 1e3))
        t0 = time.perf_counter()
        kc = ccl_manager.all_gather(outs[1], dim=2, mesh_axis=sp_axis, use_hyperparams=False)
        t1 = time.perf_counter()
        ttnn.synchronize_device(mesh_device)
        t2 = time.perf_counter()
        parts["gather_k"].append(((t1 - t0) * 1e3, (t2 - t0) * 1e3))
        t0 = time.perf_counter()
        vc = ccl_manager.all_gather(outs[2], dim=2, mesh_axis=sp_axis, use_hyperparams=False)
        t1 = time.perf_counter()
        ttnn.synchronize_device(mesh_device)
        t2 = time.perf_counter()
        parts["gather_v"].append(((t1 - t0) * 1e3, (t2 - t0) * 1e3))
        t0 = time.perf_counter()
        mask, bias, union = pre._run_route(outs[0], kc)
        t1 = time.perf_counter()
        ttnn.synchronize_device(mesh_device)
        t2 = time.perf_counter()
        parts["route"].append(((t1 - t0) * 1e3, (t2 - t0) * 1e3))
        for x in (*outs, kc, vc, mask, bias):
            ttnn.deallocate(x)
    med = lambda xs, i: sorted(x[i] for x in xs)[len(xs) // 2]
    logger.info(
        f"fused host times (last call, ms): "
        + ", ".join(f"{k}: " + "/".join(f"{n} {v * 1e3:.2f}" for n, v in d.items()) for k, d in pre.host_times.items())
    )
    breakdown = {k: {"enqueue_ms": med(v, 0), "wall_ms": med(v, 1)} for k, v in parts.items()}
    logger.info(
        f"fused breakdown rows {rows}: "
        + ", ".join(f"{k} enq {d['enqueue_ms']:.2f} / wall {d['wall_ms']:.2f} ms" for k, d in breakdown.items())
    )
    rows_out.append(
        {
            "prepass": "fused-breakdown",
            "rows": rows,
            "logical_n": logical_n,
            "q_chunk": q_chunk,
            "k_chunk": k_chunk,
            "mode": "chunk",
            "reps": reps,
            "breakdown": breakdown,
        }
    )
    path = os.environ.get("SOL_FUSED_RESULTS")
    if path:
        with open(path, "a") as f:
            for row in rows_out:
                f.write(json.dumps(row) + "\n")


# ----------------------------------------------------------------------------- primitive checks (one tile, one core)


def _prim_run(mesh_device, op, a, b, s1=1.0, s2=0.0):
    """Runs sol_prim_compute.cpp with OP=op on tiles a, b (torch [32,32]) on every device; returns device 0's tile."""
    from ....models.transformers.minimax_h3.sol_prepass_fused import KERNEL_DIR, _f32_bits

    core = ttnn.CoreCoord(0, 0)
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])
    up = lambda t: ttnn.from_torch(
        t.view(1, 1, 32, 32),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    ta, tb = up(a), up(b)
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, 32, 32]), ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh_device, ttnn.DRAM_MEMORY_CONFIG
    )
    fmt = lambda i: ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.bfloat16, page_size=2048)
    cbs = [ttnn.CBDescriptor(total_size=4096, core_ranges=grid, format_descriptors=[fmt(i)]) for i in (0, 1, 16)]
    rd_ct = [2]
    for t in (ta, tb, ta, ta):  # reader_inputs.cpp declares four accessors; pad with ta
        rd_ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    reader = ttnn.KernelDescriptor(
        kernel_source="ttnn/cpp/ttnn/kernel_lib/tests/eltwise/chain/reconfig/reader_inputs.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=grid,
        compile_time_args=rd_ct,
        runtime_args=[(core, [ta.buffer_address(), tb.buffer_address(), 1, 0])],
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source="ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=grid,
        compile_time_args=[16] + list(ttnn.TensorAccessorArgs(out).get_compile_time_args()),
        runtime_args=[(core, [out.buffer_address(), 1, 0])],
        config=ttnn.WriterConfigDescriptor(),
    )
    compute = ttnn.KernelDescriptor(
        kernel_source=os.path.join(KERNEL_DIR, "sol_prim_compute.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=grid,
        compile_time_args=[op, _f32_bits(s1), _f32_bits(s2)],
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, dst_full_sync_en=True
        ),
    )
    prog = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
    ttnn.generic_op([ta, tb, out], prog)
    ttnn.synchronize_device(mesh_device)
    return ttnn.to_torch(
        out, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=[0, 1], mesh_shape=tuple(mesh_device.shape))
    )[0, 0].float()


@GALAXY_RING
def test_sol_primitives(mesh_device, sp_axis, tp_axis, num_links, topology, is_fsdp, reset_seeds):
    """The compute primitives the route kernel is built from, one tile each, vs torch."""
    torch.manual_seed(0)
    a = (torch.randn(32, 32)).bfloat16().float()
    b = (torch.randn(32, 32)).bfloat16().float()
    ones_row = torch.zeros(32, 32)
    ones_row[0] = 1.0
    results = {}
    results["A@B"] = (_prim_run(mesh_device, 0, a, b), a @ b)
    results["A@T(B)"] = (_prim_run(mesh_device, 1, a, b), a @ b.T)
    results["ones_row@B"] = (_prim_run(mesh_device, 0, ones_row, b), ones_row @ b)
    results["A@T(ones_row)"] = (_prim_run(mesh_device, 1, a, ones_row), a @ ones_row.T)
    results["gtz(A-B)"] = (_prim_run(mesh_device, 2, a, b), (a - b > 0).float())
    results["A*s1+sqrt(B*s1+s2)"] = (
        _prim_run(mesh_device, 3, a, b.abs(), 0.5, 2.0),
        a * 0.5 + (b.abs() * 0.5 + 2.0).sqrt(),
    )
    results["A+B in dst6/7"] = (_prim_run(mesh_device, 4, a, b), a + b)
    results["relu(A*A-B)"] = (_prim_run(mesh_device, 5, a, b), (a * a - b).clamp(min=0))
    results["A*s1 in dst1"] = (_prim_run(mesh_device, 7, a, b, 0.25), a * 0.25)
    results["A*s1 in dst5 (unary, high)"] = (_prim_run(mesh_device, 6, a, b, 0.25), a * 0.25)
    bad = []
    for name, (dev, ref) in results.items():
        err = (dev - ref).abs().max().item()
        ok = torch.allclose(dev, ref.bfloat16().float(), atol=0.1 + 0.02 * ref.abs().max().item(), rtol=0.02)
        logger.info(
            f"primitive {name:22s}: max abs err {err:.4f} {'OK' if ok else 'MISMATCH'}; dev[0,:4]={dev[0,:4].tolist()} ref[0,:4]={ref[0,:4].tolist()}"
        )
        if not ok:
            bad.append(name)
    # 'A*s1 in dst5' documents the fp32-DST limitation the route kernel works around; it is reported, not asserted.
    assert not [n for n in bad if "high" not in n], bad


@GALAXY_RING
def test_sol_prepass_fused_tau_scan(mesh_device, sp_axis, tp_axis, num_links, topology, is_fsdp, reset_seeds):
    """Diagnostic: device vs reference selection density across tau (gather mode); tau=0 isolates the mean term."""
    S = tuple(mesh_device.shape)[sp_axis]
    geo = SolGeometry(shards=S, rows=1184, logical_n=9000, q_chunk=256, k_chunk=512)
    q, k, v = _structured_qkv(S, 1184)
    scale = D**-0.5
    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)
    consts = build_constants(geo, band=1, sink_tokens=300, force_first_chunk=True, scale=scale)
    cols = torch.cat([torch.arange(s * geo.nb_pad, s * geo.nb_pad + geo.nb) for s in range(S)])
    tq, tk, tv = (_upload(mesh_device, sp_axis, x) for x in (q, k, v))
    ref_dens = {
        t: torch_reference(q, k, v, geo, consts, tau=t, scale=scale)["block_mask"][:, :, : geo.nqc][..., cols]
        .mean()
        .item()
        for t in [-2, -1, -0.5, 0, 0.25, 0.5, 1, 1.5, 2, 3, 4, 6]
    }
    logger.info("reference density vs tau: " + ", ".join(f"{t}:{d:.3f}" for t, d in ref_dens.items()))
    for tau in [0.0, 0.5, 1.0, 2.0, 4.0]:
        pre = SolPrepassFused(mesh_device=mesh_device, ccl_manager=ccl_manager, sp_axis=sp_axis, tau=tau, mode="gather")
        pre.prepare(geo, 300, scale)
        out = pre(tq, tk, tv, scale=scale, report_density=False)
        ttnn.synchronize_device(mesh_device)
        dev = _to_host(mesh_device, sp_axis, tp_axis, out["block_mask"])[:, :H].float()[:, :, : geo.nqc][..., cols]
        ref = torch_reference(q, k, v, geo, consts, tau=tau, scale=scale)["block_mask"][:, :, : geo.nqc][..., cols]
        # per-row (Q chunk) density difference pattern: by Q shard and by column shard
        d_dev, d_ref = dev.mean().item(), ref.mean().item()
        by_col = [
            round(
                dev[..., s * geo.nb : (s + 1) * geo.nb].mean().item()
                - ref[..., s * geo.nb : (s + 1) * geo.nb].mean().item(),
                3,
            )
            for s in range(S)
        ]
        logger.info(
            f"tau {tau}: density dev {d_dev:.3f} ref {d_ref:.3f} agreement {(dev == ref).float().mean():.3f}; dev-ref by col shard {by_col}"
        )
        for t in out.values():
            if t is not None:
                ttnn.deallocate(t)


@GALAXY_RING
def test_sol_prepass_fused_moments(mesh_device, sp_axis, tp_axis, num_links, topology, is_fsdp, reset_seeds):
    """Diagnostic: the route program's mean / E2 / var rows (debug_moments) vs torch, head 0 of device 0."""
    S = tuple(mesh_device.shape)[sp_axis]
    geo = SolGeometry(shards=S, rows=1184, logical_n=9000, q_chunk=256, k_chunk=512)
    q, k, v = _structured_qkv(S, 1184)
    scale = D**-0.5
    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)
    consts = build_constants(geo, band=1, sink_tokens=300, force_first_chunk=True, scale=scale)
    ref = torch_reference(q, k, v, geo, consts, tau=1.0, scale=scale)
    tq, tk, tv = (_upload(mesh_device, sp_axis, x) for x in (q, k, v))
    pre = SolPrepassFused(
        mesh_device=mesh_device, ccl_manager=ccl_manager, sp_axis=sp_axis, tau=1.0, mode="gather", debug_moments=True
    )
    pre.prepare(geo, 300, scale)
    out = pre(tq, tk, tv, scale=scale)
    ttnn.synchronize_device(mesh_device)
    bias = _to_host(mesh_device, sp_axis, tp_axis, out["bias"])[0, 0].float()  # device 0, head 0 -> tiles of unit 0
    tiles = bias.view(-1, 32, geo.n_blocks_pad // 32, 32)[: 16 // (geo.n_blocks_pad // 32) + 2]  # tile rows
    flat = bias.view(bias.shape[0] // 32, 32, -1, 32).permute(0, 2, 1, 3).reshape(-1, 32, 32)  # tiles in page order
    kc = ref["pooled_k"][0].float()  # [nbg, D], head 0
    nv = consts["n_valid_blocks"]
    mean = kc.sum(0) / nv
    e2 = (kc * kc).sum(0) / nv
    var = (e2 - mean * mean).clamp(min=0)
    dev_mean = torch.cat([flat[n][0] for n in range(4)])
    dev_e2 = torch.cat([flat[4 + n][0] for n in range(4)])
    dev_var = torch.cat([flat[8 + n][0] for n in range(4)])
    for name, dv, rf in (("mean", dev_mean, mean), ("E2", dev_e2, e2), ("var", dev_var, var)):
        ratio = dv / rf.clamp(min=1e-6)
        logger.info(
            f"{name}: dev[:6]={[round(x, 4) for x in dv[:6].tolist()]} ref[:6]={[round(x, 4) for x in rf[:6].tolist()]} median ratio {ratio.median():.3f} max abs err {(dv - rf).abs().max():.4f}"
        )
    logger.info(
        f"rows 1..31 of mean tile 0 max |.|: {flat[0][1:].abs().max():.4f}; E2 tile rows 1..31: {flat[4][1:].abs().max():.4f}; var tile rows 1..31: {flat[8][1:].abs().max():.4f}"
    )
    torch.testing.assert_close(dev_mean, mean, atol=0.05, rtol=0.05)
    torch.testing.assert_close(dev_e2, e2, atol=0.05, rtol=0.05)
    torch.testing.assert_close(dev_var, var, atol=0.05, rtol=0.05)
