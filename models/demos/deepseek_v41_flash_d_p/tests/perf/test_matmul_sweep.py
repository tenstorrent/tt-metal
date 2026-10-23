# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""P.23 matmul program-config sweep (owner: program configs only; dtypes, compute kernel configs, memory configs and
the op sequence unchanged). Not a gate.

test_capture   runs the profile rung's chunk through the all-device model (warm) with tt/mm_configs.RECORD on: every
               dense ttnn.linear / ttnn.matmul call with its site, exact shapes, dtypes, memory configs, compute
               kernel config, the config MATMUL_CONFIG gave it, and the L1 the model leaves free at that moment
               (lowest occupied L1 address over the allocator base). Writes generated/ds41_p23/calls_<cfg>.json.
test_sweep     per distinct call of the capture (sites above P23_MIN_MS per chunk from the X.3 op profile): the
               current (auto) config vs explicit 2D-multicast configs (grid, in0_block_w, out subblocks, out blocks,
               transposed mcast), 1D multicast (in0 or in1), minimal_matmul (blockings), and for a batched matmul with
               per-batch weights the one-block-per-core MatmulMultiCoreReuseProgramConfig (GLM-5.3: more blocks than
               cores gives wrong results). L1 as in the model: a filler L1 buffer occupies what the model holds
               above the CBs, so a config that would clash in the model fails here. Each candidate: first run right
               after a same-shape auto run on OTHER inputs (stale-L1 check), twice (determinism), error vs the auto
               output and vs float64 torch on the device's rounded inputs; device time = the real-time profiler's
               kernel time of P23_ITERS calls on the mesh (every chip runs it, as in the model). Prints the best
               configs per call and what tt/mm_configs.program_config picks; writes generated/ds41_p23/sweep.json.
               P23_SITES (comma list of sites) selects; P23_FULL=1 keeps every candidate in the json.

    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all --no-precompile \\
        models/demos/deepseek_v41_flash_d_p/tests/perf/test_matmul_sweep.py -k capture
    DS41_MATMUL_CONFIG=auto ... -k capture   (the auto configs' L1 view)
    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all --no-precompile \\
        models/demos/deepseek_v41_flash_d_p/tests/perf/test_matmul_sweep.py -k sweep
"""

from __future__ import annotations

import itertools
import json
import math
import os
import pathlib

import torch

import ttnn
from models.demos.common.bringup.reference.golden import Golden
from models.demos.common.bringup.testing import profiler
from models.demos.common.bringup.testing.harness import begin_chunk, mesh_parametrize, spec
from models.demos.common.bringup.testing.profile import profile_rung
from models.demos.deepseek_v41_flash_d_p.tt import mm_configs as mm
from models.demos.deepseek_v41_flash_d_p.tt.settings import settings

OUT = pathlib.Path("generated/ds41_p23")
ITERS = int(os.environ.get("P23_ITERS", "20"))
DRAM = ttnn.DRAM_MEMORY_CONFIG
DT = {
    "DataType.BFLOAT16": ttnn.bfloat16,
    "DataType.FLOAT32": ttnn.float32,
    "DataType.BFLOAT8_B": ttnn.bfloat8_b,
    "DataType.BFLOAT4_B": ttnn.bfloat4_b,
}
FID = {
    "MathFidelity.LoFi": ttnn.MathFidelity.LoFi,
    "MathFidelity.HiFi2": ttnn.MathFidelity.HiFi2,
    "MathFidelity.HiFi3": ttnn.MathFidelity.HiFi3,
    "MathFidelity.HiFi4": ttnn.MathFidelity.HiFi4,
}


# ---------------------------------------------------------------------------------------------------- capture
@mesh_parametrize
def test_capture(mesh_device):
    s = spec()
    mesh = mesh_device
    rung_name = profile_rung(s)
    rung = s.rung(rung_name)
    g = Golden.for_rung(s, rung_name)
    layers = [i for i in s.layers() if i in g.layers]
    chunk = rung["chunk"]
    start = rung["seq"] - chunk
    model = s.hooks().device_model(mesh, s, layers, lm_head=False)
    state = model.new_state(rung["seq"])
    prefix = {i: g.state(i, at=start) for i in layers}
    tokens = g.tokens()[start:]
    starts = {i for k, i in enumerate(layers) if k == 0 or layers[k - 1] != i - 1}

    def run():
        for i in layers:
            state.load_prefix(i, prefix[i], start)
        begin_chunk(model, tokens, start)
        h = None
        for i in layers:
            if i in starts:
                if h is not None:
                    model.free(h)
                h = model.embed(tokens) if i == 0 else model.from_host(g.layer(g.n_chunks - 1, i)["in"].float())
            n0 = len(mm.RECORD) if mm.RECORD is not None else 0
            h2 = model.layer(i, h, start, state)
            if mm.RECORD is not None:
                for r in mm.RECORD[n0:]:
                    r["layer"] = i
            model.free(h)
            h = h2
        model.sync()
        out = model.to_host(h).float()
        model.free(h)
        return out

    run()  # compile
    mm.RECORD = []
    try:
        out = run()
        rec = mm.RECORD
    finally:
        mm.RECORD = None
    want = g.layer(g.n_chunks - 1, layers[-1])["out"].float()
    rel = float((out.double() - want.double()).norm() / want.double().norm())
    grid = mesh.compute_with_storage_grid_size()
    OUT.mkdir(parents=True, exist_ok=True)
    tag = settings.get("MATMUL_CONFIG")
    path = OUT / f"calls_{tag}.json"
    path.write_text(json.dumps({"grid": [grid.x, grid.y], "rel_out": rel, "calls": rec}, indent=1))
    print(
        f"[p23] MATMUL_CONFIG={tag}: {len(rec)} matmul calls, grid {grid.x}x{grid.y}, chunk out vs golden rel {rel:.5f}"
    )
    for r in rec:
        free = r["l1_cb_free"]
        print(
            f"[p23]  L{r['layer']:02d} {r['site']:22s} {r['op']:6s} {r['a']} {r['a_dtype'][9:]} x {r['b']} "
            f"{r['b_dtype'][9:]} -> {r['out_dtype'][9:]} {r['fidelity'][13:]} fp32acc={int(r['fp32_dest_acc_en'])} "
            f"l1acc={int(r['packer_l1_acc'])} L1 free for CBs {free / 1024:.0f} KiB  cfg={'auto' if r['config'] is None else 'tuned'}"
        )
    print(f"[p23] wrote {path}")


# ---------------------------------------------------------------------------------------------------- sweep
def _divisors(n: int, cap: int | None = None) -> list[int]:
    cap = n if cap is None else min(n, cap)
    return [d for d in range(1, cap + 1) if n % d == 0]


def _subblocks(pm: int, pn: int, dst: int) -> list[tuple[int, int]]:
    """Out subblocks (h, w), h | pm, w | pn, h * w <= dst: the largest few, w-major first."""
    c = [(h, w) for h in _divisors(pm, dst) for w in _divisors(pn, dst) if h * w <= dst]
    c.sort(key=lambda t: (-t[0] * t[1], -t[1]))
    return c[:3]


def _l1_2d(pm, pn, bw, a, b, o, fp32, ob_h=None, ob_w=None):
    ob_h, ob_w = ob_h or pm, ob_w or pn
    interm = ob_h * ob_w * 4096 if fp32 else 0  # the K-block partials (fp32 DEST, packer_l1_acc off)
    return 2 * pm * bw * a + 2 * bw * pn * b + pm * pn * o + interm


def candidates(call: dict, grid, budget: int) -> list[tuple[str, str, object]]:
    """(tag, kind, config) for one call: kind linear / matmul (program_config) or minimal (MinimalMatmulConfig)."""
    a_shape, b_shape = call["a"], call["b"]
    B = math.prod(b_shape[:-2])
    M = math.prod(a_shape[:-1]) if B == 1 else a_shape[-2]
    K, N = a_shape[-1], b_shape[-1]
    Mt, Kt, Nt = math.ceil(M / 32), K // 32, N // 32
    a, b = mm.TILE_BYTES[DT[call["a_dtype"]]], mm.TILE_BYTES[DT[call["b_dtype"]]]
    o = mm.TILE_BYTES[DT[call["out_dtype"]]]
    fp32 = call["fp32_dest_acc_en"]
    dst = 4 if fp32 else 8
    kind = call["op"]
    out = []
    if B > 1:  # batched, one weight per batch: MatmulMultiCoreReuseProgramConfig, at most one block per core
        cores = grid.x * grid.y
        for pm in _divisors(Mt):
            for pn in _divisors(Nt):
                blocks = B * (Mt // pm) * (Nt // pn)
                if blocks > cores:
                    continue
                for bw in (1, 2, 4, 8):
                    if Kt % bw or 2 * pm * bw * a + 2 * bw * pn * b + pm * pn * (o + (4096 if fp32 else 0)) > budget:
                        continue
                    for sh, sw in _subblocks(pm, pn, dst)[:2]:
                        out.append(
                            (
                                f"bmm pm{pm} pn{pn} bw{bw} sb{sh}x{sw} blocks{blocks}",
                                kind,
                                ttnn.MatmulMultiCoreReuseProgramConfig(
                                    compute_with_storage_grid_size=grid,
                                    in0_block_w=bw,
                                    out_subblock_h=sh,
                                    out_subblock_w=sw,
                                    per_core_M=pm,
                                    per_core_N=pn,
                                ),
                            )
                        )
        return out
    # 2D multicast: M over grid rows (transposed: columns), N over the other axis
    grids = set()
    for transpose in (False, True):
        gm_max, gn_max = (grid.x, grid.y) if transpose else (grid.y, grid.x)
        for gm, gn in ((gm_max, gn_max), (gm_max, max(1, gn_max // 2)), (max(1, gm_max // 2), gn_max)):
            pm, pn = math.ceil(Mt / gm), math.ceil(Nt / gn)
            grids.add((transpose, math.ceil(Mt / pm), math.ceil(Nt / pn), pm, pn))
    for transpose, gm, gn, pm, pn in sorted(grids):
        gx, gy = (gm, gn) if transpose else (gn, gm)
        for bw in (2, 4, 8, 16):
            if Kt % bw:
                continue
            for ob_h, ob_w in sorted({(pm, pn), (pm, max(_divisors(pn, max(1, pn // 2))))}):
                if _l1_2d(pm, pn, bw, a, b, o, fp32, ob_h, ob_w) > budget:
                    continue
                for sh, sw in _subblocks(ob_h, ob_w, dst)[:1]:
                    out.append(
                        (
                            f"2d {gx}x{gy}{' T' if transpose else ''} pm{pm} pn{pn} ob{ob_h}x{ob_w} bw{bw} sb{sh}x{sw}",
                            kind,
                            ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                                compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
                                in0_block_w=bw,
                                out_subblock_h=sh,
                                out_subblock_w=sw,
                                out_block_h=ob_h,
                                out_block_w=ob_w,
                                per_core_M=pm,
                                per_core_N=pn,
                                transpose_mcast=transpose,
                                fused_activation=None,
                                fuse_batch=True,
                            ),
                        )
                    )
    # 1D multicast: in0 (N over the cores, all M a core) or in1 (M over the cores, all N a core)
    cores = grid.x * grid.y
    for mcast_in0 in (True, False):
        if mcast_in0:
            pm, pn = Mt, math.ceil(Nt / cores)
        else:
            pm, pn = math.ceil(Mt / cores), Nt
        for bw in (1, 2, 4, 8, 16):
            if Kt % bw or _l1_2d(pm, pn, bw, a, b, o, fp32) > budget:
                continue
            for sh, sw in _subblocks(pm, pn, dst)[:1]:
                out.append(
                    (
                        f"1d {'in0' if mcast_in0 else 'in1'} pm{pm} pn{pn} bw{bw} sb{sh}x{sw}",
                        kind,
                        ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                            compute_with_storage_grid_size=grid,
                            in0_block_w=bw,
                            out_subblock_h=sh,
                            out_subblock_w=sw,
                            per_core_M=pm,
                            per_core_N=pn,
                            fuse_batch=True,
                            fused_activation=None,
                            mcast_in0=mcast_in0,
                        ),
                    )
                )
    # minimal_matmul (its own reader / writer; bf16 and fp32 outputs both tried, judged by the error vs auto)
    out.append(("minimal auto", "minimal", None))
    for mb, kb, nb in itertools.product((4, 8, 16), (4, 8, 16), (4, 8, 16)):
        if mb > Mt or nb > Nt or Kt % kb:
            continue
        for sh, sw in ((2, 2), (1, 4), (4, 1)):
            if mb % sh or nb % sw or sh * sw > dst:
                continue
            if 2 * mb * kb * a + 2 * kb * nb * b + mb * nb * (o + (4096 if fp32 else 0)) > budget:
                continue
            out.append(
                (
                    f"minimal M{mb} K{kb} N{nb} sb{sh}x{sw}",
                    "minimal",
                    ttnn.MinimalMatmulConfig(
                        M_block_size=mb,
                        K_block_size=kb,
                        N_block_size=nb,
                        subblock_h=sh,
                        subblock_w=sw,
                        compute_with_storage_grid_size=grid,
                    ),
                )
            )
            break
    return out


def _rep(mesh, t, dtype):
    return ttnn.from_torch(
        t.contiguous(),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=DRAM,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def _chip0(t) -> torch.Tensor:
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).double()


def _rel(x, y) -> float:
    return float((x - y).norm() / y.norm().clamp_min(1e-30))


def _filler(mesh, free_for_cbs: int):
    """An L1 buffer at the top of L1 so that [base, base + free_for_cbs) is what CBs can use, as in the model."""
    v = mm._l1_view(mesh)
    want = v["l1_cb_free"] - free_for_cbs
    if want <= 0:
        return None
    nb = ttnn.get_memory_view(mesh, ttnn.BufferType.L1).num_banks
    tiles = nb * math.ceil(want / 2048)
    t = ttnn.from_torch(
        torch.zeros(1, 1, 32 * tiles, 32),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    w = mm._l1_view(mesh)
    print(
        f"[p23] L1 filler {want / 1024:.0f} KiB a bank: CBs get {w['l1_cb_free'] / 1024:.0f} KiB "
        f"(the model: {free_for_cbs / 1024:.0f} KiB)"
    )
    return t


def _distinct(calls: list[dict]) -> dict:
    keys = {}
    for r in calls:
        k = (
            r["site"],
            r["op"],
            tuple(r["a"]),
            tuple(r["b"]),
            r["a_dtype"],
            r["b_dtype"],
            r["out_dtype"],
            r["fidelity"],
            r["fp32_dest_acc_en"],
            r["packer_l1_acc"],
            r["math_approx_mode"],
        )
        free = r.get("l1_cb_free")
        if free is None:  # a capture before l1_cb_free: block offsets are relative to the allocator base
            low = r["l1_lowest_allocated"]
            free = r["l1_bank_bytes"] if low is None else low
        e = keys.setdefault(k, {**r, "count": 0, "free_min": free})
        e["count"] += 1
        e["free_min"] = min(e["free_min"], free)
    return keys


@mesh_parametrize
def test_sweep(mesh_device):
    mesh = mesh_device
    cap = json.loads((OUT / "calls_auto.json").read_text())
    grid = mesh.compute_with_storage_grid_size()
    sel = os.environ.get("P23_SITES")
    sel = set(sel.split(",")) if sel else None
    min_ms = float(os.environ.get("P23_MIN_MS", "0.25"))
    prof_ms = _profile_ms()
    calls = _distinct(cap["calls"])
    free_all = min(e["free_min"] for e in calls.values())
    print(f"[p23] grid {grid.x}x{grid.y}; L1 free for CBs at matmul calls in the model: min {free_all / 1024:.0f} KiB")
    filler = _filler(mesh, free_all)
    budget = free_all - 16 * 1024  # small CBs besides in0 / in1 / out / interm (bias, sync) and rounding
    results = []
    torch.manual_seed(0)
    for k, call in calls.items():
        site = call["site"]
        if sel is not None and site not in sel:
            continue
        shape_key = _shape_str(call)
        per_chunk = prof_ms.get((site, shape_key))
        if sel is None and per_chunk is not None and per_chunk < min_ms:
            continue
        res = _sweep_one(mesh, grid, call, budget)
        res["profile_ms_per_chunk"] = per_chunk
        results.append(res)
    if filler is not None:
        ttnn.deallocate(filler)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sweep.json").write_text(json.dumps(results, indent=1))
    print("[p23] == summary: per call site, ms per call (device, the slowest chip) and per chunk")
    for r in results:
        b, t, p = r["auto"], r["best"], r["picked"]
        print(
            f"[p23]  {r['site']:22s} x{r['count']} {r['shape']}: auto {b['ms']:.4f} | best {t['ms']:.4f} ({t['tag']}, "
            f"vs auto {t['rel_vs_auto']:.2e}) | rule {p['ms'] if p else float('nan'):.4f} "
            f"({p['tag'] if p else 'auto'}) | chunk {b['ms'] * r['count']:.3f} -> "
            f"{(p['ms'] if p else b['ms']) * r['count']:.3f} ms"
        )


def _shape_str(call) -> str:
    def s(shape, dt):
        lead = [d for d in shape[:-2] if d != 1]
        return (
            "x".join(map(str, lead + list(shape[-2:])))
            + " "
            + {
                "DataType.BFLOAT16": "bf16",
                "DataType.FLOAT32": "fp32",
                "DataType.BFLOAT8_B": "bfp8",
                "DataType.BFLOAT4_B": "bfp4",
            }[dt]
        )

    return f"{s(call['a'], call['a_dtype'])} · {s(call['b'], call['b_dtype'])}"


def _profile_ms() -> dict:
    """(site, shape) -> device ms per chunk from the X.3 op profile, keyed by the profile's section names."""
    import gzip

    p = pathlib.Path("models/demos/deepseek_v41_flash_d_p/bringup/results/X.3_ops.json.gz")
    if not p.exists():
        return {}
    sec_site = {
        "attention": ("attn.wq_b", "attn.wo_a", "attn.wo_b"),
        "q_a": ("q_a",),
        "engram_kv": ("engram_kv",),
        "router": ("router",),
        "indexer": ("indexer.wq_b", "indexer.weights_proj"),
        "attn_hc": ("hc.fn",),
        "ffn_hc": ("hc.fn",),
        "compressor": ("compressor",),
    }
    out = {}
    for sec, rows in json.load(gzip.open(p)).items():
        name = sec.split(".", 1)[1]
        for r in rows:
            if r["op"] not in ("linear", "matmul"):
                continue
            for site in sec_site.get(name, ()):
                out[(site, r["shape"])] = out.get((site, r["shape"]), 0.0) + r["ms"]
    return out


def _sweep_one(mesh, grid, call, budget) -> dict:
    a_shape, b_shape = call["a"], call["b"]
    adt, bdt, odt = DT[call["a_dtype"]], DT[call["b_dtype"]], DT[call["out_dtype"]]
    ckc = ttnn.init_device_compute_kernel_config(
        mesh.arch(),
        math_fidelity=FID[call["fidelity"]],
        math_approx_mode=call["math_approx_mode"],
        fp32_dest_acc_en=call["fp32_dest_acc_en"],
        packer_l1_acc=call["packer_l1_acc"],
    )
    K = a_shape[-1]
    x = _rep(mesh, torch.randn(*a_shape), adt)
    x2 = _rep(mesh, torch.randn(*a_shape), adt)  # the other inputs of the stale-L1 check
    w = _rep(mesh, torch.randn(*b_shape) / K**0.5, bdt)
    truth = _chip0(x) @ _chip0(w)
    op = ttnn.linear if call["op"] == "linear" else ttnn.matmul

    def fn_of(kind, cfg):
        if kind == "minimal":
            kw = {} if cfg is None else {"config": cfg}
            return lambda t=x: ttnn.experimental.minimal_matmul(
                t, w, dtype=odt, compute_kernel_config=ckc, memory_config=DRAM, **kw
            )
        kw = {} if cfg is None else {"program_config": cfg}
        return lambda t=x: op(t, w, dtype=odt, compute_kernel_config=ckc, memory_config=DRAM, **kw)

    auto = fn_of(call["op"], None)
    ref = auto()
    ref_h = _chip0(ref)
    ttnn.deallocate(ref)
    cands = [("auto", call["op"], None)] + candidates(call, grid, budget)
    picked = mm.program_config(x, w, odt, ckc, batched=math.prod(b_shape[:-2]) > 1)
    if picked is not None:
        cands.append(("rule", "minimal" if picked[0] == "minimal" else call["op"], picked[1]))
    ok = []
    for tag, kind, cfg in cands:
        fn = fn_of(kind, cfg)
        try:
            ttnn.deallocate(auto(x2))  # same shape, other inputs: stale L1 shows as their result
            o1 = fn()
            o2 = fn()
            h1, h2 = _chip0(o1), _chip0(o2)
            chips = [ttnn.to_torch(t).double() for t in ttnn.get_device_tensors(o1)]
            ttnn.deallocate(o1)
            ttnn.deallocate(o2)
        except Exception as ex:  # a config the op refuses (L1, divisibility)
            print(f"[p23]    {tag}: FAILED {str(ex).splitlines()[0][:100]}")
            continue
        det = bool(torch.equal(h1, h2)) and all(torch.equal(c, chips[0]) for c in chips)
        ok.append(
            {
                "tag": tag,
                "kind": kind,
                "cfg": None if cfg is None else str(cfg),
                "fn": fn,
                "rel_vs_auto": _rel(h1, ref_h),
                "max_vs_auto": float((h1 - ref_h).abs().max()),
                "rel_vs_fp64": _rel(h1, truth),
                "deterministic": det,
            }
        )
    # device time: one profiled pass, a section per candidate
    profiler.enable(mesh, rt=True)
    try:
        for c in ok:
            profiler.signpost(c["tag"])
            for _ in range(ITERS):
                ttnn.deallocate(c["fn"]())
        profiler.signpost("end")
        profiler.finish_rt()
        prof = profiler.result()
    finally:
        profiler.disable()
    for c in ok:
        c["ms"] = prof["kernel_ns"].get(c["tag"], float("nan")) / ITERS / 1e6
        del c["fn"]
    base = next(c for c in ok if c["tag"] == "auto")
    # a candidate is acceptable when it is deterministic and no further from float64 than auto (+ 1 %)
    good = [
        c
        for c in ok
        if c["deterministic"] and c["rel_vs_fp64"] <= base["rel_vs_fp64"] * 1.01 + 1e-7 and c["tag"] != "rule"
    ]
    good.sort(key=lambda c: c["ms"])
    best = good[0] if good else base
    picked_run = next((c for c in ok if c["tag"] == "rule"), None)
    head = f"{call['site']} {_shape_str(call)} -> {call['out_dtype'][9:]} {call['fidelity'][13:]} fp32acc={int(call['fp32_dest_acc_en'])}"
    print(f"[p23] == {head}, x{call['count']} a chunk, L1 budget {budget / 1024:.0f} KiB, {len(ok)} configs ran")
    show = good[:8] + [base] + ([picked_run] if picked_run else [])
    for c in show:
        print(
            f"[p23]    {c['ms']:.4f} ms x{base['ms'] / c['ms']:.2f}  vs auto {c['rel_vs_auto']:.2e} "
            f"(max {c['max_vs_auto']:.2e})  vs fp64 {c['rel_vs_fp64']:.3e}  det {int(c['deterministic'])}  {c['tag']}"
        )
    bad = [c for c in ok if not c["deterministic"] or c["rel_vs_fp64"] > base["rel_vs_fp64"] * 1.01 + 1e-7]
    for c in bad[:6]:
        print(f"[p23]    REJECTED {c['tag']}: vs fp64 {c['rel_vs_fp64']:.3e}, det {int(c['deterministic'])}")
    for t in (x, x2, w):
        ttnn.deallocate(t)
    keep = ok if os.environ.get("P23_FULL") == "1" else show
    return {
        "site": call["site"],
        "shape": _shape_str(call),
        "out_dtype": call["out_dtype"],
        "count": call["count"],
        "auto": base,
        "best": best,
        "picked": picked_run,
        "rejected": [c["tag"] for c in bad],
        "configs": keep,
    }


# ---------------------------------------------------------------------------------------------------- other boxes
def _variants(call: dict, hidden_tp: int) -> list[tuple[str, list, list]]:
    """The call at other boxes' per-chip shapes: rows 640 / 512 (galaxy_8x4 at 5120 / 4096-token chunks) and the
    hidden-split widths of TP 4 (this run's TP-2 hidden slice ``hidden_tp`` -> half; 5 blocks of it likewise)."""
    out = []
    for rows in (640, 512):
        a, b = list(call["a"]), list(call["b"])
        a[-2] = rows if a[-2] >= 1024 else rows // 2  # the indexer's query rows are rows / TP
        for t in (a, b):
            for d in (-1, -2):
                if t[d] in (hidden_tp, 5 * hidden_tp):
                    t[d] //= 2
        out.append((f"rows {a[-2]}", a, b))
    return out


@mesh_parametrize
def test_rule_boxes(mesh_device):
    """The rule's config vs auto at other boxes' per-chip shapes and on a smaller grid (an 8x8 sub-grid stands in for
    another chip's grid): every call valid, the same error vs float64, deterministic, its time vs auto."""
    mesh = mesh_device
    cap = json.loads((OUT / "calls_auto.json").read_text())
    grid = mesh.compute_with_storage_grid_size()
    hidden_tp = 2560  # this run's hidden slice a chip (5120 / TP 2)
    calls = _distinct(cap["calls"])
    rows = []
    for k, call in calls.items():
        for tag, a_shape, b_shape in _variants(call, hidden_tp):
            for g in (grid, ttnn.CoreCoord(8, 8)):
                c = dict(call, a=a_shape, b=b_shape)
                rows.append(_check_rule(mesh, c, g, f"{call['site']} {tag} grid {g.x}x{g.y}"))
    bad = [r for r in rows if r and not r["ok"]]
    for r in bad:
        print(f"[p23] BOXES FAIL {r}")
    assert not bad, f"{len(bad)} rule configs failed at other shapes / grids"


def _check_rule(mesh, call, grid, label):
    a_shape, b_shape = call["a"], call["b"]
    adt, bdt, odt = DT[call["a_dtype"]], DT[call["b_dtype"]], DT[call["out_dtype"]]
    B = math.prod(b_shape[:-2])
    M = math.prod(a_shape[:-1]) if B == 1 else a_shape[-2]
    pc = mm.rule(M, a_shape[-1], b_shape[-1], adt, bdt, odt, call["fp32_dest_acc_en"], grid, B)
    if pc is None:
        print(f"[p23]  {label}: {_shape_str(call)} -> auto (no rule entry)")
        return None
    ckc = ttnn.init_device_compute_kernel_config(
        mesh.arch(),
        math_fidelity=FID[call["fidelity"]],
        math_approx_mode=call["math_approx_mode"],
        fp32_dest_acc_en=call["fp32_dest_acc_en"],
        packer_l1_acc=call["packer_l1_acc"],
    )
    K = a_shape[-1]
    x = _rep(mesh, torch.randn(*a_shape), adt)
    x2 = _rep(mesh, torch.randn(*a_shape), adt)
    w = _rep(mesh, torch.randn(*b_shape) / K**0.5, bdt)
    truth = _chip0(x) @ _chip0(w)
    op = ttnn.linear if call["op"] == "linear" else ttnn.matmul
    auto = lambda t=x: op(t, w, dtype=odt, compute_kernel_config=ckc, memory_config=DRAM)  # noqa: E731
    tuned = lambda t=x: mm._run(op, t, w, pc, odt, ckc, DRAM)  # noqa: E731
    ra = auto()
    ha = _chip0(ra)
    ttnn.deallocate(ra)
    try:
        ttnn.deallocate(auto(x2))
        o1, o2 = tuned(), tuned()
        h1, h2 = _chip0(o1), _chip0(o2)
        ttnn.deallocate(o1)
        ttnn.deallocate(o2)
    except Exception as ex:
        print(f"[p23]  {label}: {pc[0]} FAILED {str(ex).splitlines()[0][:160]}")
        return {"label": label, "ok": False, "error": str(ex).splitlines()[0][:160]}
    profiler.enable(mesh, rt=True)
    try:
        for name, fn in (("auto", auto), ("rule", tuned)):
            profiler.signpost(name)
            for _ in range(ITERS):
                ttnn.deallocate(fn())
        profiler.signpost("end")
        profiler.finish_rt()
        prof = profiler.result()
    finally:
        profiler.disable()
    ta, tr = (prof["kernel_ns"].get(n, float("nan")) / ITERS / 1e6 for n in ("auto", "rule"))
    ea, er = _rel(ha, truth), _rel(h1, truth)
    ok = bool(torch.equal(h1, h2)) and er <= ea * 1.01 + 1e-7 and h1.shape == ha.shape
    print(
        f"[p23]  {label}: {_shape_str(call)} {pc[0]}: auto {ta:.4f} -> {tr:.4f} ms, vs auto {_rel(h1, ha):.2e}, "
        f"vs fp64 {er:.3e} (auto {ea:.3e}), det {int(torch.equal(h1, h2))} {'ok' if ok else 'FAIL'}"
    )
    for t in (x, x2, w):
        ttnn.deallocate(t)
    return {"label": label, "ok": ok, "auto_ms": ta, "rule_ms": tr, "rel_fp64": er, "rel_fp64_auto": ea}
