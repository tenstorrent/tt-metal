# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Per-op performance report from the profiler's per-call records (profiler.enable(mesh, ops=True, calls=True)).

Model-agnostic: it only reads the ttnn op name, every tensor argument's / output's per-device shape and dtype, and
the CCL / compute kwargs each call was made with. Ops are grouped by (op, shapes, CCL geometry).

Main table, one row per group:
  calls, calls per layer, median / max ms per call (a call's time = its slowest chip), total ms (with the layer
  weights below), share of the total; matmul and attention ops: math utilization (FLOPs / time over the fidelity's
  peak) and DRAM utilization (bytes moved / time over the DRAM peak); CCL ops: worst-chip ingress GB/s and per link
  GB/s (ingress / (links x receive directions): 2 directions for a Ring topology, 1 for Linear).
Secondary sections: per-chip spread (median chip vs slowest chip, which chip); context scaling (the same op at a second
chunk position, when a second set of calls is given); CCL details (bytes per call, group size, axis, links, topology).

TODO: feed the report from the program real-time profiler (no per-op syncs; tests/test_realtime_probe.py) instead of
the device profiler's op mode, so the whole model can be profiled in one normal-speed pass.

Layer weights: when only representative layers were profiled, ``layer_weights`` maps each profiled layer to how many
model layers it stands for (e.g. one layer per block type x the block type's layer count); totals and calls are then
the full model's. Cost models and peaks are approximations; every assumption is listed in the report's notes.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter, defaultdict

# per-element bytes (block-float formats include the shared exponent: bfp8 1088 B / 1024, bfp4 576 B / 1024)
DTYPE_BYTES = {
    "BFLOAT16": 2.0,
    "FLOAT32": 4.0,
    "BFLOAT8_B": 1088 / 1024,
    "BFLOAT4_B": 576 / 1024,
    "UINT32": 4.0,
    "INT32": 4.0,
    "UINT16": 2.0,
    "UINT8": 1.0,
}
DTYPE_SHORT = {
    "BFLOAT16": "bf16",
    "FLOAT32": "fp32",
    "BFLOAT8_B": "bfp8",
    "BFLOAT4_B": "bfp4",
    "UINT32": "u32",
    "INT32": "i32",
    "UINT16": "u16",
    "UINT8": "u8",
}
PHASES = {"LoFi": 1, "HiFi2": 2, "HiFi3": 3, "HiFi4": 4}
# per-chip peaks (tt-perf-report's Blackhole model: worker cores x 4096 FLOP/cycle x clock / fidelity phases)
ARCH = {
    "blackhole": {"flop_per_core_cycle": 4096, "clock_hz": 1.35e9, "dram_gb_s": 512.0},
    "wormhole_b0": {"flop_per_core_cycle": 2048, "clock_hz": 1.0e9, "dram_gb_s": 288.0},
}

MATMUL_OPS = {"matmul", "linear", "experimental.minimal_matmul"}
SPARSE_SDPA_OPS = {"transformer.sparse_sdpa", "bringup.sparse_sdpa"}
INDEXER_OPS = {
    "experimental.indexer_score_dsa",
    "bringup.indexer_score_dsa",
    "bringup.ring_indexer_score_dsa",
    "experimental.ring_indexer_score_dsa",
}
LOCAL_OPS = {"mesh_partition"}  # mesh ops that move no data between chips
# ttnn CCLs that choose their own link count: ttnn.all_gather ignores num_links (and topology) and uses every link on
# the axis (all_gather_device_operation.cpp, get_num_links(mesh, axis)); reduce_scatter / all_reduce do the same when
# num_links is None
AUTO_LINK_OPS = {"all_gather", "reduce_scatter", "all_reduce"}


def family(op: str) -> str:
    if op in MATMUL_OPS:
        return "matmul"
    if op in SPARSE_SDPA_OPS or op in INDEXER_OPS or "sdpa" in op or "attention" in op.split(".")[-1]:
        return "attention"
    if op not in LOCAL_OPS and any(t in op for t in ("all_gather", "reduce_scatter", "all_reduce", "all_to_all")):
        return "ccl"
    return "other"


def _nbytes(t: dict) -> float:
    return math.prod(t["shape"]) * DTYPE_BYTES.get(t["dtype"], 2.0) if t.get("shape") else 0.0


def _short(t: dict) -> str:
    dims = list(t.get("shape", []))
    while len(dims) > 2 and dims[0] == 1:
        dims = dims[1:]
    return "x".join(map(str, dims)) + " " + DTYPE_SHORT.get(t.get("dtype", "?"), t.get("dtype", "?").lower())


def signature(call: dict) -> str:
    ins = " · ".join(_short(t) for t in call["ins"][:3])
    outs = " · ".join(_short(t) for t in call["outs"][:1])
    return f"{ins} -> {outs}" if outs else ins


# ---------------------------------------------------------------- cost models
def compute_cost(call: dict) -> dict | None:
    """FLOPs and DRAM bytes of one matmul / attention call, or None. 'assumed' lists the guesses made."""
    op, ins, outs, kw = call["op"], call["ins"], call["outs"], call["kw"]
    assumed = []
    fid = kw.get("math_fidelity")
    if fid is None:
        fid = "HiFi2"
        assumed.append("fidelity unknown: HiFi2 peak")
    if op in MATMUL_OPS and len(ins) >= 2:
        a, b = ins[0]["shape"], ins[1]["shape"]
        if len(a) < 2 or len(b) < 2:
            return None
        M, K, N = a[-2], a[-1], b[-1]
        ba, bb = math.prod(a[:-2]), math.prod(b[:-2])
        flops = 2.0 * ba * M * K * N  # batched weights (bb > 1) or a broadcast weight: the same count
        nbytes = _nbytes(ins[0]) + _nbytes(ins[1]) + sum(_nbytes(t) for t in outs[:1])
        return {"flops": flops, "bytes": nbytes, "fidelity": fid, "assumed": assumed}
    if op in SPARSE_SDPA_OPS and len(ins) >= 3:
        q, kv, idx = ins[0], ins[1], ins[2]
        _, H, Sq, D = ([1] * (4 - len(q["shape"])) + q["shape"])[-4:]
        k = idx["shape"][-1]
        dv = kw.get("v_dim") or D
        if "v_dim" not in kw:
            assumed.append("v_dim = q head dim")
        flops = 2.0 * H * Sq * k * (D + dv)
        # each query row reads its k selected rows once (shared by the heads: MQA latent)
        nbytes = (
            _nbytes(q)
            + sum(_nbytes(t) for t in outs[:1])
            + Sq * k * kv["shape"][-1] * DTYPE_BYTES.get(kv["dtype"], 2.0)
        )
        return {"flops": flops, "bytes": nbytes, "fidelity": fid, "assumed": assumed}
    if op in INDEXER_OPS and len(ins) >= 3:
        q, k = ins[0], ins[1]
        _, H, Sq, D = ([1] * (4 - len(q["shape"])) + q["shape"])[-4:]
        T = k["shape"][-2]
        tk = kw.get("kv_len") or T
        if "kv_len" not in kw:
            assumed.append("all T keys scored")
        flops = 2.0 * H * Sq * tk * D
        nbytes = _nbytes(q) + tk * D * DTYPE_BYTES.get(k["dtype"], 2.0) + Sq * tk * 2.0
        return {"flops": flops, "bytes": nbytes, "fidelity": fid, "assumed": assumed}
    return None


def ccl_cost(call: dict, mesh_shape: tuple, axis_links: dict | None = None) -> dict | None:
    """Worst-chip ingress / egress bytes of one CCL call and its link count (links x receive directions).
    axis_links: the system's link count per mesh axis, for the ttnn CCLs that pick their own (AUTO_LINK_OPS)."""
    op, ins, outs, kw = call["op"], call["ins"], call["outs"], call["kw"]
    if family(op) != "ccl" or not ins:
        return None
    assumed = []
    axis = kw.get("cluster_axis")
    if axis is None:
        G = math.prod(mesh_shape)
        assumed.append("no cluster_axis: whole mesh")
    else:
        G = mesh_shape[int(axis)]
    links = kw.get("num_links")
    auto = op in AUTO_LINK_OPS and (op == "all_gather" or links is None)
    if auto:
        axes = [int(axis)] if axis is not None else list(range(len(mesh_shape)))
        if axis_links:
            links = min(axis_links.get(a, 1) for a in axes)
            assumed.append("ttnn CCL picks its own links: all links on the axis")
        else:
            links = 1
            assumed.append("ttnn CCL picks its own links; axis link count unknown: 1")
    elif links is None:
        links = 1
        assumed.append("num_links unset: 1")
    topo = kw.get("topology")
    if topo is None or (auto and op == "all_gather"):
        topo = "Linear"
        assumed.append("topology unset / ignored: Linear (open axis)")
    dirs = 2 if "Ring" in str(topo) else 1
    b_in = _nbytes(ins[0])
    b_out = _nbytes(outs[0]) if outs else 0.0
    f = (G - 1) / G if G > 1 else 0.0
    if "all_gather" in op:
        total = b_out if b_out else b_in * G
        ingress, egress = total * f, b_in * min(2, G - 1)
    elif "reduce_scatter" in op:
        ingress = egress = b_in * f
    elif "all_reduce" in op:
        ingress = egress = 2 * b_in * f
    elif "all_to_all" in op:
        ingress = egress = b_in * f
    else:
        return None
    return {
        "ingress": ingress,
        "egress": egress,
        "group": G,
        "axis": axis,
        "links": links,
        "topology": str(topo),
        "dirs": dirs,
        "connections": links * dirs,
        "bytes_in": b_in,
        "assumed": assumed,
    }


# ---------------------------------------------------------------- report
def _ms(ns: float) -> float:
    return ns / 1e6


def build(
    calls: list[dict],
    mesh_shape: tuple,
    grid: tuple,
    arch: str = "blackhole",
    layer_weights: dict | None = None,
    context_calls: list[dict] | None = None,
    context_label: str = "",
    axis_links: dict | None = None,
) -> dict:
    """Group the calls and compute every column (see the module docstring). Returns a JSON-able dict."""
    peaks = ARCH.get(arch, ARCH["blackhole"])
    cores = grid[0] * grid[1]
    w = layer_weights or {}

    def weight(c):
        return w.get(c.get("layer"), 1)

    groups = defaultdict(list)
    for c in calls:
        cc = ccl_cost(c, mesh_shape, axis_links) if family(c["op"]) == "ccl" else None
        geo = f" [axis {cc['axis']}, {cc['links']}L, {cc['topology']}]" if cc else ""
        groups[(c["op"], signature(c) + geo)].append(c)

    def stats(rows):
        t = [_ms(max(c["ns_dev"].values())) for c in rows]
        return t

    ctx = defaultdict(list)
    for c in context_calls or []:
        cc = ccl_cost(c, mesh_shape, axis_links) if family(c["op"]) == "ccl" else None
        geo = f" [axis {cc['axis']}, {cc['links']}L, {cc['topology']}]" if cc else ""
        ctx[(c["op"], signature(c) + geo)].append(c)

    out_rows = []
    grand = sum(t * weight(c) for rows in groups.values() for c, t in zip(rows, stats(rows)))
    for (op, sig), rows in groups.items():
        t = stats(rows)
        wts = [weight(c) for c in rows]
        total = sum(ti * wi for ti, wi in zip(t, wts))
        per_layer = Counter(c.get("layer") for c in rows)
        row = {
            "op": op,
            "family": family(op),
            "shape": sig,
            "calls_profiled": len(rows),
            "calls": int(sum(wts)),
            "calls_per_layer": statistics.median(per_layer.values()),
            "layers": sorted(l for l in per_layer if l is not None),
            "median_ms": statistics.median(t),
            "max_ms": max(t),
            "total_ms": total,
            "share": total / grand if grand else 0.0,
        }
        cost = compute_cost(rows[0])
        if cost:
            peak = cores * peaks["flop_per_core_cycle"] * peaks["clock_hz"] / PHASES.get(cost["fidelity"], 2)
            fl = sum(compute_cost(c)["flops"] * wi for c, wi in zip(rows, wts))
            by = sum(compute_cost(c)["bytes"] * wi for c, wi in zip(rows, wts))
            row.update(
                math_util=fl / (total / 1e3) / peak if total else 0.0,
                dram_util=by / (total / 1e3) / (peaks["dram_gb_s"] * 1e9) if total else 0.0,
                fidelity=cost["fidelity"],
                tflops=fl / (total / 1e3) / 1e12 if total else 0.0,
                assumed=cost["assumed"],
            )
        cc = ccl_cost(rows[0], mesh_shape, axis_links)
        if cc:
            ing = sum(ccl_cost(c, mesh_shape, axis_links)["ingress"] * wi for c, wi in zip(rows, wts))
            eg = sum(ccl_cost(c, mesh_shape, axis_links)["egress"] * wi for c, wi in zip(rows, wts))
            sec = total / 1e3
            row.update(
                ingress_gb_s=ing / sec / 1e9 if sec else 0.0,
                egress_gb_s=eg / sec / 1e9 if sec else 0.0,
                per_link_gb_s=max(ing, eg) / sec / 1e9 / cc["connections"] if sec else 0.0,
                ccl={k: cc[k] for k in ("group", "axis", "links", "topology", "dirs", "connections")},
                bytes_per_call=cc["bytes_in"],
                ingress_per_call=cc["ingress"],
                assumed=cc["assumed"],
            )
        # per-chip spread: each chip's summed time for this group
        chips = defaultdict(float)
        for c, wi in zip(rows, wts):
            for chip, ns in c["ns_dev"].items():
                chips[int(chip)] += _ms(ns) * wi
        if chips:
            med = statistics.median(chips.values())
            worst = max(chips, key=chips.get)
            row["spread"] = {
                "median_chip_ms": med,
                "max_chip_ms": chips[worst],
                "max_chip": worst,
                "imbalance": chips[worst] / med - 1 if med else 0.0,
            }
        if context_calls is not None:
            other = ctx.get((op, sig))
            if other:
                tm = statistics.median(stats(other))
                row["context"] = {
                    "label": context_label,
                    "median_ms": tm,
                    "ratio": row["median_ms"] / tm if tm else float("inf"),
                }
            else:
                row["context"] = {"label": context_label, "median_ms": None, "ratio": None}
        out_rows.append(row)
    out_rows.sort(key=lambda r: -r["total_ms"])
    fam = defaultdict(float)
    for r in out_rows:
        fam[r["family"]] += r["total_ms"]
    return {
        "mesh": list(mesh_shape),
        "grid": list(grid),
        "arch": arch,
        "peaks": {
            "tflops_lofi": cores * peaks["flop_per_core_cycle"] * peaks["clock_hz"] / 1e12,
            "dram_gb_s": peaks["dram_gb_s"],
        },
        "layer_weights": {str(k): v for k, v in w.items()},
        "axis_links": {str(k): v for k, v in (axis_links or {}).items()},
        "total_ms": grand,
        "family_ms": dict(fam),
        "rows": out_rows,
    }


# ---------------------------------------------------------------- rendering
def _f(x, fmt="{:.3f}", none="-"):
    return none if x is None else fmt.format(x)


def render(rep: dict, title: str = "", top: int | None = None) -> str:
    rows = rep["rows"][:top] if top else rep["rows"]
    L = []
    L.append(f"# {title}" if title else "# Per-op report")
    w = rep.get("layer_weights") or {}
    scope = (
        f"profiled layers {', '.join(sorted(w, key=int))} weighted {dict(sorted(w.items(), key=lambda kv: int(kv[0])))} "
        "(totals and calls are the full model's)"
        if w
        else "all profiled layers, unweighted"
    )
    L.append(
        f"mesh {rep['mesh']}, {rep['grid'][0]}x{rep['grid'][1]} cores, {rep['arch']}: peak {rep['peaks']['tflops_lofi']:.0f} "
        f"TFLOP/s LoFi (/2 HiFi2, /4 HiFi4), DRAM {rep['peaks']['dram_gb_s']:.0f} GB/s per chip; {scope}"
    )
    fam = ", ".join(
        f"{k} {v:.1f} ms ({v / rep['total_ms'] * 100:.0f}%)"
        for k, v in sorted(rep["family_ms"].items(), key=lambda kv: -kv[1])
    )
    L.append(f"device total {rep['total_ms']:.1f} ms: {fam}")
    L.append("")
    L.append("## Main table (sorted by total; a call's time = its slowest chip)")
    hdr = (
        f"{'#':>3} {'op':34} {'calls':>6} {'/layer':>6} {'med ms':>8} {'max ms':>8} {'total ms':>9} {'%':>5} "
        f"{'math%':>6} {'dram%':>6} {'GB/s':>7} {'GB/s/lk':>8}  shapes"
    )
    L.append(hdr)
    L.append("-" * len(hdr))
    for i, r in enumerate(rows, 1):
        math_u = r.get("math_util")
        dram_u = r.get("dram_util")
        bw = max(r.get("ingress_gb_s", 0.0), r.get("egress_gb_s", 0.0)) if "ingress_gb_s" in r else None
        L.append(
            f"{i:>3} {r['op'][:34]:34} {r['calls']:>6} {r['calls_per_layer']:>6g} {r['median_ms']:>8.3f} "
            f"{r['max_ms']:>8.3f} {r['total_ms']:>9.2f} {r['share'] * 100:>5.1f} "
            f"{_f(math_u and math_u * 100, '{:.1f}'):>6} {_f(dram_u and dram_u * 100, '{:.1f}'):>6} "
            f"{_f(bw, '{:.1f}'):>7} {_f(r.get('per_link_gb_s'), '{:.1f}'):>8}  {r['shape']}"
        )
    L.append("")
    L.append("## Per-chip spread (groups >= 0.5% of the total; median chip vs slowest chip)")
    L.append(f"{'#':>3} {'op':34} {'median chip':>11} {'slowest':>9} {'chip':>4} {'imbal':>6}")
    for i, r in enumerate(rows, 1):
        s = r.get("spread")
        if s and r["share"] >= 0.005:
            L.append(
                f"{i:>3} {r['op'][:34]:34} {s['median_chip_ms']:>11.2f} {s['max_chip_ms']:>9.2f} {s['max_chip']:>4} "
                f"{s['imbalance'] * 100:>5.1f}%"
            )
    if any("context" in r for r in rows):
        lbl = next((r["context"]["label"] for r in rows if "context" in r), "")
        L.append("")
        L.append(f"## Context scaling (median ms per call here vs at {lbl}; ratio > 1: grows with context)")
        L.append(f"{'#':>3} {'op':34} {'here':>8} {lbl[:10]:>10} {'ratio':>6}")
        for i, r in enumerate(rows, 1):
            c = r.get("context")
            if c and r["share"] >= 0.005:
                L.append(
                    f"{i:>3} {r['op'][:34]:34} {r['median_ms']:>8.3f} {_f(c['median_ms']):>10} "
                    f"{_f(c['ratio'], '{:.2f}'):>6}"
                )
    ccl = [(i, r) for i, r in enumerate(rows, 1) if "ccl" in r]
    if ccl:
        L.append("")
        L.append("## CCL details (per call; ingress = bytes the worst chip receives; connections = links x directions)")
        L.append(
            f"{'#':>3} {'op':34} {'MB in':>7} {'MB ingress':>10} {'group':>5} {'axis':>4} {'links':>5} {'topology':>9} "
            f"{'conn':>4} {'GB/s':>7} {'GB/s/lk':>8}"
        )
        for i, r in ccl:
            c = r["ccl"]
            L.append(
                f"{i:>3} {r['op'][:34]:34} {r['bytes_per_call'] / 1e6:>7.2f} {r['ingress_per_call'] / 1e6:>10.2f} "
                f"{c['group']:>5} {str(c['axis']):>4} {c['links']:>5} {c['topology'][:9]:>9} {c['connections']:>4} "
                f"{r['ingress_gb_s']:>7.1f} {r['per_link_gb_s']:>8.1f}"
            )
    notes = sorted({a for r in rows for a in r.get("assumed", [])})
    L.append("")
    L.append("## Notes")
    L.append(
        "- math% = FLOPs / time / peak at the call's fidelity; dram% = (inputs + weights + outputs, or the gathered "
        "rows for sparse attention) / time / DRAM peak; both are per chip."
    )
    L.append(
        "- CCL GB/s = max(ingress, egress) of the worst chip / time; GB/s/lk divides by links x receive directions "
        "(Ring 2, Linear 1). All-gather ingress = (G-1)/G of the output; reduce-scatter / all-to-all = (G-1)/G of the "
        "input; all-reduce = 2x."
    )
    for a in notes:
        L.append(f"- assumed (some rows): {a}")
    return "\n".join(L)


def save(rep: dict, path) -> None:
    with open(path, "w") as f:
        json.dump(rep, f, indent=1, default=str)


def save_calls(path, **payload) -> None:
    """The raw per-call records and build() arguments, so the report can be re-rendered without the device."""
    with open(path, "w") as f:
        json.dump(payload, f, default=str)


def from_calls_file(path) -> tuple[dict, str]:
    with open(path) as f:
        d = json.load(f)
    weights = {int(k): v for k, v in (d.get("layer_weights") or {}).items()}
    links = {int(k): v for k, v in (d.get("axis_links") or {}).items()}
    for c in d["calls"] + (d.get("context_calls") or []):
        c["ns_dev"] = {int(k): v for k, v in c["ns_dev"].items()}
    rep = build(
        d["calls"],
        tuple(d["mesh"]),
        tuple(d["grid"]),
        arch=d.get("arch", "blackhole"),
        layer_weights=weights,
        context_calls=d.get("context_calls"),
        context_label=d.get("context_label", ""),
        axis_links=links,
    )
    return rep, d.get("title", "")


if __name__ == "__main__":  # python -m models.demos.common.bringup.testing.op_report <op_calls.json> [top]
    import sys

    rep, title = from_calls_file(sys.argv[1])
    print(render(rep, title=title, top=int(sys.argv[2]) if len(sys.argv) > 2 else None))
