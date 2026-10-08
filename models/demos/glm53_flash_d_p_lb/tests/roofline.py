# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Roofline view of a test_profile_ops.py profile (host only, no device).

    python models/demos/glm53_flash_d_p_lb/tests/roofline.py [generated/glm53_flash_d_p_lb/profile_ops.json]

Per chip, Blackhole p150b: a Tensix core does 4096 FLOP / cycle at LoFi (2 x 8x16x16, the matmul op's performance
model), divided by the math-fidelity phases (HiFi4 = 4, every matmul in this model); AICLK 1.35 GHz; the profiled
grid (11 x 10 = 110 cores): HiFi4 peak 152 TFLOP/s. DRAM 512 GB/s (8 GDDR6 channels). Per op row (one op, one shape,
one step, summed over layers):
  flop floor = FLOPs / HiFi4 peak     (matmul / linear from the shapes; sparse_sdpa; routed experts from routing)
  dram floor = bytes / DRAM bandwidth (inputs + output, all DRAM-interleaved in this model)
  roof = max of the two; eff = roof / measured device time; bound = which floor is higher.
CCLs (fabric bound) and ops without a model (KDA kernels, mHC, top-k, ...) get time only. Device time is the slowest
chip per call (the op profile's "ms").
"""

from __future__ import annotations

import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

CLK = 1.35e9
FLOP_PER_CYCLE_LOFI = 4096
HIFI4 = 4
DRAM_BW = 512e9
BYTES = {"bf16": 2, "fp32": 4, "bfp8": 1088 / 1024, "bfp4": 576 / 1024, "u32": 4, "i32": 4, "u16": 2, "u8": 1}
CCL = (
    "all_gather",
    "reduce_scatter",
    "all_reduce",
    "mesh_partition",
    "bringup.dispatch",
    "bringup.combine",
    "bringup.offset_cumsum",
    "point_to_point",
)
# GLM-5.3-Flash routed experts: H 4096, I 2048, top-8 of 288
H, I_MOE, TOPK = 4096, 2048, 8


def parse(shape: str):
    out = []
    for part in shape.split(" · "):
        m = re.match(r"([\dx]+) (\w+)", part.strip())
        if m:
            out.append(([int(x) for x in m.group(1).split("x")], m.group(2)))
    return out


def nbytes(dims, dt) -> float:
    return math.prod(dims) * BYTES.get(dt, 2)


def model_op(op: str, shape: str, chunk: int, chips: int):
    """(flops, bytes, kind) per call, per chip; kind in matmul / sdpa / experts / mem / ccl / none."""
    if any(op.startswith(c) or op.endswith(c) for c in CCL):
        return 0.0, 0.0, "ccl"
    ts = parse(shape)
    if "routed_expert" in op:
        pairs = chunk * TOPK / chips  # token-expert pairs per chip (uniform routing)
        flops = 2 * 3 * H * I_MOE * pairs
        w = 288 / chips * 3 * H * I_MOE * BYTES["bfp4"]
        return flops, w + 2 * pairs * H * 2, "experts"
    if "sparse_sdpa" in op and ts:
        d = ts[0][0]  # [1, heads, rows, 512]
        heads, rows, dk = d[-3], d[-2], d[-1]
        sel = 2048 + 3
        return 4 * heads * rows * sel * dk, nbytes(d, ts[0][1]) * 2 + heads * rows * sel * dk * 2 / heads, "sdpa"
    if op in ("linear", "matmul") and len(ts) >= 2:
        (a, da), (b, db) = ts[0], ts[1]
        k, n = a[-1], b[-1]
        m_rows = math.prod(a[:-1])
        batch_b = math.prod(b[:-2]) if len(b) > 2 else 1
        flops = 2 * m_rows * k * n
        out = m_rows * n * 2
        return flops, nbytes(a, da) + nbytes(b, db) + out, "matmul"
    if ts:  # elementwise / data movement: read the inputs, write one output the size of the first
        return 0.0, sum(nbytes(d, t) for d, t in ts) + nbytes(*ts[0]), "mem"
    return 0.0, 0.0, "none"


def main(path: str):
    d = json.loads(Path(path).read_text())
    chunk, chips = d["chunk"], d["chips"]
    cores = d["grid"][0] * d["grid"][1]
    peak = cores * FLOP_PER_CYCLE_LOFI / HIFI4 * CLK
    print(f"chunk {chunk} at {d['start']}, {chips} chips, {cores} cores/chip, experts {d['experts_dtype']}")
    print(
        f"per chip: HiFi4 peak {peak / 1e12:.0f} TFLOP/s, DRAM {DRAM_BW / 1e9:.0f} GB/s; ridge "
        f"{peak / DRAM_BW:.0f} FLOP/B"
    )
    tl = d.get("timeline") or {}
    if "device_timeline_ms" in tl:
        print(
            f"timeline: device {tl['device_timeline_ms']:.1f} ms = kernels {tl['kernel_ms']:.1f} + idle gaps "
            f"{tl['gap_ms']:.1f}; host dispatch {tl['host_dispatch_ms']:.1f} ms; host wall {tl['host_wall_ms']:.1f} ms"
        )
    print(f"warm chunk wall (synced once): {d['wall_ms']:.1f} ms\n")

    # aggregate rows over layers: key (step, op, shape)
    agg = defaultdict(lambda: {"ms": 0.0, "calls": 0, "gap": 0.0})
    step_ms = defaultdict(float)
    total = 0.0
    for sec, rows in d["ops"].items():
        step = sec.split(".", 1)[1] if sec.startswith("L") and "." in sec else sec
        layer = int(sec[1:].split(".")[0]) if sec.startswith("L") and sec[1:].split(".")[0].isdigit() else -1
        for r in rows:
            key = (step, r["op"], r["shape"])
            a = agg[key]
            a["ms"] += r["ms"]
            a["calls"] += r["calls"]
            a["gap"] += r.get("gap_ms", 0.0)
            a.setdefault("layers", set()).add(layer)
            step_ms[step] += r["ms"]
            total += r["ms"]
    print(f"device op time (sum over ops, slowest chip per call): {total:.1f} ms\n")

    print("per step (summed over layers):")
    for st, ms in sorted(step_ms.items(), key=lambda x: -x[1]):
        print(f"  {st:16s} {ms:8.1f} ms  {100 * ms / total:5.1f}%")

    rows = []
    for (step, op, shape), a in agg.items():
        if a["ms"] <= 0:
            continue
        f, b, kind = model_op(op, shape, chunk, chips)
        calls = a["calls"]
        tf, tb = f * calls / peak * 1e3, b * calls / DRAM_BW * 1e3
        roof = max(tf, tb)
        rows.append(
            {
                "step": step,
                "op": op,
                "shape": shape,
                "calls": calls,
                "ms": a["ms"],
                "kind": kind,
                "flop_ms": tf,
                "dram_ms": tb,
                "roof_ms": roof,
                "gap_ms": a["gap"],
                "eff": roof / a["ms"] if a["ms"] > 0 and roof > 0 else None,
                "bound": ("compute" if tf >= tb else "DRAM") if roof > 0 else kind,
            }
        )
    rows.sort(key=lambda r: -r["ms"])
    print(f"\ntop ops (summed over layers; roof = max(flop, dram) floor; eff = roof / measured):")
    print(
        f"  {'step':14s} {'op':38s} {'shape':44s} {'calls':>5s} {'ms':>8s} {'%':>5s} {'flop':>7s} {'dram':>7s} "
        f"{'eff':>5s} bound"
    )
    for r in rows[:40]:
        eff = ("> 100%" if r["eff"] > 1 else f"{100 * r['eff']:4.0f}%") if r["eff"] else "    -"
        print(
            f"  {r['step'][:14]:14s} {r['op'][:38]:38s} {r['shape'][:44]:44s} {r['calls']:5d} {r['ms']:8.2f} "
            f"{100 * r['ms'] / total:4.1f}% {r['flop_ms']:7.2f} {r['dram_ms']:7.2f} {eff} {r['bound']}"
        )

    by_kind = defaultdict(lambda: [0.0, 0.0])
    for r in rows:
        by_kind[r["kind"]][0] += r["ms"]
        by_kind[r["kind"]][1] += r["roof_ms"]
    print("\nby kind: measured ms / roofline ms (eff)")
    for k, (ms, roof) in sorted(by_kind.items(), key=lambda x: -x[1][0]):
        eff = f"{100 * roof / ms:4.0f}%" if roof and ms else "    -"
        print(f"  {k:8s} {ms:8.1f} ms  {100 * ms / total:5.1f}%   roof {roof:7.1f} ms  {eff}")

    # whole-model estimate: per-layer device time of each block type x its layer count
    counts = {"kda_dense": 3, "dsa_moe": 11, "kda_moe": 31}
    per_layer = defaultdict(lambda: defaultdict(float))
    for sec, rs in d["ops"].items():
        head = sec.split(".")[0]
        if head.startswith("L") and head[1:].isdigit():
            per_layer[int(head[1:])][sec.split(".", 1)[1]] += sum(r["ms"] for r in rs)
    kinds = {}
    for i, steps in per_layer.items():
        moe = "experts" in steps
        kda = "indexer" not in steps
        kinds.setdefault(("kda" if kda else "dsa") + ("_moe" if moe else "_dense"), []).append(sum(steps.values()))
    est = 0.0
    print("\nper-layer device time by block type (this profile) -> whole model (45 layers):")
    for k, n in counts.items():
        if k in kinds:
            v = sum(kinds[k]) / len(kinds[k])
            est += v * n
            print(f"  {k:10s} {v:7.2f} ms/layer x {n:2d} = {v * n:7.1f} ms   (profiled layers: {len(kinds[k])})")
    print(f"  estimated whole-model device op time per chunk: {est:.0f} ms")
    out = Path(path).with_name("roofline.json")
    out.write_text(json.dumps(rows, indent=1, default=list))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "generated/glm53_flash_d_p_lb/profile_ops.json")
