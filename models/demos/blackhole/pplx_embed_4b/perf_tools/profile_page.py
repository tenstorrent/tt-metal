# Data for the per-op device-profile page (claude.ai artifact EyeLiogdyYu3soMry6akYn): reads the signposted tracy
# reports of the Optimized level, keeps the page's Baseline level, adds per-signature roofline bounds and splices the
# JSON into the page's <script id="data" type="application/json"> block.
# Usage: profile_page.py <page.html> <runs.json> <out.html>
#   runs.json: {"commit": "...", "date": "YYYY-MM-DD", "batches": [{"bs": 8, "report": "<ts>", "report_dir": null,
#               "e2e": {"cold": ms, "sus": ms, "aiclk": "1062 (1050-1162)", "power": W}}, ...]}
#   report_dir is relative to the checkout (default generated/profiler/reports).
#
# Roofline per call = max(FPU time, DRAM time), or a measured floor where MEASURED_FLOORS has one:
#   FPU:  matmul 2*M*K*N, SDPA 4*B*Hq*Sq*Sk*D (half if causal) at 4096 FLOP/cycle/core * 120 cores * 1.35 GHz
#         (663.6 TFLOP/s LoFi; tech_reports/GEMM_FLOPS measures up to 96% of it), times the math-fidelity phases
#   DRAM: bytes of the op's DRAM tensors (L1 tensors free) at the attainable 450 GB/s (the best stock streaming op on a
#         P150, bf16 add; clone 390-420; datasheet 512)
# Profiles must run at 1.35 GHz (one cool-chip replay does; check tt-smi): kernel durations are cycles at 1.35 GHz,
# DRAM time is wall clock.
import csv
import json
import re
import sys
from collections import defaultdict

REPO = __file__.rsplit("/models/", 1)[0]
ISL, CLK, CORES, FLOP_CYC = 512, 1.35e9, 120, 4096
PEAK = CLK * CORES * FLOP_CYC
BW, BW_DATASHEET = 450e9, 512e9
FID = {"LoFi": 1, "HiFi2": 2, "HiFi3": 3, "HiFi4": 4}
BYTES = {"BFLOAT16": 2, "BFLOAT8_B": 1088 / 1024, "BFLOAT4_B": 576 / 1024, "FLOAT32": 4, "UINT32": 4, "INT32": 4}

# Measured per-call floors, device kernel µs at 1.35 GHz (compute only, data movement only, the full op standalone),
# Optimized level: each op's kernels at the in-model placement with the reads / writes removed, or the compute
# replaced by tile copies that keep every CB handshake (so the DM floor is an upper bound). A variant that is not
# faster than the full op standalone is not a floor (the copy compute paced it) and is dropped.
# key: (bs, label, first input's memory) -> floors.
#   SDPA: sdpa_kernel_variants.py + bench_sdpa_floors.py --parse (NEGATIVE_RESULTS §65, e7f541d kernels)
#   heads op: bench_heads_placement_ablate.py 8 8 / 16 16 / 8 32 (bs32: the quarter-batch chunk), add+RMSNorm:
#   bench_add_norm_ablate.py (a in DRAM; AN_A_L1=1: in L1), both under the device profiler, device_kernel_us.py,
#   2026-10-01. bs16's a-in-L1 add+RMSNorm does not fit standalone and takes the a-in-DRAM floors (at bs8 the two
#   placements' floors differ by 0-3 µs)
MEASURED_FLOORS = {
    (8, "", "DRAM I"): (152.8, 102.0, None),
    (16, "", "DRAM I"): (248.5, 183.5, None),
    (32, "", "DRAM I"): (480.9, 355.7, None),
    (8, "heads + QK-norm + RoPE", "L1 I"): (108.2, 106.3, 114.1),
    (16, "heads + QK-norm + RoPE", "L1 I"): (198.9, 228.4, 205.1),
    (32, "heads + QK-norm + RoPE", "L1 I"): (108.2, 105.4, 114.1),
    (8, "add + RMSNorm", "DRAM I"): (53.4, 72.1, 80.8),
    (8, "add + RMSNorm", "L1 I"): (53.4, 75.5, 82.7),
    (16, "add + RMSNorm", "DRAM I"): (82.3, 113.3, 121.0),
    (16, "add + RMSNorm", "L1 I"): (82.3, 113.3, 121.0),
    (32, "add + RMSNorm", "DRAM I"): (141.1, 241.4, 267.7),
}

DT = {"BFLOAT16": "bf16", "BFLOAT8_B": "bfp8", "BFLOAT4_B": "bfp4", "FLOAT32": "fp32", "UINT32": "u32", "INT32": "i32"}
MEM = {"INTERLEAVED": "I", "WIDTH_SHARDED": "WS", "HEIGHT_SHARDED": "HS", "BLOCK_SHARDED": "BS"}
NAME = {
    "MatmulDeviceOperation": "Matmul",
    "MinimalMatmulDeviceOperation": "MinimalMatmul",
    "BinaryNgDeviceOperation": "BinaryNg",
    "SDPAOperation": "SDPA",
    "GenericOpDeviceOperation": "GenericOp",
    "LayerNormDeviceOperation": "LayerNorm",
    "ShardedToInterleavedDeviceOperation": "ShardedToInterleaved",
    "InterleavedToShardedDeviceOperation": "InterleavedToSharded",
    "EmbeddingsDeviceOperation": "Embeddings",
    "UntilizeCodegenDeviceOperation": "UntilizeCodegen",
    "SliceDeviceOperation": "Slice",
    "ReshardDeviceOperation": "Reshard",
}
v = lambda s: re.sub(r"\[.*", "", s)


def tensor(r, i):
    dims = [v(r.get(f"INPUT_{i}_{a}_PAD[LOGICAL]", "")) for a in "WZYX"]
    if not dims[-1]:
        return None
    while len(dims) > 2 and dims[0] == "1":
        dims.pop(0)
    m = re.match(r"DEV_\d+_(L1|DRAM)_(\w+)", r[f"INPUT_{i}_MEMORY"])
    return {
        "shape": "×".join(dims),
        "dtype": DT.get(r[f"INPUT_{i}_DATATYPE"], r[f"INPUT_{i}_DATATYPE"]),
        "mem": f"{m.group(1)} {MEM.get(m.group(2), m.group(2))}",
    }


def kernels(r, col):
    raw = r.get(col, "").strip("[]")
    return [x.strip().strip("'").replace(REPO + "/", "") for x in raw.split(";") if x.strip().strip("'")]


def variant(r):
    """Implementation of this row: program config, op source dir, compute + dataflow kernel paths."""
    comp, dm = kernels(r, "COMPUTE KERNEL SOURCE"), kernels(r, "DATA MOVEMENT KERNEL SOURCE")
    pc = re.search(r"'program_config': '([A-Za-z0-9_]+)", r["ATTRIBUTES"])
    first = (comp or dm or [""])[0]
    m = re.match(r"(.*?)/(device/)?kernels", first)
    return {
        "program_config": pc.group(1) if pc else None,
        "src": m.group(1) if m else None,
        "compute": comp,
        "dataflow": dm,
    }


def classify(op, r):
    """(label, model-level group) for a row."""
    if "Matmul" in op:
        k, n = v(r["INPUT_1_Y_PAD[LOGICAL]"]), v(r["INPUT_1_X_PAD[LOGICAL]"])
        lab = {
            ("2560", "6144"): "QKV",
            ("4096", "2560"): "WO",
            ("2560", "9728"): "FF1/FF3",
            ("2560", "19456"): "FF1+FF3 (fused SwiGLU)",
            ("9728", "2560"): "FF2",
        }.get((k, n), f"K{k} N{n}")
        return lab, ("attn_mm" if lab in ("QKV", "WO") else "mlp_mm")
    if op == "GenericOpDeviceOperation":
        x0 = v(r["INPUT_0_X_PAD[LOGICAL]"])
        lab = {"6144": "heads + QK-norm + RoPE", "2560": "add + RMSNorm", "9728": "silu·mul"}.get(x0, "generic")
        return lab, {"6144": "heads", "2560": "norm", "9728": "swiglu"}.get(x0, "other")
    if op == "BinaryNgDeviceOperation":
        m = re.search(r"'binary_op_type': 'BinaryOpType::(\w+)'", r["ATTRIBUTES"])
        lab = m.group(1).lower() if m else "binary"
        return lab, ("swiglu" if lab == "mul" else "norm")
    if op == "SDPAOperation":
        return "", "sdpa"
    if op in ("LayerNormDeviceOperation", "ShardedToInterleavedDeviceOperation", "InterleavedToShardedDeviceOperation"):
        return "", "norm"
    if op == "ReshardDeviceOperation":  # bs1: the MLP norm output onto the 1D FF1 / FF3 matmul's in0 layout
        return "", "mlp_mm"
    return "", "other"


def report_rows(ts, report_dir=None):
    """Device-timed rows between the start / stop signposts of a tracy report, and the duration column."""
    path = f"{REPO}/{report_dir or 'generated/profiler/reports'}/{ts}/ops_perf_results_{ts}.csv"
    rows = list(csv.DictReader(open(path)))
    dcol = next(c for c in rows[0] if c.startswith("DEVICE KERNEL DURATION"))
    idx = {r["OP CODE"]: i for i, r in enumerate(rows) if r["OP TYPE"] == "signpost"}
    return [r for r in rows[idx["start"] + 1 : idx["stop"]] if r.get(dcol)], dcol


def sig_key(op, r):
    lab, _ = classify(op, r)
    ins = [t for t in (tensor(r, i) for i in range(3)) if t]
    return (NAME.get(op, op), lab, json.dumps(ins), int(r["CORE COUNT"]), json.dumps(variant(r)))


def build(bs, ts, e2e, base_batch, report_dir=None):
    rows, dcol = report_rows(ts, report_dir)
    sigs = {}
    for r in rows:
        op = r["OP CODE"]
        lab, grp = classify(op, r)
        ins = [t for t in (tensor(r, i) for i in range(3)) if t]
        # BinaryNg ops whose recorded rows are fewer than the batch's tokens run on the full activation
        stale = op == "BinaryNgDeviceOperation" and int(v(r["INPUT_0_Y_PAD[LOGICAL]"]) or 0) < bs * ISL
        s = sigs.setdefault(
            sig_key(op, r),
            {
                "op": NAME.get(op, op),
                "label": lab,
                "group": grp,
                "inputs": ins,
                "cores": int(r["CORE COUNT"]),
                "calls": 0,
                "total_us": 0.0,
                "max_us": 0.0,
                "min_us": 1e12,
                "stale_shape": stale,
                "impl": variant(r),
            },
        )
        us = float(r[dcol]) / 1e3
        s["calls"] += 1
        s["total_us"] += us
        s["max_us"] = max(s["max_us"], us)
        s["min_us"] = min(s["min_us"], us)
    ops = defaultdict(lambda: {"calls": 0, "total_us": 0.0, "sigs": []})
    for s in sigs.values():
        o = ops[s["op"]]
        o["calls"] += s["calls"]
        o["total_us"] += s["total_us"]
        for k in ("total_us", "max_us", "min_us"):
            s[k] = round(s[k], 2)
        o["sigs"].append(s)
    ref_ops = {o["op"]: round(o["total_us"] / 1e3, 3) for o in base_batch["ops"]}
    op_list = []
    for op, o in sorted(ops.items(), key=lambda kv: -kv[1]["total_us"]):
        o["sigs"].sort(key=lambda s: -s["total_us"])
        op_list.append(
            {
                "op": op,
                "calls": o["calls"],
                "total_us": round(o["total_us"], 2),
                "ref_ms": ref_ops.get(op),
                "sigs": o["sigs"],
            }
        )
    out = {
        "bs": bs,
        "report": ts,
        "device_ms": round(sum(o["total_us"] for o in op_list) / 1e3, 3),
        "ref_device_ms": base_batch["device_ms"],
        "e2e": e2e,
        "ref_e2e": {"cold": base_batch["e2e"]["cold"], "sus": base_batch["e2e"]["sus"]},
        "ops": op_list,
    }
    if report_dir:
        out["report_dir"] = report_dir
    return out


def dims(r, pre, i):
    d = [v(r.get(f"{pre}_{i}_{a}_PAD[LOGICAL]", "")) for a in "WZYX"]
    return [int(x) for x in d] if d[-1] else None


def prod(d):
    p = 1
    for x in d:
        p *= x
    return p


def tensors(r, generic):
    """(dims, dtype, in DRAM) of the op's tensors. A generic op lists all of its IO tensors as inputs, and its output
    again as the last one (dropped here); other ops list outputs separately."""
    out = []
    for i in range(12):
        d = dims(r, "INPUT", i)
        if d:
            out.append((d, r[f"INPUT_{i}_DATATYPE"], "DRAM" in r[f"INPUT_{i}_MEMORY"]))
    if generic:
        return out[:-1]
    for i in range(3):
        d = dims(r, "OUTPUT", i)
        if d:
            out.append((d, r[f"OUTPUT_{i}_DATATYPE"], "DRAM" in r[f"OUTPUT_{i}_MEMORY"]))
    return out


def row_roof(r, op, lab, qkv_rows, act_rows):
    """(FLOPs, DRAM bytes, FPU µs, DRAM µs) of one call"""
    generic = op == "GenericOpDeviceOperation"
    ts = tensors(r, generic)
    if generic and lab.startswith("heads") and qkv_rows:
        # size the heads op from the QKV matmul call that feeds it: the profiler records half a call's sequences for
        # its input, and in bs32's chunked calls the full-batch Q / K / V that each chunk writes part of
        sc_in = qkv_rows / prod(ts[0][0][:-1])
        new = [([prod(ts[0][0]) * sc_in], ts[0][1], ts[0][2])]
        for d, dt, dr in ts[1:]:
            if prod(d) < 1 << 20:  # cos / sin / transformation matrix
                new.append((d, dt, dr))
            else:  # head-major [B, H, S, D]: tokens = B * S
                new.append(([prod(d) * qkv_rows / (d[0] * d[2])], dt, dr))
        ts = new
    if op == "BinaryNgDeviceOperation" and act_rows:  # stale recorded shape: the op runs on the full activation
        ts = [([act_rows, d[-1]], dt, dr) if prod(d[:-1]) < act_rows else (d, dt, dr) for d, dt, dr in ts]
    if op == "EmbeddingsDeviceOperation":  # gathered rows only: read + write the output once each, plus the ids
        ts = [ts[0], ts[-1], ts[-1]]
    if op == "SliceDeviceOperation":  # only the kept rows are read, then written
        ts = [ts[-1], ts[-1]]
    dram = sum(prod(d) * BYTES.get(dt, 2) for d, dt, dr in ts if dr)
    flops = 0.0
    if "Matmul" in op:
        a, b = dims(r, "INPUT", 0), dims(r, "INPUT", 1)
        flops = 2 * prod(a[:-1]) * a[-1] * b[-1]
    elif op == "SDPAOperation":
        q, k = dims(r, "INPUT", 0), dims(r, "INPUT", 1)
        flops = 4 * prod(q[:-2]) * q[-2] * k[-2] * q[-1]
        if re.search(r"is_causal'?: '?true", r["ATTRIBUTES"]):
            flops /= 2
    t_fpu = flops * FID.get(r["MATH FIDELITY"], 1) / PEAK if flops else 0.0
    return flops, dram, t_fpu * 1e6, dram / BW * 1e6


def add_roofs(b, measured):
    """Per-signature roofline fields on batch b (measured floors where `measured` and MEASURED_FLOORS have them)."""
    rows, _ = report_rows(b["report"], b.get("report_dir"))
    act_rows = b["bs"] * ISL
    R, qkv_rows = {}, None
    for r in rows:
        op = r["OP CODE"]
        lab, _ = classify(op, r)
        if "Matmul" in op and lab == "QKV":
            qkv_rows = prod(dims(r, "INPUT", 0)[:-1])
        flops, dram, t_fpu, t_dram = row_roof(r, op, lab, qkv_rows, act_rows)
        x = R.setdefault(sig_key(op, r), defaultdict(float))
        x["calls"] += 1
        x["flops"] += flops
        x["dram_bytes"] += dram
        x["fpu_us"] += t_fpu
        x["dram_us"] += t_dram
        x["roof_us"] += max(t_fpu, t_dram)
        x["fpu_calls"] += t_fpu >= t_dram and flops > 0
        if qkv_rows and qkv_rows < act_rows and (lab == "QKV" or lab.startswith("heads")):
            x["work_frac"] = round(qkv_rows / act_rows, 3)  # bs32's chunked QKV + heads calls
    miss = 0
    for o in b["ops"]:
        o["roof_us"] = 0.0
        for s in o["sigs"]:
            for k in ("roof_us", "flops", "dram_bytes", "bound", "work_frac", "floor_compute_us", "floor_dm_us",
                      "model_fpu_us", "model_dram_us"):  # fmt: skip
                s.pop(k, None)
            x = R.get((s["op"], s["label"], json.dumps(s["inputs"]), s["cores"], json.dumps(s["impl"])))
            if x is None or x["calls"] != s["calls"]:
                miss += 1
                continue
            s["roof_us"] = round(x["roof_us"], 2)
            s["flops"] = x["flops"]
            s["dram_bytes"] = round(x["dram_bytes"])
            s["bound"] = "FPU" if x["fpu_calls"] * 2 >= x["calls"] else "DRAM"
            if "work_frac" in x:
                s["work_frac"] = x["work_frac"]
            fl = MEASURED_FLOORS.get((b["bs"], s["label"], s["inputs"][0]["mem"])) if measured else None
            if fl and s["op"] in ("SDPA", "GenericOp") and s["inputs"][0]["dtype"] == "bfp8":
                n, (c, dm, full) = s["calls"], fl
                ok = lambda f: full is None or f < full
                s["floor_compute_us"] = round(c * n, 2) if ok(c) else None
                s["floor_dm_us"] = round(dm * n, 2) if ok(dm) else None
                s["model_fpu_us"], s["model_dram_us"] = round(x["fpu_us"], 2), round(x["dram_us"], 2)
                # never below the analytic bound
                s["roof_us"] = max(s["floor_compute_us"] or 0, s["floor_dm_us"] or 0, s["roof_us"])
                s["bound"] = "compute" if (s["floor_compute_us"] or 0) >= (s["floor_dm_us"] or 0) else "DM"
            o["roof_us"] += s["roof_us"]
        o["roof_us"] = round(o["roof_us"], 2)
    b["roof_ms"] = round(sum(o["roof_us"] for o in b["ops"]) / 1e3, 3)
    return miss


def main():
    page, runs, out = sys.argv[1:4]
    t = open(page).read()
    i = t.index('<script id="data"')
    j, k = t.index(">", i) + 1, t.index("</script>", i)
    old = json.loads(t[j:k])
    base = next(L for L in old["levels"] if L["key"] == "baseline")
    cfg = json.load(open(runs))
    opt = {
        "key": "optimized",
        "name": "Optimized",
        "commit": cfg["commit"],
        "date": cfg["date"],
        "ref_name": "Baseline",
        "batches": [
            build(
                x["bs"],
                x["report"],
                x["e2e"],
                next(b for b in base["batches"] if b["bs"] == x["bs"]),
                x.get("report_dir"),
            )
            for x in cfg["batches"]
        ],
    }
    data = {k: x for k, x in old.items() if k not in ("levels", "roofline")}
    data["levels"] = [base, opt]
    data["roofline"] = {"clock_ghz": CLK / 1e9, "tflops": round(PEAK / 1e12, 1), "dram_gbs": BW / 1e9,
                        "dram_datasheet_gbs": BW_DATASHEET / 1e9}  # fmt: skip
    for L in data["levels"]:
        for b in L["batches"]:
            miss = add_roofs(b, L["key"] == "optimized")
            print(f"{L['key']} bs{b['bs']}: roofline {b['roof_ms']} ms vs device {b['device_ms']} ms, unmatched {miss}",
                  file=sys.stderr)  # fmt: skip
    open(out, "w").write(t[:j] + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + t[k:])


if __name__ == "__main__":
    main()
