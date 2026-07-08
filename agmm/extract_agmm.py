#!/usr/bin/env python3
"""Extract per-instance AllGatherMinimalMatmulAsyncOp (AGMM) data from tt-metal
multi-device op-perf CSVs for offline analysis.

For each logical AGMM instance we align the same op across all devices (each
device has its own GLOBAL CALL COUNT; the i-th AGMM on every device is the same
logical op) and take the FASTEST device's kernel duration as the performance
number (the slowest device's time is inflated by collective/dispatch wait).

Tensor roles (from the op source):
  IN0 = activation (local shard, [M, K_local])
  IN1 = weight ([K_gathered, N], force_transpose stores it transposed)
  IN2 = bias (optional, [1, N])
  IN3 = persistent_output_buffer = gathered activation ([M, K_gathered])
  IN4 = fused_ternary_a  (addcmul fusion, optional)
  IN5 = fused_ternary_b  (addcmul fusion, optional)
  (IN.. = persistent_weight_buffer for FSDP, optional)
  OUT0 = gathered-activation intermediate
  (OUT1 = gathered-weight intermediate if FSDP)
  OUT[last chunks] = matmul output(s), N split into `chunks`
"""
import csv, re, sys, json, statistics
from collections import defaultdict

OP = "AllGatherMinimalMatmulAsyncOp"


def parse_dims(v):
    """'1216[1216]' -> (padded, logical); returns (None,None) if empty."""
    v = (v or "").strip()
    if not v:
        return (None, None)
    m = re.match(r"\s*(\d+)\s*\[\s*(-?\d+)\s*\]", v)
    if m:
        return int(m.group(1)), int(m.group(2))
    try:
        n = int(v)
        return n, n
    except ValueError:
        return (None, None)


def parse_attrs(s):
    """Parse the ATTRIBUTES dict-ish string into a flat dict (incl. config fields)."""
    d = {}
    for k, v in re.findall(r"'([^']+)':\s*'([^']*)'", s):
        d[k] = v
    # nested MinimalMatmulConfig(...)
    m = re.search(r"MinimalMatmulConfig\(([^)]*)\)", s)
    if m:
        for pair in m.group(1).split(";"):
            if "=" in pair:
                kk, vv = pair.split("=", 1)
                d["cfg_" + kk.strip()] = vv.strip()
    return d


def tensor_fields(row, idx, kind, i):
    p = f"{kind}_{i}_"
    if not (row.get(idx.get(p + "LAYOUT", ""), "") or "").strip():
        return None
    w = parse_dims(row[idx[p + "W_PAD[LOGICAL]"]])
    z = parse_dims(row[idx[p + "Z_PAD[LOGICAL]"]])
    y = parse_dims(row[idx[p + "Y_PAD[LOGICAL]"]])
    x = parse_dims(row[idx[p + "X_PAD[LOGICAL]"]])
    return {
        "pad": [w[0], z[0], y[0], x[0]],
        "logical": [w[1], z[1], y[1], x[1]],
        "dtype": row[idx[p + "DATATYPE"]].strip(),
        "layout": row[idx[p + "LAYOUT"]].strip(),
        "memory": row[idx[p + "MEMORY"]].strip(),
    }


def load(path):
    with open(path) as f:
        r = csv.DictReader(f)
        raw = r.fieldnames
        idx = {c.strip(): c for c in raw}
        recs = []
        for row in r:
            if row[raw[0]].strip() != OP:
                continue
            ins = [tensor_fields(row, idx, "INPUT", i) for i in range(6)]
            outs = [tensor_fields(row, idx, "OUTPUT", i) for i in range(4)]
            recs.append({
                "gc": int(row[idx["GLOBAL CALL COUNT"]]),
                "dev": int(row[idx["DEVICE ID"]]),
                "kd_ns": float(row[idx["DEVICE KERNEL DURATION [ns]"]]),
                "cores": int(float(row[idx["CORE COUNT"]])),
                "fidelity": row[idx["MATH FIDELITY"]].strip(),
                "attrs": parse_attrs(row[idx["ATTRIBUTES"]]),
                "ins": [t for t in ins if t],
                "outs": [t for t in outs if t],
            })
    return recs


def instance_from_group(group, stage, inst_idx):
    """group = list of the same logical op across devices. Take fastest device."""
    durs = sorted(g["kd_ns"] for g in group)
    fast = min(group, key=lambda g: g["kd_ns"])
    a = fast["attrs"]

    ins, outs = fast["ins"], fast["outs"]
    n_in, n_out = len(ins), len(outs)

    ring = int(a.get("ring_size", 0))
    fsdp = a.get("fsdp_cluster_axis", "std::nullopt") not in ("std::nullopt", "", None)

    # roles by position
    act = ins[0]
    weight = ins[1]
    bias = ins[2] if n_in >= 3 else None
    # gathered-activation persistent buffer is next; addcmul ternaries follow
    has_addcmul = n_in >= 6
    tern_a = ins[4] if has_addcmul else None
    tern_b = ins[5] if has_addcmul else None

    # outputs: slot0 gathered act, optional fsdp weight, then `chunks` matmul outs
    n_matmul_out = n_out - 1 - (1 if fsdp else 0)
    chunks = max(n_matmul_out, 1)
    matmul_outs = outs[n_out - n_matmul_out:] if n_matmul_out > 0 else outs[-1:]

    M = act["logical"][2]
    K_local = act["logical"][3]
    K_gathered = K_local * ring if (K_local and ring) else None
    N = sum(o["logical"][3] for o in matmul_outs)

    flops = 2 * M * K_gathered * N if (M and K_gathered and N) else None
    min_us = durs[0] / 1000.0
    tflops = (flops / (min_us * 1e-6) / 1e12) if flops else None

    def sh(t):
        return None if not t else t["logical"]

    return {
        "stage": stage,
        "instance": inst_idx,
        "fastest_device": fast["dev"],
        "global_call_count": fast["gc"],
        # ---- performance (fastest device) ----
        "min_time_us": round(min_us, 2),
        "max_time_us": round(durs[-1] / 1000.0, 2),
        "mean_time_us": round(statistics.mean(durs) / 1000.0, 2),
        "skew_ratio": round(durs[-1] / durs[0], 3) if durs[0] else None,
        "tflops_at_min": round(tflops, 2) if tflops else None,
        "flops": flops,
        # ---- problem shape ----
        "M": M,
        "K_local": K_local,
        "K_gathered": K_gathered,
        "N": N,
        "chunks": chunks,
        "N_per_chunk": (N // chunks) if N else None,
        "activation_shape": sh(act),
        "weight_shape": sh(weight),
        "gathered_activation_shape": sh(ins[3]) if n_in >= 4 else None,
        # ---- fusions / bias ----
        "has_bias": bias is not None,
        "bias_shape": sh(bias),
        "has_addcmul": has_addcmul,
        "ternary_a_shape": sh(tern_a),
        "ternary_b_shape": sh(tern_b),
        "fsdp_fused": fsdp,
        # ---- collective config ----
        "ring_size": ring,
        "num_links": int(a.get("num_links", 0)),
        "num_workers_per_link": int(a.get("num_workers_per_link", 0)),
        "num_buffers_per_channel": int(a.get("num_buffers_per_channel", 0)),
        "topology": a.get("topology", ""),
        "cluster_axis": a.get("cluster_axis", ""),
        "force_transpose": a.get("force_transpose", ""),
        "using_persistent_weight_buffer": a.get("using_persistent_weight_buffer", ""),
        "fsdp_cluster_axis": a.get("fsdp_cluster_axis", ""),
        "fsdp_ring_size": a.get("fsdp_ring_size", ""),
        # ---- matmul kernel config ----
        "M_block_size": a.get("cfg_M_block_size", ""),
        "K_block_size": a.get("cfg_K_block_size", ""),
        "N_block_size": a.get("cfg_N_block_size", ""),
        "subblock_h": a.get("cfg_subblock_h", ""),
        "subblock_w": a.get("cfg_subblock_w", ""),
        "grid": a.get("cfg_compute_with_storage_grid_size", ""),
        # ---- dtypes / misc ----
        "act_dtype": act["dtype"],
        "weight_dtype": weight["dtype"],
        "out_dtype": matmul_outs[0]["dtype"] if matmul_outs else "",
        "act_memory": act["memory"],
        "math_fidelity": fast["fidelity"],
        "core_count": fast["cores"],
        "num_inputs": n_in,
        "num_outputs": n_out,
    }


def extract(path, stage):
    recs = load(path)
    perdev = defaultdict(list)
    for x in recs:
        perdev[x["dev"]].append(x)
    for d in perdev:
        perdev[d].sort(key=lambda z: z["gc"])
    ninst = len(next(iter(perdev.values())))
    out = []
    for i in range(ninst):
        group = [perdev[d][i] for d in perdev]
        out.append(instance_from_group(group, stage, i))
    return out


if __name__ == "__main__":
    files = sys.argv[1:] or [
        "agmm/transformer_stage1_ops_perf_results.csv",
        "agmm/transformer_stage2_ops_perf_results.csv",
    ]
    stagemap = {}
    all_rows = []
    for p in files:
        stage = "stage1" if "stage1" in p else ("stage2" if "stage2" in p else p.split("/")[-1])
        all_rows.extend(extract(p, stage))

    out_csv = "agmm/agmm_instances.csv"
    cols = list(all_rows[0].keys())
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in all_rows:
            rr = {k: (json.dumps(v) if isinstance(v, list) else v) for k, v in r.items()}
            w.writerow(rr)
    with open("agmm/agmm_instances.json", "w") as f:
        json.dump(all_rows, f, indent=2)

    print(f"Wrote {len(all_rows)} instances -> {out_csv} and agmm/agmm_instances.json\n")
    # compact preview
    hdr = ("stg", "id", "M", "Kloc", "Kgat", "N", "chnk", "bias", "addc", "ring", "lnk",
           "min_us", "TFLOPs", "skew", "fid")
    print("{:>4} {:>3} {:>5} {:>5} {:>5} {:>5} {:>4} {:>4} {:>4} {:>4} {:>3} {:>8} {:>7} {:>5} {:>6}".format(*hdr))
    for r in all_rows:
        print("{:>4} {:>3} {:>5} {:>5} {:>5} {:>5} {:>4} {:>4} {:>4} {:>4} {:>3} {:>8.1f} {:>7} {:>5} {:>6}".format(
            r["stage"], r["instance"], r["M"], r["K_local"], r["K_gathered"], r["N"], r["chunks"],
            "Y" if r["has_bias"] else "-", "Y" if r["has_addcmul"] else "-",
            r["ring_size"], r["num_links"], r["min_time_us"],
            r["tflops_at_min"] if r["tflops_at_min"] is not None else "-",
            r["skew_ratio"], r["math_fidelity"]))
