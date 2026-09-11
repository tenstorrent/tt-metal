from pathlib import Path
import csv, json, hashlib, math, collections, gzip, io

out = Path(__file__).resolve().parent
root = out.parents[2]
p = next(out.rglob("ops_perf_results*.csv.gz"))
raw = gzip.decompress(p.read_bytes())
rows = list(csv.DictReader(io.StringIO(raw.decode(), newline="")))
starts = [i for i, r in enumerate(rows) if r["OP TYPE"] == "signpost" and r["OP CODE"] == "MLA_START"]
ends = [i for i, r in enumerate(rows) if r["OP TYPE"] == "signpost" and r["OP CODE"] == "MLA_END"]
assert len(starts) == len(ends) == 1 and starts[0] < ends[0]
selected = rows[starts[0] + 1 : ends[0]]
assert all(r["OP TYPE"] == "tt_dnn_device" for r in selected)
dev = collections.defaultdict(list)
for r in selected:
    dev[int(r["DEVICE ID"])].append(r)
assert len(dev) == 8 and set(map(len, dev.values())) == {24}
identity = lambda r: (r["OP CODE"], int(r["GLOBAL CALL COUNT"]) - int(r["DEVICE ID"]), r["SUB DEVICE ID"])
seq = [identity(r) for r in next(iter(dev.values()))]
assert all([identity(r) for r in rr] == seq for rr in dev.values())
assert len(set(seq)) == 24
assert all(
    math.isfinite(float(r["DEVICE KERNEL DURATION [ns]"])) and float(r["DEVICE KERNEL DURATION [ns]"]) > 0
    for r in selected
)
assert all(not r["METAL TRACE ID"] for r in selected)
merged = []
categories = collections.Counter()
ops = collections.Counter()
for i, ident in enumerate(seq):
    rr = [v[i] for v in dev.values()]
    d = [float(r["DEVICE KERNEL DURATION [ns]"]) for r in rr]
    op = ident[0]
    assert not any(x in op for x in ["AllGather", "ReduceScatter", "AllReduce", "Matmul_RS"])
    ns = int(max(d))
    cat = (
        "Matmul"
        if "matmul" in op.lower()
        else "SDPA"
        if ("SDPA" in op or "ScaledDotProductAttention" in op)
        else "Other"
    )
    categories[cat] += ns
    ops[op] += ns
    merged.append(
        {
            "operation_index": i,
            "op": op,
            "global_call_count": ident[1],
            "duration_max_ns": ns,
            "duration_min_ns": min(d),
            "duration_mean_ns": sum(d) / 8,
            "category": cat,
        }
    )
assert sum(categories.values()) == 8585052
assert categories == {"SDPA": 7195124, "Matmul": 737823, "Other": 652105}
with (out / "direct_operation_breakdown.csv").open("w") as f:
    wr = csv.DictWriter(f, fieldnames=list(merged[0]))
    wr.writeheader()
    wr.writerows(merged)
validation = {
    "passed": True,
    "csv": str(p.relative_to(root)),
    "csv_sha256": hashlib.sha256(raw).hexdigest(),
    "csv_total_rows": len(rows),
    "signposts": {"MLA_START": len(starts), "MLA_END": len(ends)},
    "device_ids": sorted(dev),
    "selected_rows": len(selected),
    "operations_per_device": 24,
    "identical_sequence_fields": ["OP CODE", "GLOBAL CALL COUNT minus DEVICE ID", "SUB DEVICE ID"],
    "duration_validation": "all finite and strictly positive",
    "trace": "untraced first forward",
    "sum_of_per_operation_device_maxima_ns": sum(categories.values()),
    "categories_ns": dict(categories),
    "op_totals_ns": dict(ops),
    "per_device_kernel_sum_ns": {
        str(k): sum(float(r["DEVICE KERNEL DURATION [ns]"]) for r in v) for k, v in dev.items()
    },
    "program_cache_hit_values": dict(collections.Counter(r["PROGRAM CACHE HIT"] for r in selected)),
    "limitations": [
        "One instrumented first forward, no warmup/repeats; not a steady-state benchmark.",
        "Kernel-counter durations exclude host compilation gaps, but cold execution and instrumentation may affect device timing.",
        "Sum of per-operation device maxima is an accumulated operation budget, not wall-clock MLA latency or TTFT.",
        "RingJointSDPA includes distributed ring communication; CCL category zero does not mean no communication.",
        "Reference=None: execution and coverage validated, numerical accuracy not independently compared.",
    ],
}
(out / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
text = [
    "# B3 direct capture validation",
    "",
    f"Validated `{p.name}` independently using Python CSV parsing.",
    "",
    "Exactly one MLA_START/MLA_END region contains 192 device operations: 24 identical operation identities on each of eight devices. Every kernel duration is finite and positive.",
    "",
    "| Category | Device-max operation sum (ms) | Share |",
    "|---|---:|---:|",
]
for k, v in categories.most_common():
    text.append(f"| {k} | {v/1e6:.6f} | {100*v/8585052:.2f}% |")
text += [
    "| Total | 8.585052 | 100% |",
    "",
    "The independent sum matches the wrapper exactly. No separately named TP collective occurs; ring communication is included within RingJointSDPA.",
    "",
    *validation["limitations"],
]
(out / "DIRECT_BREAKDOWN.md").write_text("\n".join(text) + "\n")
print(json.dumps(validation, indent=2))
