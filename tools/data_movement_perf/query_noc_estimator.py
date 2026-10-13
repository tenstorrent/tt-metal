"""Query the actual NoC-estimator CLI and record dataset coverage with every result."""

from pathlib import Path
import json, math, subprocess, hashlib
import yaml

ROOT = Path(__file__).resolve().parents[2]
BINARY = ROOT / "generated/noc_estimator_comparison/noc_estimate"
DATA = ROOT / "tt_metal/impl/experimental/noc_estimator/latencies/noc_latencies.yaml"
EXAMPLE = ROOT / "ttnn/ttnn/operations/examples/dram_l1_transaction_size"
OUT = ROOT / "tools/data_movement_perf/noc_estimator_comparison.json"
y = yaml.safe_load(DATA.read_text())
measurements = json.loads((EXAMPLE / "core_sweep_confirmation.json").read_text())
actual = {r["cores"]: r for r in measurements["results"]}


def query(cores, transactions, pattern, txn=2048):
    args = [
        str(BINARY),
        "--arch",
        "BLACKHOLE",
        "--memory",
        "DRAM_INTERLEAVED",
        "--mechanism",
        "UNICAST",
        "--pattern",
        pattern,
        "--noc-index",
        "0",
        "--num-transactions",
        str(transactions),
        "--num-transactions-per-barrier",
        str(transactions),
        "--transaction-size-bytes",
        str(txn),
        "--num-subordinates",
        str(cores),
        "--aiclk-mhz",
        "1350",
    ]
    r = subprocess.run(args, text=True, capture_output=True, check=True)
    return dict(command=args, result=json.loads(r.stdout), stderr=r.stderr)


coverage = {}
for pattern, pid in [("ONE_FROM_ALL", 2), ("ALL_FROM_ALL", 5)]:
    entries = [
        e
        for e in y["entries"]
        if e["key"].get("arch", 2) == 3
        and e["key"].get("memory", 0) == 1
        and e["key"].get("pattern", 1) == pid
        and e["key"].get("noc_index", 0) == 0
    ]
    coverage[pattern] = dict(
        num_transactions=sorted({e["key"].get("num_transactions", 1) for e in entries}),
        num_subordinates=sorted({e["key"].get("num_subordinates", 1) for e in entries}),
    )
rows = []
for cores in [7, 11, 44, 88, 110]:
    n = math.ceil(4096 / cores)
    rows.append(
        dict(
            cores=cores,
            transactions_per_largest_shard=n,
            excel_us=actual[cores]["expected_us"],
            measured_us=actual[cores]["median_us"],
            transaction_count_clamped=n > 256,
            single_reader=query(cores, n, "ONE_FROM_ALL"),
            all_readers=query(cores, n, "ALL_FROM_ALL"),
        )
    )
probes = [dict(cores=c, **query(c, 64, "ALL_FROM_ALL")) for c in [1, 7, 11, 44, 110]]
clamps = [dict(transactions=n, **query(7, n, "ALL_FROM_ALL")) for n in [256, 373, 586, 18752]]
assert len({q["result"]["latency_cycles"] for q in probes}) == 1
assert len({q["result"]["latency_cycles"] for q in clamps}) == 1
result = dict(
    data_sha256=hashlib.sha256(DATA.read_bytes()).hexdigest(),
    coverage=coverage,
    rows=rows,
    core_count_probes=probes,
    out_of_range_probes=clamps,
    conclusions=[
        "DRAM num_subordinates is stored only as 0; no core-count interpolation is possible.",
        "Transaction counts above 256 per barrier clamp to the largest data point, rather than extrapolating.",
        "ALL_FROM_ALL is the full-grid benchmark pattern, not a fitted model for arbitrary reader counts.",
        "CLI total_bytes and achieved_GBps use transactions*payload, not all-core aggregate bytes.",
        "Benchmark reads cycle across bank pages into reused L1 scratch; our kernel fills distinct resident L1 rows.",
    ],
)
OUT.write_text(json.dumps(result, indent=2) + "\n")
print("cores | txns | single-reader us | all-reader us | measured us | clamp")
for r in rows:
    print(
        r["cores"],
        r["transactions_per_largest_shard"],
        round(r["single_reader"]["result"]["latency_us"], 3),
        round(r["all_readers"]["result"]["latency_us"], 3),
        r["measured_us"],
        r["transaction_count_clamped"],
    )
print("Coverage:", coverage)
print("Fixed-64 txn core-count probe:", [q["result"]["latency_us"] for q in probes])
print("256/373/586/18752 txn probe:", [q["result"]["latency_us"] for q in clamps])
