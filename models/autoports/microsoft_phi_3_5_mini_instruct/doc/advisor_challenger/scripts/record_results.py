"""Attach measured outcomes to the tool-generated reconciliation."""
import json
from pathlib import Path

root = Path(__file__).parents[1]
reconciliation = root / "reconciliation_dense.json"
d = json.loads(reconciliation.read_text())

def measurement(name):
    return json.loads((root / "measurements" / f"{name}.json").read_text())

query = measurement("rope_l1_query")
key = measurement("rope_l1_key")
norm = measurement("norm_11c")
incumbent = json.loads((root / "incumbent.json").read_text())
query_chains = {"dense:b14", "dense:3", "dense:2", "dense:1", "dense:4", "dense:5"}
key_chains = {"dense:b29", "dense:8", "dense:7", "dense:6", "dense:9", "dense:10"}

for chain in d["chains"]:
    cid = chain["chain"]
    if cid in query_chains or cid in key_chains:
        measured = query if cid in query_chains else key
        chain.update(
            verdict="kept", measured_ms=measured["median_ms"], repeats_ms=measured["repeats_ms"],
            oracle_passed=True, oracle_pcc=1.0, oracle_weights="real",
            combined_with=sorted(query_chains if cid in query_chains else key_chains),
            perf_report="profiles/dense_winner.csv",
        )
    elif cid in {"dense:0", "dense:11"}:
        chain.update(
            verdict="rejected", measured_ms=norm["median_ms"], repeats_ms=norm["repeats_ms"],
            oracle_passed=False, oracle_pcc=0.9999910666979231, oracle_weights="real",
            rejection="placement changed real-weight output; rejected despite speedup",
        )
    elif cid == "dense:b43":
        chain.update(
            verdict="rejected", measured_ms=incumbent["incumbent_ms"],
            repeats_ms=incumbent["repeats_ms"],
            hard_error="TT_FATAL: Sharded output not supported for GQA",
            rejection="extended SDPA->concat chain is not executable",
        )
    else:
        chain.update(verdict="below_threshold", measured_ms=None, repeats_ms=None)

for op in d["material_ops_on_le_2_cores"]:
    op.update(
        measured_ms=norm["median_ms"], repeats_ms=norm["repeats_ms"],
        oracle_passed=False, oracle_pcc=0.9999910666979231,
        sweep={"11": 0.745906, "12": 0.749010, "24": 0.748513},
        rejection="real-weight differential PCC moved",
    )
for row in d["disagreements"]:
    if row.get("bucket") == "dram_resident":
        row["verdict"] = "kept_in_dram"
        row["note"] = "advisor direction retained; de-sharding was not proposed"

reconciliation.write_text(json.dumps(d, indent=2) + "\n")
