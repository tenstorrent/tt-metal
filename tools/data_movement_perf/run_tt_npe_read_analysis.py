"""Replay captured BH DRAM reads with and without TT-NPE congestion.

Requires a separately built tt-npe and its install/lib on PYTHONPATH.
Uses recorded issue timestamps; this is a trace replay, not an independent latency prediction.
"""

import argparse
import collections
import json
from pathlib import Path
import subprocess
import tt_npe_pybind as npe

p = argparse.ArgumentParser()
p.add_argument("--cores", type=int, required=True, choices=[7, 44])
p.add_argument("--mode", choices=["fast", "none"], required=True)
p.add_argument("--step", type=int, default=32)
a = p.parse_args()
root = Path(__file__).resolve().parents[2]
folder = root / "generated/dram_l1_npe"
trace = folder / f"c{a.cores}/noc_trace_dev0_ID3072.json"
events = json.loads(trace.read_text())
reads = [r for r in events if r.get("type") == "READ"]
assert len(reads) == 4096 and sum(r["num_bytes"] for r in reads) == 8388608
assert {r["noc"] for r in reads} == {"NOC_0"}
assert len({(r["sx"], r["sy"]) for r in reads}) == a.cores
banks = collections.Counter((r["dx"], r["dy"]) for r in reads)
assert len(banks) == 8 and set(banks.values()) == {512}
cfg = npe.Config()
cfg.device_name = "P150"  # Eight-bank BH preset; generic blackhole selects seven banks.
cfg.congestion_model_name = a.mode
cfg.cycles_per_timestep = a.step
cfg.workload_json_filepath = str(trace)
cfg.workload_is_noc_trace = True
cfg.estimate_cong_impact = False  # Explicit separate none/fast runs provide the comparison.
cfg.emit_timeline_file = a.mode == "fast"
cfg.timeline_filepath = str(folder / f"c{a.cores}_{a.mode}_{a.step}_timeline.json")
cfg.set_verbosity_level(0)
wl = npe.createWorkloadFromJSON(str(trace), "P150", True)
assert wl is not None
api = npe.InitAPI(cfg)
assert api is not None
result = api.runNPE(wl)
assert isinstance(result, npe.Stats), str(result)
fields = [
    "completed",
    "estimated_cycles",
    "golden_cycles",
    "cycle_prediction_error",
    "dram_bw_util",
    "dram_bw_util_sim",
    "overall_avg_link_util",
    "overall_max_link_util",
    "overall_avg_noc0_link_util",
    "overall_max_noc0_link_demand",
    "overall_avg_niu_demand",
    "overall_max_niu_demand",
    "dram_bw_util_per_controller",
]
out = dict(
    cores=a.cores,
    mode=a.mode,
    step_cycles=a.step,
    clock_mhz=1350,
    read_count=len(reads),
    bytes=8388608,
    bank_reads={str(k): v for k, v in banks.items()},
    trace=str(trace),
    config=str(cfg),
    npe_commit=subprocess.check_output(
        ["git", "-C", str(root.parent / "tt-npe"), "rev-parse", "HEAD"], text=True
    ).strip(),
    devices={},
)
for dev, stats in result.per_device_stats.items():
    out["devices"][str(dev)] = {k: getattr(stats, k) for k in fields}
    out["devices"][str(dev)]["estimated_us"] = stats.estimated_cycles / 1350
    out["devices"][str(dev)]["trace_golden_us"] = stats.golden_cycles / 1350
    assert stats.completed
(folder / f"c{a.cores}_{a.mode}_{a.step}.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
