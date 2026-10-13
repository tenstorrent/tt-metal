"""Bounded BH interleaved DRAM-read queries. No clamping, relaxation, or extrapolation.

One final barrier is modeled. Numerical coverage does not guarantee kernel/layout equivalence.
All-reader queries require a calibrated reader-count axis; the bundled zero-only axis is rejected.
"""

from pathlib import Path
import argparse
import bisect
import json
import math
import subprocess
import yaml

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "tt_metal/impl/experimental/noc_estimator/latencies/noc_latencies.yaml"
BINARY = ROOT / "generated/noc_estimator_comparison/noc_estimate"


def bracket(values, target):
    values = sorted(set(values))
    if not values or not values[0] <= target <= values[-1]:
        return None
    i = bisect.bisect_left(values, target)
    return (values[i], values[i]) if values[i] == target else (values[i - 1], values[i])


class GuardedEstimator:
    def __init__(self, path=DATA, binary=BINARY):
        self.data = yaml.safe_load(Path(path).read_text())
        self.binary = Path(binary)

    def coverage(self, cores, transactions, size, noc=0):
        reasons = []
        if cores < 1 or transactions < 1 or size < 1 or noc not in (0, 1):
            return dict(eligible=False, reasons=["Invalid positive count/size or NoC index"])
        pattern = 2 if cores == 1 else 5
        entries = [
            e
            for e in self.data["entries"]
            if e["key"].get("arch", 2) == 3
            and e["key"].get("memory", 0) == 1
            and e["key"].get("mechanism", 0) == 0
            and e["key"].get("pattern", 1) == pattern
            and e["key"].get("noc_index", 0) == noc
            and not e["key"].get("stateful", False)
            and not e["key"].get("same_axis", False)
            and not e["key"].get("loopback", False)
        ]
        sizes = self.data["transaction_sizes"]
        size_bounds = bracket(sizes, size)
        count_bounds = bracket([e["key"].get("num_transactions", 1) for e in entries], transactions)
        sub_values = sorted({e["key"].get("num_subordinates", 1) for e in entries})
        # The one-reader pattern encodes reader count by its pattern; zero is its native metadata.
        subordinate = 0 if cores == 1 and sub_values == [0] else cores
        core_bounds = bracket(sub_values, subordinate)
        if size_bounds is None:
            reasons.append("Transaction bytes outside calibrated range")
        if count_bounds is None:
            reasons.append("Transactions per barrier outside calibrated range")
        if core_bounds is None:
            reasons.append("Reader count has no calibrated range for this pattern")
        # Require all interpolation corners and positive finite data, not merely global min/max.
        if not reasons:
            for n in set(count_bounds):
                for c in set(core_bounds):
                    matches = [
                        e
                        for e in entries
                        if e["key"].get("num_transactions", 1) == n and e["key"].get("num_subordinates", 1) == c
                    ]
                    if len(matches) != 1:
                        reasons.append("Missing or duplicate interpolation corner")
                        continue
                    # Native payload interpolation is quadratic; validate its full stored curve.
                    lat = matches[0]["latencies"]
                    if len(lat) != len(sizes) or any(not math.isfinite(v) or v <= 0 for v in lat):
                        reasons.append("Incomplete or invalid calibration curve")
        return dict(
            eligible=not reasons,
            reasons=reasons,
            pattern="ONE_FROM_ALL" if cores == 1 else "ALL_FROM_ALL",
            num_subordinates=subordinate,
            transaction_bounds=count_bounds,
            size_bounds=size_bounds,
            reader_bounds=core_bounds,
        )

    def estimate(self, cores, transactions, size, noc=0, clock_mhz=1350):
        if clock_mhz <= 0 or not math.isfinite(clock_mhz):
            raise ValueError("Positive finite clock required")
        coverage = self.coverage(cores, transactions, size, noc)
        if not coverage["eligible"]:
            return dict(source="unavailable", latency_us=None, coverage=coverage)
        args = [
            str(self.binary),
            "--arch",
            "BLACKHOLE",
            "--memory",
            "DRAM_INTERLEAVED",
            "--pattern",
            coverage["pattern"],
            "--noc-index",
            str(noc),
            "--num-transactions",
            str(transactions),
            "--num-transactions-per-barrier",
            str(transactions),
            "--transaction-size-bytes",
            str(size),
            "--num-subordinates",
            str(coverage["num_subordinates"]),
            "--aiclk-mhz",
            str(clock_mhz),
        ]
        result = subprocess.run(args, capture_output=True, text=True)
        if result.returncode or result.stderr.strip():
            return dict(source="unavailable", latency_us=None, coverage=coverage, error=result.stderr or result.stdout)
        value = json.loads(result.stdout)
        if not math.isfinite(value["latency_us"]) or value["latency_us"] <= 0:
            return dict(source="unavailable", latency_us=None, coverage=coverage, error="Invalid estimator output")
        return dict(source="bounded_estimator", latency_us=value["latency_us"], coverage=coverage, raw=value)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cores", type=int, required=True)
    p.add_argument("--transactions", type=int, required=True)
    p.add_argument("--size", type=int, required=True)
    p.add_argument("--noc", type=int, default=0)
    a = p.parse_args()
    print(json.dumps(GuardedEstimator().estimate(a.cores, a.transactions, a.size, a.noc), indent=2))
