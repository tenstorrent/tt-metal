# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Generate coarse and measurement-guided core sweeps for the transport probe.

Outputs ordinary NOC_TRANSPORT_CASES JSON plus a separate provenance/decision plan.
This is fixed work per core (64 requests), not fixed-total-work strong scaling.
"""

import argparse
from collections import defaultdict
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics

FIELDS = ("traffic", "payload_bytes", "batch", "mode", "banks")


def positive_int(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def core_limit(topology, maximum=None):
    workers = topology["workers"]
    if not workers or len({tuple(w) for w in workers}) != len(workers):
        raise ValueError("topology must contain distinct workers")
    return min(len(workers), positive_int(maximum, "max cores")) if maximum is not None else len(workers)


def coarse_counts(limit):
    """Powers of two, an upper-range 1.5x-power point if it fits, and the cap."""
    positive_int(limit, "core limit")
    counts = {1, limit}
    power = 1
    while power <= limit:
        counts.add(power)
        power *= 2
    upper = 3 * (power // 2) // 2
    if 1 <= upper <= limit:
        counts.add(upper)
    return sorted(counts)


def validate_case(case, topology, limit):
    c = {k: case.get(k, "read" if k == "traffic" else None) for k in FIELDS}
    cores = positive_int(case["cores"], "cores")
    if cores > limit:
        raise ValueError("case exceeds core limit")
    if c["traffic"] not in ("read", "write", "mixed"):
        raise ValueError("unknown traffic")
    if c["mode"] not in ("noc0", "noc1", "split") or (c["traffic"] == "mixed" and c["mode"] != "split"):
        raise ValueError("mixed requires split; modes must be noc0/noc1/split")
    if c["banks"] not in ("balanced", "hotspot"):
        raise ValueError("banks must be balanced or hotspot")
    payload = positive_int(c["payload_bytes"], "payload bytes")
    if payload > 16384 or payload % 64:
        raise ValueError("probe sweep payload must be a multiple of 64 in 64..16384")
    batch = positive_int(c["batch"], "batch")
    if 64 % batch or (c["mode"] == "split" and batch % 2):
        raise ValueError("batch must divide 64 and be even for split mode")
    if case.get("pages", 64) != 64 or case.get("iterations", 4) != 4 or case.get("placement", "row") != "row":
        raise ValueError("only the probe's 64 pages/core, 4 iterations, row placement are supported")
    if case.get("completion", "ack") not in ("ack", "remote_ack", "read_response"):
        raise ValueError("source-flush/posted completion is unsupported")
    if "logical_coords" in case:
        width = topology["worker_grid"][0]
        if case["logical_coords"] != [[i % width, i // width] for i in range(cores)]:
            raise ValueError("only row-prefix logical coordinates are supported")
    return dict(cores=cores, **c)


def coarse_cases(
    topology,
    maximum=None,
    traffic=("read", "write"),
    payloads=(128, 1024),
    batches=(16,),
    modes=("noc0", "noc1", "split"),
    banks=("balanced", "hotspot"),
):
    limit = core_limit(topology, maximum)
    cases, seen, skipped = [], set(), []
    for values in itertools.product(traffic, payloads, batches, modes, banks):
        family = dict(zip(FIELDS, values))
        # Invalid combinations are explicitly recorded, not sent to the hardware probe.
        if (family["traffic"] == "mixed" and family["mode"] != "split") or (
            family["mode"] == "split" and isinstance(family["batch"], int) and family["batch"] % 2
        ):
            skipped.append(family)
            continue
        for cores in coarse_counts(limit):
            case = validate_case(dict(cores=cores, **family), topology, limit)
            identity = json.dumps(case, sort_keys=True)
            if identity not in seen:
                seen.add(identity)
                cases.append(case)
    if not cases:
        raise ValueError("no valid scenarios; check traffic/mode/batch combinations")
    return cases, dict(core_counts=coarse_counts(limit), core_limit=limit, skipped_combinations=skipped)


def load_observations(path):
    """Accept probe JSONL, read CLI lists, or validate_transport.py output objects."""
    raw = Path(path).read_text()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return [json.loads(line) for line in raw.splitlines() if line.strip()]
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return value["cases"] if "cases" in value else [value]
    raise ValueError("expected probe rows or validation cases")


def normalized_observation(row, topology, limit):
    source = dict(row.get("case", row))
    if "traffic" in row:
        source["traffic"] = row["traffic"]
    if source.get("correct") is False or row.get("correct") is False:
        raise ValueError("incorrect measurements cannot guide refinement")
    for key, expected in (("clock_mhz", topology["clock_mhz"]), ("bank_count", len(topology["banks"]))):
        if key in source and source[key] != expected:
            raise ValueError(f"observation {key} mismatch")
    case = validate_case(source, topology, limit)
    measured = row.get("measured_us")
    if measured is None:
        measured = source["median_cycles"] / source["clock_mhz"]
    estimate = row.get("prediction", {}).get("estimated_us", row.get("estimated_us"))
    for name, value in (("measured_us", measured), ("estimated_us", estimate)):
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise ValueError(f"{name} must be finite and positive")
    return case, measured, estimate


def refine_cases(
    topology, observations, maximum=None, budget=64, saturation=0.25, duration_jump=0.20, error_threshold=0.10
):
    """Bisect flagged measured intervals, round-robin across scenario families.

    Saturation uses relative bandwidth gain / relative core-count gain, where
    bandwidth is proportional to cores / measured duration for fixed work/core.
    Thresholds are fractions. There is no extrapolation beyond measured counts.
    """
    limit = core_limit(topology, maximum)
    positive_int(budget, "budget")
    for name, threshold in (("saturation", saturation), ("duration_jump", duration_jump), ("error", error_threshold)):
        if not math.isfinite(threshold) or threshold < 0:
            raise ValueError(f"{name} threshold must be finite and nonnegative")
    groups = defaultdict(lambda: defaultdict(list))
    for row in observations:
        case, measured, estimate = normalized_observation(row, topology, limit)
        groups[tuple(case[k] for k in FIELDS)][case["cores"]].append((measured, estimate))
    if not groups:
        raise ValueError("no observations")
    queues = []
    for family, by_core in sorted(groups.items()):
        points = {}
        for cores, repeats in by_core.items():
            errors = [abs(e / m - 1) for m, e in repeats if e is not None]
            points[cores] = (statistics.median(m for m, _ in repeats), statistics.median(errors) if errors else None)
        proposals = []
        counts = sorted(points)
        for low, high in zip(counts, counts[1:]):
            if high - low < 2:
                continue
            t0, e0 = points[low]
            t1, e1 = points[high]
            efficiency = ((high / t1) / (low / t0) - 1) / (high / low - 1)
            jump = abs(t1 / t0 - 1)
            errors = [e for e in (e0, e1) if e is not None]
            error = max(errors) if errors else None
            reasons = []
            if efficiency <= saturation:
                reasons.append("bandwidth_saturation")
            if jump >= duration_jump:
                reasons.append("duration_change")
            if error is not None and error >= error_threshold:
                reasons.append("prediction_error")
            if reasons:
                proposals.append(
                    dict(
                        case=dict(cores=(low + high) // 2, **dict(zip(FIELDS, family))),
                        interval=[low, high],
                        reasons=reasons,
                        relative_bandwidth_gain_per_relative_core_gain=efficiency,
                        relative_duration_change=jump,
                        endpoint_absolute_error=error,
                    )
                )
        # Wide intervals first; scenario families get equal opportunity under the global budget.
        proposals.sort(key=lambda p: (-(p["interval"][1] - p["interval"][0]), p["interval"][0]))
        if proposals:
            queues.append(proposals)
    ordered = [queue[i] for i in range(max(map(len, queues), default=0)) for queue in queues if i < len(queue)]
    selected = ordered[:budget]
    return [p["case"] for p in selected], dict(
        core_limit=limit,
        observation_families=len(groups),
        eligible_intervals=len(ordered),
        deferred_intervals=len(ordered) - len(selected),
        decisions=selected,
        thresholds=dict(saturation=saturation, duration_jump=duration_jump, error=error_threshold),
        note="Only new midpoints inside measured intervals; rerun with accumulated observations. No proposals can mean insufficient coverage, no triggers, or adjacent integer counts.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["coarse", "refine"])
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="probe-compatible case list")
    parser.add_argument("--plan", type=Path, help="decision/provenance report; default OUTPUT.plan.json")
    parser.add_argument("--max-cores", type=int)
    parser.add_argument("--traffic", nargs="+", choices=["read", "write", "mixed"], default=["read", "write"])
    parser.add_argument("--payload-bytes", nargs="+", type=int, default=[128, 1024])
    parser.add_argument("--batch", nargs="+", type=int, default=[16])
    parser.add_argument("--modes", nargs="+", choices=["noc0", "noc1", "split"], default=["noc0", "noc1", "split"])
    parser.add_argument("--banks", nargs="+", choices=["balanced", "hotspot"], default=["balanced", "hotspot"])
    parser.add_argument(
        "--observations", type=Path, nargs="+", help="raw probe JSONL or validation JSON; refinement only"
    )
    parser.add_argument("--budget", type=int, default=64, help="maximum new cases in a refinement round")
    parser.add_argument("--saturation", type=float, default=0.25)
    parser.add_argument("--duration-jump", type=float, default=0.20)
    parser.add_argument("--error-threshold", type=float, default=0.10)
    args = parser.parse_args()
    try:
        topology = json.loads(args.topology.read_text())
        if args.stage == "coarse":
            if args.observations:
                raise ValueError("--observations is only used by refine")
            cases, details = coarse_cases(
                topology, args.max_cores, args.traffic, args.payload_bytes, args.batch, args.modes, args.banks
            )
        else:
            if not args.observations:
                raise ValueError("refine requires --observations")
            observations = [r for p in args.observations for r in load_observations(p)]
            cases, details = refine_cases(
                topology,
                observations,
                args.max_cores,
                args.budget,
                args.saturation,
                args.duration_jump,
                args.error_threshold,
            )
        plan = args.plan or args.output.with_suffix(".plan.json")
        sources = [args.topology, Path(__file__), *(args.observations or [])]
        targets = [args.output.resolve(), plan.resolve()]
        if targets[0] == targets[1] or any(p.resolve() in targets for p in sources):
            raise ValueError("output and plan must be distinct from each other and input files")
        report = dict(
            stage=args.stage,
            workload="fixed_64_requests_per_core",
            case_count=len(cases),
            details=details,
            source_sha256={str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
            status="cases_generated_not_hardware_validated",
        )
        for path, content in ((args.output, cases), (plan, report)):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n")
        print(f"{len(cases)} cases -> {args.output}; decisions -> {plan}")
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
