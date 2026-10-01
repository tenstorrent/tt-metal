# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Compare single-chip and tensor-parallel DeepSeek attention profiles.

Unlike a single-chip profile, a mesh profile contains one CSV row per device for
each logical operation. Summing those rows overstates TP latency because the
ranks execute concurrently. This tool groups the nth occurrence of an operation
on every rank and uses the slowest rank as its critical-path duration.

The five attention ``MatmulDecodeDeviceOperation`` projections are identified
from their K/N/batch attributes. Their dimensions are checked against the
expected DeepSeek TP partition:

* q_a and kv remain replicated.
* q_b and o_b split N across TP ranks.
* o_a splits its group batch across TP ranks.

Collectives are reported separately, and the final selected-ops total is the sum
of the five projection critical paths plus collective critical paths. It is not
the complete attention latency because layout, norm, RoPE, cache, and SDPA ops
are intentionally excluded.

Usage:
    python compare_attention_tp_profiles.py BASELINE.csv TP.csv
    python compare_attention_tp_profiles.py BASELINE.csv TP4.csv TP8.csv --agg median
    python compare_attention_tp_profiles.py BASELINE.csv TP.csv --skip-first 1
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

csv.field_size_limit(sys.maxsize)

_MATMUL = "MatmulDecodeDeviceOperation"
_PROJECTION_ORDER = ("q_a", "q_a+kv", "q_b", "kv", "o_a", "o_b")
_ATTR_PATTERN = re.compile(r"'(?P<key>K|M|N|batch)': '(?P<value>\d+)'")


@dataclass(frozen=True)
class MatmulSample:
    device: str
    duration_ns: float
    k: int
    m: int
    n: int
    batch: int

    @property
    def shape(self) -> str:
        return f"K={self.k},N={self.n},B={self.batch}"


@dataclass(frozen=True)
class Profile:
    path: Path
    devices: tuple[str, ...]
    projections: dict[str, dict[str, list[MatmulSample]]]
    collectives: dict[str, dict[str, list[float]]]


def _device_key(device: str) -> tuple[int, str]:
    try:
        return (int(device), device)
    except ValueError:
        return (sys.maxsize, device)


def _number(row: dict[str, str], column: str) -> float:
    try:
        return float(row.get(column) or 0)
    except ValueError:
        return 0.0


def _matmul_attributes(row: dict[str, str]) -> dict[str, int]:
    return {match.group("key"): int(match.group("value")) for match in _ATTR_PATTERN.finditer(row["ATTRIBUTES"])}


def _projection_name(k: int, n: int, batch: int, has_fused_qkv: bool) -> str | None:
    if batch == 1 and (k, n) in ((4096, 1536), (1024, 6144)):
        return "q_a+kv"
    if k == 1024 and batch == 1:
        return "q_b"
    if batch == 1 and ((k == 8192 and n in (1024, 4096)) or (k, n) == (2048, 4096)):
        return "o_b"
    if k == 4096 and batch > 1:
        return "o_a"
    if k == 4096 and n == 512 and batch == 1:
        return "kv"
    if k == 4096 and n == 1024 and batch == 1:
        return "o_a" if has_fused_qkv else "q_a"
    return None


def _is_collective(code: str) -> bool:
    return any(token in code.lower() for token in ("allgather", "allreduce", "reducescatter", "alltoall"))


def read_profile(path: Path) -> Profile:
    projections: dict[str, dict[str, list[MatmulSample]]] = defaultdict(lambda: defaultdict(list))
    collectives: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    matmuls: list[MatmulSample] = []
    devices: set[str] = set()

    with path.open(newline="") as file:
        for row in csv.DictReader(file):
            device = (row.get("DEVICE ID") or "").strip()
            if not device:
                continue
            devices.add(device)
            code = row["OP CODE"]
            duration = _number(row, "DEVICE KERNEL DURATION [ns]")

            if code == _MATMUL:
                attrs = _matmul_attributes(row)
                if not all(name in attrs for name in ("K", "M", "N", "batch")):
                    continue
                matmuls.append(
                    MatmulSample(
                        device=device,
                        duration_ns=duration,
                        k=attrs["K"],
                        m=attrs["M"],
                        n=attrs["N"],
                        batch=attrs["batch"],
                    )
                )
            elif _is_collective(code):
                collectives[code][device].append(duration)

    has_fused_qkv = any((sample.k, sample.n) in ((4096, 1536), (1024, 6144)) for sample in matmuls)
    for sample in matmuls:
        name = _projection_name(sample.k, sample.n, sample.batch, has_fused_qkv)
        if name is not None:
            projections[name][sample.device].append(sample)

    return Profile(
        path=path,
        devices=tuple(sorted(devices, key=_device_key)),
        projections={name: dict(by_device) for name, by_device in projections.items()},
        collectives={name: dict(by_device) for name, by_device in collectives.items()},
    )


def _aggregate(values: list[float], method: str) -> float:
    if not values:
        return 0.0
    if method == "median":
        return statistics.median(values)
    if method == "min":
        return min(values)
    if method == "max":
        return max(values)
    return statistics.fmean(values)


def _critical_path(
    by_device: dict[str, list[float]],
    devices: tuple[str, ...],
    method: str,
    skip_first: int,
) -> tuple[float, int]:
    """Aggregate max-rank duration for each logical occurrence."""
    present = [device for device in devices if device in by_device]
    if not present:
        return 0.0, 0
    count = min(len(by_device[device]) for device in present)
    critical = [max(by_device[device][index] for device in present) for index in range(skip_first, count)]
    return _aggregate(critical, method), len(critical)


def _profile_steps(profile: Profile) -> int:
    q_b = profile.projections.get("q_b", {})
    return min((len(samples) for samples in q_b.values()), default=1)


def _critical_path_per_step(
    by_device: dict[str, list[float]],
    devices: tuple[str, ...],
    method: str,
    skip_first: int,
    steps: int,
) -> tuple[float, int, int]:
    """Aggregate each step's sum of sequential calls, each call using max-rank time."""
    present = [device for device in devices if device in by_device]
    if not present:
        return 0.0, 0, 0
    count = min(len(by_device[device]) for device in present)
    calls_per_step = count // steps
    if calls_per_step < 1 or calls_per_step * steps != count:
        return 0.0, 0, 0
    critical = [max(by_device[device][index] for device in present) for index in range(count)]
    per_step = [sum(critical[step * calls_per_step : (step + 1) * calls_per_step]) for step in range(skip_first, steps)]
    return _aggregate(per_step, method), calls_per_step, len(per_step)


def _projection_critical(
    profile: Profile, name: str, method: str, skip_first: int, steps: int
) -> tuple[float, int, int, MatmulSample | None]:
    by_device = profile.projections.get(name, {})
    durations = {device: [sample.duration_ns for sample in samples] for device, samples in by_device.items()}
    duration, calls, count = _critical_path_per_step(durations, profile.devices, method, skip_first, steps)
    representative = next((samples[0] for samples in by_device.values() if samples), None)
    return duration, calls, count, representative


def _partition_status(base: MatmulSample, tp: MatmulSample, tp_size: int, name: str) -> str:
    if name == "q_a+kv":
        return "fused"
    if base.k != tp.k or base.m != tp.m:
        if name == "o_b" and tp.k * tp_size == base.k and tp.n == base.n:
            return "K-split"
        return "MISMATCH"
    if name in ("q_b", "o_b"):
        return "N-split" if tp.n * tp_size == base.n and tp.batch == base.batch else "MISMATCH"
    if name == "o_a":
        return "groups-split" if tp.n == base.n else "MISMATCH"
    return "replicated" if (tp.n, tp.batch) == (base.n, base.batch) else "MISMATCH"


def _fmt_us(ns: float) -> str:
    return f"{ns / 1e3:.2f}"


def _fmt_speedup(base_ns: float, tp_ns: float) -> str:
    return f"{base_ns / tp_ns:.2f}x" if tp_ns else "n/a"


def compare(base: Profile, tp: Profile, method: str, skip_first: int) -> None:
    tp_size = len(tp.devices)
    base_steps = _profile_steps(base)
    tp_steps = _profile_steps(tp)
    fused_qkv = "q_a+kv" in tp.projections
    projection_order = ("q_a+kv", "q_b", "o_a", "o_b") if fused_qkv else _PROJECTION_ORDER
    print()
    print("=" * 108)
    print(f"baseline: {base.path}  devices={','.join(base.devices) or 'none'}")
    print(f"TP:       {tp.path}  devices={','.join(tp.devices) or 'none'}  TP={tp_size}")
    print(f"timing:   {method} of per-occurrence max-rank DEVICE KERNEL DURATION; skip_first={skip_first}")
    print("=" * 108)
    print(
        f"{'projection':<12}{'baseline shape':<23}{'TP per-rank shape':<23}"
        f"{'partition':<15}{'base [us]':>11}{'TP crit [us]':>14}{'speedup':>10}{'samples':>10}"
    )
    print("-" * 108)

    base_projection_total = 0.0
    tp_projection_total = 0.0
    for name in projection_order:
        if name == "q_a+kv":
            q_ns, _, base_count, q_sample = _projection_critical(base, "q_a", method, skip_first, base_steps)
            kv_ns, _, _, kv_sample = _projection_critical(base, "kv", method, skip_first, base_steps)
            base_ns = q_ns + kv_ns
            base_calls = 2
            base_sample = (
                MatmulSample(
                    device=q_sample.device,
                    duration_ns=base_ns,
                    k=4096,
                    m=q_sample.m,
                    n=(q_sample.n + kv_sample.n),
                    batch=1,
                )
                if q_sample is not None and kv_sample is not None
                else None
            )
        else:
            base_ns, base_calls, base_count, base_sample = _projection_critical(
                base, name, method, skip_first, base_steps
            )
        tp_ns, tp_calls, tp_count, tp_sample = _projection_critical(tp, name, method, skip_first, tp_steps)
        if base_sample is None or tp_sample is None:
            missing = "baseline" if base_sample is None else "TP"
            print(f"{name:<12}{f'missing in {missing}':<61}")
            continue
        status = _partition_status(base_sample, tp_sample, tp_size, name)
        print(
            f"{name:<12}{base_sample.shape:<23}{tp_sample.shape:<23}{status:<15}"
            f"{_fmt_us(base_ns):>11}{_fmt_us(tp_ns):>14}{_fmt_speedup(base_ns, tp_ns):>10}"
            f"{f'{base_calls}:{tp_calls}':>10}"
        )
        base_projection_total += base_ns
        tp_projection_total += tp_ns

    print("-" * 108)
    print(
        f"{'PROJECTIONS':<73}{_fmt_us(base_projection_total):>11}"
        f"{_fmt_us(tp_projection_total):>14}{_fmt_speedup(base_projection_total, tp_projection_total):>10}"
    )

    tp_collective_total = 0.0
    print()
    print(f"{'TP collective':<48}{'critical [us]':>16}{'samples':>12}")
    print("-" * 76)
    if not tp.collectives:
        print("none")
    for code, by_device in sorted(tp.collectives.items()):
        duration, calls, count = _critical_path_per_step(by_device, tp.devices, method, skip_first, tp_steps)
        tp_collective_total += duration
        print(f"{code:<48}{_fmt_us(duration):>16}{f'{calls}x{count}':>12}")
    print("-" * 76)
    print(f"{'COLLECTIVES':<48}{_fmt_us(tp_collective_total):>16}")

    tp_selected_total = tp_projection_total + tp_collective_total
    print()
    print(
        "Selected critical path (projections + collectives): "
        f"baseline={_fmt_us(base_projection_total)} us, "
        f"TP={_fmt_us(tp_selected_total)} us, "
        f"speedup={_fmt_speedup(base_projection_total, tp_selected_total)}"
    )
    print("Note: this selected total excludes norm, RoPE, cache, SDPA, and layout operations.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("baseline", type=Path, help="single-chip ops_perf_results_*.csv")
    parser.add_argument("tp", type=Path, nargs="+", help="one or more TP ops_perf_results_*.csv files")
    parser.add_argument(
        "--agg",
        choices=("mean", "median", "min", "max"),
        default="mean",
        help="aggregate critical-path durations across repeated calls (default: mean)",
    )
    parser.add_argument(
        "--skip-first",
        type=int,
        default=0,
        help="discard this many initial calls of every projection/collective",
    )
    args = parser.parse_args()

    if args.skip_first < 0:
        parser.error("--skip-first must be non-negative")
    for path in (args.baseline, *args.tp):
        if not path.is_file():
            parser.error(f"profile does not exist: {path}")

    baseline = read_profile(args.baseline)
    if len(baseline.devices) != 1:
        parser.error(f"baseline must contain exactly one device, found {len(baseline.devices)}: {baseline.devices}")
    for path in args.tp:
        profile = read_profile(path)
        if len(profile.devices) < 2:
            parser.error(f"TP profile must contain at least two devices, found {len(profile.devices)} in {path}")
        compare(baseline, profile, args.agg, args.skip_first)
    return 0


if __name__ == "__main__":
    sys.exit(main())
