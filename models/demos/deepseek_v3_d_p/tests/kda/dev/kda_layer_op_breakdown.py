# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-op device timeline of one Kimi-K3 KDA layer call from a Tracy ops CSV (dev tool).

Input: ``ops_perf_results*.csv`` of ``run_safe_pytest.sh --profile`` on
``test_layer_perf.py::test_synthetic_kimi_k3_perf``. One layer call is one period of the repeating op sequence.
Traced: the ops of one trace replay session (the first replay is dropped), paired across devices by session id
(every device must hold the same sessions); untraced: the last calls of each device's stream, each anchored on the
call's first ops (``_CALL_START``) and paired from the end. Every call must hold the same op sequence, else the tool
raises. Ops are attributed by execution-ordered position, and every op of the call is listed.

Ops of one call can overlap on a device (a program's firmware can start before its predecessor ends), so FW
durations are not additive. The table therefore places each op on the call's timeline: per (device, call),
``start``/``end`` are the op's FW start/end relative to the call's first FW start, converted from cycles with the
row's own ``FW duration / cycles`` ratio, and ``advance`` is how far the op moves the latest end so far (its share of
the call span; advances sum to the span minus the first op's start). Columns are medians over all (device, call)
samples; ``FW``/``kernel`` are ``DEVICE FW/KERNEL DURATION``; ``PM ideal`` is the op's performance model where it has
one. The call's device span is reported per device and as the slowest chip per call. Usage::

    python kda_layer_op_breakdown.py <ops csv> [--replay-session N] [--calls K] [--json out.json]
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict

_FW = "DEVICE FW DURATION [ns]"
_KERNEL = "DEVICE KERNEL DURATION [ns]"
_O2O = "OP TO OP LATENCY [ns]"
_IDEAL = "PM IDEAL [ns]"
_START = "DEVICE FW START CYCLE"
_END = "DEVICE FW END CYCLE"

# (op-code substring, label); the first match wins. Positional context is resolved in _labels.
_RULES = (
    ("IndexedEntryCopy", "state copy"),
    ("MeshPartition", "export SP partition"),  # profiles before the export copy selected the heads only
    ("MatmulReduceScatter", "output projection + TP reduce-scatter (fused)"),
    ("AllGather", "TP input all-gather"),
    ("ReduceScatter", "TP reduce-scatter"),
    ("QkvCausalConv1d", "qkv conv + SP history exchange"),
    ("RelayAffinePrefix", "recurrence: SP relay"),
    ("SelectFinalCarry", "recurrence: final carry select+broadcast"),
    ("RecurrentChunkScan", "recurrence: chunk scan"),
    ("PrepareChunkRecurrence", "recurrence: prepare"),
    ("ChainAffine", "recurrence: chain affine"),
    ("AffineExclusiveScan", "recurrence: exclusive scan"),
    ("EntryCorrection", "recurrence: entry correction"),
    ("SigmoidGatedRmsNorm", "sigmoid-gated RMS norm"),
    ("Matmul", "matmul"),
    ("Binary", "eltwise binary"),
    ("Unary", "eltwise unary"),
)


def _label(op_code: str) -> str:
    for needle, label in _RULES:
        if needle.lower() in op_code.lower():
            return label
    return "other"


def _labels(op_codes: list[str]) -> list[str]:
    labels = [_label(code) for code in op_codes]
    copies = [i for i, label in enumerate(labels) if label == "state copy"]
    # forward_slot: import recurrent, import convolution, ..., export recurrent, export convolution.
    if len(copies) == 4:
        for i, name in zip(
            copies, ("import recurrent", "import convolution", "export recurrent", "export convolution")
        ):
            labels[i] = name
    # SP-sharded slabs: the two import copies are followed by the SP all-gathers of the
    # recurrent and convolution shards, then the TP input all-gather. Profiles of the earlier export path also show
    # an export SP partition before each export copy; since then the export copies select the chip's heads themselves,
    # so a profile has only the four state copies above and no partition.
    gathers = [i for i, label in enumerate(labels) if label == "TP input all-gather"]
    if len(gathers) == 3:
        for i, name in zip(gathers, ("import SP all-gather recurrent", "import SP all-gather convolution")):
            labels[i] = name
    partitions = [i for i, label in enumerate(labels) if label == "export SP partition"]
    if len(partitions) == 2:
        for i, name in zip(partitions, ("export SP partition recurrent", "export SP partition convolution")):
            labels[i] = name
    # ttKDA.forward: the fused input projection, then (after the conv) the decay-gate linear, then its bias add.
    matmuls = [i for i, label in enumerate(labels) if label == "matmul"]
    for i, name in zip(matmuls, ("input projection (fused qkv/beta/gate cols)", "decay gate linear")):
        labels[i] = name
    adds = [i for i, label in enumerate(labels) if label == "eltwise binary"]
    if len(adds) == 1:
        labels[adds[0]] = "decay gate bias add"
    return labels


def _num(row: dict, key: str) -> float | None:
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return None


# A forward_slot call starts with its two import copies followed by an all-gather (the SP gather of the recurrent
# shard, or the TP input all-gather before the slabs were sharded). The export copy before the next call's imports is
# followed by two copies, not an all-gather, so it never matches.
_CALL_START = ("IndexedEntryCopy", "IndexedEntryCopy", "AllGather")


def _anchored_calls(device_rows: list[dict]) -> list[list[dict]]:
    """Split a device's untraced op stream into calls, each starting at a _CALL_START match.

    A call runs to the next match, and the last one to the end of the stream; the caller checks that every call
    holds the same op sequence, so a truncated or foreign tail raises instead of being mislabeled."""
    codes = [r["OP CODE"] for r in device_rows]
    starts = [
        i
        for i in range(len(codes) - len(_CALL_START) + 1)
        if all(needle in codes[i + k] for k, needle in enumerate(_CALL_START))
    ]
    if not starts:
        raise ValueError(f"no call start {_CALL_START} in the device's op stream")
    bounds = starts + [len(device_rows)]
    return [device_rows[begin:end] for begin, end in zip(bounds, bounds[1:])]


def _calls(rows: list[dict], replay_session: str | None, calls: int) -> tuple[dict[int, list[list[dict]]], str]:
    by_device: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_device[int(row["DEVICE ID"])].append(row)
    traced = any(row.get("METAL TRACE REPLAY SESSION ID") for row in rows)
    result: dict[int, list[list[dict]]] = {}
    if traced:
        # Pair calls across devices by replay session, never by list index: every device must hold the same sessions.
        sessions_by_device = {
            device: sorted(
                {r["METAL TRACE REPLAY SESSION ID"] for r in device_rows if r["METAL TRACE REPLAY SESSION ID"]},
                key=int,
            )
            for device, device_rows in by_device.items()
        }
        reference_device, sessions = next(iter(sessions_by_device.items()))
        for device, device_sessions in sessions_by_device.items():
            if device_sessions != sessions:
                missing = sorted(set(sessions) ^ set(device_sessions), key=int)
                raise ValueError(
                    f"device {device} replay sessions differ from device {reference_device}'s (symmetric difference "
                    f"{missing[:10]}{'...' if len(missing) > 10 else ''}): rows were dropped; refusing to pair calls"
                )
        if replay_session is not None:
            if replay_session not in sessions:
                raise ValueError(f"replay session {replay_session} not in the profile")
            chosen = [replay_session]
        else:
            chosen = sessions[1:] if len(sessions) > 1 else sessions  # drop the first replay (first-replay jitter)
            chosen = chosen[:calls]
        for device, device_rows in by_device.items():
            device_rows.sort(key=lambda r: float(r[_START] or 0))
            result[device] = [[r for r in device_rows if r["METAL TRACE REPLAY SESSION ID"] == s] for s in chosen]
    else:
        # Untraced: the last `calls` anchored calls of each device, paired from the end of the stream.
        anchored = {}
        for device, device_rows in by_device.items():
            device_rows.sort(key=lambda r: float(r[_START] or 0))
            anchored[device] = _anchored_calls(device_rows)
        counts = {device: len(device_calls) for device, device_calls in anchored.items()}
        if len(set(counts.values())) != 1:
            raise ValueError(f"devices hold different numbers of anchored calls: {counts}; refusing to pair calls")
        available = next(iter(counts.values()))
        if calls > available:
            raise ValueError(f"asked for {calls} calls but each device holds {available}")
        result = {device: device_calls[-calls:] for device, device_calls in anchored.items()}
    return result, "traced" if traced else "untraced"


def _timeline(call: list[dict]) -> list[tuple[float, float]]:
    """(start, end) of each op in ns relative to the call's first FW start, from the cycle counters."""
    ratios = [
        _num(r, _FW) / (float(r[_END]) - float(r[_START]))
        for r in call
        if _num(r, _FW) and float(r[_END]) > float(r[_START])
    ]
    ns_per_cycle = statistics.median(ratios)
    t0 = min(float(r[_START]) for r in call)
    return [((float(r[_START]) - t0) * ns_per_cycle, (float(r[_END]) - t0) * ns_per_cycle) for r in call]


def breakdown(csv_path: str, replay_session: str | None = None, calls: int = 3) -> dict:
    with open(csv_path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    per_device, mode = _calls(rows, replay_session, calls)
    first_device = next(iter(per_device))
    codes = [r["OP CODE"] for r in per_device[first_device][0]]
    for device, device_calls in per_device.items():
        for call in device_calls:
            if [r["OP CODE"] for r in call] != codes:
                raise ValueError(f"device {device}: a call's op sequence differs from device {first_device}'s")
    labels = _labels(codes)
    n = len(codes)
    samples = defaultdict(lambda: defaultdict(list))  # position -> metric -> values (us)
    span_by_call: dict[int, list[float]] = defaultdict(list)  # call index -> span per device
    spans = []
    for device_calls in per_device.values():
        for call_index, call in enumerate(device_calls):
            times = _timeline(call)
            latest = 0.0
            for i, (row, (start, end)) in enumerate(zip(call, times)):
                metrics = samples[i]
                metrics["start"].append(start / 1e3)
                metrics["end"].append(end / 1e3)
                metrics["advance"].append(max(0.0, end - latest) / 1e3)
                latest = max(latest, end)
                for key, column in (("fw", _FW), ("kernel", _KERNEL), ("o2o", _O2O), ("ideal", _IDEAL)):
                    value = _num(row, column)
                    if value is not None and not (key == "o2o" and i == 0):
                        metrics[key].append(value / 1e3)
            spans.append(latest / 1e3)
            span_by_call[call_index].append(latest / 1e3)

    def med(values: list[float]) -> float | None:
        return statistics.median(values) if values else None

    ops = [
        {
            "position": i,
            "op_code": codes[i],
            "label": labels[i],
            **{f"{key}_median_us": med(samples[i][key]) for key in ("start", "end", "advance", "fw", "kernel", "o2o")},
            "fw_max_us": max(samples[i]["fw"]) if samples[i]["fw"] else None,
            "pm_ideal_median_us": med(samples[i]["ideal"]),
        }
        for i in range(n)
    ]
    slowest_chip = [max(values) for values in span_by_call.values()]
    return {
        "csv": csv_path,
        "mode": mode,
        "devices": len(per_device),
        "calls_per_device": len(next(iter(per_device.values()))),
        "launches_per_call": n,
        "span_us_median_over_device_calls": statistics.median(spans),
        "span_us_slowest_chip_median_over_calls": statistics.median(slowest_chip),
        "span_us_max": max(spans),
        "ops": ops,
    }


def _print(result: dict) -> None:
    print(
        f"{result['mode']}: {result['devices']} devices x {result['calls_per_device']} calls, "
        f"{result['launches_per_call']} launches/call; device span median {result['span_us_median_over_device_calls']:.1f} us, "
        f"slowest chip per call median {result['span_us_slowest_chip_median_over_calls']:.1f} us, max {result['span_us_max']:.1f} us"
    )
    header = ("#", "label", "start", "end", "advance", "FW", "FW max", "kernel", "o2o", "PM ideal")
    print(f"{header[0]:>2} {header[1]:46} " + " ".join(f"{h:>8}" for h in header[2:]))
    for op in result["ops"]:
        values = (
            op["start_median_us"],
            op["end_median_us"],
            op["advance_median_us"],
            op["fw_median_us"],
            op["fw_max_us"],
            op["kernel_median_us"],
            op["o2o_median_us"],
            op["pm_ideal_median_us"],
        )
        cells = " ".join(f"{v:8.1f}" if v is not None else f"{'-':>8}" for v in values)
        print(f"{op['position']:>2} {op['label'][:46]:46} {cells}")
    print("op codes: " + ", ".join(f"{op['position']}={op['op_code']}" for op in result["ops"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv")
    parser.add_argument("--replay-session", default=None)
    parser.add_argument("--calls", type=int, default=3)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    result = breakdown(args.csv, args.replay_session, args.calls)
    _print(result)
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(result, handle, indent=2)


if __name__ == "__main__":
    main()
