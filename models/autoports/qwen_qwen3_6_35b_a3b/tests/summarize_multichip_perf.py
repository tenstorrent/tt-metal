# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Summarize four-chip profiler evidence without treating aggregate work as latency.

Reads tracy/layer_{0,3}/{ops.csv,correctness.json,*_perf_report.csv} and
preserves the external tt-perf-report human tables. All validation precedes writes.
Run with Python only: this module never imports TTNN or accesses devices.
Use --batch 32 --doc <separate-root> for batched captures; the default B1
artifact root and summary remain unchanged. Each phase records its exact scope.
"""

import argparse
import csv
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

DOC = Path(__file__).parents[1] / "doc/multichip_decoder"
DEVICES = {"0", "1", "2", "3"}
REPLAYS = 16
BATCH_SCOPE = "one complete batch; not divided by batch size"
SEEDED_SCOPE = {"prefill": "one representative request from batch 32", "decode": BATCH_SCOPE}
DURATION = "DEVICE KERNEL DURATION [ns]"
SESSION = "METAL TRACE REPLAY SESSION ID"
GROUP_KEYS = (
    "OP CODE",
    "INPUT_0_DATATYPE",
    "INPUT_1_DATATYPE",
    "MATH FIDELITY",
    "INPUT_0_Y_PAD[LOGICAL]",
    "INPUT_0_X_PAD[LOGICAL]",
    "INPUT_1_Y_PAD[LOGICAL]",
    "INPUT_1_X_PAD[LOGICAL]",
    "INPUT_0_MEMORY",
    "INPUT_1_MEMORY",
    "CORE COUNT",
)
CATEGORIES = ("communication", "dram_matmul", "compute", "data_movement")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value, label, *, positive=False):
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label}: missing/invalid number {value!r}") from error
    require(
        math.isfinite(parsed) and (parsed > 0 if positive else parsed >= 0),
        f"{label}: invalid duration/value {value!r}",
    )
    return parsed


def read_csv(path, required):
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        require(set(required) <= set(fields), f"{path}: missing columns {set(required) - set(fields)}")
        rows = list(reader)
    require(rows, f"{path}: empty CSV")
    return rows, fields


def category(row):
    code = re.sub(r"[^a-z0-9]", "", row["OP CODE"].lower())
    if code in ("dispatchdeviceoperation", "combinedeviceoperation", "typecastdeviceoperation"):
        # Selected dispatch/combine use a size-one mesh axis for local expert
        # routing. PostCombineReduce is arithmetic and deliberately excluded.
        return "data_movement"
    # Collective+matmul fusions remain one indivisible measured communication
    # operation. Their compute time cannot be split out from this CSV.
    if any(word in code for word in ("allgather", "allreduce", "reducescatter", "alltoall", "ccl", "sendrecv")):
        return "communication"
    if "matmul" in code or "bmm" in code:
        return "dram_matmul" if "DRAM" in row.get("INPUT_1_MEMORY", "").upper() else "compute"
    if "sdpa" in code:
        # 'sdpadecode' contains the substring 'pad', but attention is compute.
        return "compute"
    if any(
        word in code
        for word in (
            "copy",
            "reshape",
            "permute",
            "transpose",
            "tilize",
            "sharded",
            "slice",
            "concat",
            "gather",
            "scatter",
            "pad",
            "repeat",
            "split",
            "view",
            "updatecache",
            "fillcache",
        )
    ):
        return "data_movement"
    return "compute"


def normal(value):
    return re.sub(r"DEV_\d+", "DEV_RANK", value or "")


def signature(row, fields):
    # Include all recorded operand/output shapes, dtypes, layouts and memories,
    # plus program hash if present. Exclude addresses, timestamps and replay IDs.
    keys = [k for k in fields if k.startswith(("INPUT_", "OUTPUT_"))]
    keys += [k for k in ("OP CODE", "MATH FIDELITY", "CORE COUNT", "PROGRAM HASH") if k in fields]
    return tuple((key, normal(row.get(key))) for key in keys)


def window(rows, phase):
    begin, end = "PERF_" + phase.upper(), "PERF_" + phase.upper() + "_END"
    starts = [i for i, row in enumerate(rows) if row["OP TYPE"] == "signpost" and row["OP CODE"] == begin]
    ends = [i for i, row in enumerate(rows) if row["OP TYPE"] == "signpost" and row["OP CODE"] == end]
    require(
        len(starts) == len(ends) == 1 and starts[0] < ends[0],
        f"{phase}: expected one ordered start/end signpost pair; got {starts}, {ends}",
    )
    selected = [row for row in rows[starts[0] + 1 : ends[0]] if row["OP TYPE"] == "tt_dnn_device"]
    require(selected, f"{phase}: empty device window")
    require(
        {r["DEVICE ID"] for r in selected} == DEVICES,
        f"{phase}: expected device IDs 0,1,2,3; got {sorted({r['DEVICE ID'] for r in selected})}",
    )
    for index, row in enumerate(selected):
        number(row.get(DURATION), f"{phase} row {index} device {row['DEVICE ID']} duration", positive=True)
    return selected


def validate_correctness(path, layer, batch=1):
    result = json.loads(path.read_text())
    mesh = result.get("multichip", {})
    require(
        result.get("passed") is True and result.get("layer") == layer,
        f"{path}: missing passing result for layer {layer}",
    )
    require(result.get("batch") == batch, f"{path}: expected batch {batch}, got {result.get('batch')}")
    fixture = result.get("profile_fixture")
    require(fixture in (None, "seeded_b32_v1"), f"{path}: unknown profile fixture")
    seeded = fixture == "seeded_b32_v1"
    require(not seeded or batch == 32, f"{path}: seeded fixture requires batch 32")
    steps = 2 if seeded else (8 if batch == 32 else 2)
    require(
        result.get("length") == 32 and result.get("decode_steps") == steps,
        f"{path}: expected length 32 and {steps} decode correctness steps",
    )
    if batch == 32:
        activation = result.get("activation_file")
        require(
            isinstance(activation, str)
            and Path(activation).name == "natural_b32.pt"
            and result.get("activation_source") == activation,
            f"{path}: batch 32 requires recorded natural_b32.pt activations",
        )
    require(
        mesh.get("decoder") == "MultichipDecoder" and mesh.get("deterministic") is True,
        f"{path}: expected deterministic MultichipDecoder evidence",
    )
    require(number(result.get("prefill_pcc"), f"{path} prefill PCC") >= 0.995, f"{path}: prefill PCC failed")
    decode = result.get("decode_pcc")
    require(isinstance(decode, list) and len(decode) == steps, f"{path}: missing decode PCC steps")
    require(all(number(p, f"{path} decode PCC") >= 0.995 for p in decode), f"{path}: decode PCC failed")
    if seeded:
        require(result.get("phase_scope") == SEEDED_SCOPE, f"{path}: seeded phase scope mismatch")
        require(result.get("step0_eager_trace_equal") is True, f"{path}: missing seeded eager/trace control")
        require(
            result.get("replay_seed_mesh_equal") == [True, True]
            and all(value is True for value in result["replay_seed_mesh_equal"]),
            f"{path}: missing two seeded replay equality controls",
        )
        require(
            mesh.get("kind") == ("linear_attention" if layer == 0 else "full_attention")
            and mesh.get("residual_layout") == "replicated"
            and mesh.get("output_replicas_equal") is True,
            f"{path}: seeded decoder layout/kind mismatch",
        )
        require(isinstance(result.get("context"), int) and result["context"] >= 34, f"{path}: invalid context")
        for field in ("seed_artifact", "seed_acceptance"):
            require(isinstance(result.get(field), str) and result[field], f"{path}: missing {field}")
        checks = result.get("prefill_state_upload_checks")
        require(isinstance(checks, list) and len(checks) == 256, f"{path}: missing seeded state upload checks")
        keys = ("conv", "recurrent") if layer == 0 else ("k", "v")
        expected = {(row, key, rank) for row in range(32) for key in keys for rank in range(4)}
        require(
            all(isinstance(check, dict) and check.get("bit_equal") is True for check in checks)
            and {(check.get("row"), check.get("tensor"), check.get("rank")) for check in checks} == expected,
            f"{path}: incomplete/failed seeded state upload roundtrips",
        )
    else:
        checks = result.get("state_checks")
        require(isinstance(checks, list) and checks, f"{path}: missing state correctness checks")
        require(
            all(number(row.get("pcc"), f"{path} state PCC") >= 0.995 for row in checks), f"{path}: state PCC failed"
        )
    require(mesh.get("runtime_fallback_audit") == "pass", f"{path}: runtime fallback audit missing/failed")
    require(
        len(mesh.get("decode_samples_ms", [])) == REPLAYS,
        f"{path}: expected 16 serialized warmed decode samples from the profile runner",
    )
    for sample in mesh["decode_samples_ms"]:
        number(sample, f"{path} warmed decode sample", positive=True)
    require(
        result.get("trace_allocation_tracking") is False,
        f"{path}: profile must be separate from trace-allocation-tracking evidence",
    )
    return result


def summarize_phase(directory, rows, fields, phase, *, duration_scope=BATCH_SCOPE):
    selected = window(rows, phase)
    repetitions = REPLAYS if phase == "decode" else 1
    by_device = defaultdict(list)
    for row in selected:
        by_device[row["DEVICE ID"]].append(row)
    samples, counts, replay_ids, groups = {}, {}, {}, []
    for device in sorted(DEVICES):
        device_rows = by_device[device]
        if phase == "decode":
            require(SESSION in fields, f"{directory}: no replay session column")
            sessions = defaultdict(list)
            for row in device_rows:
                session = (row.get(SESSION) or "").strip()
                require(session not in ("", "-", "None", "N/A"), f"{directory}: missing device {device} session ID")
                sessions[session].append(row)
            require(
                len(sessions) == REPLAYS,
                f"{directory}: device {device} has {len(sessions)} replay sessions, expected 16",
            )
            passes = list(sessions.values())
            replay_ids[device] = list(sessions)
            reference = [signature(row, fields) for row in passes[0]]
            for ordinal, replay in enumerate(passes[1:], 1):
                require(
                    [signature(row, fields) for row in replay] == reference,
                    f"{directory}: device {device} replay {ordinal} op signature/count mismatch",
                )
        else:
            passes = [device_rows]
        counts[device] = len(passes[0])
        samples[device] = [
            sum(number(row[DURATION], "duration", positive=True) for row in replay) / 1e6 for replay in passes
        ]
        grouped = defaultdict(lambda: [0, 0.0])
        for row in device_rows:
            key = (category(row), *(row.get(k, "") for k in GROUP_KEYS))
            grouped[key][0] += 1
            grouped[key][1] += float(row[DURATION]) / 1000
        for key, (count, duration) in grouped.items():
            groups.append(
                dict(
                    device_id=device,
                    category=key[0],
                    **dict(zip(GROUP_KEYS, key[1:])),
                    count_per_pass=count / repetitions,
                    kernel_us_per_pass=duration / repetitions,
                )
            )
    # Rank endpoints may use different kernels/core counts. Their repeat-local
    # signatures are checked independently; no rank equivalence is assumed.
    means = {d: statistics.mean(values) for d, values in samples.items()}
    rank = max(means, key=means.get)
    # Blocking trace replays are aligned by ordinal within each rank. Session
    # identifiers can be global or rank-local, so preserve both mappings.
    maxima = [max(samples[d][i] for d in DEVICES) for i in range(repetitions)]
    totals_by_category = {d: {c: 0.0 for c in CATEGORIES} for d in sorted(DEVICES)}
    for row in groups:
        totals_by_category[row["device_id"]][row["category"]] += row["kernel_us_per_pass"] / 1000
    report_path = directory / f"{phase}_perf_report.csv"
    report, _ = read_csv(report_path, ("Device", "OP Code", "Device Time [μs]", "Advice"))
    # tt-perf-report merges ranks into representative logical-op rows. A
    # selected Device is not an independent rank workload and may omit ranks.
    require({r["Device"] for r in report} <= DEVICES, f"{report_path}: unexpected representative device ID")
    for row in report:
        number(row["Device Time [μs]"], f"{report_path} device time", positive=True)
    human = directory / f"{phase}_perf_report.txt"
    require(human.is_file() and human.stat().st_size > 0, f"{human}: missing human-readable report")
    advice = [
        dict(op_code=code, advice=message, merged_op_rows=count)
        for (code, message), count in sorted(
            Counter((r["OP Code"], r["Advice"].strip()) for r in report if r["Advice"].strip()).items()
        )
    ]
    report_details = dict(
        scope="merged representative logical-op rows; not per-device workloads",
        merge_semantics="maximum-duration rank for noncollectives; mean across ranks for AllGather/ReduceScatter/AllReduce, labeled with the first rank",
        rows=len(report),
        bound_counts=dict(Counter(r.get("Bound", "") for r in report)),
        slow_ops=sorted({r["OP Code"] for r in report if "SLOW" in (r.get("Bound", "") + r.get("Advice", "")).upper()}),
        timing_note="Report durations are not summed for mesh latency or rank work; Total percent and op gaps can include profiler dumps between trace replays",
    )
    groups.sort(key=lambda r: (r["device_id"], -r["kernel_us_per_pass"]))
    return (
        dict(
            duration_scope=duration_scope,
            repetitions=repetitions,
            device_ops_per_pass=counts,
            per_device_kernel_sum_ms_per_pass=means,
            per_device_kernel_sum_ms_samples=samples,
            critical_rank=rank,
            critical_rank_kernel_sum_bound_ms=means[rank],
            per_replay_critical_rank_kernel_sum_bound_ms=maxima,
            mean_per_replay_critical_rank_kernel_sum_bound_ms=statistics.mean(maxima),
            median_per_replay_critical_rank_kernel_sum_bound_ms=statistics.median(maxima),
            per_device_category_kernel_sum_ms=totals_by_category,
            replay_ids_by_device=replay_ids,
            advice=advice,
            perf_report_details=report_details,
            compact_csv=f"tracy/{directory.name}/{phase}_op_groups.csv",
            report_csv=f"tracy/{directory.name}/{phase}_perf_report.csv",
            human_table=f"tracy/{directory.name}/{phase}_perf_report.txt",
        ),
        groups,
    )


def summarize(doc, batch=1):
    require(batch in (1, 32), "batch must be 1 or 32")
    require(batch == 1 or doc.resolve() != DOC.resolve(), "batch 32 requires a separate --doc artifact root")
    measurements, outputs = [], []
    for layer in (0, 3):
        directory = doc / f"tracy/layer_{layer}"
        result = validate_correctness(directory / "correctness.json", layer, batch)
        if result.get("profile_fixture") == "seeded_b32_v1":
            from models.autoports.qwen_qwen3_6_35b_a3b.tests.summarize_multichip_validation import Gate

            Gate(doc).profile_seed(result, directory / "correctness.json", layer)
        rows, fields = read_csv(directory / "ops.csv", ("OP CODE", "OP TYPE", "DEVICE ID", DURATION))
        phases = {}
        for phase in ("prefill", "decode"):
            scope = result.get("phase_scope", {}).get(phase, BATCH_SCOPE)
            phases[phase], groups = summarize_phase(directory, rows, fields, phase, duration_scope=scope)
            outputs.append((directory / f"{phase}_op_groups.csv", groups))
        measurements.append(
            dict(
                layer=layer,
                batch=batch,
                duration_scope=("phase-specific; see phases" if result.get("profile_fixture") else BATCH_SCOPE),
                profile_fixture=result.get("profile_fixture"),
                seed_artifact=result.get("seed_artifact"),
                seed_acceptance=result.get("seed_acceptance"),
                activation_source=result.get("activation_source"),
                kind=result["multichip"].get("kind"),
                phases=phases,
                correctness=f"tracy/layer_{layer}/correctness.json",
                ops_csv=f"tracy/layer_{layer}/ops.csv",
                precision_policy=result["multichip"].get("precision_policy"),
                profiled_run_host_warmed_decode_ms=result["multichip"].get("decode_traced_warmed_ms"),
                host_timing_note="Profiler-enabled run, separate warmed loop; not a clean latency or the signpost replay duration",
                roofline=None,
                roofline_note="Not inferred: generic inline kernels, padding, rereads and collectives prevent an exact byte model from this schema",
            )
        )
    summary = dict(
        schema_version=1,
        batch=batch,
        devices=sorted(DEVICES),
        expected_decode_replays=REPLAYS,
        accounting="max over ranks of per-rank mean kernel-duration sums; never sum chips as latency",
        timing_limitations=[
            "Critical-rank kernel sum is a serialized-work bound/proxy, not measured mesh latency or throughput",
            "Device durations omit dispatch gaps and may overlap; therefore no unconditional elapsed-time lower/upper bound is asserted",
            "Per-replay maxima align each rank's 16 blocking replays by ordinal; original session IDs are retained",
            "Fused collective+compute ops are charged once to communication; their internal compute cannot be separated",
            "DRAM matmul classification means a matmul weight operand resides in DRAM, not proof of bandwidth saturation",
            "Inline GenericOp work remains compute unless its op name identifies movement; consult human reports/kernel sources",
            "No roofline, clean-host speedup or full-model throughput is inferred",
            "tt-perf-report advice/bounds describe merged logical-op representatives, not per-rank work; collectives use mean-rank durations",
        ],
        measurements=measurements,
    )
    # Do not leave half-written new reports when a later layer gate fails.
    for path, groups in outputs:
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=["device_id", "category", *GROUP_KEYS, "count_per_pass", "kernel_us_per_pass"],
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(groups)
    (doc / "perf_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doc", type=Path, default=DOC)
    parser.add_argument("--batch", type=int, choices=(1, 32), default=1)
    args = parser.parse_args()
    summary = summarize(args.doc, args.batch)
    for row in summary["measurements"]:
        print(
            f"layer {row['layer']} batch {args.batch}: decode critical-rank kernel-work sum "
            f"{row['phases']['decode']['critical_rank_kernel_sum_bound_ms']:.6f}ms; "
            f"four devices, 16 complete replay signatures"
        )
    print(args.doc / "perf_summary.json")


if __name__ == "__main__":
    main()
