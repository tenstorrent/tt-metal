# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CPU-only completeness checks for stage-local native profiler evidence."""

import csv
import json
from collections import Counter

RANKS = {"0", "1", "2", "3"}
TRACE = "METAL TRACE ID"
SESSION = "METAL TRACE REPLAY SESSION ID"
UID_FIELDS = ("DEVICE ID", "GLOBAL CALL COUNT", TRACE, SESSION)
LOSS_MESSAGES = (
    "instrumentation failure:",
    "markers were dropped",
    "report will be partial",
    "skipping op with malformed json",
    "skipping cached op reference with no prior data",
    "message discarded",
    "device op(s) with no matching host data",
)
PHASE_COMPONENTS = {
    "token_out": ("model", "sampling"),
    "model_only": ("model",),
    "sampler_only": ("sampling",),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def capture_health(artifact):
    capture = artifact / "capture.log"
    require(capture.is_file() and capture.stat().st_size > 0, "Missing/nonempty capture.log required")
    findings = []
    for path in (capture, artifact / "postprocess.log"):
        if not path.is_file():
            continue
        with path.open(errors="replace") as stream:
            for line_number, line in enumerate(stream, 1):
                if any(message in line.lower() for message in LOSS_MESSAGES):
                    findings.append(f"{path.name}:{line_number}: {line.strip()[:240]}")
    require(not findings, "Incomplete profiler capture/conversion:\n" + "\n".join(findings[:20]))


def execution_uid(row):
    try:
        values = tuple(int(row[field]) for field in UID_FIELDS)
    except (ValueError, TypeError, KeyError) as error:
        raise ValueError(f"Invalid native execution UID: {[row.get(field) for field in UID_FIELDS]}") from error
    require(all(value >= 0 for value in values), f"Negative native execution UID: {values}")
    return values


def validate_executions(windows, native_rows, traces, *, steps=16):
    """Compare native and converted UIDs, including operators absent in every replay."""
    converted = Counter()
    sessions_by_phase = {}
    wanted_sessions = set()
    for phase, names in PHASE_COMPONENTS.items():
        rows = windows[phase]
        expected_traces = {str(traces[name]) for name in names}
        require({row["DEVICE ID"] for row in rows} == RANKS, f"{phase}: wrong rank set")
        require({row[TRACE] for row in rows} == expected_traces, f"{phase}: wrong trace set")
        sessions_by_phase[phase] = {}
        for name in names:
            per_rank = {}
            for rank in sorted(RANKS):
                chosen = [row for row in rows if row["DEVICE ID"] == rank and row[TRACE] == str(traces[name])]
                uids = [execution_uid(row) for row in chosen]
                sessions = {uid[3] for uid in uids}
                require(len(sessions) == steps and min(sessions) > 0, f"{phase}/{name}/{rank}: wrong sessions")
                per_rank[rank] = sorted(sessions)
                converted.update(uids)
                wanted_sessions.update((uid[0], uid[2], uid[3]) for uid in uids)
            require(len({tuple(value) for value in per_rank.values()}) == 1, f"{phase}/{name}: rank session mismatch")
            sessions_by_phase[phase][name] = per_rank
    duplicates = [uid for uid, count in converted.items() if count != 1]
    require(not duplicates, f"Duplicate converted execution UIDs: {duplicates[:20]}")
    native = Counter()
    for row in native_rows:
        if row.get(TRACE) in (None, "", "None", "N/A") or row.get(SESSION) in (None, "", "None", "N/A"):
            continue
        uid = execution_uid(row)
        if (uid[0], uid[2], uid[3]) in wanted_sessions:
            native[uid] += 1
    duplicates = [uid for uid, count in native.items() if count != 1]
    require(not duplicates, f"Duplicate native execution UIDs: {duplicates[:20]}")
    missing, extra = native - converted, converted - native
    require(
        not missing and not extra,
        f"Execution coverage mismatch: missing converted={list(missing)[:20]}; absent native={list(extra)[:20]}",
    )
    return {"native_execution_count": sum(native.values()), "sessions": sessions_by_phase}


def program_key(row):
    try:
        return int(row["DEVICE ID"]), int(row["PROGRAM HASH"])
    except (ValueError, TypeError, KeyError) as error:
        raise ValueError(f"Missing device/program hash for {row.get('OP CODE')}") from error


def load_program_metadata(path, wanted):
    """Read actual JSON, not the native CSV's last-compute-kernel fidelity scalar."""
    found = {}
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream, delimiter=";", quotechar="`")
        require("MessageName" in (reader.fieldnames or []), "Missing native MessageName column")
        for row in reader:
            message = row.get("MessageName") or ""
            if "TT_DNN_DEVICE_OP" not in message or " ->\n" not in message:
                continue
            try:
                data = json.loads(message.split(" ->\n", 1)[1])
                key = int(data["device_id"]), int(data["op_hash"])
            except (ValueError, TypeError, KeyError) as error:
                raise ValueError("Malformed native device-op metadata; inspect tracy_ops_data.csv") from error
            if key not in wanted:
                continue
            # Keep all actual operands, kernel records, attributes and precision metadata.
            metadata = {
                name: value for name, value in data.items() if name not in ("global_call_count", "program_cache_hit")
            }
            if key in found:
                for field in ("op_code", "input_tensors", "output_tensors", "kernel_info"):
                    require(found[key].get(field) == metadata.get(field), f"Ambiguous program metadata: {key}/{field}")
            else:
                found[key] = metadata
    require(not wanted - found.keys(), f"Missing native program metadata: {sorted(wanted - found.keys())[:20]}")
    return found


def validate_metadata(rows, programs):
    findings = []
    for key in sorted({program_key(row) for row in rows}):
        data = programs.get(key)
        if data is None:
            findings.append(f"{key}: missing native JSON")
            continue
        label = f"{key}/{data.get('op_code', '?')}"
        for collection in ("input_tensors", "output_tensors"):
            tensors = data.get(collection)
            if not isinstance(tensors, list):
                findings.append(f"{label}: missing {collection}")
                continue
            if data.get(collection + "_original_count", len(tensors)) != len(tensors):
                findings.append(f"{label}: truncated {collection}")
            for index, tensor in enumerate(tensors):
                shape = tensor.get("shape", {})
                storage = tensor.get("storage_type")
                valid_storage = isinstance(storage, str) and bool(storage)
                if isinstance(storage, dict):
                    memory = storage.get("memory_config", {})
                    valid_storage = "device_id" in storage and all(
                        memory.get(k) for k in ("buffer_type", "memory_layout")
                    )
                if not (
                    tensor.get("dtype") and tensor.get("layout") and all(shape.get(k) for k in "WZYX") and valid_storage
                ):
                    findings.append(f"{label}: incomplete {collection}[{index}] dtype/layout/shape/storage")
        kernels = data.get("kernel_info") or {}
        compute, movement = kernels.get("compute_kernels"), kernels.get("datamovement_kernels")
        if not isinstance(compute, list) or not isinstance(movement, list) or not (compute or movement):
            findings.append(f"{label}: missing native kernel inventory")
            continue
        for kernel in compute:
            if not all(kernel.get(field) for field in ("source", "name", "math_fidelity")):
                findings.append(f"{label}: compute kernel missing source/name/fidelity")
        for kernel in movement:
            if not all(kernel.get(field) for field in ("source", "name")):
                findings.append(f"{label}: data-movement kernel missing source/name")
        # An explicit empty compute list plus nonempty native DM kernels is the
        # evidence for no math fidelity; operation-name guesses are not exemptions.
    require(not findings, "Unsupported/incomplete native metadata; inspect:\n" + "\n".join(findings[:40]))
    for row in rows:
        data = programs[program_key(row)]
        require(row["OP CODE"] == data["op_code"], f"Operation name disagrees with native metadata: {program_key(row)}")
        for collection, prefix in (("input_tensors", "INPUT"), ("output_tensors", "OUTPUT")):
            for index, tensor in enumerate(data[collection]):
                storage = tensor["storage_type"]
                memory = storage
                if isinstance(storage, dict):
                    config = storage["memory_config"]
                    memory = (
                        f"DEV_{storage['device_id']}_{config['buffer_type'].upper()}_{config['memory_layout'].upper()}"
                    )
                expected = {"DATATYPE": tensor["dtype"], "LAYOUT": tensor["layout"], "MEMORY": memory}
                expected.update({f"{axis}_PAD[LOGICAL]": tensor["shape"][axis] for axis in "WZYX"})
                for field, value in expected.items():
                    require(
                        row.get(f"{prefix}_{index}_{field}") == value,
                        f"Missing/mismatched converted {field}: {program_key(row)}/{prefix}_{index}",
                    )
    return {f"{key[0]}:{key[1]}": value for key, value in sorted(programs.items())}


def fidelities(row, programs):
    kernels = programs[program_key(row)]["kernel_info"]["compute_kernels"]
    return json.dumps(sorted({kernel["math_fidelity"] for kernel in kernels}))
