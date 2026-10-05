# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Multi-pass perf counter capture: group tables, pass scheduling (one L1 bank per pass, the BRISC firmware
fits three groups), per-pass workload replay and the device log merge. tracy/__main__.py only calls in here."""

import hashlib
import json
import math
import re
import tempfile
import os
from pathlib import Path
from shutil import copyfile

from loguru import logger
from tracy.common import PROFILER_DEVICE_SIDE_LOG, generate_logs_folder

# Bit positions match PROFILE_PERF_COUNTERS_* in tt_metal/tools/profiler/perf_counters.hpp.
# l1_2 to l1_5 are Blackhole-only (its L1 has more client ports behind the mux).
PERF_COUNTER_GROUP_BITS = {
    "fpu": 0,
    "pack": 1,
    "unpack": 2,
    "l1_0": 3,
    "l1_1": 4,
    "instrn": 5,
    "l1_2": 6,
    "l1_3": 7,
    "l1_4": 8,
    "l1_5": 9,
}
PERF_COUNTER_L1_GROUPS = {"l1_0", "l1_1", "l1_2", "l1_3", "l1_4", "l1_5"}
PERF_COUNTER_BH_ONLY_GROUPS = {"l1_2", "l1_3", "l1_4", "l1_5"}
# Measured on Blackhole BRISC firmware: 3 groups of readout code fit, 4 overflow .text.
PERF_COUNTER_MAX_GROUPS_PER_PASS = 3
# PERF_COUNTER_PROFILER_ID in perf_counters.hpp: the timer_id the firmware tags counter rows with.
PERF_COUNTER_MARKER_ID = "9090"
# Device-log column positions used when merging passes.
PERF_COUNTER_TIMER_ID_COL = 4
PERF_COUNTER_TIMESTAMP_COL = 5
PERF_COUNTER_RUN_HOST_ID_COL = 7
PERF_COUNTER_TRACE_ID_COL = 8
PERF_COUNTER_TRACE_ID_COUNTER_COL = 9
# Environment variables that name the device architecture without opening the device.
ARCH_ENV_VARS = ("TT_METAL_DEVICE_ARCH", "TT_ARCH_NAME", "ARCH_NAME")


def schedule_perf_counter_passes(requested_groups, max_groups_per_pass=PERF_COUNTER_MAX_GROUPS_PER_PASS):
    """Split counter groups into passes: at most one L1 bank (shared mux) and max_groups_per_pass groups
    (BRISC firmware fit) per pass. Returns a list of passes, each an ordered list of group names."""
    seen = list(dict.fromkeys(g.lower() for g in requested_groups))  # dedup, preserve order
    l1 = [g for g in seen if g in PERF_COUNTER_L1_GROUPS]
    non_l1 = [g for g in seen if g not in PERF_COUNTER_L1_GROUPS]
    total = len(l1) + len(non_l1)
    if total == 0:
        return []
    # Enough passes to give every L1 bank its own pass AND keep each pass within the group cap.
    num_passes = max(len(l1), math.ceil(total / max_groups_per_pass))
    passes = [[] for _ in range(num_passes)]
    for i, g in enumerate(l1):  # one L1 bank per pass
        passes[i].append(g)
    for g in non_l1:  # fill remaining slots, least-full pass first
        target = min((p for p in passes if len(p) < max_groups_per_pass), key=len)
        target.append(g)
    return [p for p in passes if p]


def arch_l1_groups(is_blackhole, is_quasar=False):
    """L1 counter groups an architecture has: Blackhole's 2-NOC L1 exposes banks 2-5 as well. Quasar has no
    tt_perf_cnt bank on its L1 (its l1_client event counter is selected with TT_METAL_PROFILE_PERF_COUNTERS_L1_SEL)."""
    if is_quasar:
        return []
    return ["l1_0", "l1_1", "l1_2", "l1_3", "l1_4", "l1_5"] if is_blackhole else ["l1_0", "l1_1"]


def perf_counter_groups_to_bitfield(groups):
    """OR the PROFILE_PERF_COUNTERS_* bits for a list of group names."""
    bits = 0
    for g in groups:
        bits |= 1 << PERF_COUNTER_GROUP_BITS[g.lower()]
    return bits


def detect_device_arch():
    """The device architecture name in lower case, from the environment or by opening device 0; None if unknown."""
    declared = next((os.environ.get(v) for v in ARCH_ENV_VARS if os.environ.get(v)), None)
    if declared is None:
        try:
            import ttnn

            device = ttnn.open_device(device_id=0)
            declared = str(device.arch()).split(".")[-1]
            ttnn.close_device(device)
        except Exception:
            logger.debug("Failed to detect device arch via ttnn")
    return declared.strip().lower() if declared is not None else None


def resolve_perf_counter_groups(requested_groups, arch):
    """Ordered, deduplicated groups to capture; ``all`` is the architecture's full set. Groups the architecture
    does not have (l1_* on Quasar, banks 2-5 off Blackhole) raise ValueError."""
    is_blackhole = arch == "blackhole"
    is_quasar = arch == "quasar"
    if arch is None and any(g.lower() == "all" for g in requested_groups):
        raise ValueError(
            "Cannot resolve counter group 'all' without the device architecture (detection failed); "
            f"set {' or '.join(ARCH_ENV_VARS)}, or list the groups explicitly."
        )
    resolved = []
    for group in requested_groups:
        g = group.lower()
        if g == "sfpu":
            g = "fpu"
        if g == "all":
            resolved.extend(["fpu", "pack", "unpack", "instrn"] + arch_l1_groups(is_blackhole, is_quasar))
        elif g in PERF_COUNTER_GROUP_BITS:
            resolved.append(g)
        else:
            raise ValueError(
                f"Unknown counter group '{group}'. Valid groups: {', '.join(PERF_COUNTER_GROUP_BITS)}, sfpu, all"
            )
    resolved = list(dict.fromkeys(resolved))

    if is_quasar and (set(resolved) & PERF_COUNTER_L1_GROUPS):
        raise ValueError(
            "Quasar has no L1 performance counter bank; drop the l1_* groups and use "
            "TT_METAL_PROFILE_PERF_COUNTERS_L1_SEL for the l1_client event counter."
        )
    bh_only = sorted(set(resolved) & PERF_COUNTER_BH_ONLY_GROUPS)
    if bh_only and not is_blackhole:
        raise ValueError(
            f"Performance counter groups {', '.join(bh_only)} are supported only on Blackhole, "
            f"but device arch is {arch or 'undeclared'}."
        )
    return resolved


def describe_passes(passes):
    return "\n".join(
        f"  pass {i + 1}: {', '.join(p)}  (bitfield {perf_counter_groups_to_bitfield(p)})" for i, p in enumerate(passes)
    )


def plan_perf_counter_capture(requested_groups, multipass, can_replay):
    """Per-pass bitfields for a counter request. One pass is also exported via TT_METAL_PROFILE_PERF_COUNTERS;
    several passes need ``multipass`` and ``can_replay`` (this process launches the workload), else ValueError."""
    resolved = resolve_perf_counter_groups(requested_groups, detect_device_arch())
    passes = schedule_perf_counter_passes(resolved)
    bitfields = [perf_counter_groups_to_bitfield(p) for p in passes]
    if len(passes) <= 1:
        if bitfields and bitfields[0] > 0:
            os.environ["TT_METAL_PROFILE_PERF_COUNTERS"] = str(bitfields[0])
            logger.info(f"Setting performance counter groups: {resolved} (bitfield: {bitfields[0]})")
        return bitfields
    plan = describe_passes(passes)
    if not can_replay:
        raise ValueError(
            f"--no-capture-tool cannot replay the workload; these groups need {len(passes)} passes:\n{plan}"
        )
    if not multipass:
        raise ValueError(
            f"Requested counter groups {resolved} need {len(passes)} capture passes "
            f"(L1 banks share one mux; BRISC firmware fits <= {PERF_COUNTER_MAX_GROUPS_PER_PASS} "
            f"groups/pass):\n{plan}\n"
            "Re-run with --perf-counter-multipass to replay the workload once per pass and merge "
            "the results, or request fewer groups."
        )
    logger.info(f"Multi-pass perf-counter capture ({len(passes)} passes):\n{plan}")
    return bitfields


def _counter_row_fields(line):
    """Split a device-log line and return its fields if it is a perf-counter row, else None."""
    # column 4 is timer_id; perf-counter rows carry PERF_COUNTER_MARKER_ID there.
    fields = line.split(",")
    if (
        len(fields) > PERF_COUNTER_TRACE_ID_COUNTER_COL
        and fields[PERF_COUNTER_TIMER_ID_COL].strip() == PERF_COUNTER_MARKER_ID
    ):
        return fields
    return None


def _readout_key(values):
    """Canonical executed BRISC readout: chip, x, y, RISC, op, trace, replay.

    Blank trace fields denote eager execution; never collapse them onto replay 0.
    No counters are emitted by the other RISCs (perf_counters.hpp).
    """
    if not isinstance(values, (list, tuple)) or len(values) != 7:
        raise ValueError("Readout identity must contain chip/core/RISC/op/trace/replay")
    values = list(values)
    for i in (0, 1, 2, 4, 5, 6):
        value = values[i]
        if i in (5, 6) and value in (None, ""):
            values[i] = None
        elif type(value) is int and value >= 0:
            pass
        elif isinstance(value, str) and re.fullmatch(r"[0-9]+", value):
            values[i] = int(value)
        else:
            raise ValueError(f"Invalid readout identity: {values}")
    if values[3] != "BRISC" or (values[5] is None) != (values[6] is None):
        raise ValueError(f"Invalid BRISC/trace identity: {values}")
    return tuple(values)


def _groups_for_mask(bitfield):
    known = sum(1 << b for b in PERF_COUNTER_GROUP_BITS.values())
    if type(bitfield) is not int or bitfield <= 0 or bitfield & ~known:
        raise ValueError(f"Invalid counter pass mask: {bitfield}")
    groups = [g for g, bit in PERF_COUNTER_GROUP_BITS.items() if bitfield & (1 << bit)]
    if len(groups) > PERF_COUNTER_MAX_GROUPS_PER_PASS or len(set(groups) & PERF_COUNTER_L1_GROUPS) > 1:
        raise ValueError(f"Uncaptureable counter pass mask: {bitfield}")
    return groups


def _expected_counter_types(arch, groups):
    """Use the exact native readout arrays, including grant-side/retired IDs.

    Missing or changed table syntax fails closed. Installed builds must retain
    these source headers; source/native-build attestation belongs to the caller.
    """
    directories = {"blackhole": "blackhole", "wormhole_b0": "wormhole"}
    if not isinstance(arch, str) or arch not in directories:
        raise ValueError(f"Unsupported counter completeness architecture: {arch}")
    resolve_perf_counter_groups(groups, arch)
    header = Path(__file__).resolve().parents[2] / (
        f"tt_metal/hw/inc/internal/tt-1xx/{directories[arch]}/hw_counters.h"
    )
    text = re.sub(r"//[^\n]*|/\*.*?\*/", "", header.read_text(), flags=re.S)
    counters = set()
    for group in groups:
        arrays = re.findall(
            r"std::array<std::pair<PerfCounterType,\s*std::uint16_t>,\s*(\d+)>\s+"
            + re.escape(group)
            + r"_counters\s*=\s*\{(.*?)\};",
            text,
            re.S,
        )
        if len(arrays) != 1:
            raise ValueError(f"Cannot establish counter semantics for {arch}/{group}")
        count, body = arrays[0]
        names = re.findall(r"PerfCounterType::(\w+)", body)
        if not names or len(names) != int(count) or len(set(names)) != len(names) or counters.intersection(names):
            raise ValueError(f"Ambiguous counter semantics for {arch}/{group}")
        counters.update(names)
    return counters


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_pass(path, bitfield, evidence):
    if (
        not isinstance(evidence, dict)
        or type(evidence.get("schema_version")) is not int
        or evidence["schema_version"] != 1
    ):
        raise ValueError("Missing/unsupported independent execution manifest (schema_version=1 required)")
    if type(evidence.get("bitfield")) is not int or evidence["bitfield"] != bitfield:
        raise ValueError("Execution manifest does not match the scheduled pass mask")
    if evidence.get("device_log_sha256") != _file_sha256(path):
        raise ValueError("Execution manifest does not match raw device log SHA256")
    arch = evidence.get("arch")
    counters = _expected_counter_types(arch, _groups_for_mask(bitfield))
    records = evidence.get("executed_readouts")
    if not isinstance(records, list) or not records:
        raise ValueError("No independently established executed readouts")
    expected = {_readout_key(record) for record in records}
    if len(expected) != len(records):
        raise ValueError("Duplicate execution identities are ambiguous")
    observed, anchors = {}, {}
    with Path(path).open() as stream:
        if stream.readline().split(",", 1)[0].strip() != f"ARCH: {arch}":
            raise ValueError("Device log architecture does not match execution evidence")
        header = stream.readline().split(",")
        if len(header) != 15 or header[4].strip() != "timer_id" or header[14].strip() != "meta data":
            raise ValueError("Unsupported device log columns")
        for line_number, line in enumerate(stream, 3):
            fields = [field.strip() for field in line.split(",")]
            if len(fields) != 15 or not line.endswith("\n"):
                raise ValueError(f"Malformed device log row {line_number}")
            if fields[PERF_COUNTER_TIMER_ID_COL] != PERF_COUNTER_MARKER_ID:
                continue
            key = _readout_key([fields[c] for c in (0, 1, 2, 3, 7, 8, 9)])
            if key not in expected:
                raise ValueError(f"Unmatched counter readout at row {line_number}: {key}")
            if fields[11] != "TS_DATA" or not re.fullmatch(r"[0-9]+", fields[5]):
                raise ValueError(f"Invalid counter marker at row {line_number}")
            metadata = json.loads(fields[14].replace(";", ","))
            if not isinstance(metadata, dict):
                raise ValueError(f"Invalid counter metadata at row {line_number}")
            counter = metadata.get("counter type")
            if not isinstance(counter, str) or counter not in counters:
                raise ValueError(f"Unexpected/missing counter type at row {line_number}: {counter}")
            for field in ("value", "ref cnt"):
                if type(metadata.get(field)) is not int or not 0 <= metadata[field] <= 0xFFFFFFFF:
                    raise ValueError(f"Invalid {field} for {counter} at row {line_number}")
            seen = observed.setdefault(key, set())
            if counter in seen:
                raise ValueError(f"Duplicate counter record at row {line_number}: {key}/{counter}")
            seen.add(counter)
            anchors.setdefault(key, fields[5])
    for key in expected:
        missing = counters - observed.get(key, set())
        if missing:
            raise ValueError(f"Missing required counters for {key}: {sorted(missing)}")
    return arch, expected, anchors


def merge_perf_counter_device_logs(pass_csvs, out_csv, pass_bitfields=None, execution_manifests=None):
    """Validate every scheduled pass before publishing; raise ValueError on incomplete evidence.

    Execution manifests must come from independent dispatch/replay evidence,
    never from surviving counter rows. See perf_counter_multipass.md. A complete
    result is relative to that execution scope, not a final-drain/loss attestation.
    """
    if not pass_csvs or not pass_bitfields or execution_manifests is None:
        raise ValueError("Counter merge requires scheduled passes and independent execution manifests")
    if len(pass_csvs) != len(pass_bitfields) or len(pass_csvs) != len(execution_manifests):
        raise ValueError("Missing scheduled pass log or execution manifest")
    if Path(out_csv).resolve() in {Path(p).resolve() for p in pass_csvs}:
        raise ValueError("Merged output must not overwrite a raw pass log")
    used_mask, reference, anchors = 0, None, None
    for path, mask, evidence in zip(pass_csvs, pass_bitfields, execution_manifests):
        _groups_for_mask(mask)
        if used_mask & mask:
            raise ValueError("Counter groups repeated across scheduled passes")
        used_mask |= mask
        arch, executed, pass_anchors = _validate_pass(path, mask, evidence)
        if reference is None:
            reference, anchors = (arch, executed), pass_anchors
        elif reference != (arch, executed):
            raise ValueError("Executed chip/core/RISC/op/trace/replay scope differs between passes")
    # All required identities and group-specific records have passed. Raw snapshots
    # stay untouched, and disjoint groups need not have equal row counts.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=Path(out_csv).parent, delete=False) as output:
            temporary = Path(output.name)
            with Path(pass_csvs[0]).open() as base:
                for line in base:
                    output.write(line)
            for path in pass_csvs[1:]:
                with Path(path).open() as extra:
                    for line in extra:
                        fields = _counter_row_fields(line)
                        if fields:
                            key = _readout_key([fields[c].strip() for c in (0, 1, 2, 3, 7, 8, 9)])
                            fields[PERF_COUNTER_TIMESTAMP_COL] = anchors[key]
                            output.write(",".join(fields))
        os.replace(temporary, out_csv)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def run_perf_counter_passes(run_workload, env, pass_bitfields, output_folder):
    """Capture raw logs plus execution sidecars; return False for unverifiable captures.

    The workload adapter writes TT_METAL_PROFILER_EXECUTION_MANIFEST after its
    final drain. CLI callers already translate False to exit 4 before reporting.
    """
    logs_folder = generate_logs_folder(output_folder)
    device_log = logs_folder / PROFILER_DEVICE_SIDE_LOG
    pass_dir = logs_folder / "perf_counter_passes"
    try:
        pass_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        logger.error(f"Refusing to overwrite raw counter pass artifacts: {pass_dir}")
        return False
    pass_logs, manifests = [], []
    result = {"schema_version": 1, "complete": False, "pass_bitfields": pass_bitfields, "raw_pass_logs": []}
    try:
        for i, bitfield in enumerate(pass_bitfields):
            _groups_for_mask(bitfield)
            logger.info(f"Perf-counter pass {i + 1}/{len(pass_bitfields)} (bitfield {bitfield})")
            if device_log.is_file():
                device_log.unlink()
            evidence_path = pass_dir / f"pass_{i}.execution.json"
            pass_env = dict(env)
            pass_env["TT_METAL_PROFILE_PERF_COUNTERS"] = str(bitfield)
            pass_env["TT_METAL_PROFILER_EXECUTION_MANIFEST"] = str(evidence_path.resolve())
            try:
                run_workload(pass_env)
            finally:
                if device_log.is_file():
                    snap = pass_dir / f"pass_{i}.csv"
                    copyfile(device_log, snap)
                    pass_logs.append(snap)
                    result["raw_pass_logs"].append(str(snap.resolve()))
            manifests.append(json.loads(evidence_path.read_text()) if evidence_path.is_file() else None)
        merge_perf_counter_device_logs(pass_logs, device_log, pass_bitfields, manifests)
        result["merged_log"] = {"path": str(device_log.resolve()), "sha256": _file_sha256(device_log)}
        result["complete"] = True
        logger.info(f"Merged {len(pass_logs)} verified perf-counter pass logs into {device_log}")
    except (OSError, ValueError, RuntimeError, SystemExit) as error:
        result["error"] = str(error)
        logger.error(f"Incomplete perf-counter capture: {error}; raw logs retained in {pass_dir}")
    finally:
        if not result["complete"] and device_log.is_file():
            device_log.unlink()  # never leave a last-pass log masquerading as the merged result
        (pass_dir / "merge_result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result["complete"]
