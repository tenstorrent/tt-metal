# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Multi-pass perf counter capture: group tables, pass scheduling (one L1 bank per pass, the BRISC firmware
fits every group), per-pass workload replay and the device log merge. tracy/__main__.py only calls in here."""

import math
import os
import subprocess
import sys
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
# Bits 16-24 hold the Quasar l1_client selection (subport*8 + event, 0 = off), requested as l1_client=<selection>.
PERF_COUNTER_L1_CLIENT_SHIFT = 16
PERF_COUNTER_BH_ONLY_GROUPS = {"l1_2", "l1_3", "l1_4", "l1_5"}
# Every non L1 group plus one L1 bank fits the BRISC .text: 8648 of 8704 bytes on Blackhole (8692 with a harvested
# DRAM bank) and 7584 of 7712 on Wormhole with the five group mask. The one L1 bank per pass rule forces passes.
PERF_COUNTER_MAX_GROUPS_PER_PASS = 5
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
    tt_perf_cnt bank on its L1 (its l1_client event counter is requested with l1_client=<subport*8 + event>)."""
    if is_quasar:
        return []
    return ["l1_0", "l1_1", "l1_2", "l1_3", "l1_4", "l1_5"] if is_blackhole else ["l1_0", "l1_1"]


def split_l1_client_selection(requested):
    """(groups, selection): the l1_client=<subport*8 + event> entry taken out of a counter request, None without one."""
    groups, selection = [], None
    for item in requested:
        name, sep, value = item.partition("=")
        if not sep:
            groups.append(item)
            continue
        if name.strip().lower() != "l1_client":
            raise ValueError(f"Unknown counter option '{item}'; the only one is l1_client=<subport*8 + event>")
        try:
            selection = int(value, 0)
        except ValueError:
            raise ValueError(f"'{item}': the l1_client selection is a number, subport*8 + event") from None
    return groups, selection


def _llk_perf_metrics():
    try:
        from tt_llk_perf import metrics
    except ImportError:
        sys.path.append(str(Path(__file__).resolve().parents[2] / "tt_metal" / "tt-llk" / "tools" / "python"))
        from tt_llk_perf import metrics
    return metrics


def check_l1_client_selection(selection, arch):
    """The Quasar l1_client selection rules of llk::perf::l1_client_selection_is_valid; ValueError otherwise."""
    if arch != "quasar":
        raise ValueError(
            f"l1_client={selection} selects the Quasar l1_client event counter, but device arch is "
            f"{arch or 'undeclared'}."
        )
    if not _llk_perf_metrics().quasar_l1_client_selection_is_valid(selection):
        raise ValueError(
            f"l1_client={selection} is not a valid selection: subport*8 + event with 37 subports and 8 events, "
            "event 0 and the THCON events 1 to 3 excluded."
        )


def perf_counter_groups_to_bitfield(groups):
    """OR the PROFILE_PERF_COUNTERS_* bits for a list of group names."""
    bits = 0
    for g in groups:
        bits |= 1 << PERF_COUNTER_GROUP_BITS[g.lower()]
    return bits


def detect_device_arch():
    """Arch name in lower case from the environment or a ttnn child process; None if unknown. A child process on
    purpose: a ttnn import here would hold the device until this capture process exits and block the workload."""
    declared = next((os.environ.get(v) for v in ARCH_ENV_VARS if os.environ.get(v)), None)
    if declared is None:
        try:
            probe = subprocess.run(
                [sys.executable, "-c", "import ttnn; print(ttnn.get_arch_name())"],
                capture_output=True,
                text=True,
                timeout=300,
                check=True,
            )
            declared = probe.stdout.strip().splitlines()[-1]
        except (subprocess.SubprocessError, OSError, IndexError):
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
        if g == "all":
            resolved = ["fpu", "pack", "unpack", "instrn"] + arch_l1_groups(is_blackhole, is_quasar)
            break
        elif g in PERF_COUNTER_GROUP_BITS:
            resolved.append(g)
        else:
            logger.warning(f"Unknown counter group '{group}'. Valid groups: {', '.join(PERF_COUNTER_GROUP_BITS)}, all")
    resolved = list(dict.fromkeys(resolved))

    if is_quasar and (set(resolved) & PERF_COUNTER_L1_GROUPS):
        raise ValueError(
            "Quasar has no L1 performance counter bank; drop the l1_* groups and request the l1_client event "
            "counter with l1_client=<subport*8 + event>."
        )
    bh_only = sorted(set(resolved) & PERF_COUNTER_BH_ONLY_GROUPS)
    if bh_only and not is_blackhole:
        raise ValueError(
            f"Performance counter groups {', '.join(bh_only)} are supported only on Blackhole, "
            f"but device arch is {arch or 'undeclared'}."
        )
    return resolved


# Profiler DRAM buffer per RISC: 48 bytes per supported program (TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT, the tracy
# --op-support-count option). Overflow drops the tail of the run; the ops report then fails on an op count mismatch.
PROFILER_BYTES_PER_PROGRAM = 48
PROFILER_DEFAULT_PROGRAM_SUPPORT_COUNT = 1000
PROFILER_BYTES_PER_COUNTER_RECORD = 24
PROFILER_ZONE_BYTES_PER_OP = 48


def records_per_pass(passes, arch):
    """Counter records one op leaves per core for each pass, from the shared select tables; None off tt-1xx."""
    try:
        from tt_llk_perf.headers import bank_tables

        tables = bank_tables(arch)
    except Exception:
        return None
    if not tables:
        return None
    bank_of = {"fpu": "FPU", "pack": "TDMA_PACK", "unpack": "TDMA_UNPACK", "instrn": "INSTRN"}
    counts = []
    for p in passes:
        n = 0
        for g in p:
            if g in PERF_COUNTER_L1_GROUPS:
                mux = int(g.split("_")[1])
                n += sum(1 for e in tables.get("L1", []) if e.l1_mux == mux)
            else:
                n += len(tables.get(bank_of[g], []))
        counts.append(n)
    return counts


def ops_per_run(records, support_count=None):
    """How many ops per core the profiler buffer holds for a pass of `records` counter records."""
    support_count = support_count or int(
        os.environ.get("TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT", PROFILER_DEFAULT_PROGRAM_SUPPORT_COUNT)
    )
    return (PROFILER_BYTES_PER_PROGRAM * support_count) // (
        PROFILER_BYTES_PER_COUNTER_RECORD * records + PROFILER_ZONE_BYTES_PER_OP
    )


def describe_passes(passes, arch=None):
    counts = records_per_pass(passes, arch) if arch else None
    lines = []
    for i, p in enumerate(passes):
        line = f"  pass {i + 1}: {', '.join(p)}  (bitfield {perf_counter_groups_to_bitfield(p)})"
        if counts:
            line += f"  {counts[i]} records per op per core, about {ops_per_run(counts[i])} ops per run before the profiler buffer fills"
        lines.append(line)
    return "\n".join(lines)


def plan_perf_counter_capture(requested_groups, multipass, can_replay):
    """Per-pass bitfields for a counter request. One pass is also exported via TT_METAL_PROFILE_PERF_COUNTERS;
    several passes need ``multipass`` and ``can_replay`` (this process launches the workload), else ValueError."""
    requested_groups, l1_client = split_l1_client_selection(requested_groups)
    arch = detect_device_arch()
    if l1_client is not None:
        check_l1_client_selection(l1_client, arch)
    resolved = resolve_perf_counter_groups(requested_groups, arch)
    passes = schedule_perf_counter_passes(resolved)
    if l1_client and not passes:
        passes = [[]]
    l1_client_bits = (l1_client or 0) << PERF_COUNTER_L1_CLIENT_SHIFT
    bitfields = [perf_counter_groups_to_bitfield(p) | l1_client_bits for p in passes]
    if len(passes) <= 1:
        if bitfields and bitfields[0] > 0:
            os.environ["TT_METAL_PROFILE_PERF_COUNTERS"] = str(bitfields[0])
            logger.info(f"Setting performance counter groups: {resolved} (bitfield: {bitfields[0]})")
            counts = records_per_pass(passes, arch)
            if counts:
                logger.info(
                    f"{counts[0]} counter records per op per core; the profiler buffer holds about "
                    f"{ops_per_run(counts[0])} ops per run at this size, raise --op-support-count for longer runs"
                )
        return bitfields
    plan = describe_passes(passes, arch)
    if not can_replay:
        raise ValueError(
            f"--no-capture-tool cannot replay the workload; these groups need {len(passes)} passes:\n{plan}"
        )
    if not multipass:
        raise ValueError(
            f"Requested counter groups {resolved} need {len(passes)} capture passes "
            f"(L1 banks share one mux, so each bank needs its own pass; at most "
            f"{PERF_COUNTER_MAX_GROUPS_PER_PASS} groups per pass):\n{plan}\n"
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


def _op_key(fields):
    # device, core, run host id and the trace replay a traced op belongs to; every RISC of the core reads out the
    # same op, so the RISC is not part of it
    return tuple(
        fields[c].strip()
        for c in (0, 1, 2, PERF_COUNTER_RUN_HOST_ID_COL, PERF_COUNTER_TRACE_ID_COL, PERF_COUNTER_TRACE_ID_COUNTER_COL)
    )


def merge_perf_counter_device_logs(pass_csvs, out_csv):
    """Merge per-pass device logs: pass 0 whole, later passes add their counter rows re-timestamped onto pass 0."""
    base = Path(pass_csvs[0]).read_text().splitlines(keepends=True)
    anchors = {}
    for line in base:
        fields = _counter_row_fields(line)
        if fields:
            anchors.setdefault(_op_key(fields), fields[PERF_COUNTER_TIMESTAMP_COL].strip())

    merged, unanchored = list(base), 0
    for extra in pass_csvs[1:]:
        for line in Path(extra).read_text().splitlines(keepends=True):
            fields = _counter_row_fields(line)
            if not fields:
                continue
            anchor = anchors.get(_op_key(fields))
            if anchor is None:
                unanchored += 1
                continue
            fields[PERF_COUNTER_TIMESTAMP_COL] = anchor
            merged.append(",".join(fields))
    if unanchored:
        logger.warning(f"Dropped {unanchored} perf-counter rows with no matching op in pass 0")
    Path(out_csv).write_text("".join(merged))


def run_perf_counter_passes(run_workload, env, pass_bitfields, output_folder):
    """Run the workload once per pass mask and merge the device logs; False if a pass left no log."""
    logs_folder = generate_logs_folder(output_folder)
    device_log = logs_folder / PROFILER_DEVICE_SIDE_LOG
    pass_dir = logs_folder / "perf_counter_passes"
    pass_dir.mkdir(parents=True, exist_ok=True)
    pass_logs = []
    for i, bitfield in enumerate(pass_bitfields):
        logger.info(f"Perf-counter pass {i + 1}/{len(pass_bitfields)} (bitfield {bitfield})")
        if device_log.is_file():
            device_log.unlink()  # fresh per pass so each snapshot holds only that pass
        pass_env = dict(env)
        pass_env["TT_METAL_PROFILE_PERF_COUNTERS"] = str(bitfield)
        run_workload(pass_env)
        if device_log.is_file():
            snap = pass_dir / f"pass_{i}.csv"
            copyfile(device_log, snap)
            pass_logs.append(snap)
        else:
            logger.error(f"Device log missing after perf-counter pass {i + 1}: {device_log}")
    if len(pass_logs) != len(pass_bitfields):
        logger.error(
            f"Only {len(pass_logs)}/{len(pass_bitfields)} perf-counter passes produced a device log; "
            "not merging a partial capture"
        )
        return False
    merge_perf_counter_device_logs(pass_logs, device_log)
    logger.info(f"Merged {len(pass_logs)} perf-counter pass logs into {device_log}")
    return True
