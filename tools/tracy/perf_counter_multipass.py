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
PERF_COUNTER_BH_ONLY_GROUPS = {"l1_2", "l1_3", "l1_4", "l1_5"}
# The table driven readout costs a few bytes per group, so every group fits one pass next to one L1 bank.
# Measured BRISC .text with the five group mask (fpu, pack, unpack, instrn and one L1 bank): Blackhole 8684 of
# 8704 bytes, Wormhole 7600 of 7712. The cap is the number of groups such a mask holds; the one L1 bank per
# pass rule below is the hardware limit that still forces several passes.
PERF_COUNTER_MAX_GROUPS_PER_PASS = 5
# PERF_COUNTER_PROFILER_ID in perf_counters.hpp: the timer_id the firmware tags counter rows with.
PERF_COUNTER_MARKER_ID = "9090"
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
    """The device architecture name in lower case, from the environment or from ttnn in a child process; None if
    unknown. A child process on purpose: this runs in the capture process, and a ttnn import that touches the
    device there keeps the device handle until the capture process exits, so the workload it then launches
    blocks in its own open_device."""
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


# Device profiler capacity per RISC between host reads: the DRAM buffer is 48 bytes per supported program
# (TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT, default 1000, the tracy --op-support-count option), a counter record is
# 24 bytes and an op leaves about 48 bytes of zone markers on BRISC next to them. Past that the tail of the run is
# dropped and the ops report fails with a host/device op count mismatch, so say how many ops a pass holds.
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
    arch = detect_device_arch()
    resolved = resolve_perf_counter_groups(requested_groups, arch)
    passes = schedule_perf_counter_passes(resolved)
    bitfields = [perf_counter_groups_to_bitfield(p) for p in passes]
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


def merge_perf_counter_device_logs(pass_csvs, out_csv):
    """Merge per-pass device logs: pass 0 whole, later passes contribute only their perf-counter rows."""
    merged = list(Path(pass_csvs[0]).read_text().splitlines(keepends=True))
    for extra in pass_csvs[1:]:
        for line in Path(extra).read_text().splitlines(keepends=True):
            # column 4 is timer_id; perf-counter rows carry PERF_COUNTER_MARKER_ID there.
            fields = line.split(",")
            if len(fields) > 4 and fields[4].strip() == PERF_COUNTER_MARKER_ID:
                merged.append(line)
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
