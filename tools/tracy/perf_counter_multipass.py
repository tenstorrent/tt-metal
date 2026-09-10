# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Multi-pass perf counter capture: group tables, pass scheduling (one L1 bank per pass, the BRISC firmware
fits three groups), per-pass workload replay and the device log merge. tracy/__main__.py only calls in here."""

import math
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
