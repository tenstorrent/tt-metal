# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""RISC-V performance monitor metrics. Kept apart from metrics.py: that module's keys are the warehouse metric catalog
(helpers/perf/schema.py METRIC_BASES), and these are per-run profiler metrics with dynamic keys.
"""

from .metrics import bounded, pct, safe_div

# Inputs are plain numbers (a core's mcycle and minstret deltas plus one count per armed event label) instead of a
# CounterView, because the monitor has no banks and no reference counter per event. A label is the event name, or
# the names joined with "|" when one select arms several (the counter counts their OR).
RISCV_PREFIX = "riscv_"

RISCV_METRIC_LABELS = {
    "riscv_ipc_ratio": "RISC-V IPC",
    "riscv_stall_pct": "RISC-V Stall Cycles",
    "riscv_mem_stall_per_tti_push_ratio": "RISC-V Memory Stall Cycles per Tensix Push",
}
RISCV_RATIO_SUFFIXES = ("_ratio",)


def _riscv_key(label: str, suffix: str) -> str:
    return f"{RISCV_PREFIX}{label.lower().replace('|', '_or_')}{suffix}"


def _riscv_is_stall(label: str) -> bool:
    return all("STALL" in part for part in label.split("|"))


def compute_riscv_metrics(cycles, instret, counts) -> dict:
    """Metrics of one core launch (or the sum of several). counts maps event label -> count over the same window.
    A rate is a bounded per-cycle percentage, a per-instruction value an unbounded ratio. Absent inputs read None.
    """
    out = {"riscv_ipc_ratio": safe_div(instret, cycles)}
    stall_labels = [label for label in counts if _riscv_is_stall(label)]
    out["riscv_stall_pct"] = (
        pct(bounded(safe_div(counts[stall_labels[0]], cycles)))
        if stall_labels
        else None
    )
    for label, count in counts.items():
        out[_riscv_key(label, "_pct")] = pct(bounded(safe_div(count, cycles)))
        out[_riscv_key(label, "_per_instr_ratio")] = safe_div(count, instret)
    out["riscv_mem_stall_per_tti_push_ratio"] = (
        safe_div(counts["MEM_STALL_ANY"], counts["MEM_INS_BUF_WRITE"])
        if "MEM_STALL_ANY" in counts and "MEM_INS_BUF_WRITE" in counts
        else None
    )
    return out


def riscv_metric_label(key: str) -> str:
    if key in RISCV_METRIC_LABELS:
        return RISCV_METRIC_LABELS[key]
    if key.startswith(RISCV_PREFIX):
        for suffix, text in (
            ("_per_instr_ratio", "per Instruction"),
            ("_pct", "Cycle Fraction"),
        ):
            if key.endswith(suffix):
                name = (
                    key[len(RISCV_PREFIX) : -len(suffix)].upper().replace("_OR_", "|")
                )
                return f"RISC-V {name} {text}"
    return key


def riscv_is_ratio_key(key: str) -> bool:
    return key.endswith(RISCV_RATIO_SUFFIXES)
