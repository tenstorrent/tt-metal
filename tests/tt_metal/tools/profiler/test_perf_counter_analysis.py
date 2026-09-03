# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Offline tests for the perf counter decode and metric computation in tools/tracy.

These need no hardware: they feed synthetic counter captures through the same code paths the
profiler reports use.
"""

import random
import re
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "tools" / "tracy"))

from perf_counter_analysis import (
    COUNTER_TYPE_NAMES,
    PERF_COUNTER_CSV_HEADERS,
    compute_device_only_metrics,
    compute_perf_counter_metrics,
    quasar_l1_client_label,
)

QUASAR_INSTRN_CLASSES = ("CFG", "SYNC", "THCON", "XSEARCH", "INSTISSUE", "FPU", "UNPACK", "PACK")

QUASAR_CAPTURE_TYPES = (
    ["FPU_COUNTER", "SFPU_COUNTER", "MATH_COUNTER"]
    + [
        "MATH_SRC_DATA_READY",
        "DATA_HAZARD_STALLS_MOVD2A",
        "MATH_FIDELITY_STALL",
        "MATH_INSTRN_STARTED",
        "MATH_INSTRN_AVAILABLE",
        "SRCB_WRITE_AVAILABLE",
        "SRCA_WRITE_AVAILABLE",
        "UNPACK0_BUSY_THREAD0",
        "UNPACK1_BUSY_THREAD0",
        "UNPACK0_BUSY_THREAD1",
        "UNPACK1_BUSY_THREAD1",
        "MATH_INSTRN_HF_4_CYCLE",
        "MATH_INSTRN_HF_2_CYCLE",
        "MATH_INSTRN_HF_1_CYCLE",
        "SRCB_WRITE_ACTUAL",
        "SRCB_WRITE_NOT_BLOCKED_PORT",
        "SRCA_WRITE_NOT_BLOCKED_OVR",
        "SRCA_WRITE_ACTUAL",
        "SRCA_WRITE_THREAD0",
        "SRCB_WRITE_THREAD0",
        "SRCA_WRITE_THREAD1",
        "SRCB_WRITE_THREAD1",
    ]
    + [
        "PACKER_DEST_READ_AVAILABLE",
        "PACKER_BUSY",
        "DEST_READ_GRANTED_0",
        "MATH_NOT_STALLED_DEST_WR_PORT",
        "AVAILABLE_MATH",
    ]
    + [f"{cls}_INSTRN_AVAILABLE_{t}" for cls in QUASAR_INSTRN_CLASSES for t in range(4)]
    + [f"THREAD_STALLS_{t}" for t in range(4)]
    + [
        "TILE_COUNTER_STALL_PACK",
        "TILE_COUNTER_STALL_UNPACK",
        "SRCS_STALL_PACK",
        "SRCS_STALL_SFPU",
        "SRCS_STALL_UNPACK",
        "DEST_STALL_PACK",
        "DEST_STALL_SFPU",
        "DEST_STALL_MATH",
        "DEST_STALL_UNPACK",
        "SFPU_DATA_HAZARD_STALL",
        "FPU_DATA_HAZARD_STALL",
        "SRCB_STALL_UNPACK",
        "SRCA_STALL_UNPACK",
        "DVALID_STALL_MATH",
        "SRCA_STALL_MATH",
    ]
    + [f"THREAD_INSTRUCTIONS_{t}" for t in range(4)]
    + ["L1_CLIENT_UNPACK3_SBANK_POP"]
)

QUASAR_EXPECTED_METRICS = [
    "Thread 3 Stall Rate",
    "T3 Instrn Issue Rate",
    "SrcA Stall Math Rate",
    "Dest Stall Pack Rate",
    "XSEARCH Instrn Avail Rate T0",
    "INSTISSUE Instrn Avail Rate T3",
    "CFG Instrn Avail Rate T3",
    "L1_CLIENT_UNPACK3_SBANK_POP Rate",
]


def make_capture(counter_types, risc_fmt, num_riscs, seed=7):
    random.seed(seed)
    rows = []
    for n in range(num_riscs):
        for name in counter_types:
            rows.append(
                {
                    "run_host_id": 1,
                    "trace_id_count": 0,
                    "record time": 1000,
                    "core_x": 1,
                    "core_y": 1 + n,
                    "risc_type": risc_fmt.format(n),
                    "counter type": name,
                    "value": random.randint(100, 5000),
                    "ref cnt": 10000,
                }
            )
    return pd.DataFrame(rows)


def test_counter_type_names_match_enum():
    # The decode table maps enum ordinals to names; regenerating the enum without updating the
    # table silently mislabels every counter, so pin them together.
    hpp = (REPO_ROOT / "tt_metal" / "tools" / "profiler" / "perf_counters.hpp").read_text()
    enum_body = re.search(r"enum PerfCounterType : uint16_t \{(.*?)\};", hpp, re.S).group(1)
    enum_names = re.findall(r"^\s*([A-Z][A-Z0-9_]*)(?:\s*=\s*\d+)?,", enum_body, re.M)
    assert enum_names[0] == "UNDEF"
    assert len(COUNTER_TYPE_NAMES) == len(enum_names)
    for ordinal, name in COUNTER_TYPE_NAMES.items():
        assert enum_names[ordinal] == name, f"ordinal {ordinal}: table says {name}, enum says {enum_names[ordinal]}"


def test_l1_client_labels_cover_all_subport_ranges():
    assert quasar_l1_client_label(0) == "L1_CLIENT_TRISC0_UNUSED"
    assert quasar_l1_client_label(3 * 8 + 5) == "L1_CLIENT_TRISC3_FLEX_WORK_CARRY"
    assert quasar_l1_client_label(4 * 8 + 7) == "L1_CLIENT_THCON_ORDER_FIFO_ACTIVE"
    assert quasar_l1_client_label(5 * 8 + 1) == "L1_CLIENT_UNPACK0_SBANK_POP"
    assert quasar_l1_client_label(24 * 8 + 3) == "L1_CLIENT_UNPACK19_ISSUE_WORK_CARRY"
    assert quasar_l1_client_label(25 * 8 + 2) == "L1_CLIENT_PACK0_ISSUE_STALL_CARRY"
    assert quasar_l1_client_label(36 * 8 + 6) == "L1_CLIENT_PACK11_PENDING_REQS_CARRY"


def test_quasar_capture_produces_quasar_metrics_per_op():
    df = make_capture(QUASAR_CAPTURE_TYPES, "QUASAR_NEO{}_TRISC1", 4)
    stats = compute_perf_counter_metrics(df, "quasar", total_compute_cores=4)["per_op_stats"]
    for metric in QUASAR_EXPECTED_METRICS:
        assert metric in stats, metric
        values = stats[metric]["avg"]
        assert values and all(v == v for v in values.values()), f"{metric} produced NaN"


def test_quasar_capture_produces_quasar_metrics_device_only():
    df = make_capture(QUASAR_CAPTURE_TYPES, "QUASAR_NEO{}_TRISC1", 4)
    agg_metrics, _ = compute_device_only_metrics(df, "quasar")
    for metric in QUASAR_EXPECTED_METRICS:
        assert metric in agg_metrics, metric


def test_blackhole_capture_gets_no_quasar_columns():
    quasar_only = {n for n in QUASAR_CAPTURE_TYPES if "XSEARCH" in n or "INSTISSUE" in n or n.endswith("_3")}
    quasar_only |= {n for n in QUASAR_CAPTURE_TYPES if "STALL_" in n or n.startswith("L1_CLIENT_")}
    bh_types = [n for n in QUASAR_CAPTURE_TYPES if n not in quasar_only]
    df = make_capture(bh_types, "BRISC", 1)

    stats = compute_perf_counter_metrics(df, "blackhole", total_compute_cores=1)["per_op_stats"]
    agg_metrics, _ = compute_device_only_metrics(df, "blackhole")
    for keys in (stats.keys(), agg_metrics.keys()):
        leaked = [
            k
            for k in keys
            if "XSEARCH" in k or "INSTISSUE" in k or "T3" in k or "Thread 3" in k or k.startswith("L1_CLIENT_")
        ]
        assert not leaked, leaked


def test_csv_headers_are_unique():
    assert len(PERF_COUNTER_CSV_HEADERS) == len(set(PERF_COUNTER_CSV_HEADERS))
