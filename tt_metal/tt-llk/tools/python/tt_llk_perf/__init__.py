# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Host side of the tt-llk perf counters: header parsers plus the metric engine shared with the metal profiler.
Stdlib only, so tools/tracy can import it without the LLK harness.
"""

from . import (
    dram_counters,
    headers,
    metrics,
    noc_counters,
    noc_metrics,
    riscv_metrics,
    riscv_pmc,
)
from .headers import (
    BANK_KEYS,
    CounterEntry,
    bank_tables,
    counter_type_names,
    dram_counter_names,
    dram_counter_table,
    dram_layout,
    find_include_dir,
    noc_counter_names,
    noc_counter_table,
    noc_selected_counters,
    riscv_event_names,
    riscv_event_select,
    riscv_event_tables,
    riscv_select_events,
)
from .metrics import (
    METRIC_LABELS,
    RATIO_KEYS,
    RATIO_LABELS,
    CounterView,
    compute_l1_client_metrics,
    compute_metrics,
    is_ratio_label,
    metric_label,
    quasar_l1_client_label,
    quasar_l1_client_selection_is_valid,
)

__all__ = [
    "BANK_KEYS",
    "METRIC_LABELS",
    "RATIO_KEYS",
    "RATIO_LABELS",
    "CounterEntry",
    "CounterView",
    "bank_tables",
    "compute_l1_client_metrics",
    "compute_metrics",
    "counter_type_names",
    "dram_counter_names",
    "dram_counter_table",
    "dram_counters",
    "dram_layout",
    "find_include_dir",
    "headers",
    "is_ratio_label",
    "metric_label",
    "metrics",
    "noc_counters",
    "noc_metrics",
    "noc_counter_names",
    "noc_counter_table",
    "noc_selected_counters",
    "riscv_metrics",
    "riscv_pmc",
    "riscv_event_names",
    "riscv_event_select",
    "riscv_event_tables",
    "riscv_select_events",
    "quasar_l1_client_label",
    "quasar_l1_client_selection_is_valid",
]
