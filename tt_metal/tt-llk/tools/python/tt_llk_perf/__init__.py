# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Host side of the tt-llk perf counter infrastructure.

headers: parsers for tools/include/perf_counters (counter names by ordinal,
per-arch tables by bank and select). metrics: the derived-metric engine shared
by the metal profiler (tools/tracy) and the LLK perf harness. Stdlib only.
"""

from . import headers, metrics
from .headers import (
    BANK_KEYS,
    CounterEntry,
    bank_tables,
    counter_type_names,
    find_include_dir,
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
    "find_include_dir",
    "headers",
    "is_ratio_label",
    "metric_label",
    "metrics",
    "quasar_l1_client_label",
    "quasar_l1_client_selection_is_valid",
]
