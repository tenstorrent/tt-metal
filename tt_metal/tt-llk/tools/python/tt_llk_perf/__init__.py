# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Host side of the tt-llk perf counters: header parsers plus the metric engine shared with the metal profiler.
Stdlib only, so tools/tracy can import it without the LLK harness.
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
    compute_metrics,
)

__all__ = [
    "BANK_KEYS",
    "METRIC_LABELS",
    "RATIO_KEYS",
    "RATIO_LABELS",
    "CounterEntry",
    "CounterView",
    "bank_tables",
    "compute_metrics",
    "counter_type_names",
    "find_include_dir",
    "headers",
    "metrics",
]
