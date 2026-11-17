# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""NoC router counter metrics. Kept apart from metrics.py: that module's keys are the warehouse metric catalog
(helpers/perf/schema.py METRIC_BASES), and these are per-run profiler metrics.

Input is the counts of one NoC over one launch, by label: a port name (OUT_PORT1, IN_PORT1, ...) or a port name with
_VC<n> when the channels were kept apart. What one count is depends on the architecture (see count_kind of
decode_router_launch): a packet on Blackhole, a flit of any type on Quasar. A metric with no input reads None.
"""

ROUTER_PREFIX = "router_"

ROUTER_METRIC_LABELS = {
    "router_out_count": "Router Output Count",
    "router_in_count": "Router Input Count",
}


def _port_of(label: str) -> str:
    return label.split("_VC", 1)[0]


def _total(counts, prefix):
    values = [v for label, v in counts.items() if _port_of(label).startswith(prefix)]
    return sum(values) if values else None


def compute_router_metrics(counts) -> dict:
    """Counts leaving and entering the router of one NoC over one launch, over the captured ports and channels."""
    return {
        "router_out_count": _total(counts, "OUT_PORT"),
        "router_in_count": _total(counts, "IN_PORT"),
    }


def router_metric_label(key: str) -> str:
    return ROUTER_METRIC_LABELS.get(key, key)
