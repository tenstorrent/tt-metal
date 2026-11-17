# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""NoC interface counter metrics. Kept apart from metrics.py: that module's keys are the warehouse metric catalog
(helpers/perf/schema.py METRIC_BASES), and these are per-run profiler metrics.

Inputs are the counts of one NoC over one launch, by counter name, and the data word size in bytes. A metric whose
inputs were not captured (its group was off) reads None. Directions are from the tile's point of view: "in" is data
arriving (data read from elsewhere, data written by others), "out" is data leaving.
"""

from .metrics import safe_div

NOC_PREFIX = "noc_"

NOC_METRIC_LABELS = {
    "noc_read_bytes": "NoC Bytes Read",
    "noc_write_bytes": "NoC Bytes Written",
    "noc_bytes_in": "NoC Bytes In",
    "noc_bytes_out": "NoC Bytes Out",
    "noc_issued_read_bytes": "NoC Bytes Read by this Tile",
    "noc_issued_write_bytes": "NoC Bytes Written by this Tile",
    "noc_served_read_bytes": "NoC Bytes Read from this Tile",
    "noc_served_write_bytes": "NoC Bytes Written to this Tile",
    "noc_issued_read_reqs": "NoC Read Requests Issued",
    "noc_issued_write_reqs": "NoC Write Requests Issued",
    "noc_bytes_per_issued_read_req_ratio": "NoC Bytes per Issued Read Request",
    "noc_bytes_per_issued_write_req_ratio": "NoC Bytes per Issued Write Request",
}
NOC_RATIO_SUFFIXES = ("_ratio",)


def _total(counts, names):
    """Sum of the named counters, None unless every one was captured."""
    if any(n not in counts for n in names):
        return None
    return sum(counts[n] for n in names)


def _bytes(words, word_bytes):
    return None if words is None else words * word_bytes


def _ratio(numerator, denominator):
    return (
        None
        if numerator is None or denominator is None
        else safe_div(numerator, denominator)
    )


def _add(a, b):
    return None if a is None or b is None else a + b


def compute_noc_metrics(counts, word_bytes) -> dict:
    """Byte and request totals of one NoC over one launch. counts maps counter name -> count."""
    issued_read = _bytes(_total(counts, ["MST_RD_DATA_WORD_RECEIVED"]), word_bytes)
    issued_write = _bytes(
        _total(
            counts, ["MST_NONPOSTED_WR_DATA_WORD_SENT", "MST_POSTED_WR_DATA_WORD_SENT"]
        ),
        word_bytes,
    )
    served_read = _bytes(_total(counts, ["SLV_RD_DATA_WORD_SENT"]), word_bytes)
    served_write = _bytes(
        _total(
            counts,
            ["SLV_NONPOSTED_WR_DATA_WORD_RECEIVED", "SLV_POSTED_WR_DATA_WORD_RECEIVED"],
        ),
        word_bytes,
    )
    issued_read_reqs = _total(counts, ["MST_RD_REQ_SENT"])
    issued_write_reqs = _total(
        counts, ["MST_NONPOSTED_WR_REQ_SENT", "MST_POSTED_WR_REQ_SENT"]
    )
    return {
        "noc_issued_read_bytes": issued_read,
        "noc_issued_write_bytes": issued_write,
        "noc_served_read_bytes": served_read,
        "noc_served_write_bytes": served_write,
        "noc_read_bytes": _add(issued_read, served_read),
        "noc_write_bytes": _add(issued_write, served_write),
        "noc_bytes_in": _add(issued_read, served_write),
        "noc_bytes_out": _add(issued_write, served_read),
        "noc_issued_read_reqs": issued_read_reqs,
        "noc_issued_write_reqs": issued_write_reqs,
        "noc_bytes_per_issued_read_req_ratio": _ratio(issued_read, issued_read_reqs),
        "noc_bytes_per_issued_write_req_ratio": _ratio(issued_write, issued_write_reqs),
    }


def noc_metric_label(key: str) -> str:
    return NOC_METRIC_LABELS.get(key, key)


def noc_is_ratio_key(key: str) -> bool:
    return key.endswith(NOC_RATIO_SUFFIXES)
