# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""TP measurement guard (Increment 5). TP fractures a matmul and ADDS collective (ccl) ops, so the
op-count-inflation reject must discount the ccl bucket in TP regime — and ONLY in TP regime, so the
guard is byte-identical for every existing (non-TP) run."""
from agent.handlers.remeasure import _comparable

BASE = {"buckets": [{"id": "matmul", "count": 8}], "device_ms": 20.0}
TP_PROFILE = {"buckets": [{"id": "matmul", "count": 8}, {"id": "ccl", "count": 8}], "device_ms": 16.0}


def test_added_ccl_ops_rejected_off_regime():
    ok, reason = _comparable(BASE, TP_PROFILE, floor_ms=10.0)
    assert not ok and "op_count_inflated" in reason


def test_added_ccl_ops_accepted_in_tp_regime():
    ok, reason = _comparable(BASE, TP_PROFILE, floor_ms=10.0, tp_regime=True)
    assert ok and reason is None


def test_tp_regime_still_rejects_real_inflation_in_compute_ops():
    bad = {"buckets": [{"id": "matmul", "count": 20}, {"id": "ccl", "count": 4}], "device_ms": 16.0}
    ok, reason = _comparable(BASE, bad, floor_ms=10.0, tp_regime=True)
    assert not ok and "op_count_inflated" in reason
