# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
from scripts.tt_hw_planner.commands.emit_e2e import _host_op_gate_reason


def test_on_device_verdict_passes():
    r = {"ran": True, "verdict": {"on_device": True, "host_ops": []}}
    assert _host_op_gate_reason(r) is None


def test_host_ops_verdict_fails_and_names_them():
    r = {"ran": True, "verdict": {"on_device": False, "host_ops": ["aten.embedding.default", "aten.cat.default"]}}
    msg = _host_op_gate_reason(r)
    assert msg is not None
    assert "aten.embedding.default" in msg


def test_not_ran_is_non_regressive_by_default():
    assert _host_op_gate_reason({"ran": False, "reason": "no host_op_selftest hook"}) is None
    assert _host_op_gate_reason(None) is None


def test_not_ran_fails_when_required():
    r = {"ran": False, "reason": "no host_op_selftest hook"}
    assert _host_op_gate_reason(r, require=True) is not None
