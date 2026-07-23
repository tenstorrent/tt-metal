# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""TP ladder rung (Increment 2). The tp-fracture rung is offered ONLY in TP regime for a dense
memory-bound matmul after kernels are exhausted; off-regime the ladder is byte-identical to today.
A tp-fracture attempt is credited only when model source has BOTH a weight shard AND a CCL."""
import pytest

from cc_optimize import perf_mcp

_KERNELS_DONE = [
    {"op_signature": "MatmulDeviceOperation", "kernel_kind": "tt-lang"},
    {"op_signature": "MatmulDeviceOperation", "kernel_kind": "cpp"},
]
_KNOBS_DONE = [
    {"op_signature": "MatmulDeviceOperation", "kernel_kind": "shard"},
    {"op_signature": "MatmulDeviceOperation", "kernel_kind": "shard"},
]
_ALL_DONE = _KNOBS_DONE + _KERNELS_DONE
_MATMUL = {"grid": "full", "weight_dtype": "bf8_b", "bound_by": "memory"}


@pytest.fixture(autouse=True)
def _reset_regime():
    perf_mcp.set_tp_regime(False)
    yield
    perf_mcp.set_tp_regime(False)


def test_tp_rung_offered_in_regime_before_kernels():
    perf_mcp.set_tp_regime(True)
    done, rung, _ = perf_mcp._op_ladder_status(_MATMUL, "MatmulDeviceOperation", _KNOBS_DONE)
    assert not done and rung == "tp-fracture"


def test_tp_rung_off_by_default_is_structural_not_tp():
    done, rung, _ = perf_mcp._op_ladder_status(_MATMUL, "MatmulDeviceOperation", _ALL_DONE)
    assert rung == "structural"


def test_tp_fracture_precedes_kernels(monkeypatch):
    monkeypatch.setattr(perf_mcp, "_ttl_available", lambda: True)
    monkeypatch.setattr(perf_mcp, "_is_kernel_able", lambda oc: True)
    perf_mcp.set_tp_regime(True)
    done, rung, _ = perf_mcp._op_ladder_status(_MATMUL, "MatmulDeviceOperation", _KNOBS_DONE)
    assert rung == "tp-fracture"
    after_tp = _KNOBS_DONE + [{"op_signature": "MatmulDeviceOperation", "kernel_kind": "tp-fracture"}]
    done, rung, _ = perf_mcp._op_ladder_status(_MATMUL, "MatmulDeviceOperation", after_tp)
    assert rung in ("tt-lang", "cpp")


def test_kernel_rung_wedge_cap_advances(monkeypatch):
    monkeypatch.setattr(perf_mcp, "_ttl_available", lambda: True)
    monkeypatch.setattr(perf_mcp, "_is_kernel_able", lambda oc: True)
    perf_mcp.set_tp_regime(False)
    tl_wedged = _KNOBS_DONE + [
        {"op_signature": "MatmulDeviceOperation", "kernel_kind": "tt-lang", "wedged": True}
    ] * perf_mcp._MAX_KERNEL_WEDGES
    done, rung, _ = perf_mcp._op_ladder_status(_MATMUL, "MatmulDeviceOperation", tl_wedged)
    assert not done and rung == "cpp"
    both = tl_wedged + [
        {"op_signature": "MatmulDeviceOperation", "kernel_kind": "cpp", "wedged": True}
    ] * perf_mcp._MAX_KERNEL_WEDGES
    done, rung, _ = perf_mcp._op_ladder_status(_MATMUL, "MatmulDeviceOperation", both)
    assert rung == "structural"


def test_tp_candidate_requires_dense_matmul():
    perf_mcp.set_tp_regime(True)
    assert perf_mcp._tp_candidate(_MATMUL, "MatmulDeviceOperation") is True
    assert perf_mcp._tp_candidate(_MATMUL, "LayerNormDeviceOperation") is False


def test_tp_candidate_requires_memory_bound():
    perf_mcp.set_tp_regime(True)
    compute_bound = {**_MATMUL, "bound_by": "flop"}
    assert perf_mcp._tp_candidate(compute_bound, "MatmulDeviceOperation") is False


def test_tp_fracture_credited_only_with_shard_and_ccl(tmp_path, monkeypatch):
    monkeypatch.setattr(perf_mcp, "_MODEL_ROOT", tmp_path)
    monkeypatch.setattr(perf_mcp, "_KERNEL_LOG_PATH", tmp_path / "attempts.json")

    (tmp_path / "model.py").write_text("w = ShardTensorToMesh(mesh, dim=-1)\ny = ttnn.all_gather(z)\n")
    ok = perf_mcp.record_kernel_attempt("MatmulDeviceOperation", "tp-fracture", 1.0, True)
    assert ok["attempt"]["kernel_detected_in_source"] is True

    (tmp_path / "model.py").write_text("w = ShardTensorToMesh(mesh, dim=-1)\n")
    bad = perf_mcp.record_kernel_attempt("MatmulDeviceOperation", "tp-fracture", 1.0, True)
    assert bad["attempt"]["kernel_detected_in_source"] is False
