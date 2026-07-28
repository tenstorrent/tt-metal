# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The report must not assert things that are not true. It goes out for confirmation.

Two false statements were being printed for llama3_1_8b_p150:

  "(tok/s/u — N/A: not an LLM decode pipeline)"   -- it IS one; it runs a traced KV-cache decode.
      The ms branch is taken because active_bytes is 0, i.e. the physics numerator was never
      computed. Explaining a missing input by inventing a property of the model is worse than
      saying nothing.

  "modeled floor : 341.47 ms (Σ per-op roofline floors)"   -- reads as complete and arbitrary. It is
      physics (bytes/bandwidth dominates) but sums each bucket's top_ops only, so it is a LOWER
      bound over ~86% of device time.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def _sm():
    spec = importlib.util.spec_from_file_location("sm_truth_ut", _ROOT / "cc_optimize" / "summary.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["sm_truth_ut"] = m
    spec.loader.exec_module(m)
    return m


_PROFILE = {
    "device_ms": 664.2,
    "buckets": [
        {"id": "matmul", "device_ms": 560.3, "top_ops": [{"device_ms": 469.0}]},
        {"id": "reduction", "device_ms": 53.4, "top_ops": [{"device_ms": 53.4}]},
        {"id": "host_overhead", "device_ms": 46.6, "top_ops": []},
    ],
}


def test_it_does_not_claim_a_decode_model_is_not_a_decode_model():
    m = _sm()
    out = m._roofline_lines({"modeled_floor_ms": 341.47, "active_bytes": 0, "peak_bw_gbps": 512.0}, 615.69)
    txt = "\n".join(out)
    assert "not an LLM decode pipeline" not in txt, txt
    assert "active_bytes not computed" in txt, txt


def test_it_still_says_not_decode_when_that_is_actually_why():
    m = _sm()
    out = m._roofline_lines(
        {"modeled_floor_ms": 341.47, "active_bytes": 12345, "peak_bw_gbps": 512.0, "is_llm_decode": False}, 615.69
    )
    assert "not an LLM decode pipeline" in "\n".join(out)


def test_the_floor_states_the_physics_and_its_coverage():
    m = _sm()
    basis = m._floor_basis(_PROFILE)
    assert "bytes/BW" in basis, basis
    assert "covers" in basis and "%" in basis, basis


def test_the_coverage_excludes_host_overhead_and_uncounted_ops():
    """522.4 of 664.2 device_ms is covered by top_ops (host_overhead excluded) -> 79%."""
    m = _sm()
    basis = m._floor_basis(_PROFILE)
    assert "79%" in basis, basis


def test_a_profile_without_buckets_degrades_to_the_bare_basis():
    m = _sm()
    assert m._floor_basis({}) == "Σ per-op max(FLOPs/peak, bytes/BW, dispatch)"
    assert m._floor_basis(None).startswith("Σ per-op max")


def test_the_rendered_floor_line_carries_the_basis_not_an_opaque_label():
    """Testing _floor_basis alone let a mutation revert the render line to the opaque
    "(Σ per-op roofline floors)" while every test still passed."""
    m = _sm()
    out = m._roofline_lines({"modeled_floor_ms": 341.47, "active_bytes": 0, "peak_bw_gbps": 512.0}, 615.69, _PROFILE)
    line = next(l for l in out if "modeled floor" in l)
    assert "bytes/BW" in line, line
    assert "covers" in line and "%" in line, line
    assert line.strip().endswith(")"), line
