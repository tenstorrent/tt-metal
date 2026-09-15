# SPDX-License-Identifier: Apache-2.0
"""Material-gap threshold must scale with the workload's total device_ms.

A fixed 0.25ms threshold discards real per-module matmul gaps (~0.1-0.2ms on a ~2ms
module) as noise, so those ops never get optimized. The threshold must be small at
per-module scale yet never exceed the 0.25ms whole-model default at large scale.
"""
import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "perf_mcp_mg",
    str(Path(__file__).resolve().parents[1] / "cc_optimize" / "perf_mcp.py"),
)
perf_mcp = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(perf_mcp)

_mg = perf_mcp._material_gap_ms


def test_per_module_small_gap_is_material():
    # module device_ms ~2ms -> threshold ~0.06ms, so the real 0.17ms o_proj gap counts.
    thresh = _mg(2.0)
    assert thresh < 0.17, f"0.17ms gap must be material on a 2ms module (thresh={thresh})"
    assert thresh >= perf_mcp._MATERIAL_GAP_FLOOR


def test_whole_model_unchanged():
    # large device_ms -> threshold capped at the 0.25ms whole-model default (no regression).
    assert _mg(40.0) == perf_mcp._MATERIAL_GAP_MS
    assert _mg(120.0) == perf_mcp._MATERIAL_GAP_MS


def test_tiny_op_uses_noise_floor():
    # sub-ms workload -> never below the absolute noise floor.
    assert _mg(0.5) == perf_mcp._MATERIAL_GAP_FLOOR
    assert _mg(0.0) == perf_mcp._MATERIAL_GAP_FLOOR


def test_module1_matmul_gaps_now_targeted():
    # the exact freshopt5 module-1 case: 3.58ms module, matmul gaps 0.12 / 0.17ms.
    thresh = _mg(3.58)
    assert 0.1206 >= thresh and 0.1737 >= thresh, (
        f"both attention matmul gaps must clear the material bar (thresh={thresh})"
    )
