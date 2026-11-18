# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Refinement 16: mid-M compute efficiency. The 17 compute-bound model cases (text encoder at 512 tokens:
bf16 x bfp8, HiFi2; Qwen3.8 prefill M = 512 / 1024 / 2048 / 4096: bf16 x bfp4, LoFi), fp32 DEST, DRAM
interleaved, at their golden config. Same env knobs as test_genify_matmul_perf_r11.py (GENIFY_MATMUL_PERF=1,
_ONLY, _KNOBS, _FORCE, _PLAN, _CHECK)."""
import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "_genify_perf_r11", Path(__file__).with_name("test_genify_matmul_perf_r11.py")
)
_r11 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_r11)
pytestmark = _r11.pytestmark
_run = _r11.test_perf_r11

# (name, m, k, n, B dtype, fidelity, epilogue, roofline_ns)
R16_CASES = [
    ("te_gate", 512, 5120, 16384, "bfloat8_b", "HiFi2", "none", 403500),
    ("te_down", 512, 16384, 5120, "bfloat8_b", "HiFi2", "none", 403500),
    ("te_qkv", 512, 5120, 3072, "bfloat8_b", "HiFi2", "none", 75700),
    ("te_o", 512, 4096, 2560, "bfloat8_b", "HiFi2", "none", 50400),
    ("q512_in", 512, 5120, 4160, "bfloat4_b", "LoFi", "none", 51200),
    ("q512_gu", 512, 5120, 8704, "bfloat4_b", "LoFi", "swiglu", 107200),
    ("q512_down", 512, 4352, 5120, "bfloat4_b", "LoFi", "none", 53600),
    ("q512_qkvg", 512, 5120, 3584, "bfloat4_b", "LoFi", "none", 44100),
    ("q512_out", 512, 1536, 5120, "bfloat4_b", "LoFi", "none", 24400),
    ("q1024_out", 1024, 1536, 5120, "bfloat4_b", "LoFi", "none", 39200),
    ("q1024_qkvg", 1024, 5120, 3584, "bfloat4_b", "LoFi", "none", 88300),
    ("q1024_in", 1024, 5120, 4160, "bfloat4_b", "LoFi", "none", 102400),
    ("q1024_down", 1024, 4352, 5120, "bfloat4_b", "LoFi", "none", 107200),
    ("q2048_out", 2048, 1536, 5120, "bfloat4_b", "LoFi", "none", 75700),
    ("q4096_out", 4096, 1536, 5120, "bfloat4_b", "LoFi", "none", 151300),
]


@pytest.mark.parametrize("case", R16_CASES, ids=[c[0] for c in R16_CASES])
def test_perf_r16(case, device, monkeypatch, request):
    _run(case, device, monkeypatch, request)
