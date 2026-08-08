# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""GLM-4.7-REAP e2e decode perf test.

Traced sampling decode (the production decode path); asserts mean per-token decode
latency is under a target and prints `decode_mean_ms=` for the tt_hw_planner
`optimize` tool / sweeps to parse. Full production env config (incl. the decode-opt
knobs) is defaulted in conftest.py.

Run:
  export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
  ./python_env/bin/python -m pytest -svq \
    models/experimental/glm4_moe/tests/test_glm4_moe_perf.py
"""
from __future__ import annotations

import os

import pytest
from loguru import logger

# Target mean decode ms/token (B1). Baseline (FUSE-off decode winner) measured ~140.7 ms;
# generous default leaves headroom so noise doesn't flake CI. Override for tighter gating.
PERF_TARGET_MS = float(os.environ.get("GLM4_MOE_TEST_PERF_TARGET_MS", "175"))
PERF_MAX_NEW = int(os.environ.get("GLM4_MOE_TEST_PERF_MAX_NEW", "128"))


@pytest.mark.timeout(3600)  # 218B build + 128 decode steps far exceeds the default 300s
def test_glm4_moe_perf(glm4_model):
    res = glm4_model.generate(PERF_MAX_NEW, enable_trace=True, sampling=True, warmup=True)
    mean_ms = res["decode_mean_ms"]
    logger.info(
        f"[perf] decode mean={mean_ms:.1f} ms min={res['decode_min_ms']:.1f} "
        f"max={res['decode_max_ms']:.1f} prefill_s={res['prefill_s']:.3f}"
    )
    # Machine-parseable lines for the optimize tool / sweeps.
    print(f"decode_mean_ms={mean_ms:.2f}", flush=True)
    print(f"prefill_s={res['prefill_s']:.3f}", flush=True)
    assert mean_ms == mean_ms, "decode_mean_ms is NaN (need >=2 decode steps)"  # NaN guard
    assert mean_ms <= PERF_TARGET_MS, f"decode mean {mean_ms:.1f} ms exceeds target {PERF_TARGET_MS} ms"
