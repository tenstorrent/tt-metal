# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Shim: the t2p2_swiglu_bank_exchange bench lives in
ttnn/ttnn/operations/genify_matmul/perf_experiments/t2p2_swiglu_bank_exchange/ (pytest cannot collect a test file
inside the ttnn package without importing ttnn twice)."""
import importlib.util
from pathlib import Path

_D = (
    Path(__file__).resolve().parents[7]
    / "ttnn/ttnn/operations/genify_matmul/perf_experiments/t2p2_swiglu_bank_exchange"
)
_spec = importlib.util.spec_from_file_location("_t2p2swx_test_swx_op", _D / "test_swx_op.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
globals().update({k: v for k, v in vars(_mod).items() if k.startswith("test_")})
