# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Shim: the t2_raw_kwalk_hxw benches live in ttnn/ttnn/operations/genify_matmul/perf_experiments/t2_raw_kwalk_hxw/
(pytest cannot collect a test file inside the ttnn package without importing ttnn twice)."""
import importlib.util
from pathlib import Path

_D = Path(__file__).resolve().parents[7] / "ttnn/ttnn/operations/genify_matmul/perf_experiments/t2_raw_kwalk_hxw"
for _name in ("test_hxw_op", "test_hxw_bench"):
    _p = _D / f"{_name}.py"
    if _p.exists():
        _spec = importlib.util.spec_from_file_location(f"_t2hxw_{_name}", _p)
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        globals().update({k: v for k, v in vars(_mod).items() if k.startswith("test_")})
