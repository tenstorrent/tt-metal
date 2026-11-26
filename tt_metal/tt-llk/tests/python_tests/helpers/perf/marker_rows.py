# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Device log rows and the tt-metal marker decoders, for the hardware-free decoder tests."""

import importlib.util
from functools import lru_cache
from pathlib import Path

import pytest


@lru_cache(maxsize=None)
def _load_tracy_analysis(path):
    spec = importlib.util.spec_from_file_location(
        "perf_counter_analysis_under_test", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tracy_analysis():
    """tools/tracy/perf_counter_analysis.py of the enclosing tt-metal checkout; skips outside one."""
    pytest.importorskip("pandas")
    pytest.importorskip("loguru")
    path = (
        Path(__file__).resolve().parents[6]
        / "tools"
        / "tracy"
        / "perf_counter_analysis.py"
    )
    if not path.is_file():
        pytest.skip("not inside a tt-metal checkout")
    return _load_tracy_analysis(str(path))


def marker_row(run, risc, marker_id, time, data, core=(1, 2), device=0):
    """One device log row in the shape the marker decoders take."""
    return {
        "device_id": device,
        "run_host_id": run,
        "core_x": core[0],
        "core_y": core[1],
        "risc_type": risc,
        "marker_id": marker_id,
        "record_time": time,
        "data": data,
    }
