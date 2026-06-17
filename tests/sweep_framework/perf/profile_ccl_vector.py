# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""PROTOTYPE: run a single sweep vector in-process so it can be device-profiled
under ``python -m tracy``.

Why this exists
---------------
The sweeps runner's normal device-perf path enables the device profiler
*in-process* (``TT_METAL_DEVICE_PROFILER`` set after ttnn is already imported,
then per-op ``ReadDeviceProfiler`` on a forked child). For multi-device CCL ops
on FABRIC_2D that path deadlocks while building the profiler-instrumented
ethernet/dispatch kernels at mesh open (``tt::llrt::get_risc_binary`` waiters).

Every CCL/model device-perf test in the repo instead profiles via the tracy
*subprocess* wrapper (``run_device_profiler`` -> ``python -m tracy -m pytest``),
which sets ``TT_METAL_DEVICE_PROFILER`` from process start (clean init) and
harvests device data from the profiler CSV after the run. This pytest is the
single-vector entry point the sweeps runner drives for that pattern.

Driven by env vars set by ``sweeps_runner._collect_ccl_device_perf_via_tracy``:
  SWEEP_PROF_MODULE  e.g. ``model_traced.all_gather_async_model_traced``
  SWEEP_PROF_HASH    the vector's input/config hash
  SWEEP_PROF_SUITE   suite name (default ``model_traced``)
plus the usual ``TTNN_VECTORS_EXPORT_DIR`` / ``TTNN_DISPATCH_AXIS`` that select
the partition (inherited from the parent runner).
"""

import importlib
import os
import pathlib
import sys

import pytest

# Make `framework`, `sweeps`, `sweep_utils` importable the same way the runner does.
_SWEEP_ROOT = str(pathlib.Path(__file__).resolve().parents[1])  # tests/sweep_framework
if _SWEEP_ROOT not in sys.path:
    sys.path.insert(0, _SWEEP_ROOT)


class _ProfileConfig:
    # Device perf is collected by tracy at the subprocess level, NOT in-process,
    # so keep all in-process measurement OFF here (this is exactly what avoids
    # the per-op ReadDeviceProfiler that deadlocks multi-device fabric ops).
    measure_device_perf = False
    measure_perf = False
    measure_perf_with_cache = False
    measure_memory = False


def test_profile_ccl_vector():
    module_name = os.environ.get("SWEEP_PROF_MODULE")
    input_hash = os.environ.get("SWEEP_PROF_HASH")
    suite = os.environ.get("SWEEP_PROF_SUITE", "model_traced")
    if not module_name or not input_hash:
        pytest.skip("SWEEP_PROF_MODULE / SWEEP_PROF_HASH not set; nothing to profile.")

    from framework.serialize import deserialize_vector_structured
    from framework.vector_source import VectorSourceFactory
    from sweep_utils.perf_utils import run_single

    vector_source = VectorSourceFactory.create_source("vectors_export")
    vectors = vector_source.load_vectors(module_name, suite, input_hash)
    assert vectors, f"vector {input_hash} not found for module={module_name} suite={suite}"
    vector = vectors[0]
    for key in ("validity", "invalid_reason", "status", "sweep_name", "suite_name"):
        vector.pop(key, None)
    vector = deserialize_vector_structured(vector)

    test_module = importlib.import_module("sweeps." + module_name)

    # Single attempt only. The transient cold-build race (tt_elffile.cpp:405 on
    # the profiler-instrumented eth kernels) must NOT be retried in-process: a
    # second open in the same process deadlocks on the empty get_risc_binary
    # cache slot left by the first attempt's throw. The retry is done at the
    # subprocess (fresh-process) level by the caller
    # (sweeps_runner._collect_ccl_device_perf_via_tracy) so each attempt has a
    # clean get_risc_binary cache and a warmer on-disk kernel cache.
    fixture = test_module.mesh_device_fixture()
    device, _ = next(fixture)
    try:
        status, message, *_ = run_single(test_module, vector, device, _ProfileConfig())
    finally:
        try:
            next(fixture)  # run fixture teardown (close device if it opened one)
        except StopIteration:
            pass

    assert status, f"vector {input_hash} failed while profiling: {message}"
