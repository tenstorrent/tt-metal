# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Zone profile of the perf case: spin + 5 calls, traced (like eval/ccl_perf.py), with AGRN_ZONES kernels.

  AGRN_ZONES=1 TT_METAL_DEVICE_PROFILER=1 scripts/run_safe_pytest.sh --no-precompile <this file> -s
then: python tests/ttnn/unit_tests/operations/all_gather_rms_norm_kimi/zones.py"""

import os

import pytest
import torch
import ttnn

from eval import ccl_perf
from eval.golden_tests.all_gather_rms_norm_kimi.conftest import focus_mesh_or_skip
from eval.golden_tests.all_gather_rms_norm_kimi.helpers import (
    create_ttnn_gamma,
    create_ttnn_input_tensor,
    random_gamma,
    random_stacked,
)
from ttnn.operations.all_gather_rms_norm_kimi import all_gather_rms_norm_kimi, default_compute_kernel_config


@pytest.fixture
def perf_mesh(_fabric_mesh_cache):
    return _fabric_mesh_cache.get(focus_mesh_or_skip(), trace_region_size=ccl_perf.TRACE_REGION_SIZE)


def test_zone_probe(perf_mesh):
    mesh = perf_mesh
    shape = (1, 1, 640, 384)
    x = create_ttnn_input_tensor(
        random_stacked(mesh, shape, torch.bfloat16), mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )
    g = create_ttnn_gamma(random_gamma(1536, torch.bfloat16), mesh, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    cfg = default_compute_kernel_config()
    if os.environ.get("AGRN_FIDELITY"):
        cfg.math_fidelity = getattr(ttnn.MathFidelity, os.environ["AGRN_FIDELITY"])
    op = lambda: all_gather_rms_norm_kimi(x, gamma=g, cluster_axis=1, compute_kernel_config=cfg)
    spin = ccl_perf.spin_op(mesh)
    spin()
    for _ in range(3):
        op()
    ttnn.synchronize_device(mesh)
    ttnn.ReadDeviceProfiler(mesh)
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    spin()
    outs = [op() for _ in range(int(os.environ.get("AGRN_ITERS", "5")))]
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
    ttnn.ReadDeviceProfiler(mesh)
    ttnn.release_trace(mesh, tid)
