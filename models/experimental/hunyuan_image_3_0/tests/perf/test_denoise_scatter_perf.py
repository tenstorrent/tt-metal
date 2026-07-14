# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Tracy profiling for the denoise sequence-scatter path only.

Isolates ``HunyuanTtDenoiseStep._scatter`` (reshape, TILE layout, typecast, concat)
with realistic ``text_pre | img_tokens | text_post`` tensors on the 2×2 resident mesh.

Run:

    cd /path/to/tt-metal
    HUNYUAN_MODEL_DIR=/path/to/HunyuanImage-3.0 \\
    HY_VERBOSE=0 HY_NUM_LAYERS=4 \\
    python_env/bin/python -m tracy -p -r -v --op-support-count 10000 -m pytest \\
      models/experimental/hunyuan_image_3_0/tests/perf/test_denoise_scatter_perf.py -s --timeout=0

CSV filter (same signpost names as ``tt/pipeline.py``):

    python models/tt_transformers/scripts/op_perf_results.py \\
      generated/profiler/reports/*/ops_perf_results_*.csv --signpost start_scatter
"""

from __future__ import annotations

import os

os.environ.setdefault("TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT", "10000")

import pytest

from models.experimental.hunyuan_image_3_0.ref.weights import ensure_base_weights
from models.experimental.hunyuan_image_3_0.tests.perf.denoise_perf_fixtures import (
    build_denoise_perf_runtime,
    profile_denoise_region,
)

import ttnn


@pytest.fixture(scope="session", autouse=True)
def _hunyuan_checkpoint():
    return ensure_base_weights()


@pytest.fixture(scope="function")
def device_params(request):
    return {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 32768}


@pytest.mark.skipif(os.environ.get("CI") == "true", reason="perf test — run locally with tracy")
@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
def test_denoise_scatter_perf_tracy(mesh_device):
    profile_denoise_region(build_denoise_perf_runtime(mesh_device), "scatter")
