# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Tracy profiling harness for Hunyuan denoise submodules and full step.

Optimize in this order — each test isolates one region with matching signpost names
used in ``tt/pipeline.py``:

  1. patch_embed   ``start_patch_embed`` / ``stop_patch_embed``
  2. scatter       ``start_scatter`` / ``stop_scatter``
  3. backbone      ``start_backbone`` / ``stop_backbone``
  4. final_layer   ``start_final_layer`` / ``stop_final_layer`` (includes img slice)
  5. full step     ``start_denoise_step`` / ``stop_denoise_step``

**Recommended — one region (patch_embed example):**

    cd /path/to/tt-metal
    HUNYUAN_MODEL_DIR=/path/to/HunyuanImage-3.0 \\
    HY_VERBOSE=0 HY_NUM_LAYERS=4 \\
    python_env/bin/python -m tracy -p -r -v --op-support-count 50000 -m pytest \\
      models/experimental/hunyuan_image_3_0/tests/perf/test_denoise_perf_tracy.py \\
      -k test_denoise_perf_tracy_patch_embed -s --timeout=0

**All regions in one capture** (filter CSV per region with ``op_perf_results.py --signpost``):

    python_env/bin/python -m tracy -p -r -v --op-support-count 50000 -m pytest \\
      models/experimental/hunyuan_image_3_0/tests/perf/test_denoise_perf_tracy.py \\
      -k test_denoise_perf_tracy_all_regions -s --timeout=0

**Per-region CSV summary:**

    python models/tt_transformers/scripts/op_perf_results.py \\
      generated/profiler/reports/*/ops_perf_results_*.csv --signpost start_patch_embed

Environment overrides:

    HUNYUAN_MODEL_DIR=/path/to/HunyuanImage-3.0
    HY_NUM_LAYERS=4                  # backbone depth (use 32 for production-like)
    HY_DENOISE_GRID=8                # latent token grid side (8 → 64 image tokens)
    HY_DENOISE_TEXT_PRE=32           # T2I prefix text tokens
    HY_DENOISE_TEXT_POST=32          # T2I suffix text tokens
    HY_DENOISE_PERF_ITERS=3          # timed iterations inside each signpost window
    HY_DENOISE_PERF_WARMUP=1         # warmup passes before each timed window
    HY_DENOISE_RESIDENT_TEMB=1       # WIDTH_SHARDED M=32 t_emb via TimestepEmbedder (0 = host interleaved)
"""

from __future__ import annotations

import os

os.environ.setdefault("TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT", "50000")

import pytest

from models.experimental.hunyuan_image_3_0.ref.weights import ensure_base_weights
from models.experimental.hunyuan_image_3_0.tests.perf.denoise_perf_fixtures import (
    REGION_SIGNPOSTS,
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
def test_denoise_perf_tracy_patch_embed(mesh_device):
    profile_denoise_region(build_denoise_perf_runtime(mesh_device), "patch_embed")


@pytest.mark.skipif(os.environ.get("CI") == "true", reason="perf test — run locally with tracy")
@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
def test_denoise_perf_tracy_scatter(mesh_device):
    profile_denoise_region(build_denoise_perf_runtime(mesh_device), "scatter")


@pytest.mark.skipif(os.environ.get("CI") == "true", reason="perf test — run locally with tracy")
@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
def test_denoise_perf_tracy_backbone(mesh_device):
    profile_denoise_region(build_denoise_perf_runtime(mesh_device), "backbone")


@pytest.mark.skipif(os.environ.get("CI") == "true", reason="perf test — run locally with tracy")
@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
def test_denoise_perf_tracy_final_layer(mesh_device):
    profile_denoise_region(build_denoise_perf_runtime(mesh_device), "final_layer")


@pytest.mark.skipif(os.environ.get("CI") == "true", reason="perf test — run locally with tracy")
@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
def test_denoise_perf_tracy_full_step(mesh_device):
    profile_denoise_region(build_denoise_perf_runtime(mesh_device), "full")


@pytest.mark.skipif(os.environ.get("CI") == "true", reason="perf test — run locally with tracy")
@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
@pytest.mark.parametrize("region", list(REGION_SIGNPOSTS.keys()))
def test_denoise_perf_tracy_all_regions(mesh_device, region):
    """One Tracy capture profiling every denoise region sequentially."""
    rt = build_denoise_perf_runtime(mesh_device)
    profile_denoise_region(rt, region)
