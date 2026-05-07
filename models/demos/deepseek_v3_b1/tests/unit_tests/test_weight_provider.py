# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Performance benchmark for ``CacheWeightProvider.load_moe_layer`` against a warm TensorCache."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import pytest
from loguru import logger

import ttnn
from models.common.utility_functions import is_slow_dispatch
from models.demos.deepseek_v3_b1.demo.weight_provider import CacheWeightProvider
from models.demos.deepseek_v3_b1.tests.unit_tests.test_prepare_weights import _deallocate_layer
from models.demos.deepseek_v3_b1.weights.prepare import NUM_ROUTED_EXPERTS

MOE_LAYER_IDX = 3
CACHE_PERF_ROOT = Path(os.environ.get("CACHE_PERF_ROOT", str(Path.home() / ".cache" / "deepseek_v3_b1_cache_perf")))


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}],
    indirect=True,
)
def test_cache_weight_provider_load_moe_layer_perf(bh_2d_mesh_device: Any, hf_model_path: Path) -> None:
    """Benchmark ``CacheWeightProvider.load_moe_layer`` against a warm TensorCache (CAS layout)."""
    if not is_slow_dispatch():
        pytest.skip("CacheWeightProvider MoE load benchmark targets slow dispatch only")

    if bh_2d_mesh_device.shape[0] * bh_2d_mesh_device.shape[1] < 8:
        pytest.skip("Test requires 8 devices (4x2 mesh)")

    env_root = os.environ.get("CACHE_WEIGHT_PROVIDER_MOE_ROOT", "").strip()
    tensor_cache_root = Path(env_root).resolve() if env_root else CACHE_PERF_ROOT / "moe_layer_cache"
    tensor_cache_root.mkdir(parents=True, exist_ok=True)

    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((4, 2)))
    provider = CacheWeightProvider(tensor_cache_root, hf_model_path)

    total_size_mb = NUM_ROUTED_EXPERTS * 3 * 7168 * 2048 / (1024 * 1024)

    logger.info("Loading MoE layer {} from TensorCache root {}", MOE_LAYER_IDX, tensor_cache_root)
    t0 = time.perf_counter()
    loaded = provider.load_moe_layer(MOE_LAYER_IDX, submesh)
    elapsed_s = time.perf_counter() - t0
    _deallocate_layer(loaded)

    throughput_mbs = (total_size_mb / elapsed_s) if elapsed_s > 0 else 0.0
    logger.info("MoE layer {} loaded in {:.6f}s", MOE_LAYER_IDX, elapsed_s)
    logger.info("Throughput (payload MB / elapsed s): {:.1f} MB/s", throughput_mbs)
