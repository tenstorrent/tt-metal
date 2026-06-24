# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Performance benchmarks for ``CacheWeightProvider.load_moe_layer``.

Two variants:

- ``test_cache_weight_provider_load_moe_layer_perf``: DRAM-only routed experts
  against a warm TensorCache (CAS layout).
- ``test_cache_weight_provider_load_moe_layer_sram_perf``: same, but with
  SRAM-hot-experts enabled. Exercises ``prepare_compressed_sram_slots`` (and
  by extension ``CompressedTensorAssigner.assign``, ``pack_bfp_tile``, and the
  per-core L1 upload path) on top of the DRAM cache load. Intended as a
  regression gate for the assigner short-circuit and ``pack_bfp_tile`` work.

Environment variables
---------------------
``CACHE_PERF_ROOT``         Override the on-disk TensorCache root. Defaults to
                            ``~/.cache/deepseek_v3_b1_cache_perf``. Point at the
                            production demo's cache (e.g. ``--cache-path``) to
                            measure warm-path latency against real artifacts.
``BSPM_DIR``                Mirror of the demo CLI's ``--bspm-dir`` — model-specific
                            BSPM root (e.g. ``.../bit_sculpt/results/<model>``).
                            When set, the routed-expert path uses per-expert
                            mixed-precision assignments (256 unique cache entries
                            per projection per layer, matches production load
                            behaviour). When unset, falls back to uniform BFP4
                            (1 cache entry per projection, lower bound only).
``BSPM_BUDGET``             Mirror of ``--bspm-budget`` (default 3.5).
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import pytest
from loguru import logger

import ttnn
from conftest import requires_hybrid_allocator
from models.common.utility_functions import is_slow_dispatch
from models.demos.deepseek_v3_b1.demo.mesh_device_context import DEFAULT_WORKER_L1_SIZE, _worker_l1_size_for_rank
from models.demos.deepseek_v3_b1.demo.weight_provider import CacheWeightProvider
from models.demos.deepseek_v3_b1.tests.unit_tests.test_prepare_weights import _deallocate_layer
from models.demos.deepseek_v3_b1.weights.prepare import NUM_ROUTED_EXPERTS
from models.demos.deepseek_v3_b1.weights.transforms.sram_experts import (
    SramExpertCoreGrids,
    _load_routing_frequencies,
    build_sram_hot_expert_config,
)

MOE_LAYER_IDX = 3
SRAM_HOT_EXPERTS_CEILING = 64
CACHE_PERF_ROOT = Path(os.environ.get("CACHE_PERF_ROOT", str(Path.home() / ".cache" / "deepseek_v3_b1_cache_perf")))
DEFAULT_BSPM_BUDGET = 3.5


def _resolve_tensor_cache_root() -> Path:
    env_root = os.environ.get("CACHE_WEIGHT_PROVIDER_MOE_ROOT", "").strip()
    root = Path(env_root).resolve() if env_root else CACHE_PERF_ROOT / "moe_layer_cache"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _resolve_bspm_config() -> tuple[Path | None, float]:
    """Read optional BSPM directory + budget from env so callers can mirror the demo CLI.

    Set ``BSPM_DIR`` (matches ``--bspm-dir``) to point at the model-specific BSPM root
    e.g. ``/data/.../bit_sculpt/results/deepseek-r1-0528``; optionally set
    ``BSPM_BUDGET`` (matches ``--bspm-budget``, default 3.5). When ``BSPM_DIR`` is
    unset, the uniform-BFP4 fallback path is exercised (1 cache entry/projection,
    no per-expert assignment slicing) — useful for measuring the new TP8 DRAM
    compressed path's lower bound. When set, every expert/projection gets its own
    ``assignment_hash`` (256× more cache entries per layer) which matches the
    production demo's load behaviour.
    """
    bspm_dir_env = os.environ.get("BSPM_DIR", "").strip()
    bspm_dir = Path(bspm_dir_env).resolve() if bspm_dir_env else None
    bspm_budget_env = os.environ.get("BSPM_BUDGET", "").strip()
    bspm_budget = float(bspm_budget_env) if bspm_budget_env else DEFAULT_BSPM_BUDGET
    return bspm_dir, bspm_budget


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

    tensor_cache_root = _resolve_tensor_cache_root()
    bspm_dir, bspm_budget = _resolve_bspm_config()

    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((4, 2)))
    provider = CacheWeightProvider(
        tensor_cache_root,
        hf_model_path,
        bspm_dir=bspm_dir,
        bspm_budget=bspm_budget,
    )

    total_size_mb = NUM_ROUTED_EXPERTS * 3 * 7168 * 2048 / (1024 * 1024)

    bspm_mode = f"BSPM (dir={bspm_dir}, budget={bspm_budget})" if bspm_dir is not None else "uniform-BFP4"
    logger.info("Loading MoE layer {} from TensorCache root {} ({})", MOE_LAYER_IDX, tensor_cache_root, bspm_mode)
    t0 = time.perf_counter()
    loaded = provider.load_moe_layer(MOE_LAYER_IDX, submesh)
    elapsed_s = time.perf_counter() - t0
    _deallocate_layer(loaded)

    throughput_mbs = (total_size_mb / elapsed_s) if elapsed_s > 0 else 0.0
    logger.info("MoE layer {} loaded in {:.6f}s", MOE_LAYER_IDX, elapsed_s)
    logger.info("Throughput (payload MB / elapsed s): {:.1f} MB/s", throughput_mbs)


@requires_hybrid_allocator
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D, "worker_l1_size": DEFAULT_WORKER_L1_SIZE}],
    indirect=True,
)
def test_cache_weight_provider_load_moe_layer_sram_perf(bh_2d_mesh_device: Any, hf_model_path: Path) -> None:
    """Benchmark ``CacheWeightProvider.load_moe_layer`` with SRAM hot experts enabled.

    Exercises both:
      - the warm-cache DRAM routed-expert load (same as the sibling test), and
      - ``prepare_compressed_sram_slots`` for the SRAM-hot subset (assigner,
        ``pack_bfp_tile``, per-core L1 upload).

    With the single-format (``bfp4``) assigner short-circuit the SRAM-prep
    portion should be CPU-bound on ``pack_bfp_tile`` only; the test is meant
    to catch regressions in any of those phases. Skips when
    ``TT_METAL_ALLOCATOR_MODE_HYBRID=1`` is not set (required by the per-core
    L1 allocator path).
    """
    if not is_slow_dispatch():
        pytest.skip("SRAM hot experts targets slow dispatch only")

    if bh_2d_mesh_device.shape[0] * bh_2d_mesh_device.shape[1] < 8:
        pytest.skip("Test requires 8 devices (4x2 mesh)")

    tensor_cache_root = _resolve_tensor_cache_root()
    bspm_dir, bspm_budget = _resolve_bspm_config()

    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((4, 2)))

    freqs = _load_routing_frequencies()
    ranked = build_sram_hot_expert_config([MOE_LAYER_IDX], freqs)
    if MOE_LAYER_IDX not in ranked:
        pytest.skip(f"No routing frequencies available for layer {MOE_LAYER_IDX}")
    sram_hot_experts = {k: v[:SRAM_HOT_EXPERTS_CEILING] for k, v in ranked.items()}

    # 4 procs => DEFAULT_WORKER_L1_SIZE (no LM-head extension); matches the
    # `worker_l1_size` device_params above so the boundary used by
    # `prepare_compressed_sram_slots` aligns with the device's actual L1.
    worker_l1_size = _worker_l1_size_for_rank(num_procs=4)
    assert worker_l1_size == DEFAULT_WORKER_L1_SIZE, "test assumes 4-proc default L1 budget"

    provider = CacheWeightProvider(
        tensor_cache_root,
        hf_model_path,
        sram_hot_experts=sram_hot_experts,
        sram_core_grids=SramExpertCoreGrids.shared_expert_mirror(),
        worker_l1_size=worker_l1_size,
        bspm_dir=bspm_dir,
        bspm_budget=bspm_budget,
    )

    total_size_mb = NUM_ROUTED_EXPERTS * 3 * 7168 * 2048 / (1024 * 1024)
    sram_candidates = len(sram_hot_experts.get(MOE_LAYER_IDX, []))

    bspm_mode = f"BSPM (dir={bspm_dir}, budget={bspm_budget})" if bspm_dir is not None else "uniform-BFP4"
    logger.info(
        "Loading MoE layer {} (SRAM hot experts: {} candidates, {}) from TensorCache root {}",
        MOE_LAYER_IDX,
        sram_candidates,
        bspm_mode,
        tensor_cache_root,
    )
    t0 = time.perf_counter()
    loaded = provider.load_moe_layer(MOE_LAYER_IDX, submesh)
    elapsed_s = time.perf_counter() - t0
    _deallocate_layer(loaded)

    throughput_mbs = (total_size_mb / elapsed_s) if elapsed_s > 0 else 0.0
    logger.info("MoE layer {} (SRAM-hot) loaded in {:.6f}s", MOE_LAYER_IDX, elapsed_s)
    logger.info("Throughput (DRAM payload MB / elapsed s): {:.1f} MB/s", throughput_mbs)
