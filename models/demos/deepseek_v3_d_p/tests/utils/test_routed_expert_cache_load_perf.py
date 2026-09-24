# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""How fast one layer's routed-expert weights load from a prebuilt TTNN cache.

NOT RUN IN CI, deliberately. It reads a real 8.86 GiB cache off /mnt/models and reports a
wall-clock number; no threshold is asserted, because a meaningful one has to be calibrated per
filesystem and per host and that work has not been done (tenstorrent/tt-metal#<issue>). Run it by
hand when you touch the host->device upload path:

    TT_KIMI_PREFILL_TTNN_CACHE=/mnt/models/moonshotai/Kimi-K2_7-Code-Cache/Kimi-K2_7-Code-Cache-prefill \\
    RE_CACHE_PERF=1 pytest models/demos/deepseek_v3_d_p/tests/utils/test_routed_expert_cache_load_perf.py -s

Iteration 0 is a cold read unless the cache is already in the page cache; the reported median
covers iterations 1..N-1 only. This is the test that caught the 10x pinned-memory regression
(#52893): 633 ms with TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0 vs 6741 ms with it unset.
"""

import os
import statistics
import time
from pathlib import Path

import pytest
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.reference.kimi_k2_7_config import KimiK27Config
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric2d_device_params
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import ExpertMapping, extract_mesh_config, get_ep_mesh_mapper
from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import DEFAULT_ROUTED_EXPERT_WEIGHTS_DTYPE, TtRoutedExpert
from models.demos.deepseek_v3_d_p.utils.chunk_config import PREFILL_CHUNK_TOKENS_PER_CHIP
from models.demos.deepseek_v3_d_p.utils.fast_cache_checker import init_checker

KIMI_CACHE_ENV = "TT_KIMI_PREFILL_TTNN_CACHE"
OPT_IN_ENV = "RE_CACHE_PERF"
ITERS = int(os.environ.get("RE_WARM_ITERS", "5"))


@pytest.mark.skipif(
    os.environ.get(OPT_IN_ENV) != "1",
    reason=f"load-time benchmark, no pass/fail threshold yet -- set {OPT_IN_ENV}=1 to run by hand",
)
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param(
            (8, 4),
            fabric2d_device_params(),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="fabric2d-8x4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("layer_idx", [1], ids=["layer1"])
@pytest.mark.timeout(1800)
def test_routed_expert_cache_load_perf(mesh_device, device_params, layer_idx):
    root = os.environ.get(KIMI_CACHE_ENV)
    if not root:
        pytest.skip(f"set {KIMI_CACHE_ENV} to the Kimi K2.7 prefill TTNN cache root")
    rows, cols = mesh_device.shape
    cache_dir = Path(root) / f"kimi_k2_7_bh_{mesh_device.get_num_devices()}dev" / f"{rows}x{cols}"
    if not cache_dir.is_dir():
        pytest.skip(f"no cache at {cache_dir}")

    experts_per_chip = KimiK27Config.NUM_ROUTED_EXPERTS // mesh_device.get_num_devices()
    prefix = f"layer_{layer_idx}.routed_expert"
    weights_dtype = DEFAULT_ROUTED_EXPERT_WEIGHTS_DTYPE

    init_checker(cache_dir)
    # Dtype-aware on purpose: a cache built at another dtype reads as complete to a blind check,
    # and cache-only construction then loads the empty placeholder and times nothing.
    assert TtRoutedExpert.check_cache_complete(
        cache_dir, prefix, experts_per_chip, weights_dtype
    ), f"routed-expert cache incomplete for {prefix} ({weights_dtype.name}) at {cache_dir}"

    mesh_config = extract_mesh_config(mesh_device)
    global_expert_idx_tt = ttnn.from_torch(
        ExpertMapping.create_global_expert_idx_table(
            experts_per_chip=experts_per_chip,
            dispatch_group_size=mesh_config.dispatch_group_size,
            num_dispatch_groups=mesh_config.num_dispatch_groups,
        ),
        mesh_mapper=get_ep_mesh_mapper(mesh_device),
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        dtype=ttnn.uint32,
    )
    global_expert_idx_tt = ttnn.squeeze(ttnn.squeeze(global_expert_idx_tt, 0), 0)

    times = []
    for i in range(ITERS):
        ttnn.synchronize_device(mesh_device)
        start = time.perf_counter()
        expert = TtRoutedExpert(
            mesh_device=mesh_device,
            experts_per_chip=experts_per_chip,
            global_expert_idx_table=global_expert_idx_tt,
            emb_dim=KimiK27Config.EMB_SIZE,
            hidden_dim=KimiK27Config.MOE_INTERMEDIATE_SIZE,
            max_tokens=mesh_config.dispatch_group_size * PREFILL_CHUNK_TOKENS_PER_CHIP,
            torch_weights=None,
            weights_dtype=weights_dtype,
            weight_cache_path=cache_dir,
            cache_name_prefix=prefix,
            activation=ttnn.RoutedExpertActivation.Silu,
        )
        ttnn.synchronize_device(mesh_device)
        times.append(time.perf_counter() - start)
        logger.info(f"  load[{i}]: {times[-1] * 1000:.1f} ms")
        if i == 0:
            # The empty placeholder a cache miss would load has the right shape too, so this
            # catches a truncated cache; check_cache_complete above owns the miss itself.
            assert len(expert.gate_projs) == experts_per_chip
            assert tuple(expert.gate_projs[0].shape) == (KimiK27Config.EMB_SIZE, KimiK27Config.MOE_INTERMEDIATE_SIZE)

    warm = times[1:] or times
    nbytes = sum(f.stat().st_size for f in cache_dir.glob(f"{prefix}.*.tensorbin"))
    logger.info(
        f"{prefix}: {nbytes / 2**30:.2f} GiB | iter0={times[0] * 1000:.1f} ms | "
        f"warm n={len(warm)} median={statistics.median(warm) * 1000:.1f} ms "
        f"({nbytes / 1e9 / statistics.median(warm):.2f} GB/s)"
    )
