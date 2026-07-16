# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Galaxy CCL microbenchmarks and LoudBox proxies for GLM sparse-MLA tensor shapes and placements.

Thin driver over ``sparse_mla_ccl``: each production collective sparse MLA runs (``tt/mla/mla.py``) is one
``CollectivePath`` value, and every test builds the input, profiles the op(s) under the real-time profiler,
verifies reshards are lossless, and reports measured bandwidth against a fabric roofline.

Design: ``tmp/design/sparse-mla-ccl-perf.md``.
"""

import os

import pytest

import ttnn
from models.demos.deepseek_v3_d_p.reference.glm_5_1_config import glm_hf_config
from models.demos.deepseek_v3_d_p.tests.sparse_mla import sparse_mla_ccl as ccl
from models.demos.deepseek_v3_d_p.tests.sparse_mla.test_sparse_mla_perf import CHUNK_TOKENS, SCENARIOS

# Shared marks for every benchmark: perf-only, no timeout, skipped in CI (run locally on hardware).
pytestmark = [
    pytest.mark.perf,
    pytest.mark.timeout(0),
    pytest.mark.skipif(os.environ.get("CI") == "true", reason="performance test - run locally"),
]

# Single-chunk scenarios only; the looping (cold prefill) scenario is not a single CCL measurement.
_NON_LOOP_SCENARIOS = tuple(name for name, scenario in SCENARIOS.items() if not scenario["loop"])


def _scenario_id(scenario):
    return f"{scenario}_{SCENARIOS[scenario]['cache'] // 1024}k_cache"


def _workload(scenario):
    config = glm_hf_config()
    return ccl.Workload(
        chunk_tokens=CHUNK_TOKENS,
        cache_tokens=SCENARIOS[scenario]["cache"],
        num_attention_heads=config.num_attention_heads,
        kv_lora_rank=config.kv_lora_rank,
        qk_rope_head_dim=config.qk_rope_head_dim,
    )


def _run(mesh_device, path, scenario):
    assert mesh_device.arch() == ttnn.Arch.BLACKHOLE, "bandwidth assumptions apply to Blackhole only"
    workload = _workload(scenario)
    system = ccl.resolve_runtime_system(mesh_device, path)
    measurement = ccl.run_collective(mesh_device, path, workload, system)
    traffic = ccl.all_gather_roofline(path, workload, mesh_device, system)
    ccl.report(path, scenario, mesh_device, measurement, traffic)


@pytest.mark.parametrize("scenario", _NON_LOOP_SCENARIOS, ids=_scenario_id)
@pytest.mark.parametrize(
    "mesh_device,device_params",
    [ccl.ccl_mesh_param(ccl.SP_AXIS)],
    indirect=["mesh_device", "device_params"],
)
def test_kvpe_all_gather_perf(mesh_device, scenario):
    """Profile the SP all-gather used for the GLM KVPE prefix."""
    _run(mesh_device, ccl.KVPE_ALL_GATHER, scenario)


@pytest.mark.parametrize("scenario", _NON_LOOP_SCENARIOS, ids=_scenario_id)
@pytest.mark.parametrize(
    "mesh_device,device_params",
    [ccl.ccl_mesh_param(ccl.TP_AXIS)],
    indirect=["mesh_device", "device_params"],
)
def test_glm_head_to_sequence_reshard_perf(mesh_device, scenario):
    """Profile GLM's head-sharded to sequence-sharded TP redistribution."""
    _run(mesh_device, ccl.GLM_HEAD_TO_SEQUENCE, scenario)


@pytest.mark.parametrize("scenario", _NON_LOOP_SCENARIOS, ids=_scenario_id)
@pytest.mark.parametrize(
    "mesh_device,device_params",
    [ccl.ccl_mesh_param(ccl.TP_AXIS)],
    indirect=["mesh_device", "device_params"],
)
def test_glm_sequence_to_head_reshard_perf(mesh_device, scenario):
    """Profile GLM's sequence-sharded to head-sharded TP redistribution."""
    _run(mesh_device, ccl.GLM_SEQUENCE_TO_HEAD, scenario)
