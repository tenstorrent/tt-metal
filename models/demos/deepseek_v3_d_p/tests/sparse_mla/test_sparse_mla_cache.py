# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DSA (sparse MLA) cache/loading tests — the contract added by the sparse_mla_cache_loading plan.

Dense MLA cache coverage lives in cache/test_mla_cache.py. This suite verifies the sparse-specific
behavior: sparse capability is resolved explicitly (config / host weights / cache), the offline
cache build emits the indexer tensorbins, cache-only construction stays sparse (binds TtIndexer,
never dense), completeness covers the indexer files, and a sparse cache-only construct with no cache
warns and stays sparse (mirrors dense's lenient placeholder load) instead of silently going dense.

The `matches_config` test is host-only (no device); the build→cache-only→PCC test runs on a TP>=2
mesh so the dense 128-head epilogue fits (TP=1 overflows L1). Validity gating (so collected==run) is
the same as test_sparse_mla.py — here we just fix one (4,2) mesh that both variants support.
"""

import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from loguru import logger
from ttnn.device import is_blackhole

import ttnn
from models.demos.deepseek_v3_d_p.reference.cpu_deepseek_v32 import random_mla_weights
from models.demos.deepseek_v3_d_p.tt.mla import ttMLA
from models.demos.deepseek_v3_d_p.tt.mla.indexer import TtIndexer, resolve_has_indexer
from models.demos.deepseek_v3_d_p.tt.mla.rope import RotarySetup
from models.demos.deepseek_v3_d_p.utils.fast_cache_checker import init_checker, report_and_clear
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import init_kvpe_cache
from models.demos.deepseek_v3_d_p.utils.test_utils import WH_WORKER_L1_SIZE
from tests.ttnn.utils_for_testing import comp_pcc

CACHE_DIR = Path("/tmp/DS_PREFILL_sparse_mla")
SEQ_LEN = 256
SP_AXIS, TP_AXIS = 0, 1


# --------------------------------------------------------------------------------------------------
# Host-only: the config-detection path, isolated from the host/cache fallbacks. A regression where a
# variant's runtime config stops carrying the DSA fields would be masked by the PCC test below (it can
# resolve sparse via the cache), so assert matches_config / resolve_has_indexer directly.
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("variant", ["deepseek_v32", "glm_5_1"], indirect=True, ids=["deepseek_v32", "glm_5_1"])
def test_matches_config_detects_dsa(variant, config_only):
    assert TtIndexer.matches_config(config_only), f"{variant.name}: runtime config should carry DSA index_* fields"
    # No host weights, no cache, no explicit override -> still resolves sparse purely from the config.
    assert resolve_has_indexer(config_only) is True


def test_matches_config_rejects_dense():
    """A dense DeepSeek-V3 / R1-style config (no index_* fields) must not look sparse."""
    dense = SimpleNamespace(q_lora_rank=1536, hidden_size=7168)
    assert TtIndexer.matches_config(dense) is False
    assert resolve_has_indexer(dense) is False
    # ...unless a caller explicitly forces it (escape hatch).
    assert resolve_has_indexer(dense, explicit=True) is True


# --------------------------------------------------------------------------------------------------
# Device: build -> cache-only load stays sparse and reproduces the from-weights output.
# --------------------------------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def cleanup_cache():
    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    yield
    report_and_clear()


def _forward(mla, mesh_device, rope_tensors, kvpe_cache, hidden):
    """Single-shot sparse MLA forward, mirroring run_mla_inference's SP×TP input sharding."""
    shard_dims = [None, None]
    shard_dims[TP_AXIS], shard_dims[SP_AXIS] = -1, -2
    tt_hidden = ttnn.from_torch(
        hidden.unsqueeze(0),  # [1, batch, seq, hidden]
        device=mesh_device,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=shard_dims),
    )
    out = mla.forward(hidden_states=tt_hidden, rope_tensors=rope_tensors, kvpe_cache=kvpe_cache)
    return ttnn.to_torch(
        out, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=shard_dims, mesh_shape=mesh_device.shape)
    ).to(torch.bfloat16)


def _new_kvpe(config, mesh_device, mesh_shape):
    return init_kvpe_cache(
        kvpe_cache_head_dim=config.kv_lora_rank + config.qk_rope_head_dim,
        mesh_device=mesh_device,
        seq_len=SEQ_LEN,
        mesh_shape=mesh_shape,
        sp_axis=SP_AXIS,
        num_kvpe_cache_layers=1,
    )


def _build_mla(config, state_dict, mesh_device, weight_cache_path):
    return ttMLA(
        config,
        state_dict,
        mesh_device,
        layer_idx=0,
        seq_len=SEQ_LEN,
        sp_axis=SP_AXIS,
        tp_axis=TP_AXIS,
        weight_cache_path=weight_cache_path,
    )


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param(
            (4, 2),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "worker_l1_size": ttnn._ttnn.device.DEFAULT_WORKER_L1_SIZE if is_blackhole() else WH_WORKER_L1_SIZE,
            },
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(4, 2), topology="linear"),
            id="linear-4x2",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("variant", ["deepseek_v32", "glm_5_1"], indirect=True, ids=["deepseek_v32", "glm_5_1"])
@pytest.mark.skipif(not is_blackhole(), reason="DSA ops (indexer / sparse SDPA) are Blackhole-only")
@pytest.mark.timeout(0)
def test_sparse_mla_cache_only_stays_sparse(mesh_device, device_params, variant, config_only):
    """from-weights -> offline cache build -> cache-only construct: stays sparse, reproduces output,
    and the indexer cache is part of completeness. Also covers the completeness + failure-mode cases."""
    config = config_only
    config.max_seq_len = SEQ_LEN
    weights = random_mla_weights(config)  # device-vs-device round-trip: config-shaped weights suffice
    mesh_shape = list(mesh_device.shape)

    rope_tensors = RotarySetup(config, mesh_device, sp_axis=SP_AXIS, is_balanced=False).get_rope_tensors(SEQ_LEN)
    torch.manual_seed(42)
    hidden = torch.randn(1, SEQ_LEN, config.hidden_size, dtype=torch.bfloat16)

    # === from weights (no cache) ===
    mla_w = _build_mla(config, weights, mesh_device, weight_cache_path=None)
    assert mla_w._has_indexer, f"{variant.name}: from-weights construction must be sparse"
    out_weights = _forward(mla_w, mesh_device, rope_tensors, _new_kvpe(config, mesh_device, mesh_shape), hidden)

    # === offline cache build: dense + indexer tensorbins ===
    init_checker(CACHE_DIR)
    assert not ttMLA.check_cache_complete(CACHE_DIR, "layer_0.mla", has_indexer=True), "cache empty before build"
    ttMLA.build_ttnn_cache(weights, CACHE_DIR, mesh_device, config, 0, SEQ_LEN, SP_AXIS, TP_AXIS)
    init_checker(CACHE_DIR)
    assert ttMLA.check_cache_complete(CACHE_DIR, "layer_0.mla", has_indexer=True), "dense+indexer cache complete"

    # === cache-only construct: empty state dict, must stay sparse and bind TtIndexer ===
    mla_c = _build_mla(config, {}, mesh_device, weight_cache_path=CACHE_DIR)
    assert mla_c._has_indexer, f"{variant.name}: cache-only construction must stay sparse, not fall back to dense"
    assert type(mla_c._indexer).__name__ == "TtIndexer", "cache-only must bind TtIndexer, not NullIndexer"
    out_cache = _forward(mla_c, mesh_device, rope_tensors, _new_kvpe(config, mesh_device, mesh_shape), hidden)

    passed, pcc = comp_pcc(out_weights, out_cache, 0.999)
    logger.info(f"[{variant.name}] sparse cache-only vs from-weights PCC: {pcc}")
    assert passed, f"{variant.name}: cache-only output diverged from from-weights: PCC={pcc}"

    # === completeness: a missing indexer tensorbin fails the sparse check (but not the dense check) ===
    one = next(CACHE_DIR.glob("layer_0.mla.indexer_*.tensorbin"))
    one.unlink()
    init_checker(CACHE_DIR)
    assert ttMLA.check_cache_complete(CACHE_DIR, "layer_0.mla", has_indexer=False), "dense-only check still complete"
    assert not ttMLA.check_cache_complete(
        CACHE_DIR, "layer_0.mla", has_indexer=True
    ), "missing indexer tensorbin must fail the sparse completeness check"
    ttnn.synchronize_device(mesh_device)


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param(
            (4, 2),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "worker_l1_size": ttnn._ttnn.device.DEFAULT_WORKER_L1_SIZE if is_blackhole() else WH_WORKER_L1_SIZE,
            },
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(4, 2), topology="linear"),
            id="linear-4x2",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("variant", ["deepseek_v32"], indirect=True, ids=["deepseek_v32"])
@pytest.mark.skipif(not is_blackhole(), reason="DSA ops (indexer / sparse SDPA) are Blackhole-only")
@pytest.mark.timeout(0)
def test_sparse_cache_only_without_cache_warns_stays_sparse(mesh_device, device_params, variant, config_only):
    """Sparse cache-only construction with no indexer cache must WARN (not raise — mirrors dense's
    lenient placeholder load) and stay sparse, never silently fall back to dense."""
    config = config_only
    config.max_seq_len = SEQ_LEN
    warnings = []
    sink = logger.add(lambda m: warnings.append(str(m)), level="WARNING")
    try:
        mla = _build_mla(config, {}, mesh_device, weight_cache_path=CACHE_DIR)  # empty (cleaned) dir
    finally:
        logger.remove(sink)
    assert mla._has_indexer, "must stay sparse (resolved from config), not fall back to dense"
    assert type(mla._indexer).__name__ == "TtIndexer", "must bind TtIndexer, never NullIndexer"
    assert any(
        "indexer has neither host weights nor a complete cache" in m for m in warnings
    ), f"expected a loud warning about the missing indexer weights/cache; got: {warnings}"
