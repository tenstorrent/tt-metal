# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Task 2 tests for the fused all-gather + regime_a_matmul op (REGIME_A_AGMM_EXECUTION_PLAN.md):
#  - Offline host-plan tests (device-free) over D=1/2/4/8: global-K-block ownership/coverage, core
#    reservation fit + collision, and L1 sizing.
#  - D=1 correctness: the op is behaviorally identical to regime_a_matmul.

import pytest
import torch
import ttnn
from models.common.utility_functions import is_blackhole
from tests.ttnn.utils_for_testing import assert_with_pcc


def _plan(**kw):
    # Device-free host-plan preview (registered on the C++ experimental submodule).
    return ttnn._ttnn.operations.experimental.all_gather_regime_a_matmul_plan(**kw)


# ---------------------------------------------------------------- offline plan tests (no device)
@pytest.mark.parametrize("D", [1, 2, 4, 8])
@pytest.mark.parametrize("topology", ["ring", "linear"])
def test_plan_kblock_coverage_and_fit(D, topology):
    # K=6144 -> Kt=192; kb=2 -> global_k_blocks=96, divisible by all D in {1,2,4,8}.
    p = _plan(M=32, K=6144, N=3072, D=D, Ns=1, Pk=3, Sm=1, kb=2, nsb=6,
              topology=topology, num_links=2, num_workers_per_link=6)
    assert p["valid"], p["errors"]
    assert p["global_k_blocks"] == 96
    # Every global K-block owned by exactly one device (explicit ids, full coverage, no dupes/gaps).
    seen = [b for dev in p["devices"] for b in dev["local_k_blocks"]]
    assert sorted(seen) == list(range(p["global_k_blocks"]))
    assert len(seen) == len(set(seen))
    assert len(p["devices"]) == D
    assert p["core_fit"] and not p["core_collision"]
    assert p["l1_fit"]
    # D=1 reserves no fabric cores; D>1 reserves mux + workers.
    if D == 1:
        assert p["reserved_fabric_cores"] == 0
    else:
        exp_mux = 2 * (2 if topology == "ring" else 1)  # num_links * (ring?2:1)
        assert p["mux_cores"] == exp_mux
        assert p["fabric_worker_cores"] == 2 * 6
        assert p["reserved_fabric_cores"] == exp_mux + 12


def test_plan_invalid_kt_not_divisible_by_D():
    p = _plan(M=32, K=6144, N=3072, D=5, kb=2)  # Kt=192 not divisible by 5
    assert not p["valid"]
    assert any("divisible by D" in e for e in p["errors"])


def test_plan_invalid_kt_not_divisible_by_kb():
    p = _plan(M=32, K=6144, N=3072, D=2, kb=5)  # Kt=192 not divisible by 5
    assert not p["valid"]
    assert any("kb" in e for e in p["errors"])


def test_plan_core_oversubscription_detected():
    # 8*Ns*Pk*Sm = 8*4*6*2 = 384 compute cores > usable 120.
    p = _plan(M=32, K=6144, N=3072, D=4, Ns=4, Pk=6, Sm=2, kb=2, num_links=2, num_workers_per_link=6)
    assert not p["core_fit"] or p["core_collision"]
    assert not p["valid"]


def test_plan_l1_oversubscription_detected():
    # Force a large transport footprint: big Mt (M=8192 -> Mt=256) * kb * C * slots.
    p = _plan(M=8192, K=6144, N=3072, D=2, Ns=1, Pk=1, Sm=1, kb=8, nsb=0,
              C=8, transport_slots=4, num_links=2, num_workers_per_link=6)
    assert not p["l1_fit"]
    assert not p["valid"]


# ---------------------------------------------------------------- D=1 correctness (device)
@pytest.mark.skipif(not is_blackhole(), reason="regime_a_matmul is Blackhole-only")
def test_d1_matches_regime_a(device):
    M, K, N = 32, 6144, 3072
    torch.manual_seed(0)
    t0 = torch.randn(1, 1, M, K, dtype=torch.bfloat16)
    t1 = torch.randn(1, 1, K, N, dtype=torch.bfloat16)
    ref = (t0.float() @ t1.float())[0, 0]
    a = ttnn.from_torch(t0, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
    wcfg = ttnn.create_regime_a_weight_memory_config(list(t1.shape), ttnn.bfloat16, device)
    b = ttnn.from_torch(t1, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16, memory_config=wcfg)
    cfg = ttnn.RegimeAMatmulConfig(k_slices=3, n_slices=1, m_slices=1, k_block_tiles=4, n_subblock_tiles=6)

    # D=1 (K_local == K_global) => delegates to regime_a_matmul.
    out = ttnn.experimental.all_gather_regime_a_matmul_async(a, b, config=cfg)
    got = ttnn.to_torch(ttnn.from_device(out))[0, 0]
    assert_with_pcc(ref, got.float(), 0.999)

    # Byte-for-byte identical to calling regime_a_matmul directly.
    out2 = ttnn.experimental.regime_a_matmul(a, b, config=cfg)
    got2 = ttnn.to_torch(ttnn.from_device(out2))[0, 0]
    assert torch.equal(got, got2), "D=1 fused op must equal regime_a_matmul exactly"
