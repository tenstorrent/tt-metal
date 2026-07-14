# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
#
# Correctness tests for ttnn.experimental.regime_a_matmul (Blackhole DRAM-BW-optimal Regime-A matmul).
# Random BF16 inputs vs a Torch reference. in1 is DRAM width-sharded via the op's canonical helper.
# Cases follow the plan's bring-up order: Pk=1 (no reduction) -> Pk>1 (reduction) -> Ns>1 -> Sm>1 ->
# golden parity configs -> non-divisible.

import pytest
import torch
import ttnn
from models.common.utility_functions import is_blackhole
from tests.ttnn.utils_for_testing import assert_with_pcc


def _run_regime_a(device, M, K, N, Ns, Pk, Sm, kb, nsb, pcc=0.999):
    torch.manual_seed(0)
    torch_in0 = torch.randn(1, 1, M, K, dtype=torch.bfloat16)
    torch_in1 = torch.randn(1, 1, K, N, dtype=torch.bfloat16)
    torch_ref = (torch_in0.float() @ torch_in1.float())[0, 0]

    in0 = ttnn.from_torch(torch_in0, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)

    weight_mem_cfg = ttnn.create_regime_a_weight_memory_config(list(torch_in1.shape), ttnn.bfloat16, device)
    in1 = ttnn.from_torch(
        torch_in1, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16, memory_config=weight_mem_cfg
    )

    config = ttnn.RegimeAMatmulConfig(k_slices=Pk, n_slices=Ns, m_slices=Sm, k_block_tiles=kb, n_subblock_tiles=nsb)
    out = ttnn.experimental.regime_a_matmul(in0, in1, config=config)
    out_torch = ttnn.to_torch(ttnn.from_device(out))[0, 0]

    assert out_torch.shape == torch_ref.shape, f"shape {out_torch.shape} != {torch_ref.shape}"
    assert_with_pcc(torch_ref, out_torch.float(), pcc)


# (label, M, K, N, Ns, Pk, Sm, kb, nsb)
CASES = [
    ("pk1_noreduce", 32, 6144, 3072, 1, 1, 1, 4, 6),  # 8 cores, no reduction
    ("pk3_reduce", 32, 6144, 3072, 1, 3, 1, 4, 6),  # split-K reduction chain
    ("ns2", 32, 2048, 2048, 2, 2, 1, 4, 4),  # N-slice > 1
    ("sm2", 128, 6144, 4608, 1, 6, 2, 2, 1),  # M-split > 1 (in1 forward)
    ("golden_mt1", 32, 6144, 4608, 1, 12, 1, 2, 1),
    ("golden_mt2", 64, 6144, 4608, 1, 6, 1, 4, 2),
    ("golden_mt4", 128, 6144, 4608, 1, 12, 1, 2, 1),
    ("golden_mt8", 256, 6144, 4608, 1, 12, 1, 2, 1),
]


@pytest.mark.skipif(not is_blackhole(), reason="Regime-A matmul is Blackhole-only")
@pytest.mark.parametrize("label,M,K,N,Ns,Pk,Sm,kb,nsb", CASES, ids=[c[0] for c in CASES])
def test_regime_a_matmul_correctness(device, label, M, K, N, Ns, Pk, Sm, kb, nsb):
    _run_regime_a(device, M, K, N, Ns, Pk, Sm, kb, nsb)


@pytest.mark.skipif(not is_blackhole(), reason="Regime-A matmul is Blackhole-only")
def test_regime_a_matmul_cache_replay(device):
    # Run the same config twice on FRESH tensors (different buffer addresses). The second call hits the
    # program cache and must pick up the new addresses via override_runtime_arguments.
    M, K, N, cfg = 64, 6144, 4608, (1, 6, 1, 4, 2)
    Ns, Pk, Sm, kb, nsb = cfg
    for trial in range(2):
        torch.manual_seed(trial)  # different contents + fresh allocations each trial
        t0 = torch.randn(1, 1, M, K, dtype=torch.bfloat16)
        t1 = torch.randn(1, 1, K, N, dtype=torch.bfloat16)
        ref = (t0.float() @ t1.float())[0, 0]
        a0 = ttnn.from_torch(t0, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
        wcfg = ttnn.create_regime_a_weight_memory_config(list(t1.shape), ttnn.bfloat16, device)
        a1 = ttnn.from_torch(t1, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16, memory_config=wcfg)
        conf = ttnn.RegimeAMatmulConfig(k_slices=Pk, n_slices=Ns, m_slices=Sm, k_block_tiles=kb, n_subblock_tiles=nsb)
        out = ttnn.experimental.regime_a_matmul(a0, a1, config=conf)
        got = ttnn.to_torch(ttnn.from_device(out))[0, 0]
        assert_with_pcc(ref, got.float(), 0.999)


@pytest.mark.skipif(not is_blackhole(), reason="Regime-A matmul is Blackhole-only")
def test_regime_a_matmul_column_dependent_layout(device):
    # Structured in1 (each N-column has a distinct signature, constant over K) pins down N-column /
    # bank ownership: a bank-permutation or wrong N-offset scrambles columns and drops PCC even where
    # a random weight might average out. in0=ones so out[m,n] = sum_k in1[k,n].
    M, K, N, cfg = 32, 6144, 4608, (1, 12, 1, 2, 1)
    Ns, Pk, Sm, kb, nsb = cfg
    torch.manual_seed(0)
    t0 = torch.ones(1, 1, M, K, dtype=torch.bfloat16)
    # small per-column signature (keep bf16-representable): col n -> (n % 251) * 0.01
    col = ((torch.arange(N) % 251).float() * 0.01).to(torch.bfloat16)
    t1 = col.view(1, 1, 1, N).expand(1, 1, K, N).contiguous()
    ref = (t0.float() @ t1.float())[0, 0]
    a0 = ttnn.from_torch(t0, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
    wcfg = ttnn.create_regime_a_weight_memory_config(list(t1.shape), ttnn.bfloat16, device)
    a1 = ttnn.from_torch(t1, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16, memory_config=wcfg)
    conf = ttnn.RegimeAMatmulConfig(k_slices=Pk, n_slices=Ns, m_slices=Sm, k_block_tiles=kb, n_subblock_tiles=nsb)
    out = ttnn.experimental.regime_a_matmul(a0, a1, config=conf)
    got = ttnn.to_torch(ttnn.from_device(out))[0, 0]
    assert_with_pcc(ref, got.float(), 0.999)


@pytest.mark.skipif(not is_blackhole(), reason="Regime-A matmul is Blackhole-only")
def test_regime_a_matmul_nondivisible(device):
    # Kt=190, Nt=145 (neither 8-tile-aligned). v1 correctness path: the caller pads in0's K and
    # in1's K/N up to bank alignment (8 tiles) with zeros; the zero-pad K contributes nothing and the
    # pad-N output columns are sliced off. (Balanced floor/ceil tails to avoid the expanded pad reads
    # are the deferred efficiency step.)
    import torch.nn.functional as F

    M, K, N, Ns, Pk, Sm, kb, nsb = 32, 6080, 4640, 1, 12, 1, 2, 1
    TILE = 32

    def rup(x, y):
        return ((x + y - 1) // y) * y

    K_pad = rup((K + TILE - 1) // TILE, 8) * TILE  # 6144
    N_pad = rup((N + TILE - 1) // TILE, 8) * TILE  # 4864

    torch.manual_seed(0)
    torch_in0 = torch.randn(1, 1, M, K, dtype=torch.bfloat16)
    torch_in1 = torch.randn(1, 1, K, N, dtype=torch.bfloat16)
    torch_ref = (torch_in0.float() @ torch_in1.float())[0, 0]

    in0_pad = F.pad(torch_in0, (0, K_pad - K))  # pad K (last dim of activation)
    in1_pad = F.pad(torch_in1, (0, N_pad - N, 0, K_pad - K))  # pad N then K

    in0 = ttnn.from_torch(in0_pad, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
    wcfg = ttnn.create_regime_a_weight_memory_config([1, 1, K_pad, N_pad], ttnn.bfloat16, device)
    in1 = ttnn.from_torch(in1_pad, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16, memory_config=wcfg)

    config = ttnn.RegimeAMatmulConfig(k_slices=Pk, n_slices=Ns, m_slices=Sm, k_block_tiles=kb, n_subblock_tiles=nsb)
    out = ttnn.experimental.regime_a_matmul(in0, in1, config=config)
    out_torch = ttnn.to_torch(ttnn.from_device(out))[0, 0][:M, :N]  # slice valid region

    assert_with_pcc(torch_ref, out_torch.float(), 0.999)
