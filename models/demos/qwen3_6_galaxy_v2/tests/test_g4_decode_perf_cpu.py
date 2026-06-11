# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CPU-only validation for G=4 decode perf assessment scaffolding.

No device required. Run before silicon microbenches:

    export PYTHONPATH=$(pwd) && source python_env/bin/activate \\
        && python -m pytest --noconftest -v \\
            models/demos/qwen3_6_galaxy_v2/tests/test_g4_decode_perf_cpu.py
"""
from __future__ import annotations

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.qwen3_6_galaxy_v2.tests.g4_decode_perf_utils import (
    _DIM,
    _INTER,
    _MLP_PARAM_FRACTION,
    _N_LAYERS,
    DEFAULT_SNAPSHOT,
    G4_MESH_SHAPE,
    TARGET_TOK_S,
    WDTYPE_NAMES,
    _shard_cols_2d,
    _shard_w_col_parallel,
    _shard_w_row_parallel,
    analytic_mlp_floor_ms,
    evaluate_go_nogo,
    load_mlp_weights_torch,
    print_decision_table,
    tile_pad_m,
    torch_swiglu_mlp,
    unshard_cols_from_packed,
)

_PCC_THRESH = 0.99


def test_tile_pad_m():
    assert tile_pad_m(1) == 32
    assert tile_pad_m(4) == 32
    assert tile_pad_m(32) == 32
    assert tile_pad_m(33) == 64


def test_analytic_floors_match_plan():
    bf16 = analytic_mlp_floor_ms(ttnn.bfloat16)
    bf8 = analytic_mlp_floor_ms(ttnn.bfloat8_b)
    bf4 = analytic_mlp_floor_ms(ttnn.bfloat4_b)

    assert bf16.full_model_ms == pytest.approx(24.6, rel=0.05)
    assert bf8.full_model_ms == pytest.approx(12.3, rel=0.05)
    assert bf4.full_model_ms == pytest.approx(6.1, rel=0.05)

    assert bf16.full_model_tok_s < TARGET_TOK_S  # bf16 cannot hit 70 tok/s at floor
    assert bf8.full_model_tok_s > TARGET_TOK_S  # bf8 marginal/above at floor
    assert bf4.full_model_tok_s > TARGET_TOK_S * 1.5


def test_col_shard_roundtrip():
    torch.manual_seed(0)
    full = torch.randn(1, 1, 32, _DIM, dtype=torch.bfloat16)
    packed = _shard_cols_2d(full, G4_MESH_SHAPE[1])
    back = unshard_cols_from_packed(packed, G4_MESH_SHAPE[1])
    assert torch.allclose(full, back)


def test_w_col_parallel_shapes_and_values():
    """gate/up: chip c holds full K=H, output slice N/cols (column-parallel)."""
    torch.manual_seed(1)
    w = torch.randn(_DIM, _INTER, dtype=torch.bfloat16)  # [H, I]
    packed = _shard_w_col_parallel(w)
    cols = G4_MESH_SHAPE[1]
    npc = _INTER // cols
    assert packed.shape == (1, cols, _DIM, npc)
    for c in range(cols):
        assert torch.allclose(w[:, c * npc : (c + 1) * npc], packed[0, c])


def test_w_row_parallel_shapes_and_values():
    """down: chip c holds contraction slice K/cols, full output N=H (row-parallel)."""
    torch.manual_seed(2)
    w = torch.randn(_INTER, _DIM, dtype=torch.bfloat16)  # [I, H]
    packed = _shard_w_row_parallel(w)
    cols = G4_MESH_SHAPE[1]
    kpc = _INTER // cols
    assert packed.shape == (1, cols, kpc, _DIM)
    for c in range(cols):
        assert torch.allclose(w[c * kpc : (c + 1) * kpc, :], packed[0, c])


def test_g4_mlp_torch_equivalence():
    """Numpy-level proof the column/row-parallel split reconstructs full SwiGLU.

    Mirrors mlp_g4_forward op order on CPU: gather input, per-chip column-parallel
    gate/up, SwiGLU, per-chip row-parallel down, sum partials (reduce_scatter sum).
    """
    torch.manual_seed(3)
    cols = G4_MESH_SHAPE[1]
    x_full = torch.randn(1, 1, 32, _DIM) * 0.02  # [1,1,M,H]
    w1 = torch.randn(_DIM, _INTER) * 0.02  # [H, I]
    w3 = torch.randn(_DIM, _INTER) * 0.02
    w2 = torch.randn(_INTER, _DIM) * 0.02  # [I, H]

    ref = torch_swiglu_mlp(x_full, w1, w3, w2)  # full reference

    npc = _INTER // cols
    kpc = _INTER // cols
    partial_sum = torch.zeros(1, 1, 32, _DIM)
    for c in range(cols):
        w1_c = w1[:, c * npc : (c + 1) * npc]  # column-parallel
        w3_c = w3[:, c * npc : (c + 1) * npc]
        w2_c = w2[c * kpc : (c + 1) * kpc, :]  # row-parallel
        gate = x_full @ w1_c
        up = x_full @ w3_c
        ff_c = torch.nn.functional.silu(gate) * up  # [*, I/4]
        partial_sum = partial_sum + ff_c @ w2_c  # [*, H] partial
    passing, pcc = comp_pcc(ref, partial_sum, _PCC_THRESH)
    assert passing
    assert float(pcc) > 0.9999


@pytest.mark.skipif(not DEFAULT_SNAPSHOT.exists(), reason="HF snapshot not available")
def test_load_mlp_weights_shapes():
    w1, w3, w2 = load_mlp_weights_torch(DEFAULT_SNAPSHOT, layer=3)
    assert w1.shape == (_DIM, _INTER)
    assert w3.shape == (_DIM, _INTER)
    assert w2.shape == (_INTER, _DIM)


@pytest.mark.skipif(not DEFAULT_SNAPSHOT.exists(), reason="HF snapshot not available")
def test_single_layer_torch_swiglu_pcc():
    w1, w3, w2 = load_mlp_weights_torch(DEFAULT_SNAPSHOT, layer=3)
    x = torch.randn(1, 1, 1, _DIM, dtype=torch.bfloat16) * 0.02
    y = torch_swiglu_mlp(x.float(), w1.float(), w3.float(), w2.float())
    passing, pcc = comp_pcc(y, y, _PCC_THRESH)
    assert passing


@pytest.mark.skipif(not DEFAULT_SNAPSHOT.exists(), reason="HF snapshot not available")
def test_64layer_compounding_pcc_cpu():
    """CPU-only 64L residual MLP chain — validates the reference path doesn't collapse."""
    x = torch.randn(1, 1, 1, _DIM, dtype=torch.bfloat16) * 0.02
    ref = x.float()
    for layer in range(_N_LAYERS):
        w1, w3, w2 = load_mlp_weights_torch(DEFAULT_SNAPSHOT, layer)
        ref = ref + torch_swiglu_mlp(ref, w1.float(), w3.float(), w2.float())
    # Residual stream must keep the signal alive (not collapse to ~0).
    assert ref.abs().mean().item() > 1e-3
    passing, pcc = comp_pcc(ref, ref, _PCC_THRESH)
    assert passing


def test_decision_table_go_nogo_logic():
    # bf16 too slow; bf8/bf4 under 14.3ms full-model projection at B=1
    mock = {
        (ttnn.bfloat16, 1): {"mlp_ms": 0.30, "pcc": 0.999},
        (ttnn.bfloat8_b, 1): {"mlp_ms": 0.12, "pcc": 0.995},
        (ttnn.bfloat4_b, 1): {"mlp_ms": 0.08, "pcc": 0.992},
    }
    any_go, go_prec = evaluate_go_nogo(mock)
    assert any_go
    assert WDTYPE_NAMES[ttnn.bfloat16] not in go_prec
    assert WDTYPE_NAMES[ttnn.bfloat8_b] in go_prec

    slow = {(ttnn.bfloat16, 1): {"mlp_ms": 0.50, "pcc": 0.999}}
    assert not evaluate_go_nogo(slow)[0]


def test_decision_table_prints(capsys):
    mock = {(ttnn.bfloat8_b, 1): {"mlp_ms": 0.15, "pcc": 0.995}}
    print_decision_table(mock)
    out = capsys.readouterr().out
    assert "DECISION TABLE" in out
    assert "bf8" in out


def test_mlp_param_fraction():
    assert _MLP_PARAM_FRACTION == pytest.approx(0.634, rel=0.01)


def test_hardware_tests_skipped_by_default():
    """Document that device tests require G4_RUN_DEVICE=1."""
    import models.demos.qwen3_6_galaxy_v2.tests.test_g4_mlp_micro as micro

    assert hasattr(micro, "test_g4_mlp_latency_pcc")
