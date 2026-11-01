# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Milestone 1 gate: conv + SiLU as one device op, against torch.

Checks the whole contract on the real Bonsai TP=1 shape: every value head's q/k/v columns must equal
silu(sum_j tap_j * state_j), every other column must be untouched, and the shift must equal the
ring shift the layer does today.

The reference is torch, computed the way the layer computes it (Qwen36GatedDeltaNet's FIR: a multiply
plus three macs over the shift registers, then silu) -- not the new kernel's own logic.
"""
import os
import sys

import pytest
import torch
from loguru import logger

import ttnn

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, "ops"))
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("conv_silu_op", os.path.join(_here, "ops", "conv_silu", "op.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
conv_silu, geometry, tap_diagonals = _mod.conv_silu, _mod.geometry, _mod.tap_diagonals

KEY_DIM, VALUE_DIM = 2048, 6144
NUM_VALUE_HEADS, GQA, K = 48, 3, 4


def _dev(mesh, t):
    return ttnn.from_torch(
        t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.L1_MEMORY_CONFIG
    )


def _ref(qkv, state, taps, g):
    """torch reference: conv + silu on the columns each value head owns, everything else untouched."""
    conv = qkv.clone()
    nk, kd_t, v_t = g["nk"], g["kd_t"], g["v_t"]
    for hv in range(g["nv"]):
        qh = hv // GQA
        q_cols = slice(qh * kd_t * 32, (qh + 1) * kd_t * 32)
        k_cols = slice((nk + qh) * kd_t * 32, (nk + qh + 1) * kd_t * 32)
        v_cols = slice(2 * nk * kd_t * 32 + hv * v_t * 32, 2 * nk * kd_t * 32 + (hv + 1) * v_t * 32)
        for cols in (q_cols, k_cols, v_cols):
            acc = torch.zeros_like(state[0, 0, cols])
            for j in range(K):
                acc = acc + taps[j, 0, cols] * state[j, 0, cols]
            conv[0, 0, cols] = torch.nn.functional.silu(acc)
    return conv


def _ref_shift(state, qkv):
    out = state.clone()
    for j in range(K - 1):
        out[j] = state[j + 1]
    out[K - 1] = qkv
    return out


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_conv_silu_matches_torch(mesh_device):
    g = geometry(KEY_DIM, VALUE_DIM, NUM_VALUE_HEADS, GQA, K)
    logger.info(f"geometry {g}")
    torch.manual_seed(0)
    C = g["C"]
    qkv = torch.randn(1, 1, C, dtype=torch.bfloat16)
    state = torch.randn(K, 1, C, dtype=torch.bfloat16)
    taps = torch.randn(K, 1, C, dtype=torch.bfloat16)

    # the kernel convolves against diagonal tap tiles (see conv_silu.tap_diagonals)
    diags = tap_diagonals(taps)
    conv_tt, shift_tt = conv_silu(
        _dev(mesh_device, qkv),
        _dev(mesh_device, state),
        _dev(mesh_device, diags),
        KEY_DIM,
        VALUE_DIM,
        NUM_VALUE_HEADS,
        GQA,
    )
    conv = ttnn.to_torch(conv_tt).float()
    shift = ttnn.to_torch(shift_tt).float()

    want = _ref(qkv, state, taps, g).float()
    want_shift = _ref_shift(state, qkv).float()

    # diagnose: where are the infs, and do the untouched columns stay zero?
    bad = ~torch.isfinite(conv)
    logger.info(f"conv non-finite elements: {int(bad.sum())} of {conv.numel()}")
    if bad.any():
        idx = bad.nonzero()[0]
        logger.info(f"first non-finite at {idx.tolist()}; value {conv[tuple(idx)].item()}")
    logger.info(f"conv row0[0:8] = {conv[0,0,:8].tolist()}")
    logger.info(f"want row0[0:8] = {want[0,0,:8].tolist()}")
    # Is it a scale/offset on the right values, or scrambled? Check the ratio and a correlation.
    _m = (want[0,0,:].abs() > 1e-3)
    if _m.any():
        _ratio = (conv[0,0,:][_m] / want[0,0,:][_m])
        logger.info(f"conv/want ratio: median {_ratio.median():.4f} p10 {_ratio.quantile(0.1):.4f} p90 {_ratio.quantile(0.9):.4f}")
    _a = conv[0,0,:].flatten().double(); _b = want[0,0,:].flatten().double()
    _a = _a - _a.mean(); _b = _b - _b.mean()
    _pcc = float((_a*_b).sum() / ((_a*_a).sum().sqrt() * (_b*_b).sum().sqrt()))
    logger.info(f"conv vs want PCC over the whole row: {_pcc:.6f}")
    err = (conv - want).abs()
    logger.info(f"conv: max err {err.max():.4e}, mean {err.mean():.4e}")
    logger.info(f"shift: max err {(shift - want_shift).abs().max():.4e}")
    # untouched columns must be bit-identical zeros (they were never written)
    logger.info(f"conv non-head columns max |value|: {conv[0,0, 128:256].abs().max():.4e}")

    assert err.max() < 0.05, f"conv mismatch: max err {err.max():.4e}"
    assert (shift - want_shift).abs().max() < 0.05, "ring shift mismatch"
