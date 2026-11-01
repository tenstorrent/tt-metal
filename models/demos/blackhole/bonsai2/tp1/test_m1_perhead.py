# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Milestone 1 localisation: test ONE value head in isolation, so a wrong tile can be named.

The end-to-end test cannot localise the bug: the 48 heads' q/k/v slices cover all 10240 output columns
in aggregate, so "every column was written" is true whether or not each head wrote only its own slice.
This test makes the answer unambiguous by running a SINGLE head's slice through the op and requiring
that (a) that head's channels are right and (b) every other channel is untouched.

Method: build inputs where only the channels of interest are non-zero. Then any non-zero output
outside those channels is, by construction, a head writing outside its own slice.

Run: MESH_DEVICE=P150 pytest <this> -s
"""
import os
import sys

import pytest
import torch

import ttnn

_here = os.path.dirname(os.path.abspath(__file__))
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("conv_silu_op", os.path.join(_here, "ops", "conv_silu", "op.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
conv_silu, geometry, tap_diagonals = _mod.conv_silu, _mod.geometry, _mod.tap_diagonals

KEY_DIM, VALUE_DIM, NV, GQA, K = 2048, 6144, 48, 3, 4
TILE = 32


def _dev(mesh, t):
    return ttnn.from_torch(
        t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.L1_MEMORY_CONFIG
    )


def _slices(g, hv):
    """The channel ranges value head hv owns, in the FULL-row layout."""
    kd_t, v_t, nk = g["kd_t"], g["v_t"], g["nk"]
    qh = hv // GQA
    q = (qh * kd_t * TILE, (qh + 1) * kd_t * TILE)
    k = ((nk + qh) * kd_t * TILE, (nk + qh + 1) * kd_t * TILE)
    v = (2 * nk * kd_t * TILE + hv * v_t * TILE, 2 * nk * kd_t * TILE + (hv + 1) * v_t * TILE)
    return {"q": q, "k": k, "v": v}


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("hv", [0, 1, 17, 47])
@pytest.mark.parametrize("probe", ["q", "k", "v"])
def test_single_head_isolation(mesh_device, hv, probe):
    """Only head hv's `probe` channels are non-zero going in; only they may be non-zero coming out."""
    g = geometry(KEY_DIM, VALUE_DIM, NV, GQA, K)
    C = g["C"]
    torch.manual_seed(1000 + hv)
    slices = _slices(g, hv)
    lo, hi = slices[probe]

    # Non-zero ONLY in this head's probe channels, for the input row AND for every state row, so the
    # conv has something to compute. Everything else is zero, so any output there is a stray write.
    qkv = torch.zeros(1, 1, C, dtype=torch.bfloat16)
    qkv[0, 0, lo:hi] = torch.randn(hi - lo, dtype=torch.bfloat16)
    state = torch.zeros(K, 1, C, dtype=torch.bfloat16)
    for j in range(K):
        state[j, 0, lo:hi] = torch.randn(hi - lo, dtype=torch.bfloat16)
    # taps: all ones on the probe channels -> conv is a plain sum of the state rows, then silu.
    taps = torch.zeros(K, 1, C, dtype=torch.bfloat16)
    taps[:, 0, lo:hi] = 1.0

    conv, _ = conv_silu(
        _dev(mesh_device, qkv), _dev(mesh_device, state), _dev(mesh_device, tap_diagonals(taps)),
        KEY_DIM, VALUE_DIM, NV, GQA,
    )
    got = ttnn.to_torch(conv).float()[0, 0, :]

    # (a) the probe channels: silu(sum of the K state rows * 1)
    want = torch.nn.functional.silu(state[:, 0, lo:hi].float().sum(dim=0))
    err_in = (got[lo:hi] - want).abs().max().item()

    # (b) every other channel must be exactly zero
    mask = torch.ones(C, dtype=torch.bool)
    mask[lo:hi] = False
    stray = got[mask].abs().max().item()
    n_stray = int((got[mask].abs() > 1e-6).sum())

    # which 128-channel blocks are non-zero? one probe slice is 128 channels; any block that is not
    # [lo,hi) and not zero tells us where the head actually wrote.
    blocks = [b for b in range(C // 128) if got[b * 128:(b + 1) * 128].abs().max() > 1e-3]
    print(
        f"hv={hv:2d} probe={probe} cols[{lo}:{hi}]: err_in={err_in:.4e}  "
        f"stray_max={stray:.4e} ({n_stray} cols)  nonzero blocks={blocks[:12]}"
    )
    assert stray < 1e-3, (
        f"head {hv} wrote outside its {probe} slice: {n_stray} other columns non-zero, max {stray:.4e}. "
        f"That is a head writing over another head's channels (or the writer ignoring its column offset)."
    )
    assert err_in < 0.05, f"head {hv} {probe} values wrong: max err {err_in:.4e}"
