# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""ttnn.experimental.kda.gdn_decode_step against a torch reference of the GDN decode step, state in [Bmax, ...]
tensors (without state_page_idx)."""

import pytest
import torch
from loguru import logger

import ttnn
from tests.ttnn.nightly.unit_tests.operations.experimental.kda.gdn_decode_step_test_utils import (
    BMAX,
    DK,
    DV,
    MODE_IDS,
    MODES,
    NV,
    W,
    consts_dev,
    inputs,
    pack_rows,
    pcc,
    run,
    step_reference,
    tile_dev,
    token_rows,
)
from tt_py_test_utils_common.utility_functions import run_for_blackhole

pytestmark = run_for_blackhole()


def _state_threshold(fast_mode):
    # the default kernels truncate the state to TF32 in the add, so their state tracks fp32 torch more closely
    return 0.999 if fast_mode else 0.9999


@pytest.mark.parametrize("fast_mode, fp32_state", MODES, ids=MODE_IDS)
def test_gdn_decode_step_vs_reference(device, fast_mode, fp32_state):
    """Six steps of one batch row; the history is replaced by the op's own tokens after four."""
    steps = 6
    gen = torch.Generator().manual_seed(1)
    inp = inputs(1, seed=1)
    consts = consts_dev(device, inp)
    state = tile_dev(device, inp["h"].reshape(1, NV, DK, DV), ttnn.float32)
    hist = tile_dev(device, pack_rows(inp["rows"][0]).unsqueeze(0), ttnn.bfloat16)
    h_ref, rows_ref = inp["h"][0].clone(), [r.float() for r in inp["rows"][0]]
    taps = [t.float() for t in inp["tap_rows"]]
    for step in range(steps):
        row = token_rows(1, gen)[0]
        out_ref, h_ref, rows_ref = step_reference(
            rows_ref[1:], row.float(), taps, inp["dtb"].float(), inp["nea"].float(), h_ref, inp["w"].float()
        )
        out = ttnn.to_torch(
            run(
                tile_dev(device, row.reshape(1, 1, W), ttnn.bfloat16),
                consts,
                state,
                hist,
                fast_mode,
                fp32_state,
                ttnn.float32,
            )
        ).reshape(-1)
        h_t = ttnn.to_torch(state).reshape(NV, DK, DV)
        p_out, p_h = pcc(out, out_ref), pcc(h_t, h_ref)
        hist_ok = torch.equal(ttnn.to_torch(hist)[0], pack_rows([r.bfloat16() for r in rows_ref]))
        logger.info(f"step {step}: out pcc {p_out:.6f}, state pcc {p_h:.6f}, history shift exact {hist_ok}")
        assert p_out > 0.999 and p_h > _state_threshold(fast_mode) and hist_ok, (p_out, p_h, hist_ok)


@pytest.mark.parametrize("fast_mode, fp32_state", MODES, ids=MODE_IDS)
@pytest.mark.parametrize("B", [1, 2, 7, 8, 32])
def test_gdn_decode_step_batched_vs_reference(device, B, fast_mode, fp32_state):
    """B active batch rows of [BMAX, ...] state and history; rows B..BMAX-1 must not change."""
    gen = torch.Generator().manual_seed(10 + B)
    inp = inputs(BMAX, seed=10 + B)
    consts = consts_dev(device, inp)
    hist_host = torch.stack([pack_rows(inp["rows"][b], b) for b in range(BMAX)])
    state = tile_dev(device, inp["h"].clone(), ttnn.float32)
    hist = tile_dev(device, hist_host.clone(), ttnn.bfloat16)
    rows = token_rows(B, gen)
    taps = [t.float() for t in inp["tap_rows"]]
    refs, h_ref, hist_ref = [], inp["h"].clone(), hist_host.clone()
    for b in range(B):
        out_b, h_ref[b], new_rows = step_reference(
            [r.float() for r in inp["rows"][b][1:]],
            rows[b].float(),
            taps,
            inp["dtb"].float(),
            inp["nea"].float(),
            inp["h"][b],
            inp["w"].float(),
        )
        refs.append(out_b)
        hist_ref[b] = pack_rows([r.bfloat16() for r in new_rows], b)
    out = ttnn.to_torch(
        run(
            tile_dev(device, rows.reshape(1, B, W), ttnn.bfloat16),
            consts,
            state,
            hist,
            fast_mode,
            fp32_state,
            ttnn.float32,
        )
    ).reshape(B, -1)
    h_t, hist_t = ttnn.to_torch(state), ttnn.to_torch(hist)
    worst_out = min(pcc(out[b], refs[b]) for b in range(B))
    worst_state = min(pcc(h_t[b], h_ref[b]) for b in range(B))
    logger.info(f"B={B}: worst out pcc {worst_out:.6f}, worst state pcc {worst_state:.6f}")
    assert worst_out > 0.999 and worst_state > _state_threshold(fast_mode)
    assert torch.equal(hist_t[:B], hist_ref[:B]), "history shift differs from the reference"
    assert torch.equal(h_t[B:], inp["h"][B:]) and torch.equal(hist_t[B:], hist_host[B:]), "idle rows changed"


def test_gdn_decode_step_requires_conv_inputs(device, expect_error):
    inp = inputs(1, seed=0)
    consts = consts_dev(device, inp)
    x = tile_dev(device, token_rows(1, torch.Generator().manual_seed(0)).reshape(1, 1, W), ttnn.bfloat16)
    state = tile_dev(device, inp["h"].reshape(1, NV, DK, DV), ttnn.float32)
    hist = tile_dev(device, pack_rows(inp["rows"][0]).unsqueeze(0), ttnn.bfloat16)
    with expect_error(RuntimeError, "conv_hist and conv_taps are required"):
        run(x, consts, state, None, True, True)
    with expect_error(RuntimeError, "fp32_state needs fast_mode"):
        run(x, consts, state, hist, False, True)
