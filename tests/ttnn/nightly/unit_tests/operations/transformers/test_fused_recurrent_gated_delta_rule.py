# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Accuracy of ttnn.transformer.fused_recurrent_gated_delta_rule against its registered golden, on the served head
counts, batch widths and token counts the op supports today."""

import zlib

import pytest
import torch

import ttnn
from models.common.utility_functions import is_blackhole
from tests.ttnn.nightly.unit_tests.operations.transformers.gdn_decode_test_utils import (
    SHAPES,
    assert_accuracy,
    make_inputs,
)

pytestmark = pytest.mark.skipif(not is_blackhole(), reason="fused_recurrent_gated_delta_rule is Blackhole-only")

OP = ttnn.transformer.fused_recurrent_gated_delta_rule
K = V = 128
BATCHES = (1, 2, 4, 8)
TOKENS = (1, 4, 8, 12)  # decode, and speculative verify with 3, 7 and 11 draft tokens


def _to_device(device, t):
    return ttnn.from_torch(t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)


def run_op(device, x, *, output_final_state=False, output_per_token_state=False):
    """Today's calling convention: fp32 TILE interleaved tensors, q and k L2-normalised on the host (make_inputs does
    that), the state passed as a tensor. Returns torch tensors."""
    args = [_to_device(device, x[name]) for name in ("q", "k", "v", "g", "beta")]
    s0 = _to_device(device, x["initial_state"]) if x["initial_state"] is not None else None
    o, state = OP(
        *args,
        initial_state=s0,
        output_final_state=output_final_state,
        output_per_token_state=output_per_token_state,
    )
    return ttnn.to_torch(o), (ttnn.to_torch(state) if state is not None else None)


def _skip_unless_fits(device, B, HV):
    grid = device.compute_with_storage_grid_size()
    if B * HV > grid.x * grid.y:
        pytest.skip(f"B*HV = {B * HV} exceeds the {grid.x * grid.y}-core grid: one core per (b, hv) today")


def _cases():
    for name, sh in SHAPES.items():
        if not sh.supported_today:
            continue
        for B in BATCHES:
            if B * sh.num_value_heads > 110:  # the p150 grid; larger grids re-check at run time
                continue
            for T in TOKENS:
                for with_state in (True, False):
                    yield pytest.param(name, B, T, with_state, id=f"{name}-B{B}-T{T}-{'s0' if with_state else 'zero'}")


@pytest.mark.parametrize("shape, B, T, with_state", list(_cases()))
def test_decode_matches_golden(device, shape, B, T, with_state):
    """o and the state after every token (T > 1) or the final state (T = 1) against the golden at the accuracy gates."""
    sh = SHAPES[shape]
    _skip_unless_fits(device, B, sh.num_value_heads)
    seed = zlib.crc32(f"{shape}-{B}-{T}".encode())
    x = make_inputs(B, T, sh.num_key_heads, sh.num_value_heads, K, V, seed=seed, with_state=with_state)
    per_token = T > 1
    o, state = run_op(device, x, output_final_state=not per_token, output_per_token_state=per_token)
    o_ref, state_ref = ttnn.get_golden_function(OP)(
        x["q"],
        x["k"],
        x["v"],
        x["g"],
        x["beta"],
        initial_state=x["initial_state"],
        output_final_state=not per_token,
        output_per_token_state=per_token,
    )
    assert o.shape == o_ref.shape and state.shape == state_ref.shape
    assert_accuracy(o_ref, o, "o")
    if per_token:
        for t in range(T):
            assert_accuracy(state_ref[:, t], state[:, t], f"state after token {t}")
    else:
        assert_accuracy(state_ref, state, "final state")


def test_comparison_mode_passes(device):
    """The golden is wired into ttnn's comparison mode: one call with the mode on and set to raise on a miss."""
    cfg = ttnn.CONFIG
    saved = (cfg.enable_fast_runtime_mode, cfg.enable_comparison_mode, cfg.comparison_mode_should_raise_exception)
    cfg.enable_fast_runtime_mode = False
    cfg.enable_comparison_mode = True
    cfg.comparison_mode_should_raise_exception = True
    try:
        sh = SHAPES["qwen27b_tp4"]
        x = make_inputs(1, 1, sh.num_key_heads, sh.num_value_heads, K, V, seed=20261006)
        o, state = run_op(device, x, output_final_state=True)
        assert o.shape == (1, 1, sh.num_value_heads, V) and state.shape == (1, sh.num_value_heads, K, V)
    finally:
        cfg.enable_fast_runtime_mode, cfg.enable_comparison_mode, cfg.comparison_mode_should_raise_exception = saved
