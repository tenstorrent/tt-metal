# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""ttnn.transformer.fused_recurrent_gated_delta_rule against its registered golden on the served head counts, batch
widths and token counts the op supports today: accuracy, multi-token versus chained single-token identity, run-to-run
and trace-versus-eager determinism, and drift over a long chain against an fp64 reference."""

import zlib

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from tests.ttnn.nightly.unit_tests.operations.transformers.gdn_decode_test_utils import (
    PCC_MIN,
    SHAPES,
    assert_accuracy,
    chained_decode,
    make_inputs,
    metrics,
)
from ttnn.operations.transformer_golden import recurrent_gated_delta_rule

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


def _op_as_torch_fn(device):
    """The op as a torch-to-torch function in the golden's calling convention, for chained_decode."""

    def fn(q, k, v, g, beta, *, initial_state=None, output_final_state=False):
        x = {"q": q, "k": k, "v": v, "g": g, "beta": beta, "initial_state": initial_state}
        return run_op(device, x, output_final_state=output_final_state)

    return fn


@pytest.mark.parametrize("B", [1, 8])
@pytest.mark.parametrize("K_draft", [3, 7, 11])
def test_multi_token_equals_chained_single_token(device, K_draft, B):
    """One call with T = K + 1 tokens and per-token states is bit-identical to K + 1 chained single-token calls, each
    fed the previous call's state: the speculative decoder's verify path equals its decode path."""
    sh = SHAPES["qwen27b_tp4"]
    T = K_draft + 1
    x = make_inputs(B, T, sh.num_key_heads, sh.num_value_heads, K, V, seed=zlib.crc32(f"chain-{B}-{T}".encode()))
    o, states = run_op(device, x, output_per_token_state=True)
    o_chain, states_chain = chained_decode(
        _op_as_torch_fn(device), x["q"], x["k"], x["v"], x["g"], x["beta"], x["initial_state"]
    )
    assert torch.equal(o, o_chain), f"o: max |d| {(o - o_chain).abs().max().item():.3e}"
    assert torch.equal(states, states_chain), f"states: max |d| {(states - states_chain).abs().max().item():.3e}"


@pytest.mark.parametrize("shape, B, T", [("qwen27b_tp4", 8, 4), ("qwen9b_tp1", 2, 1)])
def test_repeat_runs_are_identical(device, shape, B, T):
    """Five launches of the same inputs (uploaded afresh each time) give the same bits."""
    sh = SHAPES[shape]
    x = make_inputs(B, T, sh.num_key_heads, sh.num_value_heads, K, V, seed=zlib.crc32(f"repeat-{shape}".encode()))
    per_token = T > 1
    o_first, state_first = run_op(device, x, output_final_state=not per_token, output_per_token_state=per_token)
    for launch in range(2, 6):
        o, state = run_op(device, x, output_final_state=not per_token, output_per_token_state=per_token)
        assert torch.equal(o, o_first) and torch.equal(state, state_first), f"launch {launch} differs from launch 1"


@pytest.mark.parametrize("device_params", [{"trace_region_size": 2000000}], indirect=True)
@pytest.mark.parametrize("B, T", [(1, 4), (8, 1)])
def test_trace_equals_eager(device, B, T):
    """The call captured into a trace and replayed three times gives the eager result bit for bit. The inputs are
    persistent device tensors written before the capture; the outputs are read through the captured handles after
    each replay."""
    sh = SHAPES["qwen27b_tp4"]
    x = make_inputs(B, T, sh.num_key_heads, sh.num_value_heads, K, V, seed=zlib.crc32(f"trace-{B}-{T}".encode()))
    per_token = T > 1
    inputs = {name: _to_device(device, x[name]) for name in ("q", "k", "v", "g", "beta", "initial_state")}

    def call():
        return OP(
            inputs["q"],
            inputs["k"],
            inputs["v"],
            inputs["g"],
            inputs["beta"],
            initial_state=inputs["initial_state"],
            output_final_state=not per_token,
            output_per_token_state=per_token,
        )

    o_eager, state_eager = call()
    o_eager, state_eager = ttnn.to_torch(o_eager), ttnn.to_torch(state_eager)
    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    o_traced, state_traced = call()
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    try:
        for replay in range(1, 4):
            ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
            o, state = ttnn.to_torch(o_traced), ttnn.to_torch(state_traced)
            assert torch.equal(o, o_eager) and torch.equal(state, state_eager), f"replay {replay} differs from eager"
    finally:
        ttnn.release_trace(device, trace_id)


@pytest.mark.parametrize("shape", ["qwen27b_tp4", "qwen9b_tp1"])
def test_fp64_drift(device, shape):
    """64 chained single-token steps from a random state, the state kept on the device between steps, against the
    recurrence computed in fp64: the PCC of o and of the state at every step, and the error curve in the log."""
    sh = SHAPES[shape]
    steps = 64
    x = make_inputs(1, steps, sh.num_key_heads, sh.num_value_heads, K, V, seed=zlib.crc32(f"drift-{shape}".encode()))
    state_dev = _to_device(device, x["initial_state"])
    state_ref = x["initial_state"].to(torch.float64)
    curve = []
    for t in range(steps):
        sl = slice(t, t + 1)
        o_dev, state_dev = OP(
            _to_device(device, x["q"][:, sl]),
            _to_device(device, x["k"][:, sl]),
            _to_device(device, x["v"][:, sl]),
            _to_device(device, x["g"][:, sl]),
            _to_device(device, x["beta"][:, sl]),
            initial_state=state_dev,
            output_final_state=True,
        )
        o_ref, state_ref = recurrent_gated_delta_rule(
            x["q"][:, sl],
            x["k"][:, sl],
            x["v"][:, sl],
            x["beta"][:, sl],
            x["g"][:, sl],
            initial_state=state_ref,
            output_final_state=True,
            dtype=torch.float64,
        )
        m_o, m_state = metrics(o_ref, ttnn.to_torch(o_dev)), metrics(state_ref, ttnn.to_torch(state_dev))
        curve.append((m_o["pcc"], m_state["pcc"], m_state["max_abs_rel"]))
        assert m_o["pcc"] >= PCC_MIN and m_state["pcc"] >= PCC_MIN, f"step {t}: o {m_o}, state {m_state}"
    logger.info(
        f"{shape}: after 64 steps o pcc {curve[-1][0]:.7f}, state pcc {curve[-1][1]:.7f}; state max|d|/max|ref| at "
        f"steps 1/8/16/32/64: " + " ".join(f"{curve[i][2]:.1e}" for i in (0, 7, 15, 31, 63))
    )
