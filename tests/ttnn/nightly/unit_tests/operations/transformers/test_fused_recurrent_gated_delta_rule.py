# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Device tests of ttnn.transformer.fused_recurrent_gated_delta_rule on hardware: the full accuracy matrix against its
registered golden (the served per-device head counts, HV 6, 8, 12, 24, 32 with their key-head counts; the batch widths
within the grid; the token counts of decode and of speculative verify with 3, 7 and 11 draft tokens; with and without an
initial state), the multi-token versus chained single-token identity matrix, run-to-run and trace-versus-eager
determinism, drift over 64 chained steps against an fp64 reference, and the compatibility digests of the op's bits on
fixed inputs.

Runs on hardware in the L2 nightly ("ttnn nightly transformers tests" in tests/pipeline_reorg/ops_unit_tests.yaml). The
sanity subset that gates every PR, on the simulators too, is
tests/ttnn/unit_tests/operations/transformers/test_fused_recurrent_gated_delta_rule.py; the helpers the two files share
are in tests/ttnn/unit_tests/operations/transformers/gdn_decode_test_utils.py."""

import zlib

import pytest
import torch
from loguru import logger

import ttnn
from tt_py_test_utils_common.utility_functions import is_blackhole
from tests.ttnn.unit_tests.operations.transformers.gdn_decode_test_utils import (
    K,
    OP,
    PCC_MIN,
    SHAPES,
    V,
    accuracy_cases,
    assert_compatibility_golden,
    assert_decode_matches_golden,
    assert_multi_token_equals_chained,
    case_id,
    golden_case_id,
    golden_cases,
    make_inputs,
    metrics,
    run_op,
    to_device,
)
from ttnn.operations.transformer_golden import recurrent_gated_delta_rule

pytestmark = pytest.mark.skipif(not is_blackhole(), reason="fused_recurrent_gated_delta_rule is Blackhole-only")


@pytest.mark.parametrize("shape, B, T, with_state", [pytest.param(*c, id=case_id(*c)) for c in accuracy_cases()])
def test_decode_matches_golden(device, shape, B, T, with_state):
    """o and the state after every token (T > 1) or the final state (T = 1) against the golden at the accuracy gates."""
    assert_decode_matches_golden(device, shape, B, T, with_state)


@pytest.mark.parametrize("B", [1, 8])
@pytest.mark.parametrize("K_draft", [3, 7, 11])
def test_multi_token_equals_chained_single_token(device, K_draft, B):
    """One call with T = K + 1 tokens and per-token states is bit-identical to K + 1 chained single-token calls, each
    fed the previous call's state: the speculative decoder's verify path equals its decode path."""
    assert_multi_token_equals_chained(device, B=B, K_draft=K_draft)


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
    inputs = {name: to_device(device, x[name]) for name in ("q", "k", "v", "g", "beta", "initial_state")}

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
    state_dev = to_device(device, x["initial_state"])
    state_ref = x["initial_state"].to(torch.float64)
    curve = []
    for t in range(steps):
        sl = slice(t, t + 1)
        o_dev, state_dev = OP(
            to_device(device, x["q"][:, sl]),
            to_device(device, x["k"][:, sl]),
            to_device(device, x["v"][:, sl]),
            to_device(device, x["g"][:, sl]),
            to_device(device, x["beta"][:, sl]),
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


@pytest.mark.parametrize("shape, B, T, with_state", [pytest.param(*c, id=golden_case_id(*c)) for c in golden_cases()])
def test_compatibility_goldens(device, shape, B, T, with_state):
    """The op's bits on fixed inputs against the digests stored next to gdn_decode_test_utils.py
    (fused_recurrent_gdn_goldens.json): a change of the arithmetic anywhere in the call shows here even when it stays
    within the accuracy gates. GDN_GOLDENS_UPDATE=1 rewrites the case's entry, so the diff of the JSON in the PR that
    changes the bits is the review artefact."""
    assert_compatibility_golden(device, shape, B, T, with_state)
