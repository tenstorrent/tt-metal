# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The device KDA recurrence on synthetic edge inputs (accuracy tier C) against the FP64 golden of the same inputs.

Real inputs rarely isolate the cases that break chunked low-precision kernels, so each case here targets one:
short memory (the strongest decay), long memory, both at once in one head, repeated keys, and zero values. Every
gate stays inside what Kimi-K3's bounded gate ``-5·sigmoid(x)`` can emit and is fractional (never format-exact),
so a case exercises the same argument precision a trained layer does. Each generated case asserts that it reaches
its edge, so the test cannot pass on inputs that drifted away from it.

Each case is cast to the device formats (BF16 q/k/v/gate/beta logits, FP32 state) and the golden is the FP64
recurrence of exactly those cast values, so the measured error is the device's alone. Output and final state are
gated per head (PCC + NRMSE), so one head's drift is not averaged away; zero values have an exact zero answer and
are checked absolutely. The golden costs about a second of CPU per case.
"""

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.deepseek_v3_d_p.reference.kda.ops import kda_recurrent_reference
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config
from models.demos.deepseek_v3_d_p.tests.kda.components.test_recurrence import _run_recurrence, _to_device
from models.demos.deepseek_v3_d_p.tt.kda.config import KDA_CHUNK_SIZE
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import assert_accurate

pytestmark = run_for_blackhole()

HEADS, KEY_DIM, VALUE_DIM = 2, 128, 128
GATE_LOWER_BOUND = KimiK3Config.KDA_GATE_LOWER_BOUND
# Largest |G| one chunk can accumulate: every token's log decay is above the gate bound.
CHUNK_DECAY_LIMIT = -GATE_LOWER_BOUND * KDA_CHUNK_SIZE
EDGE_CASES = ("baseline", "short_memory", "long_memory", "short_and_long_memory", "repeated_keys", "zero_values")

# (case, tokens) -> (output, state) worst-head NRMSE thresholds.
_THRESHOLDS: dict[tuple[str, int], tuple[float, float]] = {
    ("baseline", 256): (0.015, 0.04),
    ("baseline", 2048): (0.02, 0.05),
    ("short_memory", 256): (0.015, 0.015),
    ("short_memory", 2048): (0.02, 0.015),
    ("short_and_long_memory", 256): (0.015, 0.01),
    ("short_and_long_memory", 2048): (0.015, 0.015),
    ("long_memory", 256): (0.015, 0.015),
    ("long_memory", 2048): (0.02, 0.02),
    ("repeated_keys", 256): (0.02, 0.04),
    ("repeated_keys", 2048): (0.02, 0.04),
}


def _bounded_gate(logits: torch.Tensor) -> torch.Tensor:
    return GATE_LOWER_BOUND * torch.sigmoid(logits)


def _edge_case(name: str, sequence: int, *, seed: int) -> dict[str, torch.Tensor]:
    """``q, k, v, gate, beta_logits`` (and a zero ``initial_state``) for one named edge case, in FP32 ``[B, T, H, D]``.

    ``beta_logits`` is what the device recurrence takes; the recurrence uses ``beta = sigmoid(beta_logits)``.
    """
    generator = torch.Generator().manual_seed(seed)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator)

    q = randn(1, sequence, HEADS, KEY_DIM)
    k = randn(1, sequence, HEADS, KEY_DIM)
    v = randn(1, sequence, HEADS, VALUE_DIM)
    beta_logits = randn(1, sequence, HEADS)
    noise = randn(1, sequence, HEADS, KEY_DIM)
    long_memory = max(1, KEY_DIM // 16)

    if name == "baseline":
        # The whole bounded range, as random hidden states give.
        gate = _bounded_gate(2.0 * noise)
    elif name == "short_memory":
        # Every channel a few 1e-3 above -5 per token: |G_last| ≈ 159.9 per 32 tokens, never exactly 160.
        gate = _bounded_gate(8.0 + 0.5 * noise)
    elif name == "long_memory":
        # Decay ~1e-5 per token: exp(gate) rounds to 1 in BF16 (spacing 2^-8 below 1).
        gate = _bounded_gate(-13.0 + 0.5 * noise)
    elif name == "short_and_long_memory":
        # Most channels at short memory next to a few long-memory channels in the same head, with beta near 1:
        # the in-domain K3 shape that still fails on device (g1b.4.16 k3-strong-saturated).
        gate = _bounded_gate(8.0 + 0.5 * noise)
        gate[..., :long_memory] = _bounded_gate(-13.0 + 0.5 * noise[..., :long_memory])
        beta_logits = 3.5 + 1.1 * torch.sigmoid(randn(1, sequence, HEADS))  # beta = sigmoid(...) in [0.97, 0.99]
    elif name == "repeated_keys":
        # One key repeated for every token of a head: the in-chunk triangular system is as ill-conditioned
        # as KDA allows, and beta near 1 keeps every rank-1 update large.
        gate = _bounded_gate(2.0 * noise)
        k = k[:, :1].expand_as(k).clone()
        beta_logits = 2.2 + 3.0 * torch.sigmoid(randn(1, sequence, HEADS))  # beta = sigmoid(...) in [0.90, 0.995]
    else:  # zero_values
        # Zero values with a zero initial state: the exact output and state are zero, so only an absolute
        # check is meaningful.
        gate = _bounded_gate(2.0 * noise)
        v = torch.zeros_like(v)

    case = {
        "q": q,
        "k": k,
        "v": v,
        "gate": gate,
        "beta_logits": beta_logits,
        "initial_state": torch.zeros(1, HEADS, KEY_DIM, VALUE_DIM),
    }
    _assert_reaches_edge(name, case, long_memory)
    return case


def _chunk_decay(gate: torch.Tensor) -> torch.Tensor:
    """|G| of each full chunk, per head and channel; a partial tail chunk is not checked."""
    full = gate.shape[1] // KDA_CHUNK_SIZE * KDA_CHUNK_SIZE
    return -gate[:, :full].unflatten(1, (-1, KDA_CHUNK_SIZE)).sum(dim=2)


def _assert_reaches_edge(name: str, case: dict[str, torch.Tensor], long_memory: int) -> None:
    gate, beta = case["gate"], torch.sigmoid(case["beta_logits"])
    assert bool(((gate > GATE_LOWER_BOUND) & (gate < 0.0)).all()), f"{name}: gate outside ({GATE_LOWER_BOUND}, 0)"

    if name in ("short_memory", "short_and_long_memory"):
        decay = _chunk_decay(gate if name == "short_memory" else gate[..., long_memory:])
        assert bool(((decay > CHUNK_DECAY_LIMIT - 1.0) & (decay < CHUNK_DECAY_LIMIT)).all()), (
            f"{name}: chunk |G| in [{decay.min():.3f}, {decay.max():.3f}], "
            f"expected just under {CHUNK_DECAY_LIMIT:g}"
        )
    if name in ("long_memory", "short_and_long_memory"):
        slow = gate if name == "long_memory" else gate[..., :long_memory]
        assert bool((torch.exp(slow).bfloat16() == 1.0).all()), f"{name}: a long-memory decay is not 1 in BF16"
    if name == "short_and_long_memory":
        assert bool(((beta > 0.97) & (beta < 0.99)).all()), f"{name}: beta outside (0.97, 0.99)"
    if name == "repeated_keys":
        assert torch.equal(case["k"], case["k"][:, :1].expand_as(case["k"])), f"{name}: keys differ across tokens"
        assert bool((beta >= 0.9).all()), f"{name}: beta below 0.9"
    if name == "zero_values":
        assert not case["v"].any(), f"{name}: values are not zero"


@pytest.mark.parametrize("sequence,summary_group_chunks", [(256, 2), (2048, 8)], ids=["T256", "T2048"])
@pytest.mark.parametrize("name", EDGE_CASES)
@pytest.mark.use_module_device
def test_recurrence_edge_case_against_fp64_golden(
    device: ttnn.Device, name: str, sequence: int, summary_group_chunks: int
) -> None:
    case = _edge_case(name, sequence, seed=sequence)
    device_inputs = {key: case[key].bfloat16() for key in ("q", "k", "v", "gate", "beta_logits")}
    golden_output, golden_state = kda_recurrent_reference(
        *(device_inputs[key].double() for key in ("q", "k", "v", "gate")),
        torch.sigmoid(device_inputs["beta_logits"].double()),
        case["initial_state"].double(),
        dtype=torch.float64,
    )

    def flat(key: str) -> torch.Tensor:
        return device_inputs[key].reshape(1, sequence, -1)

    with ttnn.manage_config("throw_exception_on_fallback", True):
        state_tt, output_tt = _run_recurrence(
            device,
            _to_device(flat("q"), device, ttnn.bfloat16),
            _to_device(flat("k"), device, ttnn.bfloat16),
            _to_device(flat("v"), device, ttnn.bfloat16),
            _to_device(flat("gate"), device, ttnn.bfloat16),
            _to_device(device_inputs["beta_logits"], device, ttnn.bfloat16),
            _to_device(case["initial_state"].float(), device, ttnn.float32),
            summary_group_chunks=summary_group_chunks,
            local_scan_strategy="grouped",
        )
    output = ttnn.to_torch(output_tt).reshape(1, HEADS, sequence, VALUE_DIM).permute(0, 2, 1, 3).double()
    state = ttnn.to_torch(state_tt).double()

    if name == "zero_values":
        # The exact output and state are zero.
        assert not output.any() and not state.any(), (float(output.abs().max()), float(state.abs().max()))
        return
    output_threshold, state_threshold = _THRESHOLDS[(name, sequence)]
    assert_accurate(
        golden_output,
        output,
        name=f"{name} T{sequence} output",
        nrmse_threshold=output_threshold,
        nrmse_keep_dims=(2,),
    )
    assert_accurate(
        golden_state,
        state,
        name=f"{name} T{sequence} state",
        nrmse_threshold=state_threshold,
        nrmse_keep_dims=(1,),
    )
