# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Test utilities for the fused GDN decode recurrence (ttnn.transformer.fused_recurrent_gated_delta_rule).

The reference semantics live in the op's registered golden (ttnn.get_golden_function(op), implemented in
ttnn/ttnn/operations/transformer_golden.py); this module holds what is not reference semantics: the model shapes, input
generation, the chained single-token helper, the accuracy gate, the device-side accuracy check shared by the sanity and
the nightly test files, and a vendored copy of FLA's naive recurrence as the oracle the golden is checked against.

The op's device tests are split by CI pipeline; both files import this module:
  tests/ttnn/unit_tests/operations/transformers/test_fused_recurrent_gated_delta_rule.py: the sanity subset, run by
    the ttnn sanity job on every SKU, the ttsim simulators included;
  tests/ttnn/nightly/unit_tests/operations/transformers/test_fused_recurrent_gated_delta_rule.py: the full matrix,
    run by the L2 nightly on hardware.
"""

import zlib
from dataclasses import dataclass

import pytest
import torch

import ttnn
from tt_py_test_utils_common.utility_functions import comp_pcc
from ttnn.operations.transformer_golden import l2_norm

OP = ttnn.transformer.fused_recurrent_gated_delta_rule
K = V = 128  # head dimensions of every served configuration
BATCHES = (1, 2, 4, 8)
TOKENS = (1, 4, 8, 12)  # decode, and speculative verify with 3, 7 and 11 draft tokens
P150_CORES = 110  # the compute grid the case matrix is sized for; larger grids re-check at run time

# Accuracy gates: PCC, and max |delta| relative to max |reference| as a gross-error guard. The FPU reads fp32 operands
# truncated to tf32-class precision (9-10 mantissa bits), so the current kernel sits at up to 5e-3 relative error at
# every token count (1 to 12) while its PCC stays above 0.99999.
PCC_MIN = 0.9999
MAX_ABS_REL = 1e-2


@dataclass(frozen=True)
class Shape:
    num_key_heads: int
    num_value_heads: int
    supported_today: bool = True  # the current op: rank-3 g only, B * HV <= 110 cores


# Per-device head counts of served configurations. K = V = 128 throughout.
SHAPES = {
    "qwen27b_tp4": Shape(4, 12),  # Qwen3.6-27B TP 4
    "qwen27b_tp8": Shape(2, 6),  # Qwen3.6-27B TP 8
    "qwen35b_a3b_tp4": Shape(4, 8),  # Qwen3.6-35B-A3B TP 4
    "qwen9b_tp1": Shape(16, 32),  # Qwen3.5-9B single chip
    # Kimi-K3's KDA has H == HV and a per-key decay vector (rank-4 g); the current op takes rank-3 g only, so this
    # entry is the GDN-shaped stand-in at the served head count until the KDA path lands.
    "kimi_k3_sp8tp4": Shape(24, 24),  # Kimi-K3 SP 8 x TP 4
}


def make_inputs(B, T, H, HV, K, V, *, seed, normalize_qk=True, with_state=True):
    """Recurrence inputs in the layer's post-conv convention: q, k, v ~ N(0, 1), beta ~ U(0, 1) (a sigmoid output),
    g = -2 U(0, 1) (log decay), state ~ 0.05 N(0, 1). With normalize_qk the q, k rows are L2-normalised (the golden
    module's l2_norm) as the current op expects them. Returns a dict of fp32 torch tensors, token-major [B, T, heads, D].
    """
    gen = torch.Generator().manual_seed(seed)
    r = lambda *shape: torch.randn(*shape, generator=gen, dtype=torch.float32)
    q, k = r(B, T, H, K), r(B, T, H, K)
    if normalize_qk:
        q, k = l2_norm(q), l2_norm(k)
    v = r(B, T, HV, V)
    beta = torch.rand(B, T, HV, generator=gen, dtype=torch.float32)
    g = -torch.rand(B, T, HV, generator=gen, dtype=torch.float32) * 2.0
    s0 = 0.05 * r(B, HV, K, V) if with_state else None
    return {"q": q, "k": k, "v": v, "g": g, "beta": beta, "initial_state": s0}


def chained_decode(fn, q, k, v, g, beta, initial_state=None):
    """Run fn (op or golden, (q, k, v, g, beta, *, initial_state, output_final_state) -> (o, state)) one token at a
    time and stitch: o [B, T, HV, V], states [B, T, HV, K, V] (the state after each token)."""
    T = q.shape[1]
    state = initial_state
    outs, states = [], []
    for t in range(T):
        sl = slice(t, t + 1)
        o_t, state = fn(
            q[:, sl], k[:, sl], v[:, sl], g[:, sl], beta[:, sl], initial_state=state, output_final_state=True
        )
        outs.append(o_t)
        states.append(state)
    return torch.cat(outs, dim=1), torch.stack(states, dim=1)


def assert_accuracy(ref, got, label, *, pcc_min=PCC_MIN, max_abs_rel=MAX_ABS_REL):
    assert torch.isfinite(got).all(), f"{label}: non-finite values"
    pcc_passed, pcc = comp_pcc(ref, got, pcc_min)
    max_abs = (ref.to(torch.float64) - got.to(torch.float64)).abs().max().item()
    assert pcc_passed and max_abs <= max_abs_rel * ref.abs().max().item(), f"{label}: pcc {pcc}, max |delta| {max_abs}"


# ---- the device-side accuracy check shared by the sanity and the nightly test files ----------------------------------


def to_device(device, t):
    return ttnn.from_torch(t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)


def run_op(device, x, *, output_final_state=False, output_per_token_state=False):
    """Today's calling convention: fp32 TILE interleaved tensors, q and k L2-normalised on the host (make_inputs does
    that), the state passed as a tensor. Returns torch tensors."""
    args = [to_device(device, x[name]) for name in ("q", "k", "v", "g", "beta")]
    s0 = to_device(device, x["initial_state"]) if x["initial_state"] is not None else None
    o, state = OP(
        *args,
        initial_state=s0,
        output_final_state=output_final_state,
        output_per_token_state=output_per_token_state,
    )
    return ttnn.to_torch(o), (ttnn.to_torch(state) if state is not None else None)


def skip_unless_fits(device, B, HV):
    grid = device.compute_with_storage_grid_size()
    if B * HV > grid.x * grid.y:
        pytest.skip(f"B*HV = {B * HV} exceeds the {grid.x * grid.y}-core grid: one core per (b, hv) today")


def case_id(shape, B, T, with_state):
    return f"{shape}-B{B}-T{T}-{'s0' if with_state else 'zero'}"


def accuracy_cases():
    """The accuracy matrix the current op supports, as (shape, B, T, with_state) tuples: every served head count, B in
    BATCHES within the p150 grid, T in TOKENS, with and without an initial state."""
    for name, sh in SHAPES.items():
        if not sh.supported_today:
            continue
        for B in BATCHES:
            if B * sh.num_value_heads > P150_CORES:
                continue
            for T in TOKENS:
                for with_state in (True, False):
                    yield name, B, T, with_state


def assert_decode_matches_golden(device, shape, B, T, with_state):
    """o and the state after every token (T > 1) or the final state (T = 1) against the golden at the accuracy gates,
    on inputs seeded by the case."""
    sh = SHAPES[shape]
    skip_unless_fits(device, B, sh.num_value_heads)
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


def fla_naive_recurrent_gated_delta_rule(q, k, v, beta, g, scale=None, initial_state=None, output_final_state=False):
    """Vendored copy of FLA's naive_recurrent_gated_delta_rule (flash-linear-attention,
    fla/ops/gated_delta_rule/naive.py, MIT license, (c) Songlin Yang et al.), FLA order (q, k, v, beta, g). The oracle
    of the golden; vendored because no tt-metal environment installs FLA."""
    q, k, v, beta, g = map(lambda x: x.transpose(1, 2).contiguous().to(torch.float32), [q, k, v, beta, g])
    B, H, T, K, V = *k.shape, v.shape[-1]
    o = torch.zeros(B, H, T, V).to(v)
    h = torch.zeros(B, H, K, V).to(v)
    if initial_state is not None:
        h = initial_state.to(torch.float32)
    if scale is None:
        scale = 1 / (q.shape[-1] ** 0.5)
    q = q * scale
    for i in range(T):
        b_q = q[:, :, i]
        b_k = k[:, :, i]
        b_v = v[:, :, i].clone()
        h = h.clone() * g[:, :, i].exp()[..., None, None]
        b_beta = beta[:, :, i]
        b_v = b_v - (h.clone() * b_k[..., None]).sum(-2)
        b_v = b_v * b_beta[..., None]
        h = h.clone() + b_k.unsqueeze(-1) * b_v.unsqueeze(-2)
        o[:, :, i] = torch.einsum("bhd,bhdm->bhm", b_q, h)
    if not output_final_state:
        h = None
    o = o.transpose(1, 2).contiguous()
    return o, h
