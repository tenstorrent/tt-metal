# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Test utilities for the fused GDN decode recurrence (ttnn.transformer.fused_recurrent_gated_delta_rule).

The reference semantics live in the op's registered golden (ttnn.get_golden_function(op), implemented in
ttnn/ttnn/operations/transformer_golden.py); this module holds what is not reference semantics: the model shapes, input
generation, the chained single-token helper, the accuracy gate and metrics, the device-side checks shared by the sanity
and the nightly test files (accuracy against the golden, multi-token versus chained single-token identity, the
compatibility digests), and a vendored copy of FLA's naive recurrence as the oracle the golden is checked against.

The op's device tests are split by CI pipeline; both files import this module:
  tests/ttnn/unit_tests/operations/transformers/test_fused_recurrent_gated_delta_rule.py: the sanity subset, run by
    the ttnn sanity job on every SKU, the ttsim simulators included;
  tests/ttnn/nightly/unit_tests/operations/transformers/test_fused_recurrent_gated_delta_rule.py: the full matrix,
    run by the L2 nightly on hardware.
"""

import datetime
import hashlib
import json
import os
import pathlib
import subprocess
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

# The sanity job that runs the unit_tests directory has simulator SKUs (roughly 100x slower than silicon, and their
# bits are not the silicon's); tests that only add coverage on hardware carry this mark.
SIM = bool(os.environ.get("TT_METAL_SIMULATOR"))
hw_only = pytest.mark.skipif(SIM, reason="hardware test only: too slow on the simulator, or its bits are not silicon's")

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


def pcc(a, b):
    a = a.detach().to(torch.float64).flatten()
    b = b.detach().to(torch.float64).flatten()
    if torch.equal(a, b):
        return 1.0
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm()
    return float((a @ b / denom).item()) if denom > 0 else float("nan")


def metrics(ref, got):
    """PCC, max |delta| and max |delta| relative to max |ref| between a reference and a result."""
    ref64, got64 = ref.detach().to(torch.float64), got.detach().to(torch.float64)
    max_abs = (ref64 - got64).abs().max().item()
    scale = ref64.abs().max().item()
    return {"pcc": pcc(ref64, got64), "max_abs": max_abs, "max_abs_rel": max_abs / scale if scale > 0 else max_abs}


def assert_accuracy(ref, got, label, *, pcc_min=PCC_MIN, max_abs_rel=MAX_ABS_REL):
    assert torch.isfinite(got).all(), f"{label}: non-finite values"
    pcc_passed, pcc_value = comp_pcc(ref, got, pcc_min)
    max_abs = (ref.to(torch.float64) - got.to(torch.float64)).abs().max().item()
    assert (
        pcc_passed and max_abs <= max_abs_rel * ref.abs().max().item()
    ), f"{label}: pcc {pcc_value}, max |delta| {max_abs}"


def golden_digest(tensor):
    """sha256 of the fp32 bytes of a tensor: the compatibility golden of an output."""
    return hashlib.sha256(tensor.detach().to(torch.float32).contiguous().numpy().tobytes()).hexdigest()


# ---- the device-side checks shared by the sanity and the nightly test files ------------------------------------------


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


def op_as_torch_fn(device):
    """The op as a torch-to-torch function in the golden's calling convention, for chained_decode."""

    def fn(q, k, v, g, beta, *, initial_state=None, output_final_state=False):
        x = {"q": q, "k": k, "v": v, "g": g, "beta": beta, "initial_state": initial_state}
        return run_op(device, x, output_final_state=output_final_state)

    return fn


def assert_multi_token_equals_chained(device, B, K_draft, shape="qwen27b_tp4"):
    """One call with T = K_draft + 1 tokens and per-token states is bit-identical to K_draft + 1 chained single-token
    calls, each fed the previous call's state: the speculative decoder's verify path equals its decode path."""
    sh = SHAPES[shape]
    skip_unless_fits(device, B, sh.num_value_heads)
    T = K_draft + 1
    x = make_inputs(B, T, sh.num_key_heads, sh.num_value_heads, K, V, seed=zlib.crc32(f"chain-{B}-{T}".encode()))
    o, states = run_op(device, x, output_per_token_state=True)
    o_chain, states_chain = chained_decode(
        op_as_torch_fn(device), x["q"], x["k"], x["v"], x["g"], x["beta"], x["initial_state"]
    )
    assert torch.equal(o, o_chain), f"o: max |d| {(o - o_chain).abs().max().item():.3e}"
    assert torch.equal(states, states_chain), f"states: max |d| {(states - states_chain).abs().max().item():.3e}"


# ---- compatibility goldens ------------------------------------------------------------------------------------------
# The op's bits on fixed inputs, as sha256 digests stored next to this module. A change of the arithmetic anywhere in
# the call shows there even when it stays within the accuracy gates, so the diff of the JSON in the PR that changes the
# bits is the review artefact. Hardware-only: the simulator's bits are not the silicon's.
GOLDENS_PATH = pathlib.Path(__file__).with_name("fused_recurrent_gdn_goldens.json")
GOLDEN_CASES = [  # (shape, B, T); each with and without an initial state
    ("qwen27b_tp4", 1, 1),
    ("qwen27b_tp4", 8, 1),
    ("qwen27b_tp4", 1, 8),
    ("qwen27b_tp8", 1, 1),
    ("qwen9b_tp1", 1, 1),
    ("qwen35b_a3b_tp4", 4, 4),
]


def golden_case_id(shape, B, T, with_state):
    return f"{shape}/B{B}/T{T}/{'s0' if with_state else 'zero'}"


def golden_cases():
    for shape, B, T in GOLDEN_CASES:
        for with_state in (True, False):
            yield shape, B, T, with_state


def _tree_sha():
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short=11", "HEAD"],
            cwd=GOLDENS_PATH.parent,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def assert_compatibility_golden(device, shape, B, T, with_state):
    """The op's digests on the case's fixed inputs against the stored goldens. GDN_GOLDENS_UPDATE=1 rewrites the case's
    entry (digests, the PCC of each output against the golden, the tree, the device and the date) instead of checking.
    """
    sh = SHAPES[shape]
    skip_unless_fits(device, B, sh.num_value_heads)
    case = golden_case_id(shape, B, T, with_state)
    seed = zlib.crc32(f"golden-{case}".encode())
    x = make_inputs(B, T, sh.num_key_heads, sh.num_value_heads, K, V, seed=seed, with_state=with_state)
    per_token = T > 1
    o, state = run_op(device, x, output_final_state=not per_token, output_per_token_state=per_token)
    got = {"o": golden_digest(o), "state": golden_digest(state)}
    if os.environ.get("GDN_GOLDENS_UPDATE"):
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
        entry = {
            **got,
            "pcc_o": round(metrics(o_ref, o)["pcc"], 7),
            "pcc_state": round(metrics(state_ref, state)["pcc"], 7),
            "tree": _tree_sha(),
            "device": os.environ.get("GDN_GOLDENS_DEVICE", ttnn.get_arch_name()),
            "date": datetime.date.today().isoformat(),
        }
        doc = (
            json.loads(GOLDENS_PATH.read_text())
            if GOLDENS_PATH.exists()
            else {"version": 1, "op": "ttnn.transformer.fused_recurrent_gated_delta_rule", "cases": {}}
        )
        doc["cases"][case] = entry
        GOLDENS_PATH.write_text(json.dumps(doc, indent=1) + "\n")
        return
    stored = json.loads(GOLDENS_PATH.read_text())["cases"].get(case)
    assert stored is not None, f"{case}: no stored golden; GDN_GOLDENS_UPDATE=1 generates it"
    assert got == {"o": stored["o"], "state": stored["state"]}, (
        f"{case}: digests differ from the goldens taken at {stored['tree']} on {stored['device']} ({stored['date']}); "
        "GDN_GOLDENS_UPDATE=1 regenerates them when the change of bits is intended"
    )


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
