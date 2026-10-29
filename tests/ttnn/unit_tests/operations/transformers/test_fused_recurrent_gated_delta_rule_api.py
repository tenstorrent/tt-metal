# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""The v2 contract of ttnn.transformer.fused_recurrent_gated_delta_rule on today's kernel: every validation rule
and every stage-gated rejection fails on its own message, the program config, the KDA alias, the in-op norm against
host-normalised inputs, and the program cache. Host-side failures and a handful of calls, so the file is cheap on
every SKU of the ttnn sanity job, the simulators included (the shape sweep of the norm test is hardware-only)."""

import os
import zlib

import pytest
import torch

import ttnn
from tt_py_test_utils_common.utility_functions import comp_pcc, is_blackhole
from tests.ttnn.unit_tests.operations.transformers.gdn_decode_test_utils import SHAPES, make_inputs
from ttnn.operations.transformer_golden import l2_norm

pytestmark = pytest.mark.skipif(not is_blackhole(), reason="fused_recurrent_gated_delta_rule is Blackhole-only")

OP = ttnn.transformer.fused_recurrent_gated_delta_rule
KDA = ttnn.transformer.fused_recurrent_kda
Config = ttnn.transformer.FusedRecurrentGdnProgramConfig
K = V = 128
B, T, H, HV = 1, 2, 4, 12  # the valid call every rule case starts from: 27B TP 4 head counts, two tokens
_hw_only = pytest.mark.skipif(bool(os.environ.get("TT_METAL_SIMULATOR")), reason="hardware only: a shape sweep")
_VALID = object()  # "the valid call's initial_state"


def _dev(device, t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(t, dtype=dtype, layout=layout, device=device)


def _i32(device, values):
    return ttnn.from_torch(
        torch.tensor(values, dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )


def _inputs(seed=11, b=B, t=T, h=H, hv=HV, k=K, v=V, **kw):
    return make_inputs(b, t, h, hv, k, v, seed=seed, **kw)


def _op(device, x, *, repl=None, dev_repl=None, initial_state=_VALID, op=OP, **kw):
    """The op on the valid call's tensors, with torch replacements (uploaded) or device-tensor replacements."""
    repl, dev_repl = repl or {}, dev_repl or {}
    args = []
    for name in ("q", "k", "v", "g", "beta"):
        args.append(dev_repl[name] if name in dev_repl else _dev(device, repl.get(name, x[name])))
    if initial_state is _VALID:
        initial_state = _dev(device, x["initial_state"])
    return op(*args, initial_state=initial_state, **kw)


def _g4(x):
    return x["g"].unsqueeze(-1).expand(-1, -1, -1, K).contiguous()


# ---- one case per rule of the contract: (id, kwargs builder, message fragment) ---------------------------------------
# The builder returns the keyword arguments of _op for a call that breaks exactly one rule; the message is what the
# TT_FATAL text must contain (a regular expression). Permanent rules first, then the stage-gated rejections.
RULES = [
    (
        "R1 q rank",
        lambda d, x: dict(repl={"q": x["q"].reshape(B, T, H * K), "k": x["k"].reshape(B, T, H * K)}),
        r"q must be \[B,T,H,K\]",
    ),
    ("R1 k shape", lambda d, x: dict(repl={"k": x["k"][:, :, :2].contiguous()}), r"k must have q's shape"),
    ("R1 v batch", lambda d, x: dict(repl={"v": x["v"].repeat(2, 1, 1, 1)}), r"v must be \[B,T,HV,V\]"),
    ("R1 g rank", lambda d, x: dict(repl={"g": x["g"][:, :, 0].contiguous()}), r"g must be \[B,T,HV\]"),
    ("R1 beta shape", lambda d, x: dict(repl={"beta": x["beta"][:, :1].contiguous()}), r"beta must be \[B,T,HV\]"),
    (
        "R1 dtype",
        lambda d, x: dict(dev_repl={"q": _dev(d, torch.ones(B, T, H, K, dtype=torch.int32), dtype=ttnn.int32)}),
        r"q must be bfloat16 or float32",
    ),
    (
        "R2 HV % H",
        lambda d, x: dict(repl={"q": _inputs(h=5)["q"], "k": _inputs(h=5)["k"]}),
        r"HV \(12\) must be a multiple of H \(5\)",
    ),
    (
        "R3 K % 32",
        lambda d, x: dict(repl={"q": _inputs(k=100)["q"], "k": _inputs(k=100)["k"]}),
        r"K and V must be non-zero multiples of 32",
    ),
    ("R4 T > 32", lambda d, x: dict(repl=_inputs(t=33)), r"T \(33\) exceeds the maximum of 32"),
    (
        "R5 cu_seqlens B",
        lambda d, x: dict(
            repl={n: _inputs(b=2)[n] for n in ("q", "k", "v", "g", "beta")}, cu_seqlens=_i32(d, [0, 1, 2])
        ),
        r"cu_seqlens requires B == 1",
    ),
    ("R5 cu_seqlens state", lambda d, x: dict(cu_seqlens=_i32(d, [0, 1, 2])), r"one entry per sequence of cu_seqlens"),
    (
        "R6 neg eigval",
        lambda d, x: dict(allow_neg_eigval=True),
        r"allow_neg_eigval requires use_beta_sigmoid_in_kernel",
    ),
    ("R7 lower_bound", lambda d, x: dict(lower_bound=-5.0), r"lower_bound applies to the KDA gate"),
    ("R8 gate params", lambda d, x: dict(use_gate_in_kernel=True), r"use_gate_in_kernel requires A_log and dt_bias"),
    (
        "R8 dt_bias shape",
        lambda d, x: dict(
            use_gate_in_kernel=True, A_log=_dev(d, torch.zeros(HV)), dt_bias=_dev(d, torch.zeros(HV + 1))
        ),
        r"dt_bias must be \[HV\]",
    ),
    ("R8 A_log alone", lambda d, x: dict(A_log=_dev(d, torch.zeros(HV))), r"apply only with use_gate_in_kernel"),
    ("R9 inplace", lambda d, x: dict(inplace_final_state=True), r"inplace_final_state requires initial_state"),
    (
        "R10 accepted 1-D",
        lambda d, x: dict(state_indices=_i32(d, [0]), num_accepted_tokens=_i32(d, [1])),
        r"num_accepted_tokens requires 2-D state_indices",
    ),
    (
        "R11 index dtype",
        lambda d, x: dict(state_indices=_dev(d, torch.zeros(B), layout=ttnn.ROW_MAJOR_LAYOUT)),
        r"state_indices must be int32",
    ),
    (
        "R11 index columns",
        lambda d, x: dict(state_indices=_i32(d, [[0, 1, 2]])),
        r"state_indices must be int32 \[B\] or \[B, T\]",
    ),
    (
        "R12 pool rank",
        lambda d, x: dict(state_indices=_i32(d, [0]), initial_state=_dev(d, x["initial_state"][0])),
        r"initial_state is the pool",
    ),
    (
        "R13 state shape",
        lambda d, x: dict(initial_state=_dev(d, x["initial_state"][..., :64].contiguous())),
        r"initial_state must be \[B, HV, K, V\]",
    ),
    (
        "R14 output shape",
        lambda d, x: dict(output_tensor=_dev(d, torch.zeros(B, T, HV, 64))),
        r"output_tensor must be \[B, T, HV, V\]",
    ),
    (
        "R15 ring needs per-token",
        lambda d, x: dict(initial_state_block_idx=_i32(d, list(range(B * HV)))),
        r"initial_state_block_idx requires output_per_token_state",
    ),
    (
        "R15 ring vs indices",
        lambda d, x: dict(
            initial_state_block_idx=_i32(d, list(range(B * HV))),
            state_indices=_i32(d, [0]),
            output_per_token_state=True,
        ),
        r"ring mode\) and state_indices are exclusive",
    ),
    ("R16 num_slices", lambda d, x: dict(program_config=Config(num_slices=3)), r"num_slices must be 1, 2 or 4"),
    ("S2 num_slices", lambda d, x: dict(program_config=Config(num_slices=2)), r"stage 2"),
    ("S2 units_per_core", lambda d, x: dict(program_config=Config(units_per_core=2)), r"stage 2"),
    (
        "S2 grid",
        lambda d, x: dict(repl=_inputs(b=16, t=1), initial_state=_dev(d, _inputs(b=16, t=1)["initial_state"])),
        r"above the .* compute cores.*stage 2",
    ),
    ("S3a indices", lambda d, x: dict(state_indices=_i32(d, [0])), r"stage 3a"),
    ("S3a inplace", lambda d, x: dict(state_indices=_i32(d, [0]), inplace_final_state=True), r"stage 3a"),
    (
        "S4 output_tensor",
        lambda d, x: dict(output_tensor=_dev(d, torch.zeros(B, T, HV, V))),
        r"output_tensor is not implemented.*stage 4",
    ),
    (
        "S4 ROW_MAJOR",
        lambda d, x: dict(dev_repl={"q": _dev(d, x["q"], layout=ttnn.ROW_MAJOR_LAYOUT)}),
        r"ROW_MAJOR inputs.*stage 4",
    ),
    ("S5 KDA g", lambda d, x: dict(repl={"g": _g4(x)}), r"KDA.*stage 5"),
    ("S5 beta sigmoid", lambda d, x: dict(use_beta_sigmoid_in_kernel=True), r"stage 5"),
    (
        "S5 gate",
        lambda d, x: dict(use_gate_in_kernel=True, A_log=_dev(d, torch.zeros(HV)), dt_bias=_dev(d, torch.zeros(HV))),
        r"stage 5",
    ),
    ("S8 state_v_first", lambda d, x: dict(state_v_first=True), r"state_v_first.*stage 8"),
    (
        "SU cu_seqlens",
        lambda d, x: dict(cu_seqlens=_i32(d, [0, 1, 2]), initial_state=None),
        r"cu_seqlens \(varlen\) is not scheduled",
    ),
]


@pytest.mark.parametrize("case", RULES, ids=[c[0] for c in RULES])
def test_rejects(device, case, expect_error):
    """Each rule of the contract, and each stage-gated rejection, fails on its own message. Every call is valid but
    for the one rule it breaks, so the message proves the rule and not an earlier check."""
    _, build, message = case
    x = _inputs()
    with expect_error(RuntimeError, message):
        _op(device, x, **build(device, x))


def test_kda_alias_requires_rank4_g_and_rejects_until_stage_5(device, expect_error):
    assert callable(ttnn.get_golden_function(KDA))
    x = _inputs()
    with expect_error(RuntimeError, r"fused_recurrent_kda: g must be the per-key decay"):
        _op(device, x, op=KDA)
    with expect_error(RuntimeError, r"KDA.*stage 5"):
        _op(device, x, op=KDA, repl={"g": _g4(x)})


def test_program_config_default_equals_none(device):
    """A program config with every field left to the op gives the same bits as no config, and prints."""
    x = _inputs(seed=12)
    o_none, s_none = _op(device, x, output_per_token_state=True)
    o_cfg, s_cfg = _op(device, x, output_per_token_state=True, program_config=Config())
    assert torch.equal(ttnn.to_torch(o_none), ttnn.to_torch(o_cfg))
    assert torch.equal(ttnn.to_torch(s_none), ttnn.to_torch(s_cfg))
    assert repr(Config(num_slices=4)) == "FusedRecurrentGdnProgramConfig(num_slices=4, units_per_core=None)"


@pytest.mark.parametrize(
    "shape",
    ["qwen27b_tp4"] + [pytest.param(s, marks=_hw_only) for s in SHAPES if s != "qwen27b_tp4"],
)
def test_qk_l2norm_in_kernel_matches_host_norm(device, shape):
    """The op on raw q, k with use_qk_l2norm_in_kernel against the op on inputs normalised by the golden module's
    l2_norm (the same formula, eps inside the root): the device chain and torch round differently, so the gate is
    PCC and max |delta| one order tighter than the accuracy gates, not bit identity."""
    sh = SHAPES[shape]
    x = make_inputs(
        1, 1, sh.num_key_heads, sh.num_value_heads, K, V, seed=zlib.crc32(shape.encode()), normalize_qk=False
    )
    o_in, s_in = _op(device, x, output_final_state=True, use_qk_l2norm_in_kernel=True)
    o_host, s_host = _op(device, x, output_final_state=True, repl={"q": l2_norm(x["q"]), "k": l2_norm(x["k"])})
    for label, a, b in (("o", o_host, o_in), ("state", s_host, s_in)):
        ref, got = ttnn.to_torch(a).double(), ttnn.to_torch(b).double()
        passed, pcc = comp_pcc(ref, got, 0.99999)
        rel = (ref - got).abs().max().item() / ref.abs().max().item()
        assert passed and rel <= 1e-3, f"{label}: pcc {pcc}, max |delta| / max |ref| {rel:.2e}"


@_hw_only
def test_program_cache_grows_only_with_the_program(device):
    """Repeated calls and a default program config add no program-cache entries; a different batch width does."""
    x = _inputs(seed=13)
    _op(device, x, output_final_state=True)
    n = device.num_program_cache_entries()
    _op(device, x, output_final_state=True)
    assert device.num_program_cache_entries() == n, "a repeated call compiled a new program"
    _op(device, x, output_final_state=True, program_config=Config())
    assert device.num_program_cache_entries() == n, "a default program config compiled a new program"
    _op(device, _inputs(seed=13, b=2), output_final_state=True)
    assert device.num_program_cache_entries() > n, "a different B reused a program"
