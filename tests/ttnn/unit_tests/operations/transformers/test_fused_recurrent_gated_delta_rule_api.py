# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""The contract of ttnn.transformer.fused_recurrent_gated_delta_rule: every validation rule fails on its own message,
an unknown keyword is a TypeError, the program config, the in-op norm against host-normalised inputs, bf16 inputs,
the in-place final state with leading rows, the preallocated output, and the program cache. Host-side failures and a
handful of calls, so the file is cheap on every SKU of the ttnn sanity job (the shape sweep of the norm test is
hardware-only)."""

import os
import zlib

import pytest
import torch

import ttnn
from tt_py_test_utils_common.utility_functions import comp_pcc, is_blackhole
from tests.ttnn.unit_tests.operations.transformers.gdn_decode_test_utils import SHAPES, assert_accuracy, make_inputs
from ttnn.operations.transformer_golden import l2_norm

pytestmark = pytest.mark.skipif(not is_blackhole(), reason="fused_recurrent_gated_delta_rule is Blackhole-only")

OP = ttnn.transformer.fused_recurrent_gated_delta_rule
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


def _op(device, x, *, repl=None, dev_repl=None, initial_state=_VALID, **kw):
    """The op on the valid call's tensors, with torch replacements (uploaded) or device-tensor replacements."""
    repl, dev_repl = repl or {}, dev_repl or {}
    args = []
    for name in ("q", "k", "v", "g", "beta"):
        args.append(dev_repl[name] if name in dev_repl else _dev(device, repl.get(name, x[name])))
    if initial_state is _VALID:
        initial_state = _dev(device, x["initial_state"])
    return OP(*args, initial_state=initial_state, **kw)


# ---- one case per rule of the contract: (id, kwargs builder, message fragment) ---------------------------------------
# The builder returns the keyword arguments of _op for a call that breaks exactly one rule; the message is what the
# TT_FATAL text must contain (a regular expression).
RULES = [
    (
        "q rank",
        lambda d, x: dict(repl={"q": x["q"].reshape(B, T, H * K), "k": x["k"].reshape(B, T, H * K)}),
        r"q must be \[B,T,H,K\]",
    ),
    ("k shape", lambda d, x: dict(repl={"k": x["k"][:, :, :2].contiguous()}), r"k must have q's shape"),
    ("v batch", lambda d, x: dict(repl={"v": x["v"].repeat(2, 1, 1, 1)}), r"v must be \[B,T,HV,V\]"),
    ("g rank", lambda d, x: dict(repl={"g": x["g"][:, :, 0].contiguous()}), r"g must be \[B,T,HV\]"),
    ("beta shape", lambda d, x: dict(repl={"beta": x["beta"][:, :1].contiguous()}), r"beta must be \[B,T,HV\]"),
    (
        "dtype",
        lambda d, x: dict(dev_repl={"q": _dev(d, torch.ones(B, T, H, K, dtype=torch.int32), dtype=ttnn.int32)}),
        r"q must be bfloat16 or float32",
    ),
    (
        "layout",
        lambda d, x: dict(dev_repl={"q": _dev(d, x["q"], layout=ttnn.ROW_MAJOR_LAYOUT)}),
        r"q must be TILE layout",
    ),
    (
        "one dtype",
        lambda d, x: dict(dev_repl={"k": _dev(d, x["k"], dtype=ttnn.bfloat16)}),
        r"q, k and v must share one dtype",
    ),
    (
        "HV % H",
        lambda d, x: dict(repl={"q": _inputs(h=5)["q"], "k": _inputs(h=5)["k"]}),
        r"HV \(12\) must be a multiple of H \(5\)",
    ),
    (
        "K % 32",
        lambda d, x: dict(repl={"q": _inputs(k=100)["q"], "k": _inputs(k=100)["k"]}),
        r"K and V must be non-zero multiples of 32",
    ),
    ("T > 32", lambda d, x: dict(repl=_inputs(t=33)), r"T \(33\) exceeds the maximum of 32"),
    (
        "inplace without state",
        lambda d, x: dict(initial_state=None, inplace_final_state=True),
        r"requires initial_state",
    ),
    (
        "inplace vs per-token",
        lambda d, x: dict(inplace_final_state=True, output_per_token_state=True),
        r"inplace_final_state and output_per_token_state are exclusive",
    ),
    (
        "state shape",
        lambda d, x: dict(initial_state=_dev(d, x["initial_state"][..., :64].contiguous())),
        r"initial_state must be \[Bs, HV, K, V\]",
    ),
    (
        "state dtype",
        lambda d, x: dict(initial_state=_dev(d, x["initial_state"], dtype=ttnn.bfloat16)),
        r"initial_state must be \[Bs, HV, K, V\] fp32",
    ),
    (
        "state rows below B",
        lambda d, x: dict(
            repl={n: _inputs(b=2)[n] for n in ("q", "k", "v", "g", "beta")}, initial_state=_dev(d, x["initial_state"])
        ),
        r"Bs >= B",
    ),
    (
        "output shape",
        lambda d, x: dict(output_tensor=_dev(d, torch.zeros(B, T, HV, 64))),
        r"output_tensor must be \[B, T, HV, V\]",
    ),
    (
        "output dtype",
        lambda d, x: dict(output_tensor=_dev(d, torch.zeros(B, T, HV, V), dtype=ttnn.bfloat16)),
        r"output_tensor must be \[B, T, HV, V\] in the input dtype",
    ),
    (
        "ring needs per-token",
        lambda d, x: dict(initial_state_block_idx=_i32(d, list(range(B * HV)))),
        r"initial_state_block_idx requires output_per_token_state",
    ),
    ("num_slices", lambda d, x: dict(program_config=Config(num_slices=3)), r"num_slices must be 1, 2 or 4"),
    ("whole head: slices", lambda d, x: dict(program_config=Config(num_slices=2)), r"one whole head per core"),
    ("whole head: units", lambda d, x: dict(program_config=Config(units_per_core=2)), r"one whole head per core"),
    (
        "grid",
        lambda d, x: dict(repl=_inputs(b=16, t=1), initial_state=_dev(d, _inputs(b=16, t=1)["initial_state"])),
        r"exceeds compute cores",
    ),
]


@pytest.mark.parametrize("case", RULES, ids=[c[0] for c in RULES])
def test_rejects(device, case, expect_error):
    """Each rule of the contract fails on its own message. Every call is valid but for the one rule it breaks, so the
    message proves the rule and not an earlier check."""
    _, build, message = case
    x = _inputs()
    with expect_error(RuntimeError, message):
        _op(device, x, **build(device, x))


def test_unknown_keyword_is_a_type_error(device, expect_error):
    """A keyword this signature does not have (the state pool's, here) is a TypeError, not a rejection message."""
    x = _inputs()
    with expect_error(TypeError, r"incompatible function arguments"):
        _op(device, x, state_indices=_i32(device, [0]))


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
    """The op on raw q, k with use_qk_l2norm_in_kernel, and the op on inputs normalised by the golden module's l2_norm
    (the same formula, eps inside the root), both match the golden at the accuracy gates. Between themselves they agree
    to the tf32 truncation of their FPU operands: the in-kernel form feeds the FPU the raw rows and folds 1/|k| and
    scale/|q| into the products, the host form feeds it the normalised rows, so the two truncations differ by up to
    2^-10 per element and the results by a few 1e-3 of their maximum, at a PCC above 0.99999."""
    sh = SHAPES[shape]
    x = make_inputs(
        1, 1, sh.num_key_heads, sh.num_value_heads, K, V, seed=zlib.crc32(shape.encode()), normalize_qk=False
    )
    o_in, s_in = _op(device, x, output_final_state=True, use_qk_l2norm_in_kernel=True)
    o_host, s_host = _op(device, x, output_final_state=True, repl={"q": l2_norm(x["q"]), "k": l2_norm(x["k"])})
    o_ref, s_ref = ttnn.get_golden_function(OP)(
        x["q"],
        x["k"],
        x["v"],
        x["g"],
        x["beta"],
        initial_state=x["initial_state"],
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    for label, ref, got in (
        ("o, norm in kernel", o_ref, o_in),
        ("state, norm in kernel", s_ref, s_in),
        ("o, norm on host", o_ref, o_host),
        ("state, norm on host", s_ref, s_host),
    ):
        assert_accuracy(ref, ttnn.to_torch(got), label)
    for label, a, b in (("o", o_host, o_in), ("state", s_host, s_in)):
        ref, got = ttnn.to_torch(a).double(), ttnn.to_torch(b).double()
        passed, pcc = comp_pcc(ref, got, 0.99999)
        rel = (ref - got).abs().max().item() / ref.abs().max().item()
        assert passed and rel <= 5e-3, f"{label}: pcc {pcc}, max |delta| / max |ref| {rel:.2e}"


def test_bf16_inputs_match_the_golden(device):
    """bf16 q, k, v (g and beta fp32): o comes back bf16, and o and the state match the golden run on the same
    bf16-rounded values at the accuracy gate."""
    x = _inputs(seed=14)
    bf = {n: x[n].to(torch.bfloat16) for n in ("q", "k", "v")}
    dev = {n: _dev(device, bf[n], dtype=ttnn.bfloat16) for n in bf}
    o, state = _op(device, x, dev_repl=dev, output_final_state=True)
    assert o.dtype == ttnn.bfloat16
    golden = ttnn.get_golden_function(OP)
    o_ref, s_ref = golden(
        bf["q"].float(),
        bf["k"].float(),
        bf["v"].float(),
        x["g"],
        x["beta"],
        initial_state=x["initial_state"],
        output_final_state=True,
    )
    for label, ref, got in (("o", o_ref, ttnn.to_torch(o).float()), ("state", s_ref, ttnn.to_torch(state))):
        passed, pcc = comp_pcc(ref.double(), got.double(), 0.999)
        assert passed, f"{label}: pcc {pcc}"


def test_inplace_final_state_with_leading_rows(device):
    """A state of Bs = B + 2 rows: rows [0:B] come back as the final state of the plain run, the other rows are
    byte-identical to before, and the returned tensor is the caller's buffer."""
    x = _inputs(seed=15)
    torch.manual_seed(15)
    extra = 0.05 * torch.randn(2, HV, K, V, dtype=torch.float32)
    pool_t = torch.cat([x["initial_state"], extra], dim=0)
    _, plain = _op(device, x, output_final_state=True)
    pool = _dev(device, pool_t)
    _, back = _op(device, x, initial_state=pool, inplace_final_state=True)
    assert back.buffer_address() == pool.buffer_address(), "inplace_final_state returned a copy"
    got = ttnn.to_torch(pool)
    assert torch.equal(got[:B], ttnn.to_torch(plain)), "rows [0:B] differ from the final state of the plain run"
    assert torch.equal(got[B:], extra), "rows [B:Bs] were touched"


def test_output_tensor_is_written_in_place(device):
    """o lands in the caller's preallocated tensor, which is what the op returns."""
    x = _inputs(seed=16)
    o_plain, _ = _op(device, x)
    out = _dev(device, torch.zeros(B, T, HV, V))
    o, _ = _op(device, x, output_tensor=out)
    assert o.buffer_address() == out.buffer_address(), "output_tensor was not returned"
    assert torch.equal(ttnn.to_torch(out), ttnn.to_torch(o_plain))


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
