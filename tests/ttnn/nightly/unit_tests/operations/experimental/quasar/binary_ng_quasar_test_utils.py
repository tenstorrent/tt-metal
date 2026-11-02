# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Shared test harness for the Quasar binary_ng sweeps under
tests/ttnn/nightly/unit_tests/operations/experimental/quasar/. Holds the helpers originally defined in
test_binary_ng_no_bcast.py (the *_sharded_config builders, _act, _run, _run_mixed) so they can
be reused by both the no-broadcast suite and upcoming subtile-broadcast test files.

Not collected by pytest: the module deliberately has no `test_` prefix.
"""

import math

import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc, assert_with_ulp, ulp_distance


def _height_sharded_config(shard_shape, core_grid):
    return ttnn.create_sharded_memory_config(
        shard_shape,
        core_grid=core_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _block_sharded_config(shard_shape, core_grid):
    return ttnn.create_sharded_memory_config(
        shard_shape,
        core_grid=core_grid,
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _width_sharded_config(shard_shape, core_grid):
    return ttnn.create_sharded_memory_config(
        shard_shape,
        core_grid=core_grid,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


# Op table: name -> (ttnn fn, torch golden). add/subtract take the FPU compute kernel; multiply and
# divide route the SFPU compute kernel by default (is_binary_sfpu_op is true unless fast-approx mode),
# as does any fp32 op. maximum/minimum are ALWAYS-SFPU (no FPU form) and share one ckernel
# (binary_max_min.h), ported to Quasar, so they exercise the generic SFPU path beyond mul/div. Both are
# carried so the pair stays symmetric: they are the same route and the same gate treatment, so covering
# only one would leave the other gate-admitted-but-unvalidated. SFPU builds+runs on Wormhole and Quasar
# (fp32 add/sub SFPU ported to Quasar in #49883).
_OPS = {
    "add": (lambda: ttnn.experimental.quasar.add, torch.add),
    "subtract": (lambda: ttnn.experimental.quasar.subtract, torch.subtract),
    "multiply": (lambda: ttnn.experimental.quasar.multiply, torch.multiply),
    "divide": (lambda: ttnn.experimental.quasar.divide, torch.divide),
    "maximum": (lambda: ttnn.experimental.quasar.maximum, torch.maximum),
    "minimum": (lambda: ttnn.experimental.quasar.minimum, torch.minimum),
}

# PCC thresholds. Never weaken these.
_PCC = {ttnn.bfloat16: 0.997, ttnn.float32: 0.9999}

# Default shape for _run_mixed: a 16x16-tile square (256 tiles). See test_binary_ng_no_bcast.py for the
# sharded-layout configs (_MIXED_HEIGHT / _MIXED_BLOCK / _MIXED_WIDTH / ...) built on this same shape.
_MIXED_SHAPE = (16 * 32, 16 * 32)


# Fused activations exercised by the op, each with its torch golden. A lhs (pre) activation applies to
# operand A before the binary op; a post activation applies to the result. RELU is ResNet50's fused
# residual activation; SILU is Llama's SwiGLU gate (models/tt_transformers/tt/mlp.py emits
# ttnn.mul(w1_out, w3_out, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])). GELU/TANH/SQUARE/SIGMOID
# are further activations the WH-baseline matrix (QUASAR_LLK_GAPS.md Table 2) marks SUPPORTED on Quasar
# (each has a Quasar ckernel + SfpuType + an #else ARCH_QUASAR compute-API branch); they are exercised by
# test_no_bcast_activation_supported in test_binary_ng_no_bcast.py.
_ACT_GOLDEN = {
    ttnn.UnaryOpType.RELU: torch.relu,
    ttnn.UnaryOpType.SILU: torch.nn.functional.silu,
    ttnn.UnaryOpType.GELU: torch.nn.functional.gelu,
    ttnn.UnaryOpType.TANH: torch.tanh,
    ttnn.UnaryOpType.SQUARE: torch.square,
    ttnn.UnaryOpType.SIGMOID: torch.sigmoid,
}


def _act(act_type):
    return [ttnn.UnaryWithParam(act_type)] if act_type is not None else []


# The largest ULP distance from torch's result on the device's operands, per dtype and op, on Quasar (fp32 as measured
# on ttsim); the Wormhole and Blackhole runs share these bounds. bf16 takes production's bound of 1: on Quasar the SFPU
# multiply truncates and the FPU add rounds twice. maximum and minimum select an operand, so they are exact.
_ULP = {
    ttnn.float32: {"add": 0, "subtract": 0, "multiply": 0, "divide": 2, "maximum": 0, "minimum": 0},
    ttnn.bfloat16: {"add": 1, "subtract": 1, "multiply": 1, "divide": 1, "maximum": 0, "minimum": 0},
}


def _fpu_add_bf16(a, b):
    """bf16 a + b as Quasar's FPU rounds it. The FPU aligns the operand with the smaller exponent to the other's with 7
    bits below the bf16 ULP, rounds it half away from zero, and rounds the sum to nearest even."""
    a, b = a.double(), b.double()
    ea, eb = torch.frexp(a).exponent, torch.frexp(b).exponent
    a_big = ea >= eb
    big, small = torch.where(a_big, a, b), torch.where(a_big, b, a)
    # frexp's mantissa is in [0.5, 1), so the larger operand's bf16 ULP is 2**(e - 8); the FPU keeps 7 bits below it.
    step = torch.exp2((torch.maximum(ea, eb) - (8 + 7)).double())
    aligned = torch.sign(small) * torch.floor(small.abs() / step + 0.5) * step
    rounded = (big + aligned).float().to(torch.bfloat16)
    # frexp gives a zero the exponent 0, which can make the zero the larger operand.
    return torch.where((a == 0) | (b == 0), (a + b).float().to(torch.bfloat16), rounded)


# The bf16 sum as each Quasar target rounds it: twice on the hardware and on ttsim, once on craq-sim. An element is
# right when it matches either.
# TODO: keep only _fpu_add_bf16 when craq-sim rounds the FPU add as the hardware does.
_FPU_ADD_ROUNDINGS = (_fpu_add_bf16, lambda a, b: (a.float() + b.float()).to(torch.bfloat16))


def _add_goldens(ta, tb):
    """The oracles of a bf16 add, on the operands as the device holds them."""
    a, b = ttnn.to_torch(ta).float(), ttnn.to_torch(tb).float()
    return [rounding(a, b) for rounding in _FPU_ADD_ROUNDINGS]


def _matches_none(got, goldens, bound=0):
    """Per element: True where `got` is more than `bound` ULP from every golden."""
    return torch.stack([ulp_distance(got, golden) > bound for golden in goldens]).all(dim=0)


# Activations that keep the golden exact: RELU, and SQUARE in fp32.
_EXACT_ACTS = {
    ttnn.float32: (None, ttnn.UnaryOpType.RELU, ttnn.UnaryOpType.SQUARE),
    ttnn.bfloat16: (None, ttnn.UnaryOpType.RELU),
}
# The other activations, after a multiply: the largest ULP distance on Quasar (bf16 on ttsim, fp32 on craq-sim) against
# a golden that rounds each intermediate to the dtype, or production's bound where larger (TANH: 5). Beside another op
# the check is PCC only, and so is GELU's fused form, the fast lookup table (production's own bf16 bound: 30,000 ULP).
_ACT_ULP = {
    ttnn.float32: {ttnn.UnaryOpType.TANH: 5, ttnn.UnaryOpType.SIGMOID: 3, ttnn.UnaryOpType.SILU: 4},
    ttnn.bfloat16: {
        ttnn.UnaryOpType.TANH: 5,
        ttnn.UnaryOpType.SIGMOID: 5,
        ttnn.UnaryOpType.SILU: 4,
        ttnn.UnaryOpType.SQUARE: 4,
    },
}
# Below these inputs SIGMOID and SILU are under 2%, and their approximation is off by 6-7% of the value, not by a few
# ULP. As production does for approximate modes, the ULP check skips that tail, and at most _MAX_EXCLUDED of the
# elements may fall in it.
_ACT_TAIL = {ttnn.UnaryOpType.SIGMOID: -4.0, ttnn.UnaryOpType.SILU: -4.0}
_MAX_EXCLUDED = 0.05
_TORCH_DTYPE = {ttnn.bfloat16: torch.bfloat16, ttnn.float32: torch.float32}


def _golden(torch_fn, a, b, dtype_tt, lhs_act=None, rhs_act=None, post_act=None):
    """torch's result on the device's operands, with each intermediate rounded to the dtype as the device stores it:
    lhs and rhs activations before the op, a post activation after it. Also returns the elements whose activation
    inputs lie outside every activation's tail (_ACT_TAIL)."""

    def rnd(x):
        return x.to(_TORCH_DTYPE[dtype_tt]).float() if dtype_tt in _TORCH_DTYPE else x

    inputs = []
    if lhs_act is not None:
        inputs.append((lhs_act, a))
        a = rnd(_ACT_GOLDEN[lhs_act](a))
    if rhs_act is not None:
        inputs.append((rhs_act, b))
        b = rnd(_ACT_GOLDEN[rhs_act](b))
    golden = torch_fn(a, b)
    if post_act is not None:
        x = rnd(golden)
        inputs.append((post_act, x))
        golden = _ACT_GOLDEN[post_act](x)
    keep = torch.ones(golden.shape, dtype=torch.bool)
    for act, x in inputs:
        if act in _ACT_TAIL:
            keep &= torch.broadcast_to(x >= _ACT_TAIL[act], golden.shape)
    return golden, keep


def _assert_ulp(out_torch, golden, dtype_tt, ulp_threshold, allow_nonfinite=False):
    # ULP is measured in the golden's resolution, so both sides go to the output's float width: bfloat8_b is
    # compared at bfloat16 resolution, as assert_with_ulp documents. Unlike PCC, a few misplaced or stale
    # shards cannot hide in a high correlation over the rest of the tensor.
    cmp_dtype = torch.float32 if dtype_tt == ttnn.float32 else torch.bfloat16
    assert_with_ulp(
        expected_result=golden.to(cmp_dtype),
        actual_result=out_torch.to(cmp_dtype),
        ulp_threshold=ulp_threshold,
        allow_nonfinite=allow_nonfinite,
    )


def _assert_output(out_torch, golden, dtype_tt, op_name, acts=(), pcc=None, keep=None):
    """Check an output as production does: PCC first, then a ULP bound where one is known, so that a few wrong
    elements cannot hide in a high correlation. The bound is the op's, or the largest of its activations' after a
    multiply; `keep` drops the activations' tails from the ULP check."""
    assert_with_pcc(golden, out_torch, pcc or _PCC[dtype_tt])
    bound = _ULP.get(dtype_tt, {}).get(op_name)
    if bound is None:
        return
    for act in acts:
        if act in _EXACT_ACTS[dtype_tt]:
            continue
        act_bound = _ACT_ULP[dtype_tt].get(act) if op_name == "multiply" else None
        if act_bound is None:
            return
        bound = max(bound, act_bound)
    if keep is not None and not keep.all():
        excluded = 1.0 - keep.float().mean().item()
        assert excluded <= _MAX_EXCLUDED, f"{excluded:.1%} of the elements fall in an activation's tail"
        out_torch, golden = out_torch[keep], golden[keep]
    _assert_ulp(out_torch, golden, dtype_tt, bound)


# Special values, checked as production checks them (test_div_fp32): every pair of these through an op, with inf and
# NaN where torch puts them (allow_nonfinite) and the sign of zero not checked. Quasar's divide gives a number where
# IEEE gives NaN for the pairs in _DIVIDE_NOT_NAN, so those run apart.
_SPECIAL = {"1.5": 1.5, "-2": -2.0, "+0": 0.0, "-0": -0.0, "inf": math.inf, "-inf": -math.inf, "nan": math.nan}
_DIVIDE_NOT_NAN = [("+0", "-0"), ("-0", "+0"), ("-0", "-0"), ("inf", "inf"), ("-inf", "-inf"), ("nan", "+0")]


def _run_special(device, op_name, dtype_tt, pairs):
    """One 32x32 tile of special-value pairs, named by the keys of _SPECIAL, through an op, checked against torch."""
    n = 32 * 32
    reps = -(-n // len(pairs))
    a = torch.tensor(([_SPECIAL[x] for x, _ in pairs] * reps)[:n]).reshape(32, 32)
    b = torch.tensor(([_SPECIAL[y] for _, y in pairs] * reps)[:n]).reshape(32, 32)
    cfg = dict(dtype=dtype_tt, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    a_tt, b_tt = ttnn.from_torch(a, **cfg), ttnn.from_torch(b, **cfg)
    out = ttnn.to_torch(_OPS[op_name][0]()(a_tt, b_tt, dtype=dtype_tt)).float()
    golden = _OPS[op_name][1](ttnn.to_torch(a_tt).float(), ttnn.to_torch(b_tt).float())
    _assert_ulp(out, golden, dtype_tt, _ULP[dtype_tt][op_name], allow_nonfinite=True)


def _run(device, op_name, mem_config, dtype_tt, shape, lhs_act=None, rhs_act=None, post_act=None, pcc=None):
    torch.manual_seed(0)
    ttnn_fn = _OPS[op_name][0]()
    torch_fn = _OPS[op_name][1]

    # `shape` is either a single shape shared by both operands (the no-broadcast case) or a
    # 2-tuple (a_shape, b_shape) that selects a subtile broadcast: the two operands are built at
    # their own shapes and torch/ttnn broadcast for the golden. A pair is detected as a length-2
    # sequence whose first element is itself a sequence (a shape), so an ordinary (H, W) / [N,C,H,W]
    # single shape (first element an int) still takes the shared-shape path -- backward compatible.
    if isinstance(shape, (tuple, list)) and len(shape) == 2 and isinstance(shape[0], (tuple, list)):
        a_shape, b_shape = shape
    else:
        a_shape = b_shape = shape

    a = torch.randn(a_shape, dtype=torch.float32)
    b = torch.randn(b_shape, dtype=torch.float32)
    if op_name == "divide":
        # Keep the divisor away from zero so bf16 PCC is meaningful.
        b = b * 0.5 + 2.0

    a_tt = ttnn.from_torch(a, dtype=dtype_tt, device=device, layout=ttnn.TILE_LAYOUT, memory_config=mem_config)
    b_tt = ttnn.from_torch(b, dtype=dtype_tt, device=device, layout=ttnn.TILE_LAYOUT, memory_config=mem_config)
    # The golden uses the operands as the device holds them, so input quantization does not count as error.
    golden, keep = _golden(
        torch_fn, ttnn.to_torch(a_tt).float(), ttnn.to_torch(b_tt).float(), dtype_tt, lhs_act, rhs_act, post_act
    )

    kwargs = {"memory_config": mem_config, "dtype": dtype_tt}
    if post_act is not None:
        kwargs["activations"] = _act(post_act)
    if lhs_act is not None:
        kwargs["input_tensor_a_activations"] = _act(lhs_act)
    if rhs_act is not None:
        kwargs["input_tensor_b_activations"] = _act(rhs_act)

    out_tt = ttnn_fn(a_tt, b_tt, **kwargs)
    out_torch = ttnn.to_torch(out_tt)
    # Guard against a degenerate constant output silently passing the correlation check: an
    # operand-switch bug makes the kernel use the same operand twice (a, a) instead of (a, b), and for
    # divide a/a = 1.0 everywhere, which assert_with_pcc can spuriously accept. If the golden varies,
    # the output must vary too.
    golden_std = golden.float().std()
    if golden_std > 0.1:
        assert (
            out_torch.float().std() > 0.1 * golden_std
        ), "output is ~constant while the golden varies (operand-switch?)"
    _assert_output(out_torch, golden, dtype_tt, op_name, (lhs_act, rhs_act, post_act), pcc, keep)
    return out_tt


def _run_mixed(
    device,
    op_name,
    a_mem,
    b_mem,
    out_mem,
    dtype_tt,
    shape=_MIXED_SHAPE,
    lhs_act=None,
    rhs_act=None,
    post_act=None,
    pcc=None,
    output_tensor_mem=None,
    ulp_threshold=None,
):
    # Like _run, but with an INDEPENDENT memory config per operand (a, b, output) so the borrow-vs-NoC
    # routing in the DFB factory is exercised across mixed sharded/interleaved layouts (borrow only when
    # all three are L1-sharded with one memory config; otherwise every operand is NoC-read/written). Also
    # like _run, optionally fuses lhs/rhs (pre) and post activation params -- used by the sharded-broadcast-
    # operand activation-over-broadcast cases, e.g. a height-sharded broadcast operand feeding the bcast
    # reader's NoC path rather than the fully-interleaved one. output_tensor_mem preallocates the output
    # with that config and passes it as output_tensor, next to memory_config=out_mem.
    torch.manual_seed(0)
    ttnn_fn = _OPS[op_name][0]()
    torch_fn = _OPS[op_name][1]

    # `shape` is either a single shape shared by both operands (the no-broadcast case) or a 2-tuple
    # (a_shape, b_shape) that selects a subtile broadcast: the two operands are built at their own shapes
    # and torch/ttnn broadcast for the golden. Same pair-detection _run uses (a pair is a length-2
    # sequence whose first element is itself a sequence), so an ordinary (H, W) shape still takes the
    # shared-shape path -- backward compatible with every existing _run_mixed caller.
    if isinstance(shape, (tuple, list)) and len(shape) == 2 and isinstance(shape[0], (tuple, list)):
        a_shape, b_shape = shape
    else:
        a_shape = b_shape = shape

    a = torch.randn(a_shape, dtype=torch.float32)
    b = torch.randn(b_shape, dtype=torch.float32)
    if op_name == "divide":
        b = b * 0.5 + 2.0

    a_tt = ttnn.from_torch(a, dtype=dtype_tt, device=device, layout=ttnn.TILE_LAYOUT, memory_config=a_mem)
    b_tt = ttnn.from_torch(b, dtype=dtype_tt, device=device, layout=ttnn.TILE_LAYOUT, memory_config=b_mem)
    # The golden uses the operands as the device holds them, so input quantization does not count as error.
    golden, keep = _golden(
        torch_fn, ttnn.to_torch(a_tt).float(), ttnn.to_torch(b_tt).float(), dtype_tt, lhs_act, rhs_act, post_act
    )

    kwargs = {"memory_config": out_mem, "dtype": dtype_tt}
    if post_act is not None:
        kwargs["activations"] = _act(post_act)
    if lhs_act is not None:
        kwargs["input_tensor_a_activations"] = _act(lhs_act)
    if rhs_act is not None:
        kwargs["input_tensor_b_activations"] = _act(rhs_act)
    if output_tensor_mem is not None:
        kwargs["output_tensor"] = ttnn.from_torch(
            torch.zeros(golden.shape),
            dtype=dtype_tt,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=output_tensor_mem,
        )

    out_tt = ttnn_fn(a_tt, b_tt, **kwargs)
    out_torch = ttnn.to_torch(out_tt)

    # Same degenerate-constant guard as _run (catches an operand-switch / wrong-tile-pairing bug, the
    # exact failure mode the mixed borrowed/NoC routing risks).
    golden_std = golden.float().std()
    if golden_std > 0.1:
        assert (
            out_torch.float().std() > 0.1 * golden_std
        ), "output is ~constant while the golden varies (wrong tile pairing?)"
    if ulp_threshold is not None:
        _assert_ulp(out_torch, golden, dtype_tt, ulp_threshold)
    else:
        _assert_output(out_torch, golden, dtype_tt, op_name, (lhs_act, rhs_act, post_act), pcc, keep)
    return out_tt


def _run_scalar(
    device,
    op_name,
    mem_config,
    dtype_tt,
    shape,
    scalar,
    lhs_act=None,
    post_act=None,
    pcc=None,
    out_mem=None,
    infer_output_config=False,
    ulp_threshold=None,
):
    # Tensor-SCALAR sibling of _run: operand A is a tensor, the RHS is a Python number (not a tensor),
    # and the golden uses that same scalar. Same op->torch map, PCC thresholds, and degenerate-constant
    # guard as _run. On Quasar the DFB path is the only path that can run (the descriptor throws as
    # unsupported), so a passing PCC here is itself the proof the op routed to the v2/DFB tensor-scalar
    # factory -- the same implicit v2-routing check the no-broadcast / subtile-broadcast suites rely on
    # (there is no explicit DFB-path assertion helper in this module). The scalar is passed as the second
    # positional arg, which the nanobind binding routes to the tensor-scalar overload.
    torch.manual_seed(0)
    ttnn_fn = _OPS[op_name][0]()
    torch_fn = _OPS[op_name][1]

    a = torch.randn(shape, dtype=torch.float32)

    a_tt = ttnn.from_torch(a, dtype=dtype_tt, device=device, layout=ttnn.TILE_LAYOUT, memory_config=mem_config)
    # The golden uses a as the device holds it, and the Python scalar as is.
    golden, keep = _golden(torch_fn, ttnn.to_torch(a_tt).float(), scalar, dtype_tt, lhs_act, None, post_act)

    # infer_output_config omits memory_config so the op derives the output config from a's, which routes through
    # the default-config path instead of an explicit one. out_mem overrides the output config explicitly.
    kwargs = {"dtype": dtype_tt}
    if not infer_output_config:
        kwargs["memory_config"] = out_mem if out_mem is not None else mem_config
    if post_act is not None:
        kwargs["activations"] = _act(post_act)
    if lhs_act is not None:
        kwargs["input_tensor_a_activations"] = _act(lhs_act)

    out_tt = ttnn_fn(a_tt, scalar, **kwargs)
    out_torch = ttnn.to_torch(out_tt)

    # Degenerate-constant guard (as in _run): a varying golden must map to a varying output (catches a
    # broken scalar fill that leaves the RHS tile zero -> out == a, or an incoherent fill -> out ~ 0).
    golden_std = golden.float().std()
    if golden_std > 0.1:
        assert out_torch.float().std() > 0.1 * golden_std, "output is ~constant while the golden varies (scalar fill?)"
    if ulp_threshold is not None:
        _assert_ulp(out_torch, golden, dtype_tt, ulp_threshold)
    else:
        _assert_output(out_torch, golden, dtype_tt, op_name, (lhs_act, post_act), pcc, keep)
    return out_tt
