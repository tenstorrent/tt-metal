# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Shared test harness for the Quasar binary_ng sweeps under
tests/ttnn/nightly/unit_tests/operations/experimental/quasar/. Holds the helpers originally defined in
test_binary_ng_no_bcast.py (_on_quasar, the *_sharded_config builders, _act, _run, _run_mixed) so they can
be reused by both the no-broadcast suite and upcoming subtile-broadcast test files.

Not collected by pytest: the module deliberately has no `test_` prefix.
"""

import os

import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


def _on_quasar():
    # ttnn.get_arch_name() returns "invalid" under the simulator, so detect Quasar from the sim env
    # vars the simulator run sets (ARCH_NAME=quasar / CHIP_ARCH=quasar).
    return any("quasar" in os.environ.get(v, "").lower() for v in ("ARCH_NAME", "CHIP_ARCH"))


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
# as does any fp32 op. maximum is ALWAYS-SFPU (no FPU form) and its ckernel (binary_max_min.h) is ported
# to Quasar, so it exercises the generic SFPU path beyond mul/div. SFPU builds+runs on Wormhole and Quasar
# (fp32 add/sub SFPU ported to Quasar in #49883).
_OPS = {
    "add": (lambda: ttnn.experimental.quasar.add, torch.add),
    "subtract": (lambda: ttnn.experimental.quasar.subtract, torch.subtract),
    "multiply": (lambda: ttnn.experimental.quasar.multiply, torch.multiply),
    "divide": (lambda: ttnn.experimental.quasar.divide, torch.divide),
    "maximum": (lambda: ttnn.experimental.quasar.maximum, torch.maximum),
}

# PCC thresholds. NEVER weakened below what the descriptor path achieves for the same config.
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

    # Golden: lhs/rhs activations apply before the binary op (to a/b respectively), post activation after.
    a_golden = _ACT_GOLDEN[lhs_act](a) if lhs_act is not None else a
    b_golden = _ACT_GOLDEN[rhs_act](b) if rhs_act is not None else b
    golden = torch_fn(a_golden, b_golden)
    if post_act is not None:
        golden = _ACT_GOLDEN[post_act](golden)

    a_tt = ttnn.from_torch(a, dtype=dtype_tt, device=device, layout=ttnn.TILE_LAYOUT, memory_config=mem_config)
    b_tt = ttnn.from_torch(b, dtype=dtype_tt, device=device, layout=ttnn.TILE_LAYOUT, memory_config=mem_config)

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
    assert_with_pcc(out_torch, golden, pcc or _PCC[dtype_tt])
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
):
    # Like _run, but with an INDEPENDENT memory config per operand (a, b, output) so the borrow-vs-NoC
    # routing in the DFB factory is exercised across mixed sharded/interleaved layouts (borrow only when
    # all three are L1-sharded on one matching grid; otherwise every operand is NoC-read/written). Also
    # like _run, optionally fuses lhs/rhs (pre) and post activation params -- used by the sharded-broadcast-
    # operand activation-over-broadcast cases, e.g. a height-sharded broadcast operand feeding the bcast
    # reader's NoC path rather than the fully-interleaved one.
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

    # Golden: lhs/rhs activations apply before the binary op (to a/b respectively), post activation after.
    a_golden = _ACT_GOLDEN[lhs_act](a) if lhs_act is not None else a
    b_golden = _ACT_GOLDEN[rhs_act](b) if rhs_act is not None else b
    golden = torch_fn(a_golden, b_golden)
    if post_act is not None:
        golden = _ACT_GOLDEN[post_act](golden)

    a_tt = ttnn.from_torch(a, dtype=dtype_tt, device=device, layout=ttnn.TILE_LAYOUT, memory_config=a_mem)
    b_tt = ttnn.from_torch(b, dtype=dtype_tt, device=device, layout=ttnn.TILE_LAYOUT, memory_config=b_mem)

    kwargs = {"memory_config": out_mem, "dtype": dtype_tt}
    if post_act is not None:
        kwargs["activations"] = _act(post_act)
    if lhs_act is not None:
        kwargs["input_tensor_a_activations"] = _act(lhs_act)
    if rhs_act is not None:
        kwargs["input_tensor_b_activations"] = _act(rhs_act)

    out_tt = ttnn_fn(a_tt, b_tt, **kwargs)
    out_torch = ttnn.to_torch(out_tt)

    # Same degenerate-constant guard as _run (catches an operand-switch / wrong-tile-pairing bug, the
    # exact failure mode the mixed borrowed/NoC routing risks).
    golden_std = golden.float().std()
    if golden_std > 0.1:
        assert (
            out_torch.float().std() > 0.1 * golden_std
        ), "output is ~constant while the golden varies (wrong tile pairing?)"
    assert_with_pcc(out_torch, golden, pcc or _PCC[dtype_tt])
    return out_tt
