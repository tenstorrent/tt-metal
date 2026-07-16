# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Sweep for the generalized Metal 2.0 / DataflowBuffer (DFB) no-broadcast slice of the experimental
quasar binary ops. The DFB factory (binary_ng_metal_v2_factory.cpp) is a generic port of the
descriptor factory's SubtileBroadcastType::NONE tiled path; matches_metal_v2_slice routes the whole
no-broadcast tensor-tensor TILE slice to it — FPU and SFPU, interleaved and sharded, every dtype,
full lhs/rhs/post activations.

This sweep exercises the capabilities the ResNet residual-add canary (test_binary_ng_resnet_add.py)
does not: the interleaved/DRAM path (TensorAccessor NoC reads + split_work_to_cores placement +
non-borrowed rings), the generalized FPU compute beyond ADD (subtract/multiply), fp32 (fp32 dest
accumulation + unpack-to-dest), the lhs-activation post DFB self-loop, and the SFPU compute kernel.

The DFB path is arch-portable (CB-backed on Wormhole, overlay-backed on Quasar), so the same test
runs on real Wormhole and the Quasar simulator. Shapes/grids fit both Wormhole (8x8) and the Quasar
simulator (8x4).

Run on Wormhole:
    pytest tests/ttnn/nightly/unit_tests/operations/experimental/quasar/test_binary_ng_no_bcast.py

Run on the Quasar simulator (only fp32 add/sub auto-skip — no SFPU float-add primitive on Quasar, see
_run; everything else runs: the full bf16 FPU/SFPU matrix incl. lhs/rhs activations, plus fp32 mul/div):
    TT_METAL_SIMULATOR=<path>/libttsim.so TT_SIMULATOR_LOCALHOST=1 ARCH_NAME=quasar CHIP_ARCH=quasar \
        TT_METAL_SLOW_DISPATCH_MODE=1 \
        pytest tests/ttnn/nightly/unit_tests/operations/experimental/quasar/test_binary_ng_no_bcast.py
"""

import os

import pytest
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
# as does any fp32 op. SFPU builds+runs on Wormhole and Quasar (only fp32 add/sub auto-skip on Quasar — see _run).
_OPS = {
    "add": (lambda: ttnn.experimental.quasar.add, torch.add),
    "subtract": (lambda: ttnn.experimental.quasar.subtract, torch.subtract),
    "multiply": (lambda: ttnn.experimental.quasar.multiply, torch.multiply),
    "divide": (lambda: ttnn.experimental.quasar.divide, torch.divide),
}

# PCC thresholds. NEVER weakened below what the descriptor path achieves for the same config.
_PCC = {ttnn.bfloat16: 0.997, ttnn.float32: 0.9999}

# Interleaved tensor: 32x40 tiles (1280 tiles) so split_work_to_cores spreads many tiles per core on
# both an 8x8 (Wormhole) and an 8x4 (Quasar sim) worker grid -- ~20 tiles/core on 8x8, ~40/core on 8x4,
# so the compute kernel's per-tile loop runs many iterations. No operand is sharded here, so the size is
# bounded only by DRAM (the tensors stream through per-core DFB rings, never fully L1-resident).
_INTERLEAVED_SHAPE = (32 * 32, 40 * 32)

# Sharded configs: a 4-tall height-shard column and a 2x2 block grid, both fit 8x4. Each shard is a
# 4x4-tile square ([128, 128] = 16 tiles), so every core processes 16 tiles (16 compute-loop iterations).
# Each *_SHAPE is exactly shard x grid.
_HEIGHT_GRID = ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 3))})
_HEIGHT_SHARD = [4 * 32, 4 * 32]  # 16 tiles/core
_HEIGHT_SHAPE = (4 * 4 * 32, 4 * 32)  # 4 cores tall x [128,128] = [512, 128]
_BLOCK_GRID = ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (1, 1))})  # 2x2 grid
_BLOCK_SHARD = [4 * 32, 4 * 32]  # 16 tiles/core
_BLOCK_SHAPE = (2 * 4 * 32, 2 * 4 * 32)  # 2x2 cores x [128,128] = [256, 256]

# Mixed-layout tests with a SHARDED output: every operand (a, b, out) may independently be interleaved
# (DRAM) or L1-sharded, so all layouts must describe the SAME tensor shape. Use a 16x16-tile square (256
# tiles); each sharded layout below tiles it across cores that fit the Quasar 8x4 grid. The op runs on
# the sharded OUTPUT's grid, so compute tiles/core = 256 / (output shard grid) >= 16 for every case here
# (e.g. height/width output = 64 tiles/core on 4 cores; block output = 16 tiles/core on a 16-core grid).
# Mixed cases whose OUTPUT is INTERLEAVED do NOT use this shape -- they run on the full worker grid via
# split_work_to_cores, so 256 tiles would be only ~4/core; they use _BIG_SHAPE below instead.
_MIXED_SHAPE = (16 * 32, 16 * 32)
# Height-sharded: 16 tile-rows down a 4-tall column => [128, 512] shard = 64 tiles/core.
_MIXED_HEIGHT = _height_sharded_config([4 * 32, 16 * 32], ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 3))}))
# Block-sharded: 16x16 tiles on a 4x4 grid => [128, 128] shard = 16 tiles/core (x<=3, y<=3 fit 8x4).
_MIXED_BLOCK = _block_sharded_config([4 * 32, 4 * 32], ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (3, 3))}))
# Width-sharded: 16 tile-columns across a 4-wide row (full height) => [512, 128] shard = 64 tiles/core.
_MIXED_WIDTH = _width_sharded_config([16 * 32, 4 * 32], ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (3, 0))}))
# A second height-sharded spec on a DIFFERENT grid (column x=1, not x=0) for the grid-mismatch case:
# same shape and 64 tiles/core, but the grid differs from _MIXED_HEIGHT so an input on it is NoC-read,
# not borrowed.
_MIXED_HEIGHT_ALT = _height_sharded_config([4 * 32, 16 * 32], ttnn.CoreRangeSet({ttnn.CoreRange((1, 0), (1, 3))}))
_I = ttnn.DRAM_MEMORY_CONFIG
# L1-interleaved (vs DRAM-interleaved _I). is_native_L1_sharding can hold with an L1-interleaved input
# (a single L1-sharded operand satisfies it), but an interleaved operand has no shard spec to borrow, so
# such a mix must take the all-NoC path -- never borrow.
_IL1 = ttnn.L1_MEMORY_CONFIG

# --- Generality matrix: all three operands sharded with DIFFERENT strategies, and/or each operand on a
# DISTINCT core grid. All describe the same _MIXED_SHAPE (16x16 tiles); all grids fit the Quasar 8x4
# worker grid. The factory borrows only when all three operands are L1-sharded on ONE matching grid (the
# is_native case); three different strategies (or grids) is not native, so NOTHING is borrowed -- every
# operand, output included, is read/written via its sharding-aware TensorAccessor. This exercises the
# all-NoC path across strategy AND grid boundaries, which the H/B same-strategy matrix above never does.
#
# Each strategy on its canonical grid (x=0 column / y=0 row / 4x4 block) -- used where the strategies
# differ but the grids may coincide:
_GEN_HEIGHT = _MIXED_HEIGHT  # Height: [128,512] shard down column x=0 (64 tiles/core).
_GEN_WIDTH = _MIXED_WIDTH  # Width: [512,128] shard across row y=0 (64 tiles/core).
_GEN_BLOCK = _MIXED_BLOCK  # Block: [128,128] shard on the 4x4 grid (16 tiles/core).

# Three operands each on a DISTINCT, non-overlapping CoreRangeSet (and distinct strategies):
#   a = Height down column x=0 ([128,512] shard, 64 tiles/core); b = Width across row y=1 ([512,128]
#   shard, 64 tiles/core); out = Block on the 2x2 grid at (2,2)-(3,3) ([256,256] shard, 64 tiles/core).
#   No grid matches the output's, so every operand (output included) is NoC-read/written.
_GRID_A_HEIGHT = _MIXED_HEIGHT  # column x=0, rows 0..3
_GRID_B_WIDTH = _width_sharded_config([16 * 32, 4 * 32], ttnn.CoreRangeSet({ttnn.CoreRange((0, 1), (3, 1))}))  # row y=1
_GRID_C_BLOCK = _block_sharded_config(
    [8 * 32, 8 * 32], ttnn.CoreRangeSet({ttnn.CoreRange((2, 2), (3, 3))})
)  # 2x2 @(2,2)

# --- Big-shape configs for INTERLEAVED-OUTPUT mixed/generality cases -----------------------------------
# These cases run on the FULL worker grid (split_work_to_cores), not a small output shard grid, so they
# need a large total tile count to reach >=16 compute tiles/core. The output is interleaved (streamed
# through per-core DFB rings), so only the sharded INPUT operands occupy L1.
#
# A height/width shard can only spread across its column/row, which is at most 4 cores on the Quasar 8x4
# grid, so a T-tile height/width input is always T/4 tiles/core there -- it cannot be spread thinner. The
# Quasar L1 bank is ~1.19 MB and its bank-allocator budgets L1 per bank across the whole grid, so two
# sharded inputs draw from the SAME per-bank budget even when their core grids are disjoint. Two 320-tile
# (640 KB) shards therefore do not co-reside. So:
#   * Cases with ONE sharded input use _BIG_SHAPE (1280 tiles): 320 tiles/core for the lone shard fits
#     alone, and the full-grid compute is ~20/core on 8x8 Wormhole, ~40/core on 8x4 Quasar.
#   * Cases with TWO sharded inputs use _BIG_SHAPE_2SHARD (1024 tiles): each input is 256 tiles/core
#     (512 KB), so both co-reside in one ~1.19 MB bank (~1.0 MB total, leaving room for the streamed
#     output ring). Full-grid compute is 16/core on 8x8 Wormhole, 32/core on 8x4 Quasar.
# Shapes are shard x grid exactly, all grids fit 8x4 (x<=7, y<=3).
_BIG_SHAPE = (32 * 32, 40 * 32)  # 1280 tiles, single-sharded-input cases
# Height-sharded big input down column x=0: [256,1280] shard on 4 cores => 320 tiles/core.
_BIG_HEIGHT = _height_sharded_config([8 * 32, 40 * 32], ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 3))}))
# Width-sharded big input across row y=0: [1024,320] shard on 4 cores => 320 tiles/core.
_BIG_WIDTH = _width_sharded_config([32 * 32, 10 * 32], ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (3, 0))}))

_BIG_SHAPE_2SHARD = (32 * 32, 32 * 32)  # 1024 tiles, two-sharded-input cases (each shard fits paired in L1)
# Two height inputs on DISJOINT columns (x=0, x=1), each [256,1024] => 256 tiles/core: both fit one bank.
_BIG2_HEIGHT_A = _height_sharded_config([8 * 32, 32 * 32], ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 3))}))
_BIG2_HEIGHT_B = _height_sharded_config([8 * 32, 32 * 32], ttnn.CoreRangeSet({ttnn.CoreRange((1, 0), (1, 3))}))
# Distinct-grid distinct-strategy pair for H@g0.W@g1.I: height down column x=4 ([256,1024]) and width
# across row y=1 ([1024,256]). Column x=4 and row y=1 are disjoint (x=4 not in 0..3); each is 256
# tiles/core, so both fit one bank.
_BIG2_HEIGHT_G = _height_sharded_config([8 * 32, 32 * 32], ttnn.CoreRangeSet({ttnn.CoreRange((4, 0), (4, 3))}))
_BIG2_WIDTH_G = _width_sharded_config([32 * 32, 8 * 32], ttnn.CoreRangeSet({ttnn.CoreRange((0, 1), (3, 1))}))


# Fused activations exercised by the op, each with its torch golden. A lhs (pre) activation applies to
# operand A before the binary op; a post activation applies to the result. RELU is ResNet50's fused
# residual activation; SILU is Llama's SwiGLU gate (models/tt_transformers/tt/mlp.py emits
# ttnn.mul(w1_out, w3_out, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])). GELU/TANH/SQUARE/SIGMOID
# are further activations the WH-baseline matrix (QUASAR_LLK_GAPS.md Table 2) marks SUPPORTED on Quasar
# (each has a Quasar ckernel + SfpuType + an #else ARCH_QUASAR compute-API branch); they are exercised by
# test_no_bcast_activation_supported below.
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


def _run(device, op_name, mem_config, dtype_tt, shape, lhs_act=None, post_act=None, pcc=None):
    if _on_quasar():
        # The SFPU compute kernel builds and runs on Quasar: the int-SFPU op headers it does not need
        # are #ifndef ARCH_QUASAR-guarded, the no-broadcast operand switch uses copy_tile_to_dst_init_short
        # (Quasar's copy_tile_to_dst_init_short_with_dt is a no-op that cannot switch operands), and the
        # activation pack retargets via pack_init (Quasar's pack_reconfig_data_format is gasket-only).
        # bf16 multiply and divide both pass. fp32 routes SFPU: fp32 multiply/divide work on Quasar, but
        # fp32 add/sub do not — the SFPU float add/sub primitives are not yet ported to Quasar
        # (add_binary_tile is #ifndef ARCH_QUASAR; only mul/div have Quasar branches), so they fail to
        # JIT-compile. See binary_ng/QUASAR_PARITY_GAPS.md.
        if dtype_tt == ttnn.float32 and op_name in ("add", "subtract"):
            pytest.skip(
                "SFPU float add/sub not yet ported to Quasar (tenstorrent/tt-metal#49883; fp32 add/sub route "
                "SFPU; fp32 mul/div work)"
            )
        # Interleaved lhs (pre) activation hangs on the Quasar sim: the post_lhs DFB self-loop (the compute
        # kernel both produces the pre-activated operand and consumes it) on the 1-deep, async NoC
        # interleaved ring deadlocks under native timing (and corrupts, PCC ~0.66, under perturbed timing).
        # A post_lhs ring-depth bump did NOT fix it, so it is a substrate/DFB timing bug, not op ring depth.
        # Sharded lhs-activation and interleaved post-activation both pass. Tracked in tenstorrent/tt-metal#49937.
        if lhs_act is not None and not mem_config.is_sharded():
            pytest.skip(
                "Quasar sim: interleaved lhs-activation (post_lhs DFB self-loop) hangs/corrupts — "
                "substrate/DFB timing bug (tenstorrent/tt-metal#49937); sharded lhs-act + interleaved "
                "post-act pass"
            )
    torch.manual_seed(0)
    ttnn_fn = _OPS[op_name][0]()
    torch_fn = _OPS[op_name][1]

    a = torch.randn(shape, dtype=torch.float32)
    b = torch.randn(shape, dtype=torch.float32)
    if op_name == "divide":
        # Keep the divisor away from zero so bf16 PCC is meaningful.
        b = b * 0.5 + 2.0

    # Golden: lhs activation applies before the binary op, post activation after.
    a_golden = _ACT_GOLDEN[lhs_act](a) if lhs_act is not None else a
    golden = torch_fn(a_golden, b)
    if post_act is not None:
        golden = _ACT_GOLDEN[post_act](golden)

    a_tt = ttnn.from_torch(a, dtype=dtype_tt, device=device, layout=ttnn.TILE_LAYOUT, memory_config=mem_config)
    b_tt = ttnn.from_torch(b, dtype=dtype_tt, device=device, layout=ttnn.TILE_LAYOUT, memory_config=mem_config)

    kwargs = {"memory_config": mem_config, "dtype": dtype_tt}
    if post_act is not None:
        kwargs["activations"] = _act(post_act)
    if lhs_act is not None:
        kwargs["input_tensor_a_activations"] = _act(lhs_act)

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


def _run_mixed(device, op_name, a_mem, b_mem, out_mem, dtype_tt, shape=_MIXED_SHAPE, pcc=None):
    # Like _run, but with an INDEPENDENT memory config per operand (a, b, output) so the borrow-vs-NoC
    # routing in the DFB factory is exercised across mixed sharded/interleaved layouts (borrow only when
    # all three are L1-sharded on one matching grid; otherwise every operand is NoC-read/written).
    torch.manual_seed(0)
    ttnn_fn = _OPS[op_name][0]()
    torch_fn = _OPS[op_name][1]

    a = torch.randn(shape, dtype=torch.float32)
    b = torch.randn(shape, dtype=torch.float32)
    if op_name == "divide":
        b = b * 0.5 + 2.0
    golden = torch_fn(a, b)

    a_tt = ttnn.from_torch(a, dtype=dtype_tt, device=device, layout=ttnn.TILE_LAYOUT, memory_config=a_mem)
    b_tt = ttnn.from_torch(b, dtype=dtype_tt, device=device, layout=ttnn.TILE_LAYOUT, memory_config=b_mem)

    out_tt = ttnn_fn(a_tt, b_tt, memory_config=out_mem, dtype=dtype_tt)
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


@pytest.mark.parametrize("op_name", ["add", "subtract", "multiply"])
@pytest.mark.parametrize("dtype_tt", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("post_relu", [False, True])
def test_no_bcast_interleaved(device, op_name, dtype_tt, post_relu):
    # The key new path: DRAM-interleaved inputs/output route through the dual-mode reader/writer
    # (TensorAccessor NoC) + split_work_to_cores placement + non-borrowed DFB rings.
    _run(
        device,
        op_name,
        ttnn.DRAM_MEMORY_CONFIG,
        dtype_tt,
        _INTERLEAVED_SHAPE,
        post_act=ttnn.UnaryOpType.RELU if post_relu else None,
    )


@pytest.mark.parametrize("op_name", ["add", "subtract", "multiply"])
@pytest.mark.parametrize("layout", ["height", "block"])
@pytest.mark.parametrize("post_relu", [False, True])
def test_no_bcast_sharded(device, op_name, layout, post_relu):
    # Generalized sharded FPU compute beyond the canary's ADD (subtract/multiply on the borrowed-shard
    # path), height and block sharded.
    if layout == "height":
        mem_config = _height_sharded_config(_HEIGHT_SHARD, _HEIGHT_GRID)
        shape = _HEIGHT_SHAPE
    else:
        mem_config = _block_sharded_config(_BLOCK_SHARD, _BLOCK_GRID)
        shape = _BLOCK_SHAPE
    _run(device, op_name, mem_config, ttnn.bfloat16, shape, post_act=ttnn.UnaryOpType.RELU if post_relu else None)


@pytest.mark.parametrize("op_name", ["add", "multiply"])
@pytest.mark.parametrize("layout", ["interleaved", "height"])
def test_no_bcast_lhs_activation(device, op_name, layout):
    # lhs activation forces the post_lhs DFB (CBIndex c_3) — a compute-kernel self-loop (the kernel
    # both produces and consumes it), plus a post activation. Exercised on the FPU (add) and SFPU
    # (multiply) compute kernels, interleaved and sharded. The activation packs relu(a) into post_lhs,
    # which on Quasar needs the pack_init retarget (pack_reconfig_data_format is gasket-only there).
    if layout == "interleaved":
        mem_config = ttnn.DRAM_MEMORY_CONFIG
        shape = _INTERLEAVED_SHAPE
    else:
        mem_config = _height_sharded_config(_HEIGHT_SHARD, _HEIGHT_GRID)
        shape = _HEIGHT_SHAPE
    _run(
        device,
        op_name,
        mem_config,
        ttnn.bfloat16,
        shape,
        lhs_act=ttnn.UnaryOpType.RELU,
        post_act=ttnn.UnaryOpType.RELU,
    )


@pytest.mark.parametrize("layout", ["interleaved", "height"])
def test_no_bcast_lhs_silu_swiglu(device, layout):
    # Llama 3.2 1B SwiGLU: a multiply with a fused lhs SiLU -- silu(a) * b -- the exact binary-op
    # activation the model emits (models/tt_transformers/tt/mlp.py: ttnn.mul(w1_out, w3_out,
    # input_tensor_a_activations=[ttnn.UnaryOpType.SILU])). Same lhs-activation self-loop as
    # test_no_bcast_lhs_activation, but with SiLU (an SFPU activation, no packer fast-path) on the SFPU
    # multiply kernel and no post activation (SwiGLU is just silu(a)*b). bf16; interleaved and height-sharded.
    if layout == "interleaved":
        mem_config = ttnn.DRAM_MEMORY_CONFIG
        shape = _INTERLEAVED_SHAPE
    else:
        mem_config = _height_sharded_config(_HEIGHT_SHARD, _HEIGHT_GRID)
        shape = _HEIGHT_SHAPE
    _run(device, "multiply", mem_config, ttnn.bfloat16, shape, lhs_act=ttnn.UnaryOpType.SILU, pcc=0.99)


# Class-(C) coverage headroom (QUASAR_PARITY_GAPS.md §5): activations the WH-baseline matrix
# (QUASAR_LLK_GAPS.md Table 2) marks SUPPORTED on Quasar but that this op previously never exercised —
# only relu and silu were tested. gelu/tanh/square/sigmoid are the model-relevant fusions and each has a
# Quasar ckernel + SfpuType + an #else ARCH_QUASAR compute-API branch, so they should fuse today. This
# validates that matrix claim end-to-end through the op; a failure means either an op-layer gap or an
# over-optimistic matrix cell (both worth knowing).
_LLK_SUPPORTED_ACTS = [
    ttnn.UnaryOpType.GELU,
    ttnn.UnaryOpType.TANH,
    ttnn.UnaryOpType.SQUARE,
    ttnn.UnaryOpType.SIGMOID,
]


@pytest.mark.parametrize("act", _LLK_SUPPORTED_ACTS)
@pytest.mark.parametrize("position", ["post", "lhs"])
@pytest.mark.parametrize("layout", ["interleaved", "height"])
def test_no_bcast_activation_supported(device, act, position, layout):
    # Each activation fused on the SFPU multiply kernel, as a post activation (act(a*b)) and as an lhs
    # (pre) activation (act(a)*b — the post_lhs DFB self-loop path that needs Quasar's pack_init retarget).
    # bf16; interleaved and height-sharded. These activations are accurate in bf16, so _run uses the file's
    # default _PCC[bf16] (0.997) — not the looser 0.99 the divide/silu tests need.
    #
    # gelu is the exception on Quasar: tanh/square/sigmoid compile + pass (sim-certified), but the Quasar
    # gelu LLK bridge (hw/ckernels/quasar/.../llk_sfpu/ckernel_sfpu_gelu.h) fails to JIT-compile —
    # gelu_init() calls _sfpu_load_config32_ *unqualified*, but it lives in namespace ckernel::math
    # (cmath_common.h); the sibling topk bridge calls it qualified, so only gelu is affected. It still
    # runs on Wormhole. TODO: un-skip when the LLK bridge is fixed (tenstorrent/tt-metal#49314). See
    # binary_ng/QUASAR_PARITY_GAPS.md §5 and QUASAR_LLK_GAPS.md (Table 2, gelu row) for the one-line fix.
    if _on_quasar() and act == ttnn.UnaryOpType.GELU:
        pytest.skip("Quasar gelu LLK bridge fails to compile (_sfpu_load_config32_ unqualified in ckernel_sfpu_gelu.h)")
    if layout == "interleaved":
        mem_config = ttnn.DRAM_MEMORY_CONFIG
        shape = _INTERLEAVED_SHAPE
    else:
        mem_config = _height_sharded_config(_HEIGHT_SHARD, _HEIGHT_GRID)
        shape = _HEIGHT_SHAPE
    act_kwarg = {"post_act": act} if position == "post" else {"lhs_act": act}
    _run(device, "multiply", mem_config, ttnn.bfloat16, shape, **act_kwarg)


@pytest.mark.parametrize("dtype_tt", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("layout", ["interleaved", "height"])
def test_no_bcast_sfpu_divide(device, dtype_tt, layout):
    # SFPU compute kernel (double-DST stride) via divide, on interleaved and sharded. Unlike fp32
    # add/sub (no SFPU float-add primitive on Quasar), fp32 divide routes SFPU and IS supported, so
    # both bf16 and fp32 are exercised here.
    if layout == "interleaved":
        mem_config = ttnn.DRAM_MEMORY_CONFIG
        shape = _INTERLEAVED_SHAPE
    else:
        mem_config = _height_sharded_config(_HEIGHT_SHARD, _HEIGHT_GRID)
        shape = _HEIGHT_SHAPE
    _run(device, "divide", mem_config, dtype_tt, shape, pcc=0.99)


# (a_mem, b_mem, out_mem) permutations beyond the uniform III / SSS already covered by
# test_no_bcast_interleaved / test_no_bcast_sharded. The id encodes the layout of each operand:
#   I = DRAM-interleaved (NoC), IL1 = L1-interleaved (NoC), H/B/W = height/block/width sharded. The
#   factory borrows ONLY when all three operands are L1-sharded on one matching grid (the is_native
#   case); any interleaved operand, or a strategy/grid mismatch, takes the all-NoC path where every
#   operand (output included) is read/written via its own sharding-aware TensorAccessor.
_MIXED_CASES = [
    # NOTE: the interleaved-OUTPUT mixed cases (H.I.I, I.H.I, H.H.I, W.I.I) live in _MIXED_INTERLEAVED_OUT_CASES
    # below -- they run on the full worker grid (not the output shard grid), so they need _BIG_SHAPE to reach
    # >=16 compute tiles/core. Every case in THIS list has a SHARDED output (op runs on that small grid).
    # --- Hard tier: output SHARDED with an interleaved input => not all-sharded => all NoC. The output
    #     is NoC-written and the inputs NoC-read via sharding-aware TensorAccessors over a linear
    #     split_work_to_cores partition (nothing borrowed; HAS_SHARDING off). ---
    pytest.param(_I, _MIXED_HEIGHT, _MIXED_HEIGHT, id="I.H.H"),  # a interleaved -> all NoC
    pytest.param(_MIXED_HEIGHT, _I, _MIXED_HEIGHT, id="H.I.H"),  # b interleaved -> all NoC
    pytest.param(_I, _I, _MIXED_HEIGHT, id="I.I.H"),  # both inputs interleaved -> all NoC into sharded out
    pytest.param(_I, _I, _MIXED_BLOCK, id="I.I.B"),  # both inputs interleaved -> all NoC into BLOCK-sharded out
    # --- L1-interleaved inputs (not DRAM): is_native_L1_sharding is true for IL1.IL1.<sharded> (equal
    #     input configs + sharded output), but neither input has a shard spec, so the factory must take
    #     the all-NoC path, NOT borrow -- a borrow-from-native gate would null-deref here. ---
    pytest.param(_IL1, _IL1, _MIXED_HEIGHT, id="IL1.IL1.H"),  # both inputs L1-interleaved -> all NoC
    pytest.param(_IL1, _IL1, _MIXED_BLOCK, id="IL1.IL1.B"),  # both inputs L1-interleaved -> all NoC
    pytest.param(_IL1, _MIXED_HEIGHT, _MIXED_HEIGHT, id="IL1.H.H"),  # a L1-interleaved -> all NoC
    # --- Grid-mismatch: input-a sharded on a DIFFERENT grid than the sharded output => grids do not all
    #     match (not is_native) => nothing borrowed, every operand NoC-read/written. ---
    pytest.param(_MIXED_HEIGHT_ALT, _MIXED_HEIGHT, _MIXED_HEIGHT, id="Halt.H.H"),  # S(grid!=out).S.S
    # --- Width sharding: one uniform all-width case (borrowed) + mixed-width case (all NoC). The
    #     interleaved-output W.I.I case is in _MIXED_INTERLEAVED_OUT_CASES (needs _BIG_SHAPE). ---
    pytest.param(_MIXED_WIDTH, _MIXED_WIDTH, _MIXED_WIDTH, id="W.W.W"),  # uniform width, one grid (all borrowed)
    pytest.param(_I, _MIXED_WIDTH, _MIXED_WIDTH, id="I.W.W"),  # a interleaved -> all NoC into width-out
]


@pytest.mark.parametrize("op_name", ["add", "multiply"])
@pytest.mark.parametrize("a_mem, b_mem, out_mem", _MIXED_CASES)
def test_no_bcast_mixed_layout(device, op_name, a_mem, b_mem, out_mem):
    # Mixed sharded/interleaved (and width-sharded) layouts, one config per operand. bf16 add takes the
    # FPU compute kernel; bf16 multiply takes the SFPU kernel. Grids fit the Quasar 8x4 grid.
    pcc = 0.99 if op_name == "multiply" else None
    _run_mixed(device, op_name, a_mem, b_mem, out_mem, ttnn.bfloat16, pcc=pcc)


# --- Generality cases: all-three-sharded with DIFFERENT strategies, and distinct-grid-per-operand. ---
# With three different strategies (or grids) the operands are not all on one matching grid, so the config
# is not is_native and NOTHING is borrowed -- every operand, the output included, is read/written via its
# sharding-aware TensorAccessor. The id is a.b.out where H/W/B is the strategy; a "@gN" suffix marks an
# operand on a distinct (non-canonical) grid.
_GENERALITY_CASES = [
    # All three sharded, three DIFFERENT strategies, output strategy varies (Block, then Height, then
    # Width) -- every operand NoC-read/written across a strategy boundary:
    pytest.param(_GEN_HEIGHT, _GEN_WIDTH, _GEN_BLOCK, id="H.W.B"),  # out=Block, a=Height, b=Width
    pytest.param(_GEN_WIDTH, _GEN_BLOCK, _GEN_HEIGHT, id="W.B.H"),  # out=Height, a=Width, b=Block
    pytest.param(_GEN_BLOCK, _GEN_HEIGHT, _GEN_WIDTH, id="B.H.W"),  # out=Width, a=Block, b=Height
    # Distinct core grid AND strategy per operand: a=Height@col0, b=Width@row1, out=Block@2x2(2,2). No
    # grids match, so every operand is NoC-read/written across both a grid and a strategy boundary -- the
    # broadest sharded->sharded case.
    pytest.param(_GRID_A_HEIGHT, _GRID_B_WIDTH, _GRID_C_BLOCK, id="H@g0.W@g1.B@g2"),
    # NOTE: the distinct-grid sharded-inputs -> INTERLEAVED-output case (H@g0.W@g1.I) lives in
    # _MIXED_INTERLEAVED_OUT_CASES below -- it runs on the full worker grid and needs _BIG_SHAPE.
]


@pytest.mark.parametrize("op_name", ["add", "multiply"])
@pytest.mark.parametrize("a_mem, b_mem, out_mem", _GENERALITY_CASES)
def test_no_bcast_mixed_strategy_and_grid(device, op_name, a_mem, b_mem, out_mem):
    # Prove the factory has NO restriction on the non-borrowed operands: different shard STRATEGIES per
    # operand (Height/Width/Block) and DIFFERENT core grids per operand both route through the NoC
    # (TensorAccessor) path. bf16 add => FPU compute kernel; bf16 multiply => SFPU kernel. On Quasar this
    # passing is itself proof of v2 routing (the descriptor path cannot run on Quasar); on Wormhole the
    # descriptor path would also pass, so there it checks numerical correctness rather than v2 selection.
    pcc = 0.99 if op_name == "multiply" else None
    _run_mixed(device, op_name, a_mem, b_mem, out_mem, ttnn.bfloat16, pcc=pcc)


# --- Interleaved-OUTPUT mixed/generality cases (pulled out of _MIXED_CASES / _GENERALITY_CASES) --------
# The output is interleaved, so the op does NOT run on a small output shard grid -- it splits over the
# FULL worker grid via split_work_to_cores. To reach >=16 compute tiles/core there each case runs at a big
# shape (last param): _BIG_SHAPE (1280 tiles) for the single-sharded-input cases, _BIG_SHAPE_2SHARD (1024
# tiles) for the two-sharded-input cases so both input shards co-reside in one Quasar L1 bank (see the
# big-shape config block above). Routing is preserved exactly: an interleaved output means not-all-sharded,
# so every operand (any sharded input included) is NoC-read/written -- nothing is borrowed.
_MIXED_INTERLEAVED_OUT_CASES = [
    pytest.param(_BIG_HEIGHT, _I, _I, _BIG_SHAPE, id="H.I.I"),  # S.I.I: one height input -> all NoC into out
    pytest.param(_I, _BIG_HEIGHT, _I, _BIG_SHAPE, id="I.H.I"),  # I.S.I: one height input -> all NoC into out
    pytest.param(_BIG_WIDTH, _I, _I, _BIG_SHAPE, id="W.I.I"),  # one width input -> all NoC into out
    # H.H.I: two height inputs on DISJOINT columns (x=0 and x=1); _BIG_SHAPE_2SHARD keeps each at 256
    # tiles/core so both fit one Quasar L1 bank. all NoC into interleaved out.
    pytest.param(_BIG2_HEIGHT_A, _BIG2_HEIGHT_B, _I, _BIG_SHAPE_2SHARD, id="H.H.I"),  # S.S.I
    # Generality: distinct-grid AND distinct-strategy sharded inputs feeding an interleaved output. Height
    # on column x=4 and width on row y=1 are disjoint (x=4 not in 0..3); _BIG_SHAPE_2SHARD keeps each at 256
    # tiles/core so both fit one bank. Confirms split_work_to_cores NoC-reads arbitrary grids/strategies.
    pytest.param(_BIG2_HEIGHT_G, _BIG2_WIDTH_G, _I, _BIG_SHAPE_2SHARD, id="H@g0.W@g1.I"),
]


@pytest.mark.parametrize("op_name", ["add", "multiply"])
@pytest.mark.parametrize("a_mem, b_mem, out_mem, shape", _MIXED_INTERLEAVED_OUT_CASES)
def test_no_bcast_mixed_interleaved_out(device, op_name, a_mem, b_mem, out_mem, shape):
    # Mixed sharded/interleaved layouts whose OUTPUT is interleaved: the op runs on the full worker grid
    # (split_work_to_cores), so the big per-case shape gives >=16 compute tiles/core. Same all-NoC routing
    # as the interleaved-output entries that used to sit in _MIXED_CASES/_GENERALITY_CASES; only the tile
    # count grows. bf16 add => FPU compute kernel; bf16 multiply => SFPU kernel. Grids fit the Quasar 8x4.
    pcc = 0.99 if op_name == "multiply" else None
    _run_mixed(device, op_name, a_mem, b_mem, out_mem, ttnn.bfloat16, shape=shape, pcc=pcc)
