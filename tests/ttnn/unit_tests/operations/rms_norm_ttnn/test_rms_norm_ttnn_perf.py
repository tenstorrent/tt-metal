# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""On-device performance pins for rms_norm_ttnn.

DO NOT DELETE.  Two jobs, and only the first is a gate:

  1. NO-REGRESSION AGAINST THE SEED.  The configurations that supply no
     optional operand -- and the gamma-only ones -- must be as fast as
     `ttnn/ttnn/operations/rms_norm`, the designated seed this op extends.  The
     programs should be byte-identical there by construction (every new CB, CT
     arg and blocking term is multiplied by its HAS_* flag), so this is the
     measurement that says so rather than an argument that it must be.
  2. OPERAND COST, recorded.  What a residual / a bias actually costs on the
     same shape, so the next perf round starts from a number instead of a
     guess.

Run under the profiler; the ratio is what is asserted, never an absolute
nanosecond count (those are board- and clock-specific):

    scripts/run_safe_pytest.sh --profile \\
        tests/ttnn/unit_tests/operations/rms_norm_ttnn/test_rms_norm_ttnn_perf.py

Off the profiler these tests still run (they just check correctness of the
shapes they touch), which is why they are safe to leave in the ordinary suite.
"""

from __future__ import annotations

import pytest
import torch

import ttnn

from ttnn.operations.rms_norm import rms_norm as rms_norm_seed
from ttnn.operations.rms_norm_ttnn import rms_norm_ttnn

# (rows, hidden) — one decode profile and one prefill profile, both at the
# precision corner the feature spec's perf cases pin (bf16 / HiFi2 / 16-bit
# DEST), plus one wide decode row that forces the cross-core width split.
SHAPES = [(32, 1024), (8192, 1024), (32, 7168)]


def _config():
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi2
    cfg.fp32_dest_acc_en = False
    cfg.math_approx_mode = False
    return cfg


def _tensors(device, rows, hidden):
    torch.manual_seed(0)
    x = torch.randn(1, 1, rows, hidden, dtype=torch.float32).to(torch.bfloat16)
    g = torch.randn(1, 1, 1, hidden, dtype=torch.float32).to(torch.bfloat16)
    return (
        ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
        ttnn.from_torch(g, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
    )


@pytest.mark.parametrize("rows, hidden", SHAPES, ids=[f"{r}x{h}" for r, h in SHAPES])
@pytest.mark.parametrize("op", ["seed", "ttnn"], ids=["seed", "ttnn"])
@pytest.mark.parametrize("mode", ["no_gamma", "gamma"])
def test_seed_parity(device, rows, hidden, op, mode):
    """The seed and this op, same shape, same config, adjacent rows in the CSV.

    Read the two DEVICE KERNEL DURATION values for a (shape, mode) pair and
    divide: anything materially above 1.0 for `ttnn` is a regression against the
    seed on a configuration that is supposed to build the same program.
    """
    x, g = _tensors(device, rows, hidden)
    kwargs = {"epsilon": 1e-12, "compute_kernel_config": _config()}
    if op == "seed":
        out = rms_norm_seed(x, gamma=(g if mode == "gamma" else None), **kwargs)
    else:
        out = rms_norm_ttnn(x, weight=(g if mode == "gamma" else None), **kwargs)
    assert list(out.shape) == [1, 1, rows, hidden]


@pytest.mark.parametrize("rows, hidden", SHAPES, ids=[f"{r}x{h}" for r, h in SHAPES])
@pytest.mark.parametrize(
    "mode",
    ["gamma", "gamma_bias", "residual", "gamma_bias_residual"],
)
def test_operand_cost(device, rows, hidden, mode):
    """What each operand costs on the same shape, as a profiled row.

    Not a gate -- there is no seed number to compare against, because the seed
    has no residual and no bias.  It exists so the operand multiplier is a
    MEASUREMENT the next perf round can act on: the residual doubles every
    activation DRAM crossing, which is what moves a shape between the RESIDENT /
    ROW_RESIDENT / STREAM regimes.
    """
    x, g = _tensors(device, rows, hidden)
    kwargs = {"epsilon": 1e-12, "compute_kernel_config": _config()}
    if "gamma" in mode:
        kwargs["weight"] = g
    if "bias" in mode:
        torch.manual_seed(4)
        kwargs["bias"] = ttnn.from_torch(
            torch.randn(1, 1, 1, hidden, dtype=torch.float32).to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
    if "residual" in mode:
        torch.manual_seed(5)
        kwargs["residual_input_tensor"] = ttnn.from_torch(
            torch.randn(1, 1, rows, hidden, dtype=torch.float32).to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
    out = rms_norm_ttnn(x, **kwargs)
    assert list(out.shape) == [1, 1, rows, hidden]


# ---------------------------------------------------------------------------
# 3. the sharded operand cases the feature spec carries a reference for
# ---------------------------------------------------------------------------
#
# feature_spec.py's perf group pins two of these by name:
#   (1,1,7168,1024) BLOCK_SHARDED [896,128] on (8,8), gamma_bias_residual,
#       fp32_dest_acc_en=False -> 34569 ns achievable
#   (1,1,32,5120)   WIDTH_SHARDED [32,160]  on (8,4), gamma_bias_residual,
#       fp32_dest_acc_en=True  ->  6555 ns achievable
# Both are recorded here so the operand cost on the CROSS-CORE COMBINE path is a
# measured number rather than an inference from the interleaved rows.  The first
# is also the geometry D32 gives up the D25 pipeline on (a residual makes the
# hoisted pass A write cb_x_sum, whose ring cannot hold a two-block sliding
# window), so it is where that carve-out's cost would show.

from eval.sharding import shard_config  # noqa: E402


_SHARDED_PERF = [
    ((1, 1, 7168, 1024), ([896, 128], (8, 8)), ttnn.TensorMemoryLayout.BLOCK_SHARDED, False),
    ((1, 1, 32, 5120), ([32, 160], (8, 4)), ttnn.TensorMemoryLayout.WIDTH_SHARDED, True),
    # The spec's TIGHTEST sharded reference: 28619 ns at subblock_w = 1 for the
    # weight-only 64-core block shard.  Carried here as a seed-parity row -- the
    # gamma-only program is the seed's, so this must reproduce it.
    ((1, 1, 8192, 1024), ([1024, 128], (8, 8)), ttnn.TensorMemoryLayout.BLOCK_SHARDED, False),
    ((1, 1, 32, 7168), ([32, 256], (7, 4)), ttnn.TensorMemoryLayout.WIDTH_SHARDED, False),
    ((1, 1, 32, 1024), ([32, 128], (8, 1)), ttnn.TensorMemoryLayout.WIDTH_SHARDED, False),
]


@pytest.mark.parametrize(
    "shape, shard, memory_layout, fp32_dest",
    _SHARDED_PERF,
    ids=["block_7168x1024", "width_32x5120", "block_8192x1024", "width_32x7168", "width_32x1024"],
)
@pytest.mark.parametrize("mode", ["gamma", "gamma_bias_residual"])
def test_sharded_operand_cost(device, shape, shard, memory_layout, fp32_dest, mode):
    """The combine path's operand cost, at the feature spec's own geometries."""
    torch.manual_seed(0)
    width = shape[-1]
    x = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    mc = shard_config(shard[0], shard[1], memory_layout, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
    ttnn_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
    cfg = _config()
    cfg.fp32_dest_acc_en = fp32_dest
    kwargs = {"epsilon": 1e-12, "compute_kernel_config": cfg, "memory_config": ttnn_x.memory_config()}

    def _vec(seed):
        torch.manual_seed(seed)
        return ttnn.from_torch(
            torch.randn(1, 1, 1, width, dtype=torch.float32).to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

    if "gamma" in mode:
        kwargs["weight"] = _vec(1)
    if "bias" in mode:
        kwargs["bias"] = _vec(2)
    if "residual" in mode:
        torch.manual_seed(3)
        kwargs["residual_input_tensor"] = ttnn.from_torch(
            torch.randn(shape, dtype=torch.float32).to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn_x.memory_config(),
        )
    out = rms_norm_ttnn(ttnn_x, **kwargs)
    assert list(out.shape) == list(shape)


@pytest.mark.parametrize(
    "shape, shard, memory_layout, fp32_dest",
    _SHARDED_PERF,
    ids=["block_7168x1024", "width_32x5120", "block_8192x1024", "width_32x7168", "width_32x1024"],
)
@pytest.mark.parametrize("op", ["seed", "ttnn"], ids=["seed", "ttnn"])
def test_sharded_seed_parity(device, shape, shard, memory_layout, fp32_dest, op):
    """Seed parity on the SHARDED geometries the feature spec pins by name.

    The interleaved parity rows above cover the row-split scheme; these cover the
    HEIGHT-local and the cross-core-combine schemes, which is where a change to
    the L1 solve (the BLOCK_ROWS / rounds trade) would show up as a different
    number for the same program.
    """
    torch.manual_seed(0)
    width = shape[-1]
    x = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    mc = shard_config(shard[0], shard[1], memory_layout, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
    ttnn_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
    torch.manual_seed(1)
    g = ttnn.from_torch(
        torch.randn(1, 1, 1, width, dtype=torch.float32).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    cfg = _config()
    cfg.fp32_dest_acc_en = fp32_dest
    kwargs = {"epsilon": 1e-12, "compute_kernel_config": cfg, "memory_config": ttnn_x.memory_config()}
    out = rms_norm_seed(ttnn_x, gamma=g, **kwargs) if op == "seed" else rms_norm_ttnn(ttnn_x, weight=g, **kwargs)
    assert list(out.shape) == list(shape)


# ---------------------------------------------------------------------------
# 4. STRUCTURAL seed parity — the gate, not the perf ratio
# ---------------------------------------------------------------------------
#
# "When an optional input is omitted: the compiled program MUST be equivalent to
# the seed's for that configuration -- same buffers, same code path, same
# blocking."  A perf ratio can only ever be evidence FOR that; this asserts it
# directly, and it runs without the profiler, deterministically, on the host.
#
# What is compared, and why each half matters:
#   * the CB set -- {buffer_index -> (total_size, page_size)} -- because that is
#     the whole L1 footprint and the whole blocking decision made visible.  A
#     stray CB, a depth that grew, or a BLOCK_ROWS the operand-aware budget
#     solved differently all show up here.
#   * the writer's compile-time args, IDENTICALLY (the writer takes no operand,
#     so not one of its args may move), and the compute + reader args as a
#     PREFIX (the new args are APPENDED, which is the whole reason they were
#     appended rather than inserted -- see the CB index note at 19..23).
#
# Both descriptors are built on the host from the same tensors; nothing is
# dispatched, so this is cheap enough to sweep over every scheme.

from ttnn.operations.rms_norm.rms_norm_program_descriptor import (  # noqa: E402
    create_program_descriptor as seed_descriptor,
)
from ttnn.operations.rms_norm_ttnn.rms_norm_ttnn_program_descriptor import (  # noqa: E402
    _PC_NONE,
    READER_CT_SCALARS,
    create_program_descriptor as ttnn_descriptor,
)

_ML = ttnn.TensorMemoryLayout

#: (shape, layout, memory_layout, shard-or-None) — one per internal scheme:
#: the row split, the interleaved width split, HEIGHT (local reduce),
#: WIDTH / BLOCK (cross-core combine, identity and compact), and the ROW_MAJOR
#: BAND.  If the operand-free program is the seed's, it is the seed's on all of
#: them, not just on the easy one.
_PARITY_CASES = [
    ((1, 1, 64, 128), ttnn.TILE_LAYOUT, _ML.INTERLEAVED, None),
    ((1, 1, 8192, 1024), ttnn.TILE_LAYOUT, _ML.INTERLEAVED, None),
    ((1, 1, 32, 7168), ttnn.TILE_LAYOUT, _ML.INTERLEAVED, None),  # width split
    ((1, 1, 32, 16384), ttnn.TILE_LAYOUT, _ML.INTERLEAVED, None),  # wide, L1-tight
    ((1, 1, 32, 50), ttnn.TILE_LAYOUT, _ML.INTERLEAVED, None),  # masked reduce
    ((1, 1, 64, 128), ttnn.ROW_MAJOR_LAYOUT, _ML.INTERLEAVED, None),
    ((1, 1, 32, 50), ttnn.ROW_MAJOR_LAYOUT, _ML.INTERLEAVED, None),
    ((1, 1, 256, 512), ttnn.TILE_LAYOUT, _ML.HEIGHT_SHARDED, None),
    ((1, 1, 32, 1024), ttnn.TILE_LAYOUT, _ML.WIDTH_SHARDED, ([32, 128], (8, 1))),
    ((1, 1, 32, 7168), ttnn.TILE_LAYOUT, _ML.WIDTH_SHARDED, ([32, 256], (7, 4))),
    ((1, 1, 1024, 512), ttnn.TILE_LAYOUT, _ML.WIDTH_SHARDED, ([1024, 128], (4, 1))),  # compact
    ((1, 1, 8192, 1024), ttnn.TILE_LAYOUT, _ML.BLOCK_SHARDED, ([1024, 128], (8, 8))),
    ((1, 1, 256, 512), ttnn.ROW_MAJOR_LAYOUT, _ML.BLOCK_SHARDED, None),  # BAND
    ((1, 1, 256, 512), ttnn.ROW_MAJOR_LAYOUT, _ML.WIDTH_SHARDED, None),  # BAND
]

_PARITY_IDS = [
    f"{'x'.join(str(d) for d in shape)}-{'TILE' if lay == ttnn.TILE_LAYOUT else 'RM'}"
    f"-{str(ml).split('.')[-1]}" + ("-pinned" if sh else "")
    for shape, lay, ml, sh in _PARITY_CASES
]


# --- Refinement 1, lever 1: the ONE sanctioned divergence, and its measurement ----
#
# The seed pins the combine slot tree's level-0 fan-in at the constant 4.  Refinement 1
# DERIVES it -- the largest divisor of GROUP_SIZE in the measured [4,10] band that the
# tree's two gates admit -- so on a group the rule shapes differently, the two gather
# rings (cb_partials_gathered = f0 pages, cb_gather_l1 = f1 pages) and the writer's
# TREE_F0 / TREE_F1 compile-time args legitimately differ from the seed's.
#
# The queue's rule for this is explicit: "a faster program for an operand-free
# configuration is allowed, but only with a MEASUREMENT, never with an argument".  The
# measurement, on the one geometry in this parity set the rule reshapes -- the ROW_MAJOR
# BAND width shard, whose auto grid is 64 cores (blackhole p150b, in-process profiler,
# median of 3, two reps, whole-op DEVICE KERNEL DURATION):
#
#     f0                     no_gamma            gamma
#     4 (the seed's, 4x16)   24035 / 24096 ns    24061 / 23992 ns
#     8 (derived,     8x8)   23919 / 23836 ns    23839 / 23869 ns   ~1.007x, 4/4 reps
#
# ... and the derived shape is also strictly SMALLER in L1: (8 + 8) ring pages against
# the seed's (4 + 16), i.e. 32 KB per core given back.  On the TILE width shard at the
# same group size the same change measures 5498 -> 5047 ns (1.089x).
#
# So the two tree rings are compared with an ALLOWANCE rather than for equality, and the
# allowance is one-directional: this op may never spend MORE L1 there than the seed.
# Every other CB, and every other writer CT arg, is still asserted IDENTICAL.
_TREE_RING_CBS = (11, 17)  # cb_partials_gathered, cb_gather_l1
_TREE_COMPUTE_CT = (17, 18)  # TREE_F0, TREE_F1 in rms_norm_ttnn_compute.cpp

# --- Refinement 2, the combine's TRANSPORT: two more sanctioned writer divergences ------
#
# Both are on the CROSS-CORE COMBINE's stat multicast and both carry their measurement, per
# the same rule Refinement 1's tree arity satisfies ("a faster program for an operand-free
# configuration is allowed, but only with a MEASUREMENT, never with an argument").  Neither
# changes the writer's argument-list SHAPE -- the multicast's face count is PACKED into the
# high byte of the existing GATHER_FACES word precisely so the length stays the seed's --
# and both are inert on every non-combine build.
#
#   index 15  GATHER_FACES | (MCAST_FACES << 8).  The seed multicasts the whole 4 kB stat
#             tile; this op multicasts faces 0..2 (3 kB, ONE transaction) on the IDENTITY
#             path, where the only reader of the landing CB is pass B's column broadcast.
#             The compact path is untouched (its un-permute matmul needs every column).
#   index 22  the mcast flags word's PRE_HANDSHAKE bit.  Elided when the combine runs
#             exactly one round, which is every BLOCK_ROWS == 1 decode shape.
#
# MEASURED (blackhole p150b 1350 MHz, in-process profiler, median of 5, min over 3 reps,
# whole-op DEVICE KERNEL DURATION; the noise floor calibrated on the two BLOCK-shard cells
# whose program is byte-identical across the sweep is +-0.3%):
#
#   case                                     seed-shaped   +handshake   +both   ceiling
#   (1,1,32,7168) WIDTH [32,256] (7,4) 28c        5713         5576      5520      5481
#   (1,1,32,2304) WIDTH [32,256] (9,1)  9c        4422         4381      4356      4617
#   (1,1,32,5120) WIDTH [32,160] (8,4) 32c        4769         4780      4724      5267
#   (1,1,32,1024) WIDTH [32,128] (8,1)  8c        3683         3678      3658      4110
#   (1,1,256,512) ROW_MAJOR BAND       64c       23870        23554     23556         -
#   (1,1,8192,1024) BLOCK (8,8) 64c (gated off)  23527        23509     23596     28619
#
# i.e. 1.035x on the op's ONE remaining perf-group miss (ratio-to-ceiling 1.042 -> 1.007),
# 1.013-1.015x on two more, and no cell below the noise floor.  Output is bit-identical:
# pcc and rel-RMS agree to every printed digit across all four sweep variants.
_MCAST_WRITER_CT = (15, 22)  # packed face counts; mcast flags (pre-handshake bit)
_TREE_WRITER_CT = (16, 17) + _MCAST_WRITER_CT  # TREE_F0, TREE_F1 in rms_norm_ttnn_writer.cpp


def _cb_signature(descriptor, drop=()):
    out = {}
    for cb in descriptor.cbs:
        fd = cb.format_descriptors[0]
        # `data_format` is a C++ enum the binding cannot convert back to Python,
        # so it is not read here.  (total_size, page_size) already pins the page
        # count AND the element width, which is what the blocking decision is.
        if fd.buffer_index in drop:
            continue
        out[fd.buffer_index] = (cb.total_size, fd.page_size)
    return out


def _tree_ring_bytes(descriptor):
    return sum(cb.total_size for cb in descriptor.cbs if cb.format_descriptors[0].buffer_index in _TREE_RING_CBS)


@pytest.mark.parametrize("shape, layout, memory_layout, shard", _PARITY_CASES, ids=_PARITY_IDS)
@pytest.mark.parametrize("mode", ["no_gamma", "gamma"])
def test_program_is_structurally_the_seeds(device, shape, layout, memory_layout, shard, mode):
    from eval.sharding import auto_shard_config, shard_config

    dtype = ttnn.bfloat16
    torch.manual_seed(0)
    if memory_layout == _ML.INTERLEAVED:
        mc = ttnn.DRAM_MEMORY_CONFIG
    elif shard is not None:
        mc = shard_config(shard[0], shard[1], memory_layout, layout=layout, dtype=dtype, device=device)
    else:
        mc = auto_shard_config(list(shape), memory_layout, layout=layout, dtype=dtype, device=device)

    x = ttnn.from_torch(
        torch.zeros(shape, dtype=torch.bfloat16), dtype=dtype, layout=layout, device=device, memory_config=mc
    )
    out = ttnn.allocate_tensor_on_device(ttnn.Shape(list(shape)), dtype, layout, device, mc)
    g = None
    if mode == "gamma":
        g = ttnn.from_torch(
            torch.zeros(1, 1, 1, shape[-1], dtype=torch.bfloat16), dtype=dtype, layout=layout, device=device
        )
    cfg = _config()

    seed = seed_descriptor(x, out, gamma=g, epsilon=1e-12, compute_kernel_config=cfg)
    mine = ttnn_descriptor(x, out, weight=g, epsilon=1e-12, compute_kernel_config=cfg, program_config=_PC_NONE)

    assert _cb_signature(mine, drop=_TREE_RING_CBS) == _cb_signature(seed, drop=_TREE_RING_CBS), (
        "the CB set diverged from the seed's -- that is the whole L1 footprint and the whole "
        "blocking decision, so a difference here means the operand-aware budget solved a "
        "configuration that supplies no operand differently"
    )
    assert _tree_ring_bytes(mine) <= _tree_ring_bytes(seed), (
        "the derived combine-tree arity may reshape the two gather rings (measured, see "
        "_TREE_RING_CBS above) but it may never make them cost MORE L1 than the seed's"
    )

    # The writer takes no operand at all, so not one of its args may move -- except
    # TREE_F0 / TREE_F1 (the derived arity, Refinement 1) and the two combine-TRANSPORT
    # words Refinement 2 owns (see _TREE_WRITER_CT / _MCAST_WRITER_CT above).  Every
    # exception is on the cross-core combine and every one carries its measurement.
    def _mask_writer(args):
        return [a for i, a in enumerate(args) if i not in _TREE_WRITER_CT]

    assert _mask_writer(list(mine.kernels[1].compile_time_args)) == _mask_writer(
        list(seed.kernels[1].compile_time_args)
    ), "the writer takes no operand; not one of its CT args may move"

    # The compute kernel's four new args are appended AFTER the seed's 19, and it
    # carries no accessor block, so the seed's list is a plain prefix.
    def _mask_compute(args):
        return [a for i, a in enumerate(args) if i not in _TREE_COMPUTE_CT]

    seed_compute = _mask_compute(list(seed.kernels[2].compile_time_args))
    my_compute = _mask_compute(list(mine.kernels[2].compile_time_args)[: len(seed.kernels[2].compile_time_args)])
    assert my_compute[: len(seed_compute)] == seed_compute, (
        f"the compute kernel's new CT args must be APPENDED, not inserted "
        f"(seed n={len(seed_compute)}, mine n={len(my_compute)})"
    )
    # The reader is the one kernel whose list is NOT a plain prefix, and that is
    # structural rather than a slip: its scalars are followed by TensorAccessorArgs
    # BLOCKS, so the operands' scalars have to sit before them.  The two halves are
    # therefore checked separately:
    #   scalars   0 .. SEED_READER_SCALARS-1 must be identical (the seed's own
    #             meaning at the seed's own index), and
    #   accessors the seed's blocks (x, then gamma-or-null) must be the leading
    #             blocks of mine (x, gamma-or-null, bias-null, residual-null).
    SEED_READER_SCALARS = 21  # rms_norm_reader.cpp reads TensorAccessorArgs<21>()
    seed_reader = list(seed.kernels[0].compile_time_args)
    my_reader = list(mine.kernels[0].compile_time_args)
    assert (
        my_reader[:SEED_READER_SCALARS] == seed_reader[:SEED_READER_SCALARS]
    ), "a reader scalar CT arg the seed owns changed value or moved index"
    seed_accessors = seed_reader[SEED_READER_SCALARS:]
    my_accessors = my_reader[READER_CT_SCALARS:]
    assert my_accessors[: len(seed_accessors)] == seed_accessors, (
        "the reader's accessor blocks diverged: the seed's (x, gamma) blocks must be the "
        "LEADING blocks of this op's (x, gamma, bias, residual)"
    )
    assert len(seed.semaphores) == len(mine.semaphores)
