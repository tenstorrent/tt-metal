# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Standalone block-based MAX-pool-along-ROW LLK test (experimental).

Exercises the experimental ``reduce_block_max_row`` LLKs
(``experimental/llk_{math,unpack_AB}_reduce_custom{,_runtime}.h``), which today
are covered only inside the fuser and the ``sdpa_reinits`` chain. Semantics:

    out[row] = max over the full block width (``block_ct_dim`` tiles, 32 cols
               each) of operand A.

The row-max lands in **column [0]** of the output tile; per the op's contract
the packer's REDUCE_ROW mask leaves the other columns unspecified, so the test
validates column [0] alone (mirroring ``ReduceBlockMaxRowGolden``).

The scaler B is a single face of 1.0 in F0 (the op's contract), so the pool is a
pure MAX and the scaler is a no-op multiplier.

Coverage (Blackhole + Wormhole B0):
  * compile-time path, ``block_ct_dim`` sweep, bf16 and fp32-dest;
  * runtime path (dynamic ``block_ct_dim``);
  * reinit_short / reinit_minimal / reinit (addrmod-only) re-arm after a clobbering op,
    guarding the reconfig-escape path. Compile-time reinit_short/minimal are
    Blackhole-only lib fns; runtime reinit_short and runtime reinit run on both arches,
    runtime reinit_minimal is Blackhole-only.
  * trigger handshake (respect_trigger, + overlap_first_half on the runtime family):
    the unpack splits the block reduce into two half-width MOP runs gated by HW
    semaphores; the PACK thread plays the producer. Both arches. Validates split-MOP
    correctness + semaphore balance (not the SDPA live-data overlap; see the C++ note).
  * runtime ``block_ct_dim`` limit: 127 reduces, 0 and 128 assert at init on both arches, and a
    reinit_short re-arm with 128 asserts in the MOP reprogram; a trigger split with an odd
    width asserts at init.
  * operand-shape guard: 16x16, 32x16 and 8x32 operands are rejected at init (compile-time
    and runtime families) and by the reinit_short re-arm.
"""

import os

import pytest
import torch
from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.device import LLKAssertException
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import ReduceBlockMaxRowGolden, get_golden_generator
from helpers.llk_params import DestAccumulation
from helpers.param_config import parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    CLOBBER_OP,
    OVERLAP_FIRST_HALF,
    REDUCE_BLOCK_CT_DIM,
    REDUCE_MATH_SKIP_EXECUTE,
    REDUCE_OPERAND_SHAPE,
    REDUCE_PACK_SKIP_EXECUTE,
    REDUCE_REINIT_BLOCK_CT_DIM,
    REDUCE_REINIT_OPERAND_SHAPE,
    REDUCE_UNPACK_SKIP_EXECUTE,
    REINIT_MODE,
    RESPECT_TRIGGER,
    USE_RUNTIME,
)
from helpers.tile_shape import TileShape, construct_tile_shape
from helpers.tilize_untilize import tilize, untilize
from helpers.utils import tolerances

# 32x32 bf16 tile.
TILE_DIM = 32
ELEMENTS_PER_TILE = TILE_DIM * TILE_DIM

FULL_TILE_SHAPE = construct_tile_shape((32, 32))

# Widest block the runtime reduce accepts on every arch (its compile-time sibling static_asserts
# 1 <= block_ct_dim <= 127). block_ct_dim goes straight into the MOP outer-loop count, which keeps 7 bits
# on Wormhole, and the block is walked by the unpacker's tile counter, whose input address keeps 8
# bits on both arches: 128 silently reduces nothing on Wormhole, 257 aliases tile 0 on Blackhole.
MAX_BLOCK_CT_DIM = 127

# Inputs are always bf16 (the op's contract); only the DEST/output precision varies.
FORMATS = [
    InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b),
    InputOutputFormat(DataFormat.Float16_b, DataFormat.Float32),
]

# REINIT_MODE C++ selector: 0 = plain init, 1 = reinit_short, 2 = reinit_minimal,
# 3 = reinit (addrmod-only, all four; runtime only).
_REINIT_DEFINE = {"none": 0, "short": 1, "minimal": 2, "full": 3}

# CLOBBER_OP C++ selector: op run between init and reinit to overwrite the reduce config,
# so the reinit path must actually restore it (reconfig-escape guard).
#   0 = none.
#   1 = eltwise binary (ELWADD) init — clobbers ALL addrmods (incl. ADDR_MOD_3) + the MOP;
#       paired with reinit_short (which reprograms the MOP and all addrmods).
#   2 = scramble ADDR_MOD_1/2/6 only, preserving ADDR_MOD_3 + MOP; paired with reinit_minimal
#       (which restores only 1/2/6 and relies on 3/MOP being intact, as matmul/sub_exp do).
#   3 = scramble ADDR_MOD_1/2/3/6 (every addrmod the reduce consumes), preserving the MOP +
#       replay buffer; paired with the addrmod-only reinit ("full"), which reprograms exactly
#       those four and nothing else.
_CLOBBER_DEFINE = {"none": 0, "eltwise": 1, "minimal_safe": 2, "addrmod_all": 3}

# The clobber must match the reinit's restore contract, or the reduce hangs.
_CLOBBER_FOR_REINIT = {
    "short": "eltwise",
    "minimal": "minimal_safe",
    "full": "addrmod_all",
}


def _dest_acc(output_format):
    return (
        DestAccumulation.Yes
        if output_format == DataFormat.Float32
        else DestAccumulation.No
    )


def _make_reduce_block_max_config(
    formats,
    block_ct_dim,
    use_runtime=False,
    reinit="none",
    clobber="none",
    respect_trigger=False,
    overlap_first_half=False,
    expected_nondeterministic=False,
    operand_shape=FULL_TILE_SHAPE,
    staged_ct_dim=None,
    reinit_block_ct_dim=None,
    reinit_operand_shape=None,
    unpack_skip_execute=False,
    math_skip_execute=False,
    pack_skip_execute=False,
):
    """Build one reduce_block_max_row variant plus its golden (row-major, per-row max in
    column [0]).

    ``operand_shape`` is the ``TileShape`` the unpack/math inits are told about; the
    staged operand is always 32x32 tiles, so a shape the op does not implement reaches
    the inits with the same L1 image the full-tile case uses.

    ``staged_ct_dim`` is how many tiles are staged (and the golden's width) when it
    differs from the ``block_ct_dim`` the kernel is built with. Only the guard tests use
    it: a block the inits must reject never reads its operand, so staging it in full
    would only exhaust L1.

    ``reinit_block_ct_dim`` / ``reinit_operand_shape`` are what the reinit_short re-arm is
    told (default: the init values); the reinit-guard tests hand it a width or shape the
    op rejects, and set ``unpack_skip_execute`` so the unpack delivers no tiles: their
    math stops on the re-arm's assert, and tiles left in the source banks survive the
    per-test TRISC soft reset and stall the next kernel. ``math_skip_execute`` is the
    mirror image for the trigger odd-width guard: its unpack stops at init, and a math MOP
    left waiting for SrcA is a stalled Tensix instruction the soft reset does not recover.
    ``pack_skip_execute`` is set by every guard test whose math stops at init: that math
    never signals math-done, and a packer left on the MATH_PACK wait is a stalled Tensix
    instruction that makes the next kernel on the core pack zeros.
    """
    dest_acc = _dest_acc(formats.output_format)
    staged_ct_dim = block_ct_dim if staged_ct_dim is None else staged_ct_dim

    # Build operand A row-major (32 rows x staged_ct_dim*32 cols), so the canonical
    # ReduceBlockMaxRowGolden (which reshapes its operand to `dimensions` row-major)
    # is unambiguous. A ~ U[0, 1] keeps every row-max well inside bf16's range.
    dimensions = [TILE_DIM, staged_ct_dim * TILE_DIM]
    src_A_rowmajor = (
        torch.empty(TILE_DIM * staged_ct_dim * TILE_DIM)
        .uniform_(0.0, 1.0)
        .to(torch.bfloat16)
    )

    # Canonical golden: the established fuser oracle for this op. Returns a row-major
    # tensor with the per-row max in column [0], every other column 0.
    generate_golden = get_golden_generator(ReduceBlockMaxRowGolden)
    golden_flat = generate_golden(
        src_A_rowmajor.clone(), staged_ct_dim, formats.output_format, dimensions
    )

    # Device operand is tilized per 32x32 tile (the layout the C++ reads from L1).
    src_A = torch.cat(
        [
            tilize(
                src_A_rowmajor.reshape(TILE_DIM, staged_ct_dim * TILE_DIM)[
                    :, t * TILE_DIM : (t + 1) * TILE_DIM
                ].reshape(-1),
                formats.input_format,
            )
            for t in range(staged_ct_dim)
        ]
    )
    # Scaler B: a single tile of 1.0 (F0 holds the scaler; the op multiplies by it).
    src_B = torch.ones(ELEMENTS_PER_TILE, dtype=torch.bfloat16)

    configuration = TestConfig(
        "sources/reduce_block_max_test.cpp",
        formats,
        templates=[
            REDUCE_BLOCK_CT_DIM(block_ct_dim),
            USE_RUNTIME(use_runtime),
            REINIT_MODE(_REINIT_DEFINE[reinit]),
            CLOBBER_OP(_CLOBBER_DEFINE[clobber]),
            RESPECT_TRIGGER(respect_trigger),
            OVERLAP_FIRST_HALF(overlap_first_half),
            REDUCE_OPERAND_SHAPE(operand_shape),
            REDUCE_REINIT_BLOCK_CT_DIM(
                block_ct_dim if reinit_block_ct_dim is None else reinit_block_ct_dim
            ),
            REDUCE_REINIT_OPERAND_SHAPE(
                operand_shape if reinit_operand_shape is None else reinit_operand_shape
            ),
            REDUCE_UNPACK_SKIP_EXECUTE(unpack_skip_execute),
            REDUCE_MATH_SKIP_EXECUTE(math_skip_execute),
            REDUCE_PACK_SKIP_EXECUTE(pack_skip_execute),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=staged_ct_dim,
            tile_count_B=1,
            tile_count_res=1,
            sfpu=False,
        ),
        dest_acc=dest_acc,
        expected_nondeterministic=expected_nondeterministic,
    )

    golden_rowmajor = golden_flat.reshape(TILE_DIM, staged_ct_dim * TILE_DIM)
    return configuration, golden_rowmajor


def _run_reduce_block_max(
    formats,
    block_ct_dim,
    use_runtime=False,
    reinit="none",
    clobber="none",
    respect_trigger=False,
    overlap_first_half=False,
    expect_mismatch=False,
    expected_nondeterministic=False,
):
    """Run one reduce_block_max_row variant and validate column [0] against golden.

    ``expect_mismatch`` inverts the verdict: the run must diverge from golden. Used by
    the negative controls, which pin that a clobber left un-repaired really does corrupt
    the reduce (so the reinit cases are not passing for free).

    ``expected_nondeterministic`` exempts the variant from the --bit-exact-runs
    determinism check: a run with a deliberately corrupt HW config leaves DEST
    addressing undefined, so its output is not bit-reproducible by contract (and is
    arch-dependent: stable on Wormhole, a coin-flip on Blackhole). The single-run
    functional check above still applies.
    """
    configuration, golden_rowmajor = _make_reduce_block_max_config(
        formats,
        block_ct_dim,
        use_runtime=use_runtime,
        reinit=reinit,
        clobber=clobber,
        respect_trigger=respect_trigger,
        overlap_first_half=overlap_first_half,
        expected_nondeterministic=expected_nondeterministic,
    )

    res_from_L1 = configuration.run().result

    assert (
        len(res_from_L1) == ELEMENTS_PER_TILE
    ), f"Expected one {ELEMENTS_PER_TILE}-element output tile, got {len(res_from_L1)}"

    # Untilize the device output tile to row-major so both sides share layout.
    res = untilize(
        torch.tensor(res_from_L1, dtype=torch.float32), formats.output_format
    ).reshape(TILE_DIM, TILE_DIM)
    tol = tolerances[formats.output_format]

    # Only column [0] of each physical row is defined (the row-max); the packer's
    # REDUCE_ROW mask leaves every other lane unspecified, so validate column [0].
    mismatches = [
        (pr, float(res[pr, 0]), float(golden_rowmajor[pr, 0]))
        for pr in range(TILE_DIM)
        if abs(float(res[pr, 0]) - float(golden_rowmajor[pr, 0]))
        > tol.atol + tol.rtol * abs(float(golden_rowmajor[pr, 0]))
    ]

    if expect_mismatch:
        assert mismatches, (
            "expected the un-repaired clobber to corrupt the reduce, but every row "
            f"matched golden (block_ct_dim={block_ct_dim}, reinit={reinit}, "
            f"clobber={clobber}); this negative test is no longer meaningful"
        )
        return

    if mismatches:
        pr, d, g = mismatches[0]
        raise AssertionError(
            f"reduce_block_max_row mismatch at row {pr}: device={d} golden={g} "
            f"(block_ct_dim={block_ct_dim}, {len(mismatches)} row(s) off)"
        )


# Math fidelity is intentionally not swept: reduce_block_max_row is a pure MAX pool
# (scaler fixed at 1.0) and the reduce LLKs take no fidelity template, so HiFi2/HiFi4
# would produce identical kernels.


@parametrize(
    formats=FORMATS,
    block_ct_dim=[1, 2, 3, 4, 8],
)
def test_reduce_block_max(formats, block_ct_dim):
    """Compile-time block_ct_dim sweep, bf16 and fp32-dest."""
    _run_reduce_block_max(formats, block_ct_dim)


@parametrize(
    formats=FORMATS,
    block_ct_dim=[1, 2, 4, 8],
)
def test_reduce_block_max_runtime(formats, block_ct_dim):
    """Runtime (dynamic block_ct_dim) path."""
    _run_reduce_block_max(formats, block_ct_dim, use_runtime=True)


@parametrize(
    formats=FORMATS,
    block_ct_dim=[2, 4],
    reinit=["short", "minimal"],
)
def test_reduce_block_max_reinit(formats, block_ct_dim, reinit):
    """Reinit / reprogram after a clobbering op (reconfig-escape guard).

    A clobber runs between init and reinit to overwrite the reduce config, then the
    reinit must restore it for the reduce to match golden. The clobber matches each
    reinit's restore contract (``_CLOBBER_FOR_REINIT``):
      * reinit_short pairs with an eltwise binary op (reprograms all addrmods + MOP,
        as matmul / sub_exp do in the SDPA inner loop);
      * reinit_minimal pairs with ``minimal_safe`` (an ADDR_MOD_1/2/6-only scramble
        leaving ADDR_MOD_3 + MOP intact — the narrow escape it expects).

    Compile-time short/minimal reinit lib fns are Blackhole-only; skip on WH.
    """
    if get_chip_architecture() != ChipArchitecture.BLACKHOLE:
        pytest.skip(
            "compile-time reduce_block_max_row reinit is a Blackhole-only lib path"
        )
    _run_reduce_block_max(
        formats, block_ct_dim, reinit=reinit, clobber=_CLOBBER_FOR_REINIT[reinit]
    )


@parametrize(
    formats=FORMATS,
    block_ct_dim=[2, 4],
    reinit=["short", "minimal", "full"],
)
def test_reduce_block_max_reinit_runtime(formats, block_ct_dim, reinit):
    """Runtime reinit paths: reinit_short_runtime and reinit_runtime on both arches;
    reinit_minimal_runtime is a Blackhole-only lib fn.

    full is _llk_math_reduce_block_max_row_reinit_runtime_, the addrmod-only
    re-arm that reprograms all four addrmods the reduce consumes (ADDR_MOD_1/2/3/6) and
    nothing else. Its clobber (addrmod_all) scrambles exactly those four and leaves
    the MOP intact, so dropping any one of them from the lib fn breaks this test.
    """
    if reinit == "minimal" and get_chip_architecture() != ChipArchitecture.BLACKHOLE:
        pytest.skip(
            "runtime reduce_block_max_row reinit_minimal is a Blackhole-only lib path"
        )
    _run_reduce_block_max(
        formats,
        block_ct_dim,
        use_runtime=True,
        reinit=reinit,
        clobber=_CLOBBER_FOR_REINIT[reinit],
    )


@parametrize(
    block_ct_dim=[2, 4],
    reinit=["none", "minimal"],
)
def test_reduce_block_max_reinit_negative_control(block_ct_dim, reinit):
    """Negative control for the `addrmod_all` clobber, so the reinit cases above
    cannot silently go vacuous.

    The clobber zeroes ADDR_MOD_1/2/3/6, which only reinit_runtime ("full")
    reprograms in full. This test pins that the escape is real:
      * `none`: clobber with no re-arm at all must diverge from golden;
      * `minimal`: reinit_minimal_runtime restores 1/2/6 but deliberately not
        ADDR_MOD_3, so it must also diverge. If it ever passed, the "full" case would
        no longer prove that reinit_runtime restores ADDR_MOD_3.

    bf16 only: the escape is a DEST addressing failure, independent of DEST precision.
    """
    if reinit == "minimal" and get_chip_architecture() != ChipArchitecture.BLACKHOLE:
        pytest.skip(
            "runtime reduce_block_max_row reinit_minimal is a Blackhole-only lib path"
        )
    _run_reduce_block_max(
        FORMATS[0],
        block_ct_dim,
        use_runtime=True,
        reinit=reinit,
        clobber="addrmod_all",
        expect_mismatch=True,
        expected_nondeterministic=True,
    )


# block_ct_dim must be even: respect_trigger splits the unpack MOP into two half-block
# runs (outerloop = block_ct_dim / 2), so an odd width would drop a tile.
@parametrize(
    formats=FORMATS,
    block_ct_dim=[2, 4, 8],
)
def test_reduce_block_max_trigger(formats, block_ct_dim):
    """Compile-time trigger handshake: the unpack splits the block reduce into two
    half-width MOP runs separated by a HW semaphore wait on FPU_SFPU, with the PACK
    thread posting the data-ready token. Validates split-MOP Z-counter continuity and
    semaphore post/wait/get balance; the reduce must still match golden. Both arches."""
    _run_reduce_block_max(formats, block_ct_dim, respect_trigger=True)


@parametrize(
    formats=FORMATS,
    block_ct_dim=[2, 4, 8],
    overlap_first_half=[False, True],
)
def test_reduce_block_max_trigger_runtime(formats, block_ct_dim, overlap_first_half):
    """Runtime trigger handshake. overlap_first_half switches run()#1's gate from
    FPU_SFPU to the early UNPACK_MATH_DONE token (so the first-half reduce overlaps
    the second-half pack); the PACK thread posts both tokens. Both arches."""
    _run_reduce_block_max(
        formats,
        block_ct_dim,
        use_runtime=True,
        respect_trigger=True,
        overlap_first_half=overlap_first_half,
    )


def _expect_llk_assert_on(
    configuration, golden_rowmajor, what, triscs=("TRISC0", "TRISC1")
):
    """The reduce LLKs must LLK_ASSERT on ``what`` on every thread in ``triscs``; the harness
    surfaces the ebreak as LLKAssertException. If the guard is silent the kernel runs anyway,
    so report what it packed into column [0] against the staged operand's row maxima: that is
    the wrong result the guard exists to prevent."""
    if os.environ.get("TT_LLK_DISABLE_ASSERTS") == "1":
        pytest.skip("LLK_ASSERT is compiled out (TT_LLK_DISABLE_ASSERTS=1)")
    try:
        res_from_L1 = configuration.run().result
    except LLKAssertException as hit:
        # Every thread that programs the geometry / loop count carries the guard: by default the
        # unpack init (TRISC0) and the math MOP config (TRISC1) must each have stopped on it.
        stacks = str(hit)
        missing = [t for t in triscs if f"{t} STACK TRACE" not in stacks]
        assert (
            not missing
        ), f"{what}: the LLK assert fired, but not on {missing}:{stacks}"
        return
    res = untilize(
        torch.tensor(res_from_L1, dtype=torch.float32), DataFormat.Float16_b
    ).reshape(TILE_DIM, TILE_DIM)
    col0 = [float(res[pr, 0]) for pr in range(TILE_DIM)]
    golden_col0 = [float(golden_rowmajor[pr, 0]) for pr in range(TILE_DIM)]
    rows_off = sum(1 for d, g in zip(col0, golden_col0) if d != g)
    pytest.fail(
        f"reduce_block_max_row init accepted {what}: no LLK assert fired and the kernel ran; "
        f"column [0] differs from the staged operand's row max in {rows_off}/{TILE_DIM} rows "
        f"(device rows 0-7 = {col0[:8]}, staged row max = {golden_col0[:8]})"
    )


@parametrize(block_ct_dim=[0, MAX_BLOCK_CT_DIM, MAX_BLOCK_CT_DIM + 1])
def test_reduce_block_max_runtime_block_ct_dim_limit(block_ct_dim):
    """The runtime reduce programs block_ct_dim straight into the MOP outer-loop count
    (7 bits on Wormhole, so 128 unpacks and pools nothing and the packed row max is stale
    DEST; 0 programs no loop at all, with the same stale result) and walks the block with
    the unpacker's 8-bit tile counter (so 257 aliases tile 0 on Blackhole); the runtime
    inits must hold the same 1 <= block_ct_dim <= 127 contract their compile-time siblings
    static_assert, on both arches. The widest legal block must reduce correctly; 0 and one
    past the limit must LLK_ASSERT at init instead of programming the count. A rejected
    block never reads its operand, so a single tile is staged."""
    (block_ct_dim,) = block_ct_dim
    if 1 <= block_ct_dim <= MAX_BLOCK_CT_DIM:
        _run_reduce_block_max(FORMATS[0], block_ct_dim, use_runtime=True)
        return
    configuration, golden_rowmajor = _make_reduce_block_max_config(
        FORMATS[0],
        block_ct_dim,
        use_runtime=True,
        staged_ct_dim=1,
        pack_skip_execute=True,
    )
    _expect_llk_assert_on(
        configuration,
        golden_rowmajor,
        f"block_ct_dim={block_ct_dim} outside the 1..{MAX_BLOCK_CT_DIM} range",
    )


# Operand geometries the op does not implement: it pools two full 16-row column faces per
# face-row (two GMPOOLs with a 16-row SrcA step) and strides whole 32x32 / 16x32 tiles, so one
# face column (16x16, 32x16) or a short face (8x32) pools SrcA rows the unpacker never wrote.
_UNSUPPORTED_OPERAND_SHAPES = [
    TileShape(16, 16, 1, 1),  # 16x16
    TileShape(16, 16, 2, 1),  # 32x16
    TileShape(8, 16, 1, 2),  # 8x32
]


@parametrize(use_runtime=[False, True], operand_shape=_UNSUPPORTED_OPERAND_SHAPES)
def test_reduce_block_max_shape_guard(use_runtime, operand_shape):
    """The unpack and math inits must LLK_ASSERT on a tile geometry the op does not
    implement (validate_tensor_shape_reduce_block_max_row_), on the compile-time and the
    runtime family alike. A guard that admits the shape lets the op pool SrcA rows the
    unpacker never delivered and pack a wrong row max with no diagnostic."""
    configuration, golden_rowmajor = _make_reduce_block_max_config(
        FORMATS[0],
        block_ct_dim=1,
        use_runtime=use_runtime,
        operand_shape=operand_shape,
        pack_skip_execute=True,
    )
    _expect_llk_assert_on(
        configuration, golden_rowmajor, f"operand shape {operand_shape}"
    )


def test_reduce_block_max_reinit_short_runtime_rejects_width():
    """A valid init followed by a reinit_short_runtime re-arm with block_ct_dim=128: the MOP
    reprogram (TRISC1 only; the unpack is not re-armed) must assert instead of writing the
    wrapped count."""
    configuration, golden_rowmajor = _make_reduce_block_max_config(
        FORMATS[0],
        block_ct_dim=2,
        use_runtime=True,
        reinit="short",
        clobber=_CLOBBER_FOR_REINIT["short"],
        reinit_block_ct_dim=MAX_BLOCK_CT_DIM + 1,
        unpack_skip_execute=True,
        pack_skip_execute=True,
    )
    _expect_llk_assert_on(
        configuration,
        golden_rowmajor,
        f"reinit_short_runtime with block_ct_dim={MAX_BLOCK_CT_DIM + 1}",
        triscs=("TRISC1",),
    )


@parametrize(use_runtime=[False, True])
def test_reduce_block_max_reinit_short_rejects_shape(use_runtime):
    """A valid init followed by a reinit_short re-arm told the operand is a single 16x16
    face: the MOP reprogram (TRISC1 only) must assert. The compile-time reinit_short is a
    Blackhole-only lib path; the runtime one runs on both arches."""
    (use_runtime,) = use_runtime
    if not use_runtime and get_chip_architecture() != ChipArchitecture.BLACKHOLE:
        pytest.skip(
            "compile-time reduce_block_max_row reinit_short is a Blackhole-only lib path"
        )
    configuration, golden_rowmajor = _make_reduce_block_max_config(
        FORMATS[0],
        block_ct_dim=2,
        use_runtime=use_runtime,
        reinit="short",
        clobber=_CLOBBER_FOR_REINIT["short"],
        reinit_operand_shape=TileShape(16, 16, 1, 1),
        unpack_skip_execute=True,
        pack_skip_execute=True,
    )
    _expect_llk_assert_on(
        configuration,
        golden_rowmajor,
        "reinit_short with a 16x16 operand shape",
        triscs=("TRISC1",),
    )


def test_reduce_block_max_trigger_runtime_rejects_odd_width():
    """respect_trigger splits the unpack into two block_ct_dim/2 runs, so an odd width
    delivers one tile fewer than the math MOP waits for and the reduce hangs; the unpack
    init (TRISC0) must assert on it instead."""
    configuration, golden_rowmajor = _make_reduce_block_max_config(
        FORMATS[0],
        block_ct_dim=3,
        use_runtime=True,
        respect_trigger=True,
        math_skip_execute=True,
    )
    _expect_llk_assert_on(
        configuration,
        golden_rowmajor,
        "respect_trigger with block_ct_dim=3",
        triscs=("TRISC0",),
    )
