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
  * compile-time path, ``block_ct_dim`` sweep, bf16 and fp32-dest, 32x32 tiles.
    16x32 (bf16 only) is Blackhole-only: Wormhole unpack still hardcodes ``y_dim = 4``;
  * runtime path (dynamic ``block_ct_dim``), including the Blackhole 16x32 sweep;
  * reinit_short / reinit_minimal / reinit (addrmod-only) re-arm after a clobbering op,
    guarding the reconfig-escape path. 32x32 only. Compile-time reinit_short/minimal are
    Blackhole-only lib fns; runtime reinit_short and runtime reinit run on both arches,
    runtime reinit_minimal is Blackhole-only.
  * trigger handshake (respect_trigger, + overlap_first_half on the runtime family):
    the unpack splits the block reduce into two half-width MOP runs gated by HW
    semaphores; the PACK thread plays the producer. 32x32 only, both arches. Validates
    split-MOP correctness + semaphore balance (not the SDPA live-data overlap; see the
    C++ note).

``perf_reduce_block_max.py`` reuses ``_run_reduce_block_max`` with ``is_perf=True``.
"""

import pytest
import torch
from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.data_format_inference import dest_acc_for
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import ReduceBlockMaxRowGolden, get_golden_generator
from helpers.llk_params import DestAccumulation, PerfRunType
from helpers.param_config import parametrize
from helpers.perf.core import create_test_or_perf_config
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import (
    CLOBBER_OP,
    LOOP_FACTOR,
    NUM_FACES_C_DIM,
    NUM_FACES_R_DIM,
    OVERLAP_FIRST_HALF,
    REDUCE_BLOCK_CT_DIM,
    REINIT_MODE,
    RESPECT_TRIGGER,
    TILE_COUNT,
    USE_RUNTIME,
)
from helpers.tile_constants import MAX_FACE_R_DIM, MAX_NUM_FACES_C_DIM
from helpers.tilize_untilize import tilize, untilize

# Tiles are 32 columns wide: 32x32 (4 faces) or a 16x32 tiny tile (2 faces, one face-row).
TILE_DIM = 32

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


def block_max_num_faces(formats):
    """Face counts for one block-max format.

    A 32-bit Dest does not support the 16x32 tiny tile, so fp32-dest stays at 4 faces.
    16x32 (2 faces) is Blackhole-only: Wormhole unpack still hardcodes ``y_dim = 4``
    and strides a full 32x32 tile between block tiles.
    """
    if dest_acc_for(formats.output_format) == DestAccumulation.Yes:
        return [4]
    if get_chip_architecture() != ChipArchitecture.BLACKHOLE:
        return [4]
    return [2, 4]


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
    num_faces=4,
    *,
    is_perf=False,
    perf_report=None,
    run_types=None,
    loop_factor=1,
):
    """Run one reduce_block_max_row variant.

    Functional runs validate column [0] against golden with an exact compare.
    ``is_perf=True`` returns after ``configuration.run(perf_report)`` and does not
    check golden. ``respect_trigger`` and ``overlap_first_half`` must stay false on
    that path: the pack thread posts those semaphores once, before the loop, so
    repeating the unpack/math body ``loop_factor`` times with either flag set
    deadlocks or skews the measurement.

    ``num_faces`` is 4 (32x32) or 2 (16x32, Blackhole bf16 only). ``loop_factor``
    repeats the TILE_LOOP body (1 for functional, 32 for perf). ``run_types`` selects
    the perf pipeline modes; functional runs use ``L1_TO_L1``.

    ``expect_mismatch`` inverts the verdict: the run must diverge from golden. Used by
    the negative controls, which pin that a clobber left un-repaired really does corrupt
    the reduce (so the reinit cases are not passing for free).

    ``expected_nondeterministic`` exempts the variant from the --bit-exact-runs
    determinism check: a run with a deliberately corrupt HW config leaves DEST
    addressing undefined, so its output is not bit-reproducible by contract (and is
    arch-dependent: stable on Wormhole, a coin-flip on Blackhole). The single-run
    functional check above still applies.
    """
    if is_perf and (respect_trigger or overlap_first_half):
        raise AssertionError(
            "perf only covers respect_trigger=False and overlap_first_half=False; "
            "the pack thread posts those semaphores once"
        )

    dest_acc = dest_acc_for(formats.output_format)
    num_faces_r_dim = num_faces // MAX_NUM_FACES_C_DIM
    tile_rows = MAX_FACE_R_DIM * num_faces_r_dim
    tile_dimensions = [tile_rows, TILE_DIM]
    elements_per_tile = tile_rows * TILE_DIM

    # One permutation of 0..width-1 per row. The row max is width-1 only when every
    # column is read, so a stride into the next tile or into scaler B (all 1.0) fails.
    # Integers through 255 are exact in bf16 (block_ct_dim <= 8 => width <= 256).
    width = block_ct_dim * TILE_DIM
    dimensions = [tile_rows, width]
    src_A_rowmajor = (
        torch.stack(
            [torch.randperm(width, dtype=torch.float32) for _ in range(tile_rows)]
        )
        .reshape(-1)
        .to(torch.bfloat16)
    )

    # Canonical golden: the established fuser oracle for this op. Returns a row-major
    # tensor with the per-row max in column [0], every other column 0.
    generate_golden = get_golden_generator(ReduceBlockMaxRowGolden)
    golden_flat = generate_golden(
        src_A_rowmajor.clone(), block_ct_dim, formats.output_format, dimensions
    )

    # Device operand is tilized per tile (the layout the C++ reads from L1).
    src_A = torch.cat(
        [
            tilize(
                src_A_rowmajor.reshape(tile_rows, block_ct_dim * TILE_DIM)[
                    :, t * TILE_DIM : (t + 1) * TILE_DIM
                ].reshape(-1),
                formats.input_format,
                num_faces=num_faces,
                tile_dimensions=tile_dimensions,
            )
            for t in range(block_ct_dim)
        ]
    )
    # Scaler B: a single tile of 1.0 (F0 holds the scaler; the op multiplies by it).
    src_B = torch.ones(elements_per_tile, dtype=torch.bfloat16)

    if is_perf and perf_report is None:
        raise ValueError("perf_report must be provided when is_perf=True")

    templates = [
        REDUCE_BLOCK_CT_DIM(block_ct_dim),
        USE_RUNTIME(use_runtime),
        REINIT_MODE(_REINIT_DEFINE[reinit]),
        CLOBBER_OP(_CLOBBER_DEFINE[clobber]),
        RESPECT_TRIGGER(respect_trigger),
        OVERLAP_FIRST_HALF(overlap_first_half),
    ]
    runtimes = [
        TILE_COUNT(block_ct_dim),
        NUM_FACES_R_DIM(num_faces_r_dim, num_faces_r_dim),
        NUM_FACES_C_DIM(MAX_NUM_FACES_C_DIM),
        LOOP_FACTOR(loop_factor),
    ]
    config_kwargs = (
        {} if is_perf else {"expected_nondeterministic": expected_nondeterministic}
    )
    configuration = create_test_or_perf_config(
        is_perf=is_perf,
        run_types=run_types or [PerfRunType.L1_TO_L1],
        test_config_kwargs=dict(
            test_name="sources/reduce_block_max_test.cpp",
            formats=formats,
            templates=templates,
            runtimes=runtimes,
            variant_stimuli=StimuliConfig(
                src_A,
                formats.input_format,
                src_B,
                formats.input_format,
                formats.output_format,
                tile_count_A=block_ct_dim,
                tile_count_B=1,
                tile_count_res=1,
                num_faces=num_faces,
                tile_dimensions=tile_dimensions,
                use_dense_tile_dimensions=True,
                sfpu=False,
            ),
            dest_acc=dest_acc,
            **config_kwargs,
        ),
    )
    if is_perf:
        configuration.run(perf_report)
        return

    res_from_L1 = configuration.run().result

    assert (
        len(res_from_L1) == elements_per_tile
    ), f"Expected one {elements_per_tile}-element output tile, got {len(res_from_L1)}"

    golden_rowmajor = golden_flat.reshape(tile_rows, block_ct_dim * TILE_DIM)

    # Untilize the device output tile to row-major so both sides share layout.
    res = untilize(
        torch.tensor(res_from_L1, dtype=torch.float32),
        formats.output_format,
        tile_dimensions=tile_dimensions,
    ).reshape(tile_rows, TILE_DIM)

    # Only column [0] of each physical row is defined (the row-max); the packer's
    # REDUCE_ROW mask leaves every other lane unspecified. The stimulus is an integer
    # permutation, so column [0] must match exactly; a tolerance would hide a bad stride.
    mismatches = [
        (pr, float(res[pr, 0]), float(golden_rowmajor[pr, 0]))
        for pr in range(tile_rows)
        if float(res[pr, 0]) != float(golden_rowmajor[pr, 0])
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
    num_faces=block_max_num_faces,
)
def test_reduce_block_max(formats, block_ct_dim, num_faces):
    """Compile-time block_ct_dim sweep, bf16 and fp32-dest.

    32x32 on both arches. 16x32 bf16 is Blackhole-only.
    """
    _run_reduce_block_max(formats, block_ct_dim, num_faces=num_faces)


@parametrize(
    formats=FORMATS,
    block_ct_dim=[1, 2, 4, 8],
    num_faces=block_max_num_faces,
)
def test_reduce_block_max_runtime(formats, block_ct_dim, num_faces):
    """Runtime (dynamic block_ct_dim) path, including the Blackhole 16x32 sweep."""
    _run_reduce_block_max(formats, block_ct_dim, use_runtime=True, num_faces=num_faces)


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
