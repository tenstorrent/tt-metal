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
  * reinit_short / reinit_minimal re-arm after the init, guarding the reconfig-escape
    path. Compile-time reinit_short/minimal are Blackhole-only lib fns; runtime
    reinit_short runs on both arches, runtime reinit_minimal is Blackhole-only.
"""

import pytest
import torch
from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import ReduceBlockMaxRowGolden, get_golden_generator
from helpers.llk_params import DestAccumulation
from helpers.param_config import parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    CLOBBER_OP,
    REDUCE_BLOCK_CT_DIM,
    REINIT_MODE,
    USE_RUNTIME,
)
from helpers.tilize_untilize import tilize, untilize
from helpers.utils import tolerances

# 32x32 bf16 tile.
TILE_DIM = 32
ELEMENTS_PER_TILE = TILE_DIM * TILE_DIM

# Inputs are always bf16 (the op's contract); only the DEST/output precision varies.
FORMATS = [
    InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b),
    InputOutputFormat(DataFormat.Float16_b, DataFormat.Float32),
]

# REINIT_MODE C++ selector: 0 = plain init, 1 = reinit_short, 2 = reinit_minimal.
_REINIT_DEFINE = {"none": 0, "short": 1, "minimal": 2}

# CLOBBER_OP C++ selector: op run between init and reinit to overwrite the reduce config,
# so the reinit path must actually restore it (reconfig-escape guard).
#   0 = none.
#   1 = eltwise binary (ELWADD) init — clobbers ALL addrmods (incl. ADDR_MOD_3) + the MOP;
#       paired with reinit_short (which reprograms the MOP and all addrmods).
#   2 = scramble ADDR_MOD_1/2/6 only, preserving ADDR_MOD_3 + MOP; paired with reinit_minimal
#       (which restores only 1/2/6 and relies on 3/MOP being intact, as matmul/sub_exp do).
_CLOBBER_DEFINE = {"none": 0, "eltwise": 1, "minimal_safe": 2}

# The clobber must match the reinit's restore contract, or the reduce hangs.
_CLOBBER_FOR_REINIT = {"short": "eltwise", "minimal": "minimal_safe"}


def _dest_acc(output_format):
    return (
        DestAccumulation.Yes
        if output_format == DataFormat.Float32
        else DestAccumulation.No
    )


def _run_reduce_block_max(
    formats,
    block_ct_dim,
    use_runtime=False,
    reinit="none",
    clobber="none",
):
    dest_acc = _dest_acc(formats.output_format)

    # Build operand A row-major (32 rows x block_ct_dim*32 cols), so the canonical
    # ReduceBlockMaxRowGolden (which reshapes its operand to `dimensions` row-major)
    # is unambiguous. A ~ U[0, 1] keeps every row-max well inside bf16's range.
    dimensions = [TILE_DIM, block_ct_dim * TILE_DIM]
    src_A_rowmajor = (
        torch.empty(TILE_DIM * block_ct_dim * TILE_DIM)
        .uniform_(0.0, 1.0)
        .to(torch.bfloat16)
    )

    # Canonical golden: the established fuser oracle for this op. Returns a row-major
    # tensor with the per-row max in column [0], every other column 0.
    generate_golden = get_golden_generator(ReduceBlockMaxRowGolden)
    golden_flat = generate_golden(
        src_A_rowmajor.clone(), block_ct_dim, formats.output_format, dimensions
    )

    # Device operand is tilized per 32x32 tile (the layout the C++ reads from L1).
    src_A = torch.cat(
        [
            tilize(
                src_A_rowmajor.reshape(TILE_DIM, block_ct_dim * TILE_DIM)[
                    :, t * TILE_DIM : (t + 1) * TILE_DIM
                ].reshape(-1),
                formats.input_format,
            )
            for t in range(block_ct_dim)
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
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=block_ct_dim,
            tile_count_B=1,
            tile_count_res=1,
            sfpu=False,
        ),
        dest_acc=dest_acc,
    )

    res_from_L1 = configuration.run().result

    assert (
        len(res_from_L1) == ELEMENTS_PER_TILE
    ), f"Expected one {ELEMENTS_PER_TILE}-element output tile, got {len(res_from_L1)}"

    golden_rowmajor = golden_flat.reshape(TILE_DIM, block_ct_dim * TILE_DIM)

    # Untilize the device output tile to row-major so both sides share layout.
    res = untilize(
        torch.tensor(res_from_L1, dtype=torch.float32), formats.output_format
    ).reshape(TILE_DIM, TILE_DIM)
    tol = tolerances[formats.output_format]

    # Only column [0] of each physical row is defined (the row-max); the packer's
    # REDUCE_ROW mask leaves every other lane unspecified, so validate column [0].
    for pr in range(TILE_DIM):
        g = float(golden_rowmajor[pr, 0])
        d = float(res[pr, 0])
        assert abs(d - g) <= tol.atol + tol.rtol * abs(g), (
            f"reduce_block_max_row mismatch at row {pr}: device={d} golden={g} "
            f"(block_ct_dim={block_ct_dim})"
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
    reinit=["short", "minimal"],
)
def test_reduce_block_max_reinit_runtime(formats, block_ct_dim, reinit):
    """Runtime reinit paths: reinit_short_runtime on both arches;
    reinit_minimal_runtime is a Blackhole-only lib fn."""
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
