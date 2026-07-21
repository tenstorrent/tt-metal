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
  * reinit_short / reinit_minimal (compile-time = Blackhole-only lib fns; runtime
    = both arches) re-arm after the init, guarding the reconfig-escape path.
"""

import pytest
import torch
from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import DestAccumulation, MathFidelity
from helpers.param_config import parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import StimuliSpec, generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    BLOCK_CT_DIM,
    MATH_FIDELITY,
    REINIT_MODE,
    USE_RUNTIME,
)
from helpers.utils import tolerances

# 32x32 bf16 tile, stored as 4 faces of 16x16 (2x2 face grid).
TILE_DIM = 32
FACE_DIM = 16
FACES_PER_ROW = 2  # num_faces_c_dim
FACE_SIZE = FACE_DIM * FACE_DIM
ELEMENTS_PER_TILE = TILE_DIM * TILE_DIM

# Inputs are always bf16 (the op's contract); only the DEST/output precision varies.
FORMATS = [
    InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b),
    InputOutputFormat(DataFormat.Float16_b, DataFormat.Float32),
]

# REINIT_MODE C++ selector: 0 = plain init, 1 = reinit_short, 2 = reinit_minimal.
_REINIT_DEFINE = {"none": 0, "short": 1, "minimal": 2}


def _dest_acc(output_format):
    return (
        DestAccumulation.Yes
        if output_format == DataFormat.Float32
        else DestAccumulation.No
    )


def _defined_output_indices():
    """Flat (tilized) indices of the defined row-max lanes in the 32x32 output tile.

    The row-max for each of the 32 physical rows lands in column [0], which in the
    4-face tile layout is: face-row ``pr // 16`` (faces {0,1} or {2,3}), row
    ``pr % 16``, col 0 → flat index (face_row * FACES_PER_ROW) * FACE_SIZE + within*16.
    """
    indices = []
    for pr in range(TILE_DIM):
        face_row = pr // FACE_DIM
        within = pr % FACE_DIM
        indices.append((face_row * FACES_PER_ROW) * FACE_SIZE + within * FACE_DIM)
    return indices


def _row_max_golden(src_a, block_ct_dim):
    """Per-row MAX across the whole block width (block_ct_dim tiles x 32 cols).

    Operand A is block_ct_dim contiguous 32x32 tiles in tilized (4-face) layout.
    Returns a flat 1024-element tile with the row-max placed in column [0] of each
    physical row (tilized layout); other lanes are left 0 (unspecified on HW).
    Mirrors helpers.golden_generators.ReduceBlockMaxRowGolden semantics.
    """
    src = src_a.to(torch.float32)
    out = torch.zeros(ELEMENTS_PER_TILE, dtype=torch.float32)
    for pr in range(TILE_DIM):
        face_row = pr // FACE_DIM
        within = pr % FACE_DIM
        row_vals = []
        for t in range(block_ct_dim):
            tile_base = t * ELEMENTS_PER_TILE
            # The two faces holding physical row `pr` (left cols 0-15, right cols 16-31).
            for fc in range(FACES_PER_ROW):
                face = face_row * FACES_PER_ROW + fc
                face_base = tile_base + face * FACE_SIZE + within * FACE_DIM
                row_vals.append(src[face_base : face_base + FACE_DIM])
        out_idx = (face_row * FACES_PER_ROW) * FACE_SIZE + within * FACE_DIM
        out[out_idx] = torch.max(torch.cat(row_vals))
    return out


def _run_reduce_block_max(
    formats, math_fidelity, block_ct_dim, use_runtime=False, reinit="none"
):
    dest_acc = _dest_acc(formats.output_format)
    # Operand A is a single block of block_ct_dim tiles, laid out tile-major
    # (each 32x32 tile contiguous), i.e. a 32-row x (block_ct_dim*32)-col operand.
    input_dimensions = [TILE_DIM, block_ct_dim * TILE_DIM]

    # A ~ U[0, 1] keeps every row-max well inside bf16's dynamic range.
    src_A, tile_cnt_A, _, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        spec_A=StimuliSpec.uniform(low=0.0, high=1.0),
    )
    # Scaler B: a single tile of 1.0 (F0 holds the scaler; the op multiplies by it).
    src_B = torch.ones(ELEMENTS_PER_TILE, dtype=torch.bfloat16)

    golden = _row_max_golden(src_A, block_ct_dim)

    configuration = TestConfig(
        "sources/reduce_block_max_test.cpp",
        formats,
        templates=[
            BLOCK_CT_DIM(block_ct_dim),
            MATH_FIDELITY(math_fidelity),
            USE_RUNTIME(use_runtime),
            REINIT_MODE(_REINIT_DEFINE[reinit]),
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

    res = torch.tensor(res_from_L1, dtype=torch.float32)
    tol = tolerances[formats.output_format]

    # Only column [0] of each physical row is defined (the row-max); the packer's
    # REDUCE_ROW mask leaves every other lane unspecified, so validate those alone.
    for pr, idx in enumerate(_defined_output_indices()):
        g = float(golden[idx])
        d = float(res[idx])
        assert abs(d - g) <= tol.atol + tol.rtol * abs(g), (
            f"reduce_block_max_row mismatch at physical row {pr} (idx {idx}): "
            f"device={d} golden={g} "
            f"(block_ct_dim={block_ct_dim}, fidelity={math_fidelity.name})"
        )


@parametrize(
    formats=FORMATS,
    math_fidelity=[MathFidelity.HiFi2, MathFidelity.HiFi4],
    block_ct_dim=[1, 2, 3, 4, 8],
)
def test_reduce_block_max(formats, math_fidelity, block_ct_dim):
    """Compile-time block_ct_dim sweep, bf16 and fp32-dest."""
    _run_reduce_block_max(formats, math_fidelity, block_ct_dim)


@parametrize(
    formats=FORMATS,
    math_fidelity=[MathFidelity.HiFi2, MathFidelity.HiFi4],
    block_ct_dim=[1, 2, 4, 8],
)
def test_reduce_block_max_runtime(formats, math_fidelity, block_ct_dim):
    """Runtime (dynamic block_ct_dim) path."""
    _run_reduce_block_max(formats, math_fidelity, block_ct_dim, use_runtime=True)


@parametrize(
    formats=FORMATS,
    math_fidelity=[MathFidelity.HiFi2, MathFidelity.HiFi4],
    block_ct_dim=[2, 4],
    reinit=["short", "minimal"],
)
def test_reduce_block_max_reinit(formats, math_fidelity, block_ct_dim, reinit):
    """Reinit / reprogram after init (reconfig-escape guard).

    Compile-time short/minimal reinit lib fns are Blackhole-only; skip on WH.
    """
    if get_chip_architecture() != ChipArchitecture.BLACKHOLE:
        pytest.skip(
            "compile-time reduce_block_max_row reinit is a Blackhole-only lib path"
        )
    _run_reduce_block_max(formats, math_fidelity, block_ct_dim, reinit=reinit)


@parametrize(
    formats=FORMATS,
    math_fidelity=[MathFidelity.HiFi2, MathFidelity.HiFi4],
    block_ct_dim=[2, 4],
    reinit=["short", "minimal"],
)
def test_reduce_block_max_reinit_runtime(formats, math_fidelity, block_ct_dim, reinit):
    """Runtime reinit paths: reinit_short_runtime on both arches;
    reinit_minimal_runtime is a Blackhole-only lib fn."""
    if reinit == "minimal" and get_chip_architecture() != ChipArchitecture.BLACKHOLE:
        pytest.skip(
            "runtime reduce_block_max_row reinit_minimal is a Blackhole-only lib path"
        )
    _run_reduce_block_max(
        formats, math_fidelity, block_ct_dim, use_runtime=True, reinit=reinit
    )
