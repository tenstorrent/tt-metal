# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Tests for the Blackhole-only experimental SDPA weighted-reduce op sdpa_weighted_reduce
(api/compute/experimental/sdpa_weighted_reduce.h -> weighted_reduce, merged tt-metal #51361).

The op is a lightweight "weighted row reduction" matmul for the DSA indexer SDPA path. Per the
header, each chunk computes:

    out[1, 32] = weights[1, 8] x qk[8, 32]
    out[p] = sum_{h=0..7} weights[h] * qk[h, p]        (p = 0..31)

realized with two raw TTI_MVMUL (Dst[i,j] += SrcB[i,k] * SrcA[k,j], weights -> SrcB row 0,
qk -> SrcA two 8-row faces). Only DEST logical row 0 == out[0:32] is defined; the rest of the
DEST tile is undefined. See the golden-derivation block at the top of
sources/sdpa_weighted_reduce_test.cpp for the exact instruction-level derivation and how this
test drives the header's MATH core.

The physical mapping of the reduced row onto the 16x16 Dest faces is Blackhole hardware detail
we cannot validate in this environment (no BH card). To stay independent of it, the 8 head
weights are filled with a single constant W and the whole qk tile with a single constant Q, so
every output lane is analytically identical regardless of the exact face/lane grouping:

    out[p] = sum_{h=0..7} W * Q = NUM_HEADS * W * Q      (NUM_HEADS == 8)

DEFINED-LANE SCOPE (why only 16 columns are checked)
-----------------------------------------------------
This tt-llk unit test cannot include the api/ helpers the header's raw two-PACR face-stepping
pack relies on, so its C++ driver runs the header's MATH core verbatim (the two TTI_MVMUL) and
then does a standard full-tile _llk_pack_ (see sources/sdpa_weighted_reduce_test.cpp). Under that
path only ONE lane group is provably defined:

  - MVMUL #1 (ADDR_MOD_6) writes the reduced row into DEST row 0, cols 0..15 == logical row 0,
    cols 0..15 (Dest face0 row 0).
  - ADDR_MOD_6 then advances DEST by 8 rows, so MVMUL #2 (qk face1) writes DEST row 8 -- still
    inside face0, NOT the physical face1 (DEST rows 16..31) that carries logical row 0 cols
    16..31. The header's real pack reaches that data with a custom face-stepping addrmod; a plain
    full-tile pack does not, and that Dest<->face mapping is Blackhole hardware detail we cannot
    validate here (no BH card).

So, per the "validate only defined lanes" rule, we assert logical row 0 columns 0..15 (the 16
lanes MVMUL #1 defines) against NUM_HEADS * W * Q. Columns 16..31 of that row -- and every other
row -- are undefined by this MATH+plain-pack path and are not checked. (The sibling
test_sdpa_reduce_row.py likewise checks only its single defined lane, column 0.)
"""

import torch
from conftest import blackhole_only, skip_for_coverage
from helpers.format_config import DataFormat
from helpers.golden_generators import TILE_DIM
from helpers.llk_params import (
    DestAccumulation,
    format_dict,
)
from helpers.param_config import (
    input_output_formats,
    parametrize,
)
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    TILE_COUNT,
)
from helpers.tilize_untilize import tilize_block, untilize_block
from helpers.utils import passed_test

# The header reduces over exactly 8 head weights: weights[1, 8], qk[8, 32]. Fixed by the op.
# Blackhole-only experimental SDPA LLK: guard WH/Quasar from building BH-only headers.
pytestmark = blackhole_only

NUM_HEADS = 8

# One Dest face is 16 columns wide. MVMUL #1 defines exactly logical row 0, cols 0..15 (face0);
# see the "DEFINED-LANE SCOPE" note in the module docstring for why only these 16 lanes are checked.
FACE_WIDTH = 16

# Constants that fill the weights (W) and qk (Q) tiles. Chosen small so NUM_HEADS * W * Q stays
# exactly representable in Float16_b, with one pair exercising a non-trivial magnitude.
FILL_CONSTANTS = (
    (1.0, 1.5),  # W, Q -> 8 * 1.0 * 1.5 = 12.0
    (0.5, 3.0),  # W, Q -> 8 * 0.5 * 3.0 = 12.0 via different factors
)


@skip_for_coverage
@parametrize(
    formats=input_output_formats(
        [
            DataFormat.Float16_b
        ],  # Only Float16_b is supported for the SDPA weighted reduce
        same=True,
    ),
    dest_acc=[DestAccumulation.No],  # Only Float16_b / non-fp32 dest is supported
    fill_constants=list(FILL_CONSTANTS),
)
def test_sdpa_weighted_reduce(
    formats,
    dest_acc,
    fill_constants,
):
    weight_value, qk_value = fill_constants
    input_dimensions = [TILE_DIM, TILE_DIM]  # single [32, 32] tile per operand

    # generate_stimuli lays out the two operand buffers / tile counts the harness expects; we
    # overwrite the data with constants so the reduction is layout-independent (see docstring).
    #   buffer_A = weights (-> SrcB), buffer_B = qk (-> SrcA), matching the C++ operand mapping.
    weights_src, tile_cnt_w, qk_src, tile_cnt_qk = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )

    torch_format = format_dict[formats.input_format]

    # generate_stimuli returns flat 1D operand buffers; build the constant operand tiles at the
    # [32, 32] logical shape directly so the row/col indexing below is well-defined.
    # Weights: only row 0, cols 0..7 (the 8 head weights) are read by the MVMULs; everything else
    # is zero (the header notes rows 1..7 and cols 8.. are zero, so only Dst row 0 is populated).
    weights_tile = torch.zeros(input_dimensions, dtype=torch_format)
    weights_tile[0, :NUM_HEADS] = float(weight_value)

    # qk: fill the whole tile with the constant Q. The MVMULs only read rows 0..7 (two faces), so
    # a uniform fill makes every read equal to Q regardless of the exact face mapping.
    qk_tile = torch.full(input_dimensions, float(qk_value), dtype=torch_format)

    # Tilize both operands the way the kernel reads them.
    weights_src = tilize_block(
        weights_tile, input_dimensions, stimuli_format=formats.input_format
    ).flatten()
    qk_src = tilize_block(
        qk_tile, input_dimensions, stimuli_format=formats.input_format
    ).flatten()

    # GOLDEN
    # *******************************************************
    # out[p] = sum_{h=0..7} W * Q = NUM_HEADS * W * Q, identical for every column p of row 0.
    reduced_value = float(weight_value) * float(qk_value) * NUM_HEADS

    # Golden row of the reduced value. Only the first FACE_WIDTH lanes (logical row 0, cols 0..15)
    # are asserted -- the lanes MVMUL #1 provably defines under this MATH+plain-pack path; see the
    # "DEFINED-LANE SCOPE" note in the module docstring.
    golden_row = torch.full((TILE_DIM,), reduced_value, dtype=torch_format)
    # *******************************************************

    configuration = TestConfig(
        "sources/sdpa_weighted_reduce_test.cpp",
        formats,
        templates=[],  # dims are fixed constexpr in the C++ driver; no compile-time knobs
        runtimes=[
            TILE_COUNT(tile_cnt_qk),
        ],
        variant_stimuli=StimuliConfig(
            weights_src,
            formats.input_format,
            qk_src,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_w,
            tile_count_B=tile_cnt_qk,
            tile_count_res=1,
        ),
        unpack_to_dest=False,  # matmul unpack feeds SrcA/SrcB; inputs must not go straight to dest
        dest_acc=dest_acc,
    )
    res_from_L1 = configuration.run().result

    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])
    res_tensor = untilize_block(res_tensor, formats.output_format, input_dimensions)

    # Validate ONLY the lanes the test's MATH core provably defines: logical row 0, columns 0..15.
    # See the golden-derivation note below and the C++ driver header for why only these 16 lanes
    # are defined here; every other lane is undefined by this MATH+plain-pack path and not checked.
    assert passed_test(
        golden_row[:FACE_WIDTH],
        res_tensor[0, :FACE_WIDTH],
        formats.output_format,
    )
