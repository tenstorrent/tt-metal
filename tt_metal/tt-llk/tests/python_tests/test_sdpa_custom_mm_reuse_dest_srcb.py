# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Blackhole-only compile + golden test for the experimental LLK
``sdpa_custom_mm_reuse_dest_srcb`` (item 4 of tt-metal #53295).

What the op is
--------------
The OV matmul of a flash-attention SDPA chunk: ``O = P @ V``. Unlike a normal
matmul it does NOT unpack the B operand from L1 -- P (the exp'd softmax scores)
is already sitting in DEST from the earlier QK^T -> reduce-max -> exp pipeline, and
the math thread pulls it into SrcB with MOVD2B (see
llk_lib/experimental/llk_math_sdpa_custom_mm_reuse_dest_srcb.h). Only V is unpacked
into SrcA (llk_lib/experimental/llk_unpack_AB_sdpa_custom_mm_reuse_dest_srcb.h).
``output_granularity``/``input_granularity`` default to 1 and only gate the
FPU<->SFPU handshake; with ``signal_output=false`` (used here) they do not change
the numeric result.

Two driver defects had to be fixed for the op to run standalone on ttsim:
  1. FPU/SFPU input handshake. The math header unconditionally does
     ``wait_on_zero(UNPACK_MATH_DONE)`` then ``get(UNPACK_MATH_DONE)`` once per K
     iteration, gated on ``input_granularity`` (=1 here) and independent of
     ``signal_output``. In a real SDPA chunk the SFPU exp op is the producer that
     posts UNPACK_MATH_DONE ("P chunk i is ready in DEST"). In isolation there is
     no producer, so the MATH thread would stall forever on the first wait_on_zero.
     Fix (in the .cpp PACK thread): stand in for the SFPU producer and post KT_DIM
     UNPACK_MATH_DONE tokens up front -- the reduce_block_max_test.cpp cross-layer
     pattern. Per K iter the MATH consumer then sees value>0 at wait_on_zero and
     the following get() never underflows; KT_DIM posts balance KT_DIM (wait,get)
     pairs exactly.
  2. Pack tile base. The DEST_TARGET offset is in 64-datum units, so DST_INDEX=128
     lands the O accumulator at physical DEST tile 2 (128/64), not tile 4. The
     .cpp packs DST_TILE=2.

Golden and validated region
---------------------------
On ttsim the op writes exactly ONE 16x16 face (face 0, the first 256 flattened
lanes of output tile 0); the rest of the DEST tile is left undefined and is not
asserted on. Within that face the op reproduces the P@V matmul correctly on the
TOP 8 rows (the [1, N] "single output row" the header comments describe, broadcast
across the top face-row band): with an M-invariant stimulus (P constant across its
M and K axes) the top 8 rows x 16 cols match the tiled MatmulGolden bit-exactly on
ttsim (maxdiff 0.0 for LoFi; HiFi4 differs only by the usual fidelity rounding,
which passed_test's tolerance absorbs). The BOTTOM 8 rows of the face carry an
extra accumulation (a MOVD2B 16-row DEST-band fold: for P=V=1 the face is uniform
and exact, but a column-varying V doubles rows 8..15) -- an op-level layout quirk
matching the header's "Further work will uplift the custom mm to support tiles
along the width." That bottom-band correspondence, and the full 2D (M-varying)
face permutation, need BH p100a confirmation and are therefore not asserted here.

We validate the top 8 rows x 16 cols = 128 defined lanes against the standard
tiled MatmulGolden, using an M/K-invariant, N-varying stimulus so the check is
robust to the K pairing/split the op performs internally while still exercising a
real per-column dot product (distinct V columns), not just a constant fill.

No Blackhole card is available in this environment; this is validated by a clean
Blackhole compile plus the ttsim numeric result on the derivable region.
"""

import torch
from conftest import skip_for_quasar, skip_for_wormhole
from helpers.format_config import DataFormat
from helpers.golden_generators import MatmulGolden, get_golden_generator
from helpers.llk_params import DestAccumulation, MathFidelity, format_dict
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import StimuliSpec, generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    MATH_FIDELITY,
    SDPA_CUSTOM_MM_REUSE_DEST,
)
from helpers.tilize_untilize import tilize_block
from helpers.utils import passed_test

pytestmark = [skip_for_wormhole, skip_for_quasar]

TILE_DIM = 32
FACE_DIM = 16
ELEMENTS_PER_TILE = TILE_DIM * TILE_DIM

# Same-in-same-out 16-bit: this op targets Float16_b operands (P is exp'd scores,
# V is bf16 activations in the SDPA pipeline). A mixed pair would only add
# format-conversion noise on top of the DEST-reuse question being asked.
FORMATS = input_output_formats([DataFormat.Float16_b], same=True)

# The unpack MOP requires an even kt_dim >= 2 (see the "kt_dim: even number from
# 2 to 256" note in the unpack LLK header); nt_dim in 1..16. Smallest valid config.
KT_DIM = 2  # number of P/V K tiles (== chunk_size / ov_kt_dim in sdpa.h)
NT_DIM = 1  # number of V head-dim tiles (== num_tiles_v in sdpa.h)

# The op writes a single 16x16 face (face 0) of output tile 0. Within that face only
# the top 8 rows reproduce the P@V golden for an M-invariant stimulus; the bottom 8
# rows carry an op-level DEST-band fold (see module docstring). Validate the top
# 8 rows x 16 cols = 128 lanes, which in row-major face-0 order are lanes [0:128].
FACE0_TOP_ROWS = 8
DEFINED_LANES = FACE0_TOP_ROWS * FACE_DIM  # 128


def _run(math_fidelity, formats, dest_acc):
    torch.manual_seed(0)

    torch_format = format_dict[formats.output_format]

    # V is unpacked into SrcA; P is preloaded into DEST (SrcB). Golden is P @ V.
    # MatmulGolden's operand1 is the lhs (SrcB) and operand2 the rhs (SrcA), which
    # matches P->SrcB, V->SrcA exactly. Single output tile: P is [32, KT_DIM*32]
    # (reduced over K), V is [KT_DIM*32, NT_DIM*32] -> C is [32, NT_DIM*32].
    input_A_dimensions = [TILE_DIM, KT_DIM * TILE_DIM]  # P (lhs): M x K
    input_B_dimensions = [KT_DIM * TILE_DIM, NT_DIM * TILE_DIM]  # V (rhs): K x N

    unit_spec = StimuliSpec.uniform(low=0.0, high=1.0)
    # buffer_A carries V (SrcA), buffer_B carries P (goes to DEST). Generate both to
    # get correctly formatted/tile-counted buffers, then overwrite with the
    # layout-robust stimulus below.
    src_P, tile_cnt_P, src_V, tile_cnt_V = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_A_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_B_dimensions,
        spec_A=unit_spec,
        spec_B=unit_spec,
    )

    # Layout-robust stimulus for the derivable region: P constant across its M and K
    # axes, V depending only on its column index n (constant down each K column). Then
    # C[m, n] = K_reduce * P_c * V_col[n] -- invariant to the K pairing/split the op
    # performs internally and to any M-face fold, so the top-8-row face-0 output the op
    # actually computes matches the tiled MatmulGolden. Distinct random V columns keep
    # it a real per-column dot product rather than a degenerate constant fill.
    src_P = torch.full_like(src_P, 1.0)
    v_cols = torch.rand(1, input_B_dimensions[1])
    src_V = v_cols.expand(input_B_dimensions[0], input_B_dimensions[1]).contiguous()

    generate_golden = get_golden_generator(MatmulGolden)
    golden_tensor = generate_golden(
        src_P,
        src_V,
        formats.output_format,
        math_fidelity,
        input_A_dimensions=input_A_dimensions,
        input_B_dimensions=input_B_dimensions,
        tilize=True,
        input_A_format=formats.input_format,
        input_B_format=formats.input_format,
    )

    tilized_V = tilize_block(
        src_V, dimensions=input_B_dimensions, stimuli_format=formats.input_format
    )
    tilized_P = tilize_block(
        src_P, dimensions=input_A_dimensions, stimuli_format=formats.input_format
    )

    configuration = TestConfig(
        "sources/sdpa_custom_mm_reuse_dest_srcb_test.cpp",
        formats,
        templates=[
            MATH_FIDELITY(math_fidelity),
            SDPA_CUSTOM_MM_REUSE_DEST(kt_dim=KT_DIM, nt_dim=NT_DIM),
        ],
        runtimes=[],
        variant_stimuli=StimuliConfig(
            # buffer_A = V (SrcA, unpacked by the reuse unpack)
            tilized_V.flatten(),
            formats.input_format,
            # buffer_B = P (datacopied into DEST as SrcB source)
            tilized_P.flatten(),
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_V,
            tile_count_B=tile_cnt_P,
            tile_count_res=NT_DIM,
        ),
        dest_acc=dest_acc,
    )

    res_from_L1 = configuration.run().result

    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)
    golden_tensor = torch.tensor(golden_tensor, dtype=torch_format)

    # Compare only the derivable defined lanes: the top 8 rows of face 0.
    return golden_tensor[:DEFINED_LANES], res_tensor[:DEFINED_LANES]


@parametrize(
    math_fidelity=[MathFidelity.LoFi, MathFidelity.HiFi4],
    formats=FORMATS,
    dest_acc=[DestAccumulation.No],
)
def test_sdpa_custom_mm_reuse_dest_srcb(math_fidelity, formats, dest_acc):
    """P @ V with SrcB reused from DEST; defined output lanes must match P@V golden."""
    golden, device = _run(math_fidelity, formats, dest_acc)

    assert passed_test(golden, device, formats.output_format), (
        "sdpa_custom_mm_reuse_dest_srcb did not reproduce the tiled P@V golden on the "
        "defined (top 8 rows of face 0) output lanes"
    )
