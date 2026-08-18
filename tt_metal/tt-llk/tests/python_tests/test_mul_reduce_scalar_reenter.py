# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Is the mul_reduce_scalar family re-enterable? (Blackhole only)

``mul_reduce_scalar_chunked_tile`` (api/compute/experimental/mul_reduce_scalar.h) ships in
``HW_JIT_API_HEADERS`` with no in-tree caller and no test. Two attempts at a full chunked
driver were reverted -- every bf16 variant returned 5-30x golden -- and the fault was
localised to a **single batch's** reduce rather than the cross-batch accumulation: packing
``DEST[0]`` for num_tiles=3/dst_capacity=2 read 37120 where one tile's sum is ~512. The
accumulator fill and a missing UNPACK/MATH barrier were both tried on silicon and both left
the output byte-identical, so neither was the cause.

That left one structural difference from the working non-chunked driver
(``test_mul_reduce_scalar.py``, green across 54 variants): the chunked form issues the
per-batch inits -- ``_llk_math_mul_reduce_scalar_init_`` on math, ``_llk_unpack_AB_init_``
plus ``switch_to_reduce`` on unpack -- once per batch instead of once per kernel. So the
standing hypothesis is that the family is **not re-enterable**: a second init accumulates
addrmod/counter state instead of re-establishing it.

This file tests exactly that, in isolation, without rebuilding the chunked driver -- which
is the cheap way to confirm or kill the hypothesis before any more effort goes into the
sweep. The kernel runs the known-good non-chunked sequence ``REDUCE_PASSES`` times over the
**same input**, re-issuing what the chunked loop re-issues, and packs each pass's scalar
separately. Identical input must give identical scalars.

What this established (BH p100a)
--------------------------------
The hypothesis is CONFIRMED, and sharper than it was stated. Re-entry is not broken in
general -- it is broken specifically when there is no DEST-section boundary in between:

  * ``passes=1``, either mode                  -> correct (the control).
  * ``passes=2``, ``single_dest_section=False`` -> correct, and bit-identical between the
    two passes. A ``dest_section_done`` / ``wait_for_dest_available`` pair between reduces
    fully restores whatever the second init does not.
  * ``passes=2``, ``single_dest_section=True``  -> WRONG, every variant: 9.27x to 9.93x
    golden across both output formats, both fidelities and num_tiles 1..3.

The second bullet is why the original one-line hypothesis ("the family is not
re-enterable") was too coarse, and the third is the defect. The chunked op is structured
exactly like the third case: ``mul_reduce_scalar_chunked_tile`` documents that the caller
"must ... acquire DST before calling" and then re-enters every batch inside that one
section, with no pack handshake between -- ``if (batch > 0) mul_reduce_scalar_init(...)``
is the only restoration it attempts.

That matches the reverted driver's signature closely enough to call it the same defect: it
reported 5-30x golden and "not a clean multiple of anything", and this reproduces 9.3-9.9x,
also non-integer. The value here is that it is ~40 lines of driver rather than a full
chunked implementation, so whoever fixes the LLK has a minimal reproducer.

Which is why the failing combination is xfail rather than deleted: it records a real defect
in a promoted LLK and flips to XPASS the moment re-entry inside one DEST section starts
restoring state. The fix belongs in the LLK -- or in the compute API, if the answer is that
the chunked op must close the section per batch -- not here.
"""

import pytest
import torch
from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import DestAccumulation, MathFidelity, format_dict
from helpers.param_config import parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    MATH_FIDELITY,
    NUM_FACES_C_DIM,
    NUM_FACES_R_DIM,
    REDUCE_PASSES,
    TILE_COUNT,
)
from helpers.tile_shape import construct_tile_shape
from helpers.utils import tolerances

# Inputs are always bf16; only the DEST/output precision varies. Mirrors the non-chunked
# test, and bf16 output is where the reverted chunked driver went wrong.
FORMATS = [
    InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b),
    InputOutputFormat(DataFormat.Float16_b, DataFormat.Float32),
]

# Full tile only. Tiny-tile geometry is swept by the non-chunked test and is orthogonal to
# whether a second init re-establishes state.
TILE_DIMENSIONS = [32, 32]


def _dest_acc(output_format):
    """Native fp32 DEST is required whenever the output is Float32."""
    return (
        DestAccumulation.Yes
        if output_format == DataFormat.Float32
        else DestAccumulation.No
    )


_REENTER_XFAIL = (
    "re-entering mul_reduce_scalar inside one DEST section does not restore state: the "
    "second reduce returns ~9.3-9.9x golden. This is the defect behind the reverted "
    "mul_reduce_scalar_chunked_tile driver, which re-enters per batch inside a single "
    "section exactly like this"
)


@parametrize(
    formats=FORMATS,
    math_fidelity=[MathFidelity.HiFi2, MathFidelity.HiFi4],
    num_tiles=[1, 2, 3],
    passes=[1, 2],
    single_dest_section=[False, True],
)
def test_mul_reduce_scalar_reenter(
    request, formats, math_fidelity, num_tiles, passes, single_dest_section
):
    """Re-running the multiply+reduce sequence must reproduce the same scalar.

    ``single_dest_section`` is the axis that matters. False puts a full DEST-section
    boundary (dest_section_done / wait_for_dest_available, i.e. a pack handshake) between
    passes; True keeps them in one section, which is what the chunked op does. The boundary
    re-establishes state the chunked form never re-establishes, so a False-only result
    would not answer the question -- it would only show that re-entry works when MATH has
    gone through the packer in between.
    """
    if get_chip_architecture() != ChipArchitecture.BLACKHOLE:
        pytest.skip("mul_reduce_scalar is a Blackhole-only experimental LLK")

    # Marker, not pytest.xfail(): the body has to run for this to report XPASS once the
    # defect is fixed, which is the point of keeping it.
    if single_dest_section and passes > 1:
        request.node.add_marker(pytest.mark.xfail(reason=_REENTER_XFAIL, strict=False))

    tile_shape = construct_tile_shape(TILE_DIMENSIONS)
    elements_per_tile = tile_shape.total_tile_size()
    dest_acc = _dest_acc(formats.output_format)
    input_dimensions = [num_tiles * TILE_DIMENSIONS[0], TILE_DIMENSIONS[1]]

    src_A, tile_cnt_A, _, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        tile_dimensions=TILE_DIMENSIONS,
    )

    # B == 1.0 everywhere, as in the non-chunked test: A * B == A, so the fused op reduces
    # to sum(A) and the golden stays independent of fidelity.
    src_B = torch.ones(
        tile_cnt_B * elements_per_tile, dtype=format_dict[formats.input_format]
    )

    golden_scalar = float(
        (src_A.to(torch.float32) * src_B.to(torch.float32)).sum().item()
    )

    configuration = TestConfig(
        "sources/mul_reduce_scalar_reenter_test.cpp",
        formats,
        templates=[
            MATH_FIDELITY(math_fidelity),
            REDUCE_PASSES(passes, single_dest_section=single_dest_section),
        ],
        runtimes=[
            TILE_COUNT(num_tiles),
            NUM_FACES_R_DIM(tile_shape.num_faces_r_dim, tile_shape.num_faces_r_dim),
            NUM_FACES_C_DIM(tile_shape.num_faces_c_dim, tile_shape.num_faces_c_dim),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=1 if single_dest_section else passes,
            num_faces=tile_shape.total_num_faces(),
            face_r_dim=tile_shape.face_r_dim,
            tile_dimensions=TILE_DIMENSIONS,
            use_dense_tile_dimensions=True,
            sfpu=False,
        ),
        dest_acc=dest_acc,
    )

    res_from_L1 = configuration.run().result

    # The reduced scalar is element [0] of each output tile; every other lane is
    # unspecified. In single-section mode there is one tile, holding the LAST pass's
    # scalar -- which is the one that answers the question.
    packed = 1 if single_dest_section else passes
    scalars = [
        float(res_from_L1[pass_index * elements_per_tile])
        for pass_index in range(packed)
    ]

    tol = tolerances[formats.output_format]

    def close(value):
        return abs(value - golden_scalar) <= tol.atol + tol.rtol * abs(golden_scalar)

    context = (
        f"num_tiles={num_tiles}, fidelity={math_fidelity.name}, "
        f"out={formats.output_format.name}, passes={passes}, "
        f"single_dest_section={single_dest_section}, golden={golden_scalar}"
    )

    if single_dest_section:
        # Only the last pass's scalar exists. If passes > 1 this IS the re-entry result.
        assert close(scalars[0]), (
            f"the last of {passes} reduce(s) sharing one DEST section returned "
            f"{scalars[0]} ({context}). At passes=1 that is a bug in this driver; at "
            "passes>1 it means re-entering the reduce inside one DEST section does not "
            "restore state -- which is exactly what mul_reduce_scalar_chunked_tile does "
            "per batch, and would be the cause of the reverted chunked driver's 5-30x "
            "results."
        )
        return

    assert close(scalars[0]), (
        f"pass 0 is already wrong (device={scalars[0]}, {context}). Pass 0 is the "
        "non-chunked sequence this driver shares with test_mul_reduce_scalar.py, so this "
        "is a bug in this driver rather than a finding about re-entry -- fix it before "
        "reading anything into the pass>0 result."
    )

    if passes > 1:
        wrong = [
            (index, value)
            for index, value in enumerate(scalars[1:], start=1)
            if not close(value)
        ]
        assert not wrong, (
            "the mul_reduce_scalar family is NOT re-enterable: pass 0 reduced correctly "
            f"to {scalars[0]} but {[f'pass {i}={v}' for i, v in wrong]} did not, over "
            f"identical input ({context}). A second "
            "_llk_math_mul_reduce_scalar_init_ therefore does not re-establish state -- "
            "which is exactly what mul_reduce_scalar_chunked_tile does once per batch, "
            "and so is the cause of the reverted chunked driver's 5-30x results."
        )

        assert scalars[1] == scalars[0], (
            f"both passes are within tolerance of the golden but not bit-identical "
            f"(pass 0={scalars[0]}, pass 1={scalars[1]}, {context}). The reduce is "
            "deterministic given identical input, so a difference means re-entry perturbs "
            "the result without breaking it outright."
        )
