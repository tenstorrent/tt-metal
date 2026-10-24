# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""``_llk_unpack_AB_<ROW>`` with an explicit ``bcast_row_idx``.

One 32x32 tile, ``Res = A + row_bcast(B, idx)``; the golden is ``BroadcastGolden`` with its
``row_idx`` (row ``idx`` of the tile is row ``idx % 16`` of faces ``(0, 1)`` for ``idx < 16`` and
of faces ``(2, 3)`` otherwise; every output face repeats its column half of that row).

The guard cases pin the two checks in the ROW branch of ``_llk_unpack_AB_``: an index past the
tile (which otherwise silently reads a row of whatever follows the tile in L1) and a BFP operand B
with a non-zero index (the base-address shift cannot select a row of a shared-exponent tile). Both
rely on ``LLK_ASSERT`` being compiled in, which the harness does unless ``TT_LLK_DISABLE_ASSERTS``
is set.
"""

import os

import pytest
import torch
from conftest import skip_for_quasar
from helpers.device import TRISC_CORES, LLKAssertException, commit_tensix_soft_reset
from helpers.format_config import DataFormat
from helpers.golden_generators import (
    BroadcastGolden,
    EltwiseBinaryGolden,
    get_golden_generator,
)
from helpers.llk_params import (
    BroadcastType,
    DestAccumulation,
    MathFidelity,
    MathOperation,
    format_dict,
)
from helpers.param_config import input_output_formats, parametrize, runtime
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import BuildMode, TestConfig
from helpers.test_variant_parameters import (
    BCAST_ROW_IDX,
    BROADCAST_TYPE,
    MATH_FIDELITY,
    MATH_OP,
    UNPACK_ISSUED,
)
from helpers.tilize_untilize import tilize_block
from helpers.utils import passed_test

# The driver includes llk_unpack_AB.h, which only the Wormhole and Blackhole LLK trees provide.
pytestmark = skip_for_quasar

FACE_DIM = 16
INPUT_DIMENSIONS = [32, 32]
# One source for the kernel templates and the golden.
BCAST_OP = MathOperation.Elwadd
BCAST_FIDELITY = MathFidelity.LoFi

ROW_BCAST_FORMATS = input_output_formats(
    [DataFormat.Float16_b, DataFormat.Bfp8_b], same=True
)


def _make_config(formats, bcast_row_idx):
    if TestConfig.SPEED_OF_LIGHT:
        # Speed-of-light folds the runtime parameters into compile-time constants and empties
        # RuntimeParams; the cross-thread UNPACK_ISSUED gate needs the mutable L1 copy.
        pytest.skip("requires mutable runtime arguments for the cross-thread gate")
    src_A, _, src_B, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=INPUT_DIMENSIONS,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=INPUT_DIMENSIONS,
    )
    src_A_tilized = tilize_block(
        src_A, dimensions=INPUT_DIMENSIONS, stimuli_format=formats.input_format
    ).flatten()
    src_B_tilized = tilize_block(
        src_B, dimensions=INPUT_DIMENSIONS, stimuli_format=formats.input_format
    ).flatten()

    configuration = TestConfig(
        "sources/unpack_AB_bcast_row_idx_test.cpp",
        formats,
        templates=[
            BROADCAST_TYPE(BroadcastType.Row),
            MATH_OP(mathop=BCAST_OP),
            MATH_FIDELITY(BCAST_FIDELITY),
        ],
        runtimes=[BCAST_ROW_IDX(bcast_row_idx), UNPACK_ISSUED(0)],
        variant_stimuli=StimuliConfig(
            src_A_tilized,
            formats.input_format,
            src_B_tilized,
            formats.input_format,
            formats.output_format,
            tile_count_A=1,
            tile_count_B=1,
            tile_count_res=1,
        ),
        dest_acc=DestAccumulation.No,
        unpack_to_dest=False,
    )
    return configuration, src_A_tilized, src_B_tilized


def valid_row_indices(formats):
    # The base-address row select cannot address a row of a shared-exponent (BFP) tile, so a BFP
    # operand B is only meaningful at bcast_row_idx == 0.
    if formats.input_format == DataFormat.Bfp8_b:
        return [0]
    return [0, 1, 15, 16, 31]


def rejected_row_indices(formats):
    # 32 is the first index past the tile; a BFP B trips the format check on any non-zero index.
    if formats.input_format == DataFormat.Bfp8_b:
        return [5, 40]
    return [32, 40]


@parametrize(
    formats=ROW_BCAST_FORMATS,
    bcast_row_idx=runtime(rejected_row_indices),
)
def test_unpack_AB_bcast_row_idx_rejected(formats, bcast_row_idx):
    """The unpacker must refuse an index it cannot address rather than read a wrong row."""
    if os.environ.get("TT_LLK_DISABLE_ASSERTS") == "1":
        pytest.skip("needs LLK_ASSERT compiled in")
    if TestConfig.TEST_TARGET.run_simulator:
        # The harness polls the mailboxes for the whole simulator timeout before it looks for the
        # assert, longer than the simulator job's per-test timeout.
        pytest.skip(
            "assert detection on the simulator takes longer than the test timeout"
        )

    # Math and pack wait on the UNPACK_ISSUED gate, so when the unpacker stops in the assert they
    # are spinning on L1 (reset cleanly) rather than stalled on a Tensix instruction (not reset,
    # hangs the next kernel).
    configuration, _, _ = _make_config(formats, bcast_row_idx)
    raises = pytest.raises(LLKAssertException)  # allow-pytest.raises: no expect_error
    try:
        with raises:
            configuration.run()
    finally:
        # Hold the TRISCs in reset: the next test writes its runtime block before it resets them,
        # and a non-zero value at the gate's offset would release the still-spinning math/pack
        # threads into Tensix instructions that the soft reset does not clear. The compile
        # producer skips inside run() after building the ELFs, with no device context.
        if TestConfig.BUILD_MODE != BuildMode.PRODUCE:
            commit_tensix_soft_reset(
                1, cores=TRISC_CORES, location=TestConfig.TENSIX_LOCATION
            )


@parametrize(
    formats=ROW_BCAST_FORMATS,
    bcast_row_idx=runtime(valid_row_indices),
)
def test_unpack_AB_bcast_row_idx(formats, bcast_row_idx):
    configuration, src_A_tilized, src_B_tilized = _make_config(formats, bcast_row_idx)

    # BroadcastGolden quantizes B in its own format before the row select, so the binary golden
    # must not quantize it again (input_format_B=None).
    broadcast_golden = get_golden_generator(BroadcastGolden)
    src_B_broadcast = broadcast_golden(
        BroadcastType.Row,
        src_B_tilized,
        formats.input_format,
        num_faces=4,
        tile_cnt=1,
        face_r_dim=FACE_DIM,
        row_idx=bcast_row_idx,
    )
    binary_golden = get_golden_generator(EltwiseBinaryGolden)
    golden_tensor = binary_golden(
        BCAST_OP,
        src_A_tilized,
        src_B_broadcast,
        formats.output_format,
        BCAST_FIDELITY,
        input_format=formats.input_format,
        input_format_B=None,
    )

    res_from_L1 = configuration.run().result
    assert len(res_from_L1) == len(golden_tensor)
    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])
    assert passed_test(golden_tensor, res_tensor, formats.output_format)
