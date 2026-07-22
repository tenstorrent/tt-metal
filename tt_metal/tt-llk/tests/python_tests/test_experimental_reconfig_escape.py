# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Reconfig-escape / empty-``uninit`` sweep for the experimental LLKs (BH + WH B0).

Several experimental LLKs have an empty or no-op ``uninit`` even though their
``init`` clobbers persistent HW state (ADDR_MODs, cfg regs such as
``THCON_SEC0_REG2_Haloize_mode``). This is the #1 contract hole in the
experimental-LLK inventory: state could leak between an experimental op and the
op next to it. Each parametrized case runs a two-op sequence and validates the
SECOND op (run 1); run 0 is packed to a scratch buffer and discarded.

``sequence`` selects the two ops and therefore which invariant is pinned:

    FORWARD (X -> datacopy): experimental op X, then a canonical A2D datacopy of
        a fresh, known tile (buffer_A[1]), validated against its input. Pins that
        X's empty ``uninit`` leaves no escape the datacopy's own init doesn't
        re-establish.
    REVERSE (datacopy -> X): a canonical datacopy first, then experimental op X,
        validating X. Pins that X's own init is self-sufficient and does not
        silently rely on leftover clean state from a preceding canonical op.
    REPEAT (X -> X): experimental op X twice, validating the second. Pins that
        X's empty ``uninit`` is safe for back-to-back reuse.

REVERSE / REPEAT validate the experimental op itself, so they only run for the
polluters with a clean single-tile golden: ``matmul_custom_no_mop`` (0) and the
SDPA sub+bcast-col pair (2). ``reduce_block_max_row`` (1) is FORWARD-only — its
validated-op behavior is already covered by the fuser reduce test.

Why a green result pins the invariant:

  * The MATH polluters clobber ADDR_MODs; the datacopy init reprograms exactly
    the ADDR_MODs it consumes (ADDR_MOD_0/2/3), so those clobbers cannot corrupt
    the copy.
  * The UNPACK polluter (``sub_bcast_col``) RMWs ``Haloize_mode`` (transpose
    within face); the victim's ``_llk_unpack_A_init_`` rewrites ``Haloize_mode``,
    so the transpose bit cannot leak.
  * Under fp32 DEST, the reduce polluter's unpack init additionally RMWs
    ``ALU_ACC_CTRL_Zero_Flag_disabled_src`` (and ``ALU_FORMAT_SPEC`` SrcA) and its
    uninit does NOT restore them; the victim's own datacopy format reconfig +
    init is what must reset them. The sweep runs both bf16 and fp32 DEST so this
    cfg-reg hole is armed, not just the ADDR_MOD / Haloize ones.

If any case ever fails, that is a genuine reconfig escape (an op left state that
the following op does not reset) — a real finding to document, not to mask.
"""

from dataclasses import dataclass
from enum import IntEnum

import pytest
import torch
from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import (
    BroadcastGolden,
    DataCopyGolden,
    EltwiseBinaryGolden,
    MatmulGolden,
    get_golden_generator,
)
from helpers.llk_params import (
    BroadcastType,
    DestAccumulation,
    MathFidelity,
    MathOperation,
)
from helpers.param_config import parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import StimuliSpec, generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import MATH_FIDELITY, TemplateParameter
from helpers.tilize_untilize import tilize_block, untilize_block
from helpers.utils import passed_test

# 32x32 tile: 4 faces of 16x16.
ELEMENTS_PER_TILE = 1024
TILE_DIMENSIONS = [32, 32]
FACE_R_DIM = 16
NUM_FACES = 4

# Experimental polluter selector (matches the POLLUTER switch in the CPP kernel).
# Scenario 2 runs the paired SDPA fused sub+bcast-col op, which exercises BOTH the
# eltwise_binary_custom (MATH) and unpack_AB_sub_bcast_col_custom (UNPACK) empty uninits
# at once — they only run correctly as a pair.
POLLUTERS = {
    0: "matmul_custom_no_mop",
    1: "reduce_block_max_row",
    2: "eltwise_binary_custom + unpack_AB_sub_bcast_col_custom (SDPA sub+bcast-col)",
}


class Sequence(IntEnum):
    """Two-op ordering; matches the SEQUENCE switch in the CPP kernel."""

    FORWARD = 0  # experimental op -> canonical datacopy (validate datacopy)
    REVERSE = 1  # canonical datacopy -> experimental op (validate experimental)
    REPEAT = 2  # experimental op -> experimental op (validate the second)


@dataclass
class POLLUTER(TemplateParameter):
    """Selects which experimental op X is."""

    polluter: int = 0

    def convert_to_cpp(self) -> str:
        # A #define (not a constexpr) so the CPP kernel can switch on it with #if.
        return f"#define POLLUTER {self.polluter}"


@dataclass
class SEQUENCE(TemplateParameter):
    """Selects the two-op ordering (FORWARD / REVERSE / REPEAT)."""

    sequence: int = 0

    def convert_to_cpp(self) -> str:
        return f"#define SEQUENCE {int(self.sequence)}"


def _sequences(polluter):
    # reduce_block_max_row (1) is FORWARD-only: validating it as the run-1 op would
    # duplicate the fuser reduce test (masked-scalar pack + ReduceBlockMaxRowGolden).
    if polluter == 1:
        return [Sequence.FORWARD]
    return [Sequence.FORWARD, Sequence.REVERSE, Sequence.REPEAT]


def _dest_accs(sequence):
    # fp32 DEST only widens the escape surface for FORWARD: the ALU_ACC_CTRL /
    # ALU_FORMAT_SPEC hole is "experimental init sets it, the following canonical op
    # must reset it". In REVERSE/REPEAT the experimental op both sets and consumes it,
    # so there is no cross-op escape to pin — bf16 is sufficient there.
    if sequence == Sequence.FORWARD:
        return [DestAccumulation.No, DestAccumulation.Yes]
    return [DestAccumulation.No]


def _tilize_tile(tile, input_format):
    """Tilize one 32x32 tile (row-major -> face layout) so an experimental unpacker
    that expects tiled input reads it correctly."""
    return tilize_block(
        tile,
        TILE_DIMENSIONS,
        input_format,
        num_faces=NUM_FACES,
        tile_dimensions=TILE_DIMENSIONS,
        face_r_dim=FACE_R_DIM,
    ).flatten()


def _tilize_two_tile_buffer(src, input_format):
    """Tilize both tiles of a [64, 32] buffer independently; buffer[0] then holds a
    correctly-tiled tile for the experimental op (buffer[1] feeds the discarded
    run-0 datacopy in REVERSE and is otherwise unused)."""
    t0 = _tilize_tile(src[:ELEMENTS_PER_TILE], input_format)
    t1 = _tilize_tile(src[ELEMENTS_PER_TILE : 2 * ELEMENTS_PER_TILE], input_format)
    return torch.cat([t0, t1])


def _sdpa_sub_bcast_col_golden(src_A_tile, src_B_tile, formats, math_fidelity):
    """Golden for the SDPA fused SUB with column-broadcast srcB (single tile, ct_dim=1).
    Mirrors test_eltwise_bcast_col_custom: broadcast srcB across the face column, then
    subtract from srcA."""
    fmt = formats.input_format
    src_B_tilized = _tilize_tile(src_B_tile, fmt)
    src_B_broadcasted_tilized = get_golden_generator(BroadcastGolden)(
        BroadcastType.Column,
        src_B_tilized,
        fmt,
        num_faces=NUM_FACES,
        tile_cnt=1,
        face_r_dim=FACE_R_DIM,
    )
    src_B_golden = untilize_block(
        src_B_broadcasted_tilized,
        fmt,
        TILE_DIMENSIONS,
        num_faces=NUM_FACES,
        tile_dimensions=TILE_DIMENSIONS,
        face_r_dim=FACE_R_DIM,
    ).flatten()
    return get_golden_generator(EltwiseBinaryGolden)(
        MathOperation.Elwsub,
        src_A_tile,
        src_B_golden,
        formats.output_format,
        math_fidelity,
    )


@parametrize(
    formats=[InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b)],
    # Two high-fidelity levels: HiFi2 and HiFi4 both take the matmul_no_mop high-fidelity
    # branch (identical ADDR_MOD_5 fidelity-increment programming; ADDR_MOD_6 is throttle-gated
    # and not written at THROTTLE_LEVEL 0), but drive a different fidelity-phase replay count,
    # so the polluter runs a longer MVMUL walk over its clobbered ADDR_MOD state at HiFi4.
    math_fidelity=[MathFidelity.HiFi2, MathFidelity.HiFi4],
    polluter=list(POLLUTERS.keys()),
    # Conditional axes: reduce is FORWARD-only; fp32 DEST is swept only for FORWARD.
    sequence=_sequences,
    dest_acc=_dest_accs,
)
def test_experimental_reconfig_escape(
    formats, math_fidelity, polluter, sequence, dest_acc
):
    if get_chip_architecture() == ChipArchitecture.QUASAR:
        pytest.skip(
            "experimental reconfig-escape sweep targets Blackhole and Wormhole B0"
        )

    # buffer_A[0]/buffer_B[0] feed the experimental op; buffer_A[1] is the fresh, known
    # tile the canonical datacopy reads. Two A tiles + two B tiles laid out contiguously.
    input_dimensions = [2 * TILE_DIMENSIONS[0], TILE_DIMENSIONS[1]]
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        spec_A=StimuliSpec.uniform(low=0.0, high=1.0),
        spec_B=StimuliSpec.uniform(low=0.0, high=1.0),
    )

    # Build the run-1 golden and the stimuli the kernel actually consumes. FORWARD always
    # validates the canonical datacopy of buffer_A[1]; REVERSE/REPEAT validate the
    # experimental op (which consumes tiled operands out of buffer_A[0]/buffer_B[0]).
    untilize_result = False
    if sequence == Sequence.FORWARD:
        stim_A, stim_B = src_A, src_B
        victim_tile = src_A[ELEMENTS_PER_TILE : 2 * ELEMENTS_PER_TILE]
        golden = get_golden_generator(DataCopyGolden)(
            victim_tile,
            formats.output_format,
            input_format=formats.input_format,
        )
    elif polluter == 0:
        # matmul_custom_no_mop: A2 = buffer_A[0] x buffer_B[0]. Kernel consumes tiled
        # operands; the golden tilizes its own output to match the HW tiled pack layout.
        stim_A = _tilize_two_tile_buffer(src_A, formats.input_format)
        stim_B = _tilize_two_tile_buffer(src_B, formats.input_format)
        golden = get_golden_generator(MatmulGolden)(
            src_A[:ELEMENTS_PER_TILE],
            src_B[:ELEMENTS_PER_TILE],
            formats.output_format,
            math_fidelity,
            input_A_dimensions=TILE_DIMENSIONS,
            input_B_dimensions=TILE_DIMENSIONS,
            tilize=True,
            input_A_format=formats.input_format,
            input_B_format=formats.input_format,
        )
    else:  # polluter == 2: SDPA sub+bcast-col
        stim_A = _tilize_two_tile_buffer(src_A, formats.input_format)
        stim_B = _tilize_two_tile_buffer(src_B, formats.input_format)
        golden = _sdpa_sub_bcast_col_golden(
            src_A[:ELEMENTS_PER_TILE],
            src_B[:ELEMENTS_PER_TILE],
            formats,
            math_fidelity,
        )
        untilize_result = True

    configuration = TestConfig(
        "sources/experimental_reconfig_escape_test.cpp",
        formats,
        templates=[
            MATH_FIDELITY(math_fidelity),
            POLLUTER(polluter=polluter),
            SEQUENCE(sequence=int(sequence)),
        ],
        variant_stimuli=StimuliConfig(
            stim_A,
            formats.input_format,
            stim_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=1,
            sfpu=False,
        ),
        dest_acc=dest_acc,
    )

    res_from_L1 = configuration.run().result

    if untilize_result:
        res_from_L1 = untilize_block(
            res_from_L1,
            formats.output_format,
            TILE_DIMENSIONS,
            num_faces=NUM_FACES,
            tile_dimensions=TILE_DIMENSIONS,
            face_r_dim=FACE_R_DIM,
        ).flatten()

    assert (
        len(res_from_L1) == ELEMENTS_PER_TILE
    ), f"Expected one {ELEMENTS_PER_TILE}-element output tile, got {len(res_from_L1)}"

    res_tensor = torch.tensor(res_from_L1, dtype=torch.float32)
    assert passed_test(golden, res_tensor, formats.output_format), (
        f"reconfig escape (sequence={sequence.name}, polluter='{POLLUTERS[polluter]}', "
        f"dest_acc={dest_acc.name}): validated run-1 op diverged from its golden"
    )
