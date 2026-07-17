# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Mixed-tile transition test for `tilize_uninit` + the C2 reconfig retarget.

Closes the last open `tilize_uninit` (C1) cell: "normal->tiny / tiny->normal with
reconfig" (axis F = tile-size transition, axis E = with-reconfig). A real
`unpack_tilize` at geometry G0 leaves the SrcA baseline in the G0 tilize state;
the transition under test re-establishes a DIFFERENT geometry G1 using
`_llk_unpack_tilize_uninit_(G0)` (PR C1) followed by
`_llk_unpack_reconfig_data_format_srca_impl_<FACE_ROW_MAJOR>(G1)` (PR C2) — the
only restore path that both re-commits the canonical SrcA Y-stride AND retargets
the geometry (Tile_x_dim_cntx0 from G1.face_r_dim, descriptor Z-dim from
G1.num_faces).

The kernel reads the four canonical SrcA registers back and LLK_ASSERTs them
against the expected G1 baseline on-device. Because a SrcA datacopy victim's
correctness is *entirely* a function of these registers — and Phases 1/2 already
prove "correct registers => correct data" behaviourally — checking them directly
verifies the transition for BOTH directions without needing a second operand /
golden (a single-geometry stimuli harness can't lay out two tile shapes cleanly).

`do_restore=False` is the negative control: with no transition the registers stay
in the G0 tilize state, so the kernel asserts the G1 baseline is NOT reproduced
(Tile_x_dim / Z-dim differ), proving the transition is load-bearing.

WH-only: exercises the 3-arg WH test wrapper `_llk_unpack_tilize_uninit_wrapper_(dst,
num_faces, face_r_dim)` and tiny `face_r_dim < 16` geometry. The BH library uninit does
honor `face_r_dim` (via its `TensorShape` param), but the BH wrapper is 2-arg and drops
it, so this branch cannot be expressed on BH without extending that wrapper (matches
`test_unpack_tilize_uninit_restore_tiny`).
"""

from dataclasses import dataclass

from conftest import skip_for_blackhole, skip_for_coverage
from helpers.format_config import DataFormat
from helpers.llk_params import DestAccumulation
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    NUM_FACES,
    TEST_FACE_DIMS,
    TemplateParameter,
)

pytestmark = skip_for_coverage

REGULAR_FACE_R_DIM = 16
REGULAR_NUM_FACES = 4
TINY_NUM_FACES = 2


@dataclass
class DO_RESTORE(TemplateParameter):
    """Whether the G0->G1 transition (uninit + reconfig) runs before the readback."""

    do_restore: bool = True

    def convert_to_cpp(self) -> str:
        return f"constexpr bool DO_RESTORE = {str(self.do_restore).lower()};"


@dataclass
class VICTIM_GEOMETRY(TemplateParameter):
    """Target (G1) tile geometry the transition must re-establish."""

    victim_num_faces: int = REGULAR_NUM_FACES
    victim_face_r_dim: int = REGULAR_FACE_R_DIM

    def convert_to_cpp(self) -> str:
        return (
            f"constexpr std::uint32_t VICTIM_NUM_FACES = {self.victim_num_faces};\n"
            f"constexpr std::uint32_t VICTIM_FACE_R_DIM = {self.victim_face_r_dim};"
        )


@skip_for_blackhole
@parametrize(
    formats=input_output_formats(
        [
            DataFormat.Float16_b,
            DataFormat.Float16,
            DataFormat.Float32,
        ],
        same=False,
    ),
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    direction=["normal_to_tiny", "tiny_to_normal"],
    # The tiny side's face row count (the normal side is always 16 / 4 faces).
    tiny_face_r_dim=[8, 4, 2, 1],
    do_restore=[True, False],
)
def test_tilize_transition_reconfig(
    formats,
    dest_acc,
    direction,
    tiny_face_r_dim,
    do_restore,
):
    # G0 = polluter tilize geometry (run-0); G1 = target geometry the transition
    # must re-establish.
    if direction == "normal_to_tiny":
        g0_face_r_dim, g0_num_faces = REGULAR_FACE_R_DIM, REGULAR_NUM_FACES
        g1_face_r_dim, g1_num_faces = tiny_face_r_dim, TINY_NUM_FACES
    else:  # tiny_to_normal
        g0_face_r_dim, g0_num_faces = tiny_face_r_dim, TINY_NUM_FACES
        g1_face_r_dim, g1_num_faces = REGULAR_FACE_R_DIM, REGULAR_NUM_FACES

    # Stimuli are sized for the run-0 (G0) tilize input; the run-0 output is
    # discarded to scratch and the test validates registers, so the exact data
    # does not matter (no golden).
    g0_is_tiny = g0_face_r_dim < REGULAR_FACE_R_DIM
    g0_dims = [g0_face_r_dim, 32] if g0_is_tiny else [32, 32]

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=g0_dims,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=g0_dims,
        face_r_dim=g0_face_r_dim,
        num_faces=g0_num_faces,
    )

    stimuli_kwargs = dict(
        tile_count_A=tile_cnt_A,
        tile_count_B=tile_cnt_B,
        tile_count_res=tile_cnt_A,
        num_faces=g0_num_faces,
        face_r_dim=g0_face_r_dim,
    )
    if g0_is_tiny:
        # Tiny tile is dense (num_faces * face_r_dim * 16 row-major elements);
        # do NOT pad to a full 32x32 tile (matches the tiny restore test).
        pass
    else:
        # Full 32x32 tiles needed in L1 for the regular tilize input.
        stimuli_kwargs["write_full_tiles"] = True

    configuration = TestConfig(
        "sources/tilize_transition_reconfig_test.cpp",
        formats,
        templates=[
            DO_RESTORE(do_restore=do_restore),
            VICTIM_GEOMETRY(
                victim_num_faces=g1_num_faces, victim_face_r_dim=g1_face_r_dim
            ),
        ],
        runtimes=[
            NUM_FACES(g0_num_faces),
            TEST_FACE_DIMS(face_r_dim=g0_face_r_dim, face_c_dim=16),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            **stimuli_kwargs,
        ),
        dest_acc=dest_acc,
        L1_to_L1_iterations=2,
    )

    # All verification is on-device: the kernel LLK_ASSERTs that the SrcA baseline
    # matches the canonical G1 state when do_restore=True, and that it does NOT
    # match (a load-bearing leak) when do_restore=False. A failed assert surfaces
    # here as a test failure.
    configuration.run()
