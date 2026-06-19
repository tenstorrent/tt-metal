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

The kernel reads the four canonical SrcA registers back and DEVICE_PRINTs the
(expected G1 baseline, actual) pairs. Because a SrcA datacopy victim's
correctness is *entirely* a function of these registers — and Phases 1/2 already
prove "correct registers => correct data" behaviourally — reading them directly
verifies the transition for BOTH directions without needing a second operand /
golden (a single-geometry stimuli harness can't lay out two tile shapes cleanly).

`do_restore=False` is the negative control: with no transition the registers stay
in the G0 tilize state, so the host sees the G1 baseline is NOT reproduced
(Tile_x_dim / Z-dim differ), proving the transition is load-bearing.

WH-only: exercises the 3-arg `_llk_unpack_tilize_uninit_(dst, num_faces,
face_r_dim)` and tiny `face_r_dim < 16` geometry, which BH's 2-arg uninit cannot
express (matches `test_unpack_tilize_uninit_restore_tiny`).
"""

import re
from dataclasses import dataclass

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

from conftest import skip_for_blackhole, skip_for_coverage

pytestmark = skip_for_coverage

REGULAR_FACE_R_DIM = 16
REGULAR_NUM_FACES = 4
TINY_NUM_FACES = 2

# "[RISC|UNPACK|...] T1 txdim exp=256 act=64" -> ("txdim", 256, 64)
_T1_LINE_RE = re.compile(r"T1\s+(\w+)\s+exp=(-?\d+)\s+act=(-?\d+)")


def _parse_t1(lines: list[str]) -> dict[str, tuple[int, int]]:
    out: dict[str, tuple[int, int]] = {}
    for line in lines:
        m = _T1_LINE_RE.search(line)
        if m:
            out[m.group(1)] = (int(m.group(2)), int(m.group(3)))
    return out


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
        ],
        same=True,
    ),
    dest_acc=[DestAccumulation.No],
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
        requires_device_print=True,
    )

    parsed = _parse_t1(configuration.run().device_print_lines)

    keys = ("ystride", "zstride", "txdim", "zdim")
    for key in keys:
        assert key in parsed, f"missing T1 {key} device print; parsed={parsed}"

    if do_restore:
        # The transition must reproduce the canonical G1 SrcA baseline exactly.
        for key in keys:
            exp, act = parsed[key]
            assert exp == act, (
                f"transition (uninit + reconfig) failed to retarget {key} to G1: "
                f"expected(G1)={exp}, actual(register)={act} "
                f"(direction={direction}, tiny_face_r_dim={tiny_face_r_dim})"
            )
    else:
        # Negative control: without the transition the SrcA baseline is the G0
        # tilize state, so the full G1 baseline must NOT be reproduced (the
        # geometry-bearing Tile_x_dim / Z-dim differ between G0 and G1).
        all_match = all(parsed[key][0] == parsed[key][1] for key in keys)
        assert not all_match, (
            "expected a state leak without the transition, but the SrcA registers "
            f"already matched the G1 baseline: parsed={parsed} "
            f"(direction={direction}, tiny_face_r_dim={tiny_face_r_dim})"
        )
