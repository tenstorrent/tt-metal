# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Direct helper-correctness check for the PR #45127 canonical-baseline helpers (C4).

`canonical_unpA_y_stride`, `canonical_unpA_z_stride`, and
`canonical_unpA_tile_x_dim_cntx` are the single source of truth that the three
restore paths (`_llk_unpack_tilize_uninit_`, `_llk_unpack_bcastA_B_uninit_`,
`_llk_unpack_reconfig_data_format_srca_impl_<FACE_ROW_MAJOR>`) write back. The
whole "recompute-canonical" restore design is only correct if those helpers
reproduce the EXACT register values that `configure_unpack_AB`
(`_llk_unpack_hw_configure_`) programs for a given (dst_format, face_r_dim,
num_faces). Every restore test exercises the helpers transitively; this test
pins them down directly across the full (format × num_faces × face_r_dim) grid.

The kernel (`unpack_canonical_baseline_check_test.cpp`) runs `hw_configure`,
reads back the four canonical SrcA registers, asserts each equals the helper
value on-device (LLK_ASSERT), and DEVICE_PRINTs the (expected, actual) pairs.
This test parses those prints and re-asserts the equality on the host, so a
helper drift is caught both ways and is diagnosable from the dumped values.

`configure_unpack_AB` computes the Z-stride and Tile_x_dim_cntx0 with its OWN
inline formulas (not via the helpers — cunpack_common.h:807/928), so the
expected/actual agreement is a genuine cross-check of two independent code
paths, not a tautology.
"""

import re

from helpers.format_config import DataFormat
from helpers.llk_params import DestAccumulation
from helpers.param_config import input_output_formats, parametrize
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import NUM_FACES, TEST_FACE_DIMS

from conftest import skip_for_coverage

pytestmark = skip_for_coverage

# "[RISC|UNPACK|...] C4 ystride exp=16 act=16" -> ("ystride", 16, 16)
_C4_LINE_RE = re.compile(r"C4\s+(\w+)\s+exp=(-?\d+)\s+act=(-?\d+)")


def _parse_c4(lines: list[str]) -> dict[str, tuple[int, int]]:
    out: dict[str, tuple[int, int]] = {}
    for line in lines:
        m = _C4_LINE_RE.search(line)
        if m:
            out[m.group(1)] = (int(m.group(2)), int(m.group(3)))
    return out


@parametrize(
    # `same=True` so the inferred SrcA dst register format is well-defined. The
    # formats span all canonical x-stride buckets: Float16->2, Float32->4, and
    # the "else->1" bucket (Float16_b and Bfp8_b). Bfp8_b is applicable here even
    # though the tilize unpacker can't read Bfp8_b input: this test only
    # `configure_unpack_AB`s the unpacker and reads the descriptor/stride
    # registers back (no actual tilize), so it directly checks the helper buckets
    # the Bfp8_b dst-format code into the 1-byte x-stride like configure does.
    formats=input_output_formats(
        [
            DataFormat.Float16_b,
            DataFormat.Float16,
            DataFormat.Float32,
            DataFormat.Bfp8_b,
        ],
        same=True,
    ),
    dest_acc=[DestAccumulation.No],
    # Tile-descriptor Z-dim restore axis.
    num_faces=[4, 2, 1],
    # Stride / Tile_x_dim axis: 16 = normal, <16 = tiny.
    face_r_dim=[16, 8, 4, 2, 1],
)
def test_unpack_canonical_baseline(
    formats,
    dest_acc,
    num_faces,
    face_r_dim,
):
    outcome = TestConfig(
        "sources/unpack_canonical_baseline_check_test.cpp",
        formats,
        runtimes=[
            NUM_FACES(num_faces),
            TEST_FACE_DIMS(face_r_dim=face_r_dim, face_c_dim=16),
        ],
        dest_acc=dest_acc,
        requires_device_print=True,  # the readback values are surfaced via DEVICE_PRINT
    ).run()

    parsed = _parse_c4(outcome.device_print_lines)

    for key in ("ystride", "zstride", "txdim", "zdim"):
        assert key in parsed, (
            f"missing C4 {key} device print; "
            f"got lines: {outcome.device_print_lines}"
        )
        exp, act = parsed[key]
        assert exp == act, (
            f"canonical helper vs configure_unpack_AB mismatch for {key}: "
            f"expected(helper)={exp}, actual(register)={act} "
            f"(num_faces={num_faces}, face_r_dim={face_r_dim})"
        )
