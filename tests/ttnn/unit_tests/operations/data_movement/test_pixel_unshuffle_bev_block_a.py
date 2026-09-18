# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# ttnn.pixel_unshuffle — golden verification for every distinct BEV op configuration.
#
# The full BEV model issues ten pixel_unshuffle calls but only FOUR distinct programs
# (the ops perf report shows exactly four PROGRAM HASH values). Cameras 1-4 are the
# 1536x1536 deformed cameras and are byte-identical to one another; camera 5 is the
# larger 1280x2304 cylinder camera. Each camera contributes an r=4 call followed by an
# r=2 call, so the ten calls are 4x config 1, 4x config 2, 1x config 3, 1x config 4.
#
#  cfg | camera | in shape            | r | out shape          | calls | kernel ns (pre-opt)
#  ----|--------|---------------------|---|--------------------|-------|--------------------
#   1  |  1-4   | 1 x 1 x 1536 x 1536 | 4 | 1 x 16 x 384 x 384 |   4   | 666,436 (mean 669,116)
#   2  |  1-4   | 1 x 2 x  768 x  768 | 2 | 1 x  8 x 384 x 384 |   4   | 334,623 (mean 334,560)
#   3  |   5    | 1 x 1 x 1280 x 2304 | 4 | 1 x 16 x 320 x 576 |   1   | 820,179
#   4  |   5    | 1 x 2 x  640 x 1152 | 2 | 1 x  8 x 320 x 576 |   1   | 410,577
#
# All four: BFLOAT16, ROW_MAJOR, DRAM-interleaved input -> L1-interleaved output,
# SPATIAL_MAJOR, 64 cores. Configs 1 and 2 are also exactly the two block A
# (single camera) calls, which together were ~1.0 ms, 14.08% of block A inference.
# Pre-opt the ten calls totalled 5,245,462 ns, 9.01% of the 58.239 ms inference.
#
# Sources: docs/pixel_unshuffle_op_details_full_bev_model.md (scoped indices 11, 24,
# 1130, 1145) and docs/pixel_unshuffle_op_details_block_a_single_camera.md.
#
# pixel_unshuffle is a pure gather — it moves elements and never computes on them — so
# the device result must be **bit-exact** against the golden. The assertion is exact
# equality, not PCC: for a kernel rewrite any numeric change is a bug, not a tolerance
# question.
#
# Golden (reshape -> permute -> reshape), SPATIAL_MAJOR ordering:
#   x: [N, C, H, W] -> reshape [N, C, H/r, r, W/r, r]
#   permute (0,3,5,1,2,4) -> reshape [N, C*r^2, H/r, W/r]
#   c_out = rh*(r*C) + rw*C + c_in        (matches ONNX SpaceToDepth)

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_equal

try:  # signposts are only meaningful under `python -m tracy`; keep the test standalone-safe
    from tracy import signpost
except ImportError:  # pragma: no cover

    def signpost(header, message=None):
        pass


SPATIAL_MAJOR = ttnn.PixelUnshuffleChannelOrder.SPATIAL_MAJOR


def golden_pixel_unshuffle_spatial_major(x, r):
    """reshape -> permute -> reshape reference, SPATIAL_MAJOR ordering.

    dtype-preserving: the golden of a bfloat16 input is exactly representable in
    bfloat16, so it can be compared bit-for-bit against the device result.
    """
    N, C, H, W = x.shape
    t = x.reshape(N, C, H // r, r, W // r, r)  # [N, C, h_out, rh, w_out, rw]
    t = t.permute(0, 3, 5, 1, 2, 4).contiguous()  # [N, rh, rw, C, h_out, w_out]
    return t.reshape(N, C * r * r, H // r, W // r)


@pytest.mark.parametrize(
    "N,C,H,W,r",
    [
        # cameras 1-4 (deformed), 4 calls each in the full model
        pytest.param(1, 1, 1536, 1536, 4, id="cfg1_cam1to4_r4_C1_1536x1536"),
        pytest.param(1, 2, 768, 768, 2, id="cfg2_cam1to4_r2_C2_768x768"),
        # camera 5 (cylinder), 1 call each
        pytest.param(1, 1, 1280, 2304, 4, id="cfg3_cam5_r4_C1_1280x2304"),
        pytest.param(1, 2, 640, 1152, 2, id="cfg4_cam5_r2_C2_640x1152"),
    ],
)
def test_pixel_unshuffle_bev_block_a(device, N, C, H, W, r):
    tag = f"pixel_unshuffle_r{r}_C{C}_{H}x{W}"
    torch.manual_seed(1234)
    x = torch.randn(N, C, H, W, dtype=torch.bfloat16)
    golden = golden_pixel_unshuffle_spatial_major(x, r)

    tt_in = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    # Signposts bracket only the op under measurement, so the ops perf report can be
    # scoped to it and compared against the pre-optimisation numbers.
    signpost(f"{tag}-start")
    tt_out = ttnn.pixel_unshuffle(
        tt_in,
        downscale_factor=r,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        channel_order=SPATIAL_MAJOR,
    )
    signpost(f"{tag}-end")

    # Output contract, against the IR's declared output operand
    assert list(tt_out.shape) == [N, C * r * r, H // r, W // r], f"shape {list(tt_out.shape)}"
    assert tt_out.dtype == ttnn.bfloat16, f"dtype {tt_out.dtype}"
    assert tt_out.layout == ttnn.ROW_MAJOR_LAYOUT, f"layout {tt_out.layout}"
    assert tt_out.memory_config().buffer_type == ttnn.BufferType.L1, "output must land in L1"

    assert_equal(golden, ttnn.to_torch(tt_out))
