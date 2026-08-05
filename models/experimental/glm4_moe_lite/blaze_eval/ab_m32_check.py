# SPDX-License-Identifier: Apache-2.0
"""Does blaze's DRAMStreamingMatmul work at m=32?

Its shipped tests only cover m in {1, 4, 8}, where tile_h == m gives a non-standard tile.
The A/B wants m=32 (a standard 32x32 tile) so the ttnn side can be expressed at all. Confirm
blaze is correct there before blaming the A/B rig for a low PCC.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path("/home/ttuser/sdawle/tt-blaze/tests/blaze/micro_ops/common")))

import pytest
import ttnn

from test_dram_streaming_matmul import _run_and_compare


@pytest.mark.parametrize("m", [1, 8, 32])
def test_glm_oproj_m(device, m):
    _run_and_compare(device, k=5120, n=2048, m=m, weight_dtype=ttnn.bfloat8_b)
