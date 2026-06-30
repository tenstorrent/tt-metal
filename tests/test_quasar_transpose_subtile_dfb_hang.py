# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Repro for the craq-sim sub-tile DFB credit bug (see craq_sim_subtile_dfb_credit_bug.md).

This reproduces ON PLAIN MAIN (no experimental/quasar changes): it drives a row-major,
height-sharded WH transpose, which is exactly the resnet50-fold transpose. The transpose's
row-major reader fills the input dataflow buffer (DFB tile entry = 2048 B) from many SUB-TILE
NOC stick reads (W=224 elems * 2 B = 448 B per stick) and posts the tile credit itself with an
explicit push_back. On Quasar/craq-sim the sim auto-credits a full tile per 448 B NOC op, which
races the kernel's explicit credit:

  * cross-core sub-tile reads -> the can_post occupancy gate never clears -> the reader hart
    spins on `pc -= 4` forever (watcher waypoint NARW; with TTSIM_QSR_DFB_TRACE=1 the trace
    grows without bound), OR
  * the ring desyncs (posted >> acked) and the consumer's wait_front never advances ->
    "Device 0: Not done phys cores: ..." forever.

IMPORTANT — this test only reproduces on the Quasar simulator. Run it with slow dispatch and
forced JIT compile (fast dispatch is unsupported on Quasar):

    TT_METAL_FORCE_JIT_COMPILE=1 TT_METAL_SLOW_DISPATCH_MODE=1 \
      pytest tests/ttnn/unit_tests/operations/test_quasar_transpose_subtile_dfb_hang.py -q

Add TTSIM_QSR_DFB_TRACE=1 to emit the [qsr-rocc-issue]/[qsr-dfb-rocc-counter] credit traces;
the spin shows the same (src_coord,dst_coord,dst) read re-issued over and over.

EXPECTED RESULT TODAY: the test HANGS (does not finish) on Quasar. With a correct sub-tile
credit model in the sim, it should complete and pass PCC.

Key difference from the existing tests/.../misc/test_transpose.py::test_fold_transpose: that
test builds its input in TILE_LAYOUT, so the transpose runs the tiled (full-tile-transfer)
kernels and does NOT exercise the sub-tile stick reader. Here we force ROW_MAJOR_LAYOUT so the
row-major sub-tile reader path is used.
"""

import pytest
import torch

import ttnn

from loguru import logger
from tests.ttnn.utils_for_testing import assert_with_pcc


# Resnet50-fold transpose shape. W=224 -> 448 B row-major stick, well under the 2048 B DFB tile,
# so every reader NOC read is sub-tile -> triggers the per-op auto-credit / explicit-credit race.
@pytest.mark.parametrize(
    "input_shape",
    [
        (16, 4, 256, 224),  # the resnet50 fold transpose input
    ],
)
def test_quasar_transpose_wh_rm_sharded_subtile_dfb_hang(device, input_shape):
    torch.manual_seed(2005)
    N, C, H, W = input_shape

    # Height-shard the row-major input across the device grid (same as the fold).
    compute_grid_size = device.compute_with_storage_grid_size()
    num_cores = min(N, compute_grid_size.x * compute_grid_size.y)
    shard_grid = ttnn.num_cores_to_corerangeset(num_cores, compute_grid_size, True)
    sharded_config = ttnn.create_sharded_memory_config_(
        input_shape,
        shard_grid,
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.ShardOrientation.ROW_MAJOR,
    )

    x = torch.rand(input_shape).bfloat16().float()

    # ROW_MAJOR_LAYOUT is what selects the sub-tile stick reader (the buggy path).
    ttnn_input = ttnn.from_torch(
        x,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        device=device,
        memory_config=sharded_config,
    )

    logger.info("Issuing WH row-major sharded transpose(2,3) — expected to hang on craq-sim today")
    xtt = ttnn.experimental.quasar.transpose(ttnn_input, 2, 3)  # WH transpose

    tt_out = ttnn.to_torch(xtt.cpu().to(ttnn.ROW_MAJOR_LAYOUT))
    ref = x.transpose(2, 3)
    assert_with_pcc(ref, tt_out, 0.9999)
