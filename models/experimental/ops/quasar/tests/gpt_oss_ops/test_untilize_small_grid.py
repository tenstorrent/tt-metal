# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# ---------------------------------------------------------------------------
# MANUAL companion to the generated test_untilize.py — do NOT regenerate this file
# from a capture.
#
# The captured call untilizes a [1, 1, 32, 262144] bfloat8_b tensor: one tile row of 8192 tiles,
# all handled by one core on the 2-node Quasar emulator, where the reader does one NoC read per
# tile. That is slow, and more so with the MX formats (the captured bfloat8_b runs as MX under
# TTNN_GRAPH_OPS_BF8_AS_MXFP8x). The full-width case is left out of the ``-m emulator`` subset
# (conftest.py, ``_fits_emulator``) and still runs outside it; the cases below stand in for it inside
# the subset.
#
# What the capture exercises is the *block* untilize factory (``UntilizeMultiCoreBlockProgramFactory``:
# reader_unary_interleaved_wh_multicore, untilize_wh, writer_unary_stick_layout_wh_multicore). A
# non-sharded input reaches it two ways (untilize_device_operation.cpp, select_program_factory):
#   * ``enough_space_height`` is false: one row of input + output CBs does not fit the L1 left for
#     a program (untilize.cpp: (input tile + output tile) * tiles_per_row >= available L1). This is
#     the capture's route, and the only one on a 2-core grid;
#   * a row over 32 tiles that splits over more cores than a plain row split. On a 2-core grid that
#     never happens (the split is one block per row), so a narrow row falls to the plain multicore
#     factory and does not exercise these kernels.
#
# So the row has to be wide enough to overflow L1, and no wider. A bfloat8_b/MX tile is 1056 B in and
# a bf16 tile 2048 B out: 3104 B per tile against the 3,860,928 B bank of a 4 MB part is 1244
# tiles; the 3 MB part (2,812,352 B) needs 907. 1248 tiles (39936 columns) is the narrowest row that
# overflows both, and it is 6.5x narrower than the capture. The bf16 case needs 4096 B per tile,
# so 1248 tiles overflows it too.
#
#   00  captured dtype   (bfloat8_b; MX under TTNN_GRAPH_OPS_BF8_AS_MXFP8x, bf16 under
#                         TTNN_GRAPH_OPS_BF8_AS_BF16)
#   01  the same shape in bfloat16 (baseline: compare its time against 00)
#
# Check ``kernel_names.txt`` of a watcher run: reader_unary_interleaved_wh_multicore_metal2.cpp /
# untilize_wh_metal2.cpp mean the block factory ran; reader_unary_start_id_metal2.cpp /
# untilize_variable_num_blocks_metal2.cpp mean the width fell short.
# ---------------------------------------------------------------------------
"""Small (2-node) variants of ``ttnn.untilize``: the captured wide-row call cut to 1248 tiles per row."""

import pytest

import ttnn
from models.experimental.ops.quasar.tests.gpt_oss_ops import graph_case as G

_OP = ttnn.untilize

_DRAM = {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None}


def _case(case_id, shape, dtype):
    return {
        "id": case_id,
        "op": "ttnn.untilize",
        "count": 1,
        "args": [{"k": "t", "shape": shape, "dtype": dtype, "layout": "TILE", "mem": dict(_DRAM)}],
        "kwargs": {"use_multicore": {"k": "lit", "v": True}},
        "outs": [None],
    }


_WIDTH = 39936  # 1248 tiles: the narrowest row whose input + output CBs overflow L1 on both Quasar L1 sizes

CASES = [
    _case(f"00_32x{_WIDTH}_bf8_int-dram", [1, 1, 32, _WIDTH], "BFLOAT8_B"),
    _case(f"01_32x{_WIDTH}_bf16_int-dram", [1, 1, 32, _WIDTH], "BFLOAT16"),
]


# 1248 tiles on one core takes minutes on the emulator, and longer with the watcher on; the repo default
# (pytest.ini: 300 s) timed the bf16 baseline out under the watcher. The marker overrides the ini and any --timeout.
@pytest.mark.timeout(1200)
@G.with_default_mesh()
@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_untilize_small_grid(ttnn_mesh_device, reset_seeds, case):
    G.run_case(_OP, case, ttnn_mesh_device)
