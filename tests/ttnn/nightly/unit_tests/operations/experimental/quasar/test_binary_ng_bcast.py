# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Subtile broadcast for the Quasar binary_ng DFB path (exercises unary_bcast).

Black-box: same op/golden/PCC as the no-broadcast suite; only the input shapes select the broadcast
type. This first slice drives ROW subtile broadcast (unary_bcast<BroadcastType::ROW>) end-to-end on the
DFB path: reader delivers the partial tile, the compute broadcasts the single valid row across the tile
via the intermediate llk_post DFB, then the binary op runs.

Shape convention: a ROW broadcast requires the broadcasting operand's LOGICAL height (dim[-2]) to be 1
(get_subtile_broadcast_type keys off logical dim[-2]==1 -> ROW_A / ROW_B). That single logical row
tilizes into row 0 of a 32-row tile and unary_bcast<ROW> replicates it across the tile, matching torch's
[1,W] -> [H,W] broadcast. The full operand is [2,2,64,128] = 2x2 batch/channel x 2 tile-rows x 4 tile-cols
(32 tiles), so the case also exercises OUTER broadcast across N and C (the broadcasting operand's N=C=1
reused across batch/channel) combined with the subtile ROW fill -- more representative than a [1,1,H,W]
full operand, which leaves the outer dims trivial. The broadcasting operand is [1,1,1,128] = one (padded)
tile-row x 4 tile-cols.

Run on the Quasar simulator:
    unset TT_METAL_DISABLE_SFPLOADMACRO
    TT_METAL_SIMULATOR=<path>/libttsim.so TT_SIMULATOR_LOCALHOST=1 ARCH_NAME=quasar CHIP_ARCH=quasar \
        TT_METAL_SLOW_DISPATCH_MODE=1 \
        pytest tests/ttnn/nightly/unit_tests/operations/experimental/quasar/test_binary_ng_bcast.py
"""
import pytest
import ttnn
from tests.ttnn.nightly.unit_tests.operations.experimental.quasar.binary_ng_quasar_test_utils import _run


# a is the full [H,W]; the other operand broadcasts. ROW_B: b has one (logical) row. ROW_A: a has one.
# subtract (non-commutative) is included alongside add so an lhs/rhs (BCAST_INPUT) operand swap would
# flip the sign and fail PCC -- add alone cannot catch it. add/subtract are bf16-FPU and use the FPU ROW
# compute kernel; multiply/divide/maximum are bf16-SFPU (is_binary_sfpu_op) and drive the SFPU ROW compute
# kernel (eltwise_binary_sfpu_row_bcast_dfb.cpp) -- the first SFPU consumer of the Quasar unary_bcast
# primitive. maximum (always-SFPU, no FPU form; ckernel binary_max_min.h is Quasar-ported) proves the
# widened gate admits the generic SFPU-ROW path beyond mul/div. divide (non-commutative) also guards
# against an operand swap, and its reciprocal-approx bf16 PCC uses the same relaxed 0.99 threshold as the
# no-broadcast divide sweep; add/subtract/multiply/maximum keep the default (standard bf16 PCC).
@pytest.mark.parametrize("op_name", ["add", "subtract", "multiply", "divide", "maximum"])
@pytest.mark.parametrize(
    "a_shape,b_shape,bcast",
    [
        ([2, 2, 64, 128], [1, 1, 1, 128], "ROW_B"),  # b: single row -> broadcasts down
        ([1, 1, 1, 128], [2, 2, 64, 128], "ROW_A"),  # a: single row -> broadcasts down
    ],
)
def test_bcast_row_interleaved(device, op_name, a_shape, b_shape, bcast):
    pcc = 0.99 if op_name == "divide" else None
    _run(device, op_name, ttnn.DRAM_MEMORY_CONFIG, ttnn.bfloat16, (a_shape, b_shape), pcc=pcc)
