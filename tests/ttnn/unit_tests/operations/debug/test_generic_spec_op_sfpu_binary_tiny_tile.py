# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""SFPU binary ops (add/sub/mul_binary_tile) on tiny Tile([M, 32]) tiles, through ttnn.experimental.generic_op.

The compute kernel copies tile 0 of a into DST 0 and tile 0 of b into DST B_DST, then runs op_binary_tile(0, 1, 0) and
packs DST 0. On Quasar a tiny-tile copy places DST tile i at row i * max(16, tile rows) (_set_dst_write_addr_by_rows_),
but calculate_sfpu_binary computes operand offsets for its TILE_SHAPE, which the compute API leaves at Tile32x32
(64 rows per DST index). So with B_DST = 1 the SFPU reads its second operand from the wrong rows. The result is wrong
with no assert. With B_DST = 64 / max(16, tile rows) both views agree, which confirms the cause.

Observed on Quasar craq-sim with watcher + LLK asserts, B_DST = 1, rows 16, 8 and 1:
- add and sub return exactly a (the SFPU's second operand reads as zero).
- mul returns PCC 0.
- full_tile_offset is exact (PCC 0.99999) for all three ops.
On BH ttsim, B_DST = 1 is exact at rows 32 and 16. That ttsim build aborts on any tile with fewer than 16 rows per face
("UnsupportedFunctionality: tensix_unpacr_nop: bank_clr_ctrl=1"), so those cases are skipped on the BH simulator.
"""

import os

import pytest
import torch
import ttnn
from tt_py_test_utils_common.utility_functions import is_blackhole, is_quasar

from tests.ttnn.utils_for_testing import comp_pcc

E = ttnn.experimental

CORE = ttnn.CoreCoord(0, 0)

DM_SOURCE = """
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    Noc noc;
    DataflowBuffer in0(dfb::in0);
    DataflowBuffer in1(dfb::in1);
    DataflowBuffer out(dfb::out);
    const auto a = TensorAccessor(tensor::a);
    const auto b = TensorAccessor(tensor::b);
    const auto result = TensorAccessor(tensor::result);
    in0.reserve_back(1);
    in1.reserve_back(1);
    noc.async_read(a, in0, in0.get_entry_size(), {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read(b, in1, in1.get_entry_size(), {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    in0.push_back(1);
    in1.push_back(1);
    out.wait_front(1);
    noc.async_write(out, result, out.get_entry_size(), {.offset_bytes = 0}, {.page_id = 0});
    noc.async_write_barrier();
    out.pop_front(1);
}
"""

COMPUTE_SOURCE = """
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t b_dst = get_arg(args::b_dst);
    DataflowBuffer in0(dfb::in0);
    DataflowBuffer in1(dfb::in1);
    DataflowBuffer out(dfb::out);
    compute_kernel_hw_startup(dfb::in0, dfb::in1, dfb::out);
    in0.wait_front(1);
    in1.wait_front(1);
    tile_regs_acquire();
    copy_init(dfb::in0);
    copy_tile(dfb::in0, 0, 0);
    copy_init(dfb::in1);
    copy_tile(dfb::in1, 0, b_dst);
    OP_binary_tile_init();
    OP_binary_tile(0, 1, 0);
    tile_regs_commit();
    out.reserve_back(1);
    tile_regs_wait();
    pack_tile(0, dfb::out);
    tile_regs_release();
    out.push_back(1);
    in0.pop_front(1);
    in1.pop_front(1);
}
"""

GOLDEN = {"add": torch.add, "sub": torch.sub, "mul": torch.mul}


def run(device, op, rows, b_dst):
    tile = ttnn.Tile([rows, 32])
    torch.manual_seed(0)
    torch_a = torch.randn(rows, 32).bfloat16()
    torch_b = torch.randn(rows, 32).bfloat16()

    def to_device(t):
        return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, tile=tile, device=device)

    tensors = {"a": to_device(torch_a), "b": to_device(torch_b), "result": to_device(torch.zeros(rows, 32).bfloat16())}
    dfbs = [
        E.DataflowBufferSpec(
            unique_id=name,
            entry_size=tile.get_tile_size(ttnn.bfloat16),
            num_entries=1,
            data_format=ttnn.bfloat16,
            tile=tile,
        )
        for name in ("in0", "in1", "out")
    ]
    spec = E.ProgramSpec(
        name="sfpu_binary_tiny_tile",
        kernels=[
            E.KernelSpec(
                unique_id="dm",
                source=E.KernelSpec.SourceCode(code=DM_SOURCE),
                hw_config=E.create_reader_datamovement_config(disable_dfb_implicit_sync_for_all=True),
                dfb_bindings=[E.producer_of(n, n) for n in ("in0", "in1")] + [E.consumer_of("out", "out")],
                tensor_bindings=[E.TensorBinding(tensor_parameter_name=t, accessor_name=t) for t in tensors],
            ),
            E.KernelSpec(
                unique_id="compute",
                source=E.KernelSpec.SourceCode(code=COMPUTE_SOURCE.replace("OP_", f"{op}_")),
                hw_config=E.ComputeHardwareConfig(),
                dfb_bindings=[E.consumer_of(n, n) for n in ("in0", "in1")] + [E.producer_of("out", "out")],
                compile_time_args={"b_dst": b_dst},
            ),
        ],
        dataflow_buffers=dfbs,
        tensor_parameters=[E.TensorParameter(unique_id=t, spec=v.spec) for t, v in tensors.items()],
        work_units=[E.WorkUnitSpec(kernels=["dm", "compute"], target_nodes=CORE)],
    )
    E.generic_op(list(tensors.values()), spec, E.ProgramRunArgs(tensor_args=tensors))
    expected = GOLDEN[op](torch_a.float(), torch_b.float())
    result = ttnn.to_torch(tensors["result"]).float()
    _, pcc = comp_pcc(expected, result, 0.999)
    return pcc, result, torch_a.float()


@pytest.mark.parametrize("op", ["add", "sub", "mul"])
@pytest.mark.parametrize("rows", [32, 16, 8, 1])
@pytest.mark.parametrize("b_at", ["dst1", "full_tile_offset"])
def test_generic_spec_op_sfpu_binary_tiny_tile(device, op, rows, b_at):
    dst_rows = max(16, rows * 2)  # DST rows of a two-face Tile([rows, 32]) copy: at least one 16-row face
    b_dst = 1 if b_at == "dst1" else 64 // dst_rows
    if b_at == "full_tile_offset" and (b_dst == 1 or not is_quasar()):
        pytest.skip("The full-tile DST offset is a Quasar-only workaround")
    if rows < 16 and is_blackhole() and os.environ.get("TT_METAL_SIMULATOR"):
        pytest.skip("BH ttsim aborts on faces under 16 rows (tensix_unpacr_nop: bank_clr_ctrl=1)")
    pcc, result, a = run(device, op, rows, b_dst)
    print(f"SFPU_BINARY op={op} rows={rows} b_dst={b_dst} pcc={pcc} equals_a={torch.equal(result, a)}")
    if is_quasar() and rows < 32 and b_at == "dst1":
        # Silent: the second operand is read from the wrong DST rows (module docstring).
        if op == "mul":
            assert pcc < 0.5, pcc
        else:
            assert torch.equal(result, a), pcc
    else:
        assert pcc > 0.999, pcc
