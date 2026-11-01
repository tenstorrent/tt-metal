# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""A tiny Tile([M, 32]) through a compute self-loop DFB, through ttnn.experimental.generic_op.

"direct": the compute kernel copies the DM-fed tile into DST and packs it to out.
"self_loop": it packs the tile into mid, a DFB the same compute kernel produces and consumes, then copies it back from
mid and packs it to out. Both must reproduce the input exactly.

Observed: exact on Quasar craq-sim with watcher + LLK asserts for rows 32, 16, 8 and 1, and on BH ttsim for rows 32
and 16. That ttsim build aborts on faces under 16 rows, so the [8, 32] and [1, 32] cases are skipped on the BH simulator.
"""

import os

import pytest
import torch
import ttnn
from tt_py_test_utils_common.utility_functions import is_blackhole

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
    DataflowBuffer out(dfb::out);
    const auto a = TensorAccessor(tensor::a);
    const auto result = TensorAccessor(tensor::result);
    in0.reserve_back(1);
    noc.async_read(a, in0, in0.get_entry_size(), {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    in0.push_back(1);
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
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

template <typename Src, typename Dst>  // DataflowBuffer is a class template on Quasar
void copy_pack(Src& src, Dst& dst, uint32_t src_id, uint32_t dst_id) {
    src.wait_front(1);
    tile_regs_acquire();
    copy_init(src_id);
    copy_tile(src_id, 0, 0);
    tile_regs_commit();
    dst.reserve_back(1);
    tile_regs_wait();
    pack_init(dst_id);
    pack_tile(0, dst_id);
    tile_regs_release();
    dst.push_back(1);
    src.pop_front(1);
}

void kernel_main() {
    DataflowBuffer in0(dfb::in0);
    DataflowBuffer out(dfb::out);
    compute_kernel_hw_startup(dfb::in0, dfb::out);
#ifdef SELF_LOOP
    DataflowBuffer mid(dfb::mid);
    copy_pack(in0, mid, dfb::in0, dfb::mid);
    copy_pack(mid, out, dfb::mid, dfb::out);
#else
    copy_pack(in0, out, dfb::in0, dfb::out);
#endif
}
"""


def run(device, rows, self_loop):
    tile = ttnn.Tile([rows, 32])
    torch.manual_seed(0)
    torch_a = torch.randn(rows, 32).bfloat16()

    def to_device(t):
        return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, tile=tile, device=device)

    tensors = {"a": to_device(torch_a), "result": to_device(torch.zeros(rows, 32).bfloat16())}
    names = ["in0", "out"] + (["mid"] if self_loop else [])
    dfbs = [
        E.DataflowBufferSpec(
            unique_id=name,
            entry_size=tile.get_tile_size(ttnn.bfloat16),
            num_entries=1,
            data_format=ttnn.bfloat16,
            tile=tile,
        )
        for name in names
    ]
    compute_bindings = [E.consumer_of("in0", "in0"), E.producer_of("out", "out")]
    if self_loop:
        compute_bindings += [E.producer_of("mid", "mid"), E.consumer_of("mid", "mid")]
    spec = E.ProgramSpec(
        name="tiny_tile_self_loop",
        kernels=[
            E.KernelSpec(
                unique_id="dm",
                source=E.KernelSpec.SourceCode(code=DM_SOURCE),
                hw_config=E.create_reader_datamovement_config(disable_dfb_implicit_sync_for_all=True),
                dfb_bindings=[E.producer_of("in0", "in0"), E.consumer_of("out", "out")],
                tensor_bindings=[E.TensorBinding(tensor_parameter_name=t, accessor_name=t) for t in tensors],
            ),
            E.KernelSpec(
                unique_id="compute",
                source=E.KernelSpec.SourceCode(code=COMPUTE_SOURCE),
                hw_config=E.ComputeHardwareConfig(),
                compiler_options=E.KernelSpec.CompilerOptions(defines={"SELF_LOOP": "1"} if self_loop else {}),
                dfb_bindings=compute_bindings,
            ),
        ],
        dataflow_buffers=dfbs,
        tensor_parameters=[E.TensorParameter(unique_id=t, spec=v.spec) for t, v in tensors.items()],
        work_units=[E.WorkUnitSpec(kernels=["dm", "compute"], target_nodes=CORE)],
    )
    E.generic_op(list(tensors.values()), spec, E.ProgramRunArgs(tensor_args=tensors))
    return torch_a.float(), ttnn.to_torch(tensors["result"]).float()


@pytest.mark.parametrize("rows", [32, 16, 8, 1])
@pytest.mark.parametrize("path", ["direct", "self_loop"])
def test_generic_spec_op_tiny_tile_self_loop(device, rows, path):
    if rows < 16 and is_blackhole() and os.environ.get("TT_METAL_SIMULATOR"):
        pytest.skip("BH ttsim aborts on faces under 16 rows (tensix_unpacr_nop: bank_clr_ctrl=1)")
    expected, result = run(device, rows, path == "self_loop")
    face_ok = [torch.equal(expected[:, c : c + 16], result[:, c : c + 16]) for c in (0, 16)]
    print(f"TINY_SELF_LOOP rows={rows} path={path} cols0-15_exact={face_ok[0]} cols16-31_exact={face_ok[1]}")
    assert all(face_ok), face_ok
