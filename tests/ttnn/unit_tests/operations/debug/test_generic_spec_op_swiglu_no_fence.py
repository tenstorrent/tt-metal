# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""tt-blaze's matmul_swiglu without its FPU_SFPU fence, through ttnn.experimental.generic_op.

On Blackhole, blaze runs SiLU on the gate tiles from the PACK thread while MATH computes the up matmul, ordered by the
FPU_SFPU Tensix semaphore. Quasar has no such semaphore (its one compute semaphore pairs UNPACK and PACK). This is the
portable form: in one DEST section, MATH computes the gate tiles, applies silu_tile to them, computes the up tiles
next to them and multiplies the two with mul_binary_tile, so MATH's own order replaces the fence. The result,
silu(a @ w_gate) * (a @ w_up) for an activation tile and OUT_W output tiles, is compared with torch.
"""

import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc

E = ttnn.experimental

CORE = ttnn.CoreCoord(0, 0)
OUT_W = 2  # output tiles
TILE_BYTES = 2 * 32 * 32

DM_SOURCE = """
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t out_w = get_arg(args::out_w);
    Noc noc;
    auto load = [&](auto& dfb, const auto& tensor, uint32_t num_tiles) {
        dfb.reserve_back(num_tiles);
        for (uint32_t t = 0; t < num_tiles; ++t) {
            noc.async_read(tensor, dfb, dfb.get_entry_size(), {.page_id = t}, {.offset_bytes = t * dfb.get_entry_size()});
        }
        noc.async_read_barrier();
        dfb.push_back(num_tiles);
    };
    DataflowBuffer act(dfb::act);
    DataflowBuffer gate(dfb::gate);
    DataflowBuffer up(dfb::up);
    DataflowBuffer out(dfb::out);
    load(act, TensorAccessor(tensor::act), 1);
    load(gate, TensorAccessor(tensor::gate), out_w);
    load(up, TensorAccessor(tensor::up), out_w);
    const auto result = TensorAccessor(tensor::result);
    out.wait_front(out_w);
    for (uint32_t t = 0; t < out_w; ++t) {
        noc.async_write(out, result, out.get_entry_size(), {.offset_bytes = t * out.get_entry_size()}, {.page_id = t});
    }
    noc.async_write_barrier();
    out.pop_front(out_w);
}
"""

COMPUTE_SOURCE = """
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t out_w = get_arg(args::out_w);
    DataflowBuffer act(dfb::act);
    DataflowBuffer gate(dfb::gate);
    DataflowBuffer up(dfb::up);
    DataflowBuffer out(dfb::out);
    compute_kernel_hw_startup(dfb::act, dfb::gate, dfb::out);
    act.wait_front(1);
    gate.wait_front(out_w);
    up.wait_front(out_w);

    tile_regs_acquire();
    // Gate tiles into DST [0, out_w), then SiLU on them: MATH issues both, so the SFPU reads finished matmul results.
    matmul_init(dfb::act, dfb::gate, /*transpose=*/0);
    for (uint32_t i = 0; i < out_w; ++i) {
        matmul_tiles(dfb::act, dfb::gate, 0, i, i);
    }
    silu_tile_init();
    for (uint32_t i = 0; i < out_w; ++i) {
        silu_tile(i);
    }
    // Up tiles into DST [out_w, 2 * out_w), then gate *= up.
    matmul_init(dfb::act, dfb::up, /*transpose=*/0);
    for (uint32_t i = 0; i < out_w; ++i) {
        matmul_tiles(dfb::act, dfb::up, 0, i, out_w + i);
    }
    mul_binary_tile_init();
    for (uint32_t i = 0; i < out_w; ++i) {
        mul_binary_tile(i, out_w + i, i);
    }
    tile_regs_commit();

    out.reserve_back(out_w);
    tile_regs_wait();
    for (uint32_t i = 0; i < out_w; ++i) {
        pack_tile(i, dfb::out);
    }
    tile_regs_release();
    out.push_back(out_w);

    act.pop_front(1);
    gate.pop_front(out_w);
    up.pop_front(out_w);
}
"""


def test_generic_spec_op_swiglu_without_fence(device):
    torch.manual_seed(0)
    torch_act = torch.randn(32, 32).bfloat16()
    torch_gate = (torch.randn(32, 32 * OUT_W) / 4).bfloat16()
    torch_up = (torch.randn(32, 32 * OUT_W) / 4).bfloat16()
    expected = torch.nn.functional.silu(torch_act.float() @ torch_gate.float()) * (torch_act.float() @ torch_up.float())

    def tile_tensor(t):
        return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    tensors = {
        "act": tile_tensor(torch_act),
        "gate": tile_tensor(torch_gate),
        "up": tile_tensor(torch_up),
        "result": tile_tensor(torch.zeros(32, 32 * OUT_W).bfloat16()),
    }
    entries = {"act": 1, "gate": OUT_W, "up": OUT_W, "out": OUT_W}
    inputs = ["act", "gate", "up"]
    spec = E.ProgramSpec(
        name="swiglu_no_fence",
        kernels=[
            E.KernelSpec(
                unique_id="dm",
                source=E.KernelSpec.SourceCode(code=DM_SOURCE),
                hw_config=E.create_reader_datamovement_config(disable_dfb_implicit_sync_for_all=True),
                dfb_bindings=[E.producer_of(n, n) for n in inputs] + [E.consumer_of("out", "out")],
                tensor_bindings=[E.TensorBinding(tensor_parameter_name=t, accessor_name=t) for t in tensors],
                compile_time_args={"out_w": OUT_W},
            ),
            E.KernelSpec(
                unique_id="compute",
                source=E.KernelSpec.SourceCode(code=COMPUTE_SOURCE),
                hw_config=E.ComputeHardwareConfig(),
                dfb_bindings=[E.consumer_of(n, n) for n in inputs] + [E.producer_of("out", "out")],
                compile_time_args={"out_w": OUT_W},
            ),
        ],
        dataflow_buffers=[
            E.DataflowBufferSpec(unique_id=n, entry_size=TILE_BYTES, num_entries=e, data_format_metadata=ttnn.bfloat16)
            for n, e in entries.items()
        ],
        tensor_parameters=[E.TensorParameter(unique_id=t, spec=v.spec) for t, v in tensors.items()],
        work_units=[E.WorkUnitSpec(kernels=["dm", "compute"], target_nodes=CORE)],
    )
    E.generic_op(list(tensors.values()), spec, E.ProgramRunArgs(tensor_args=tensors))

    assert_with_pcc(expected, ttnn.to_torch(tensors["result"]).float(), 0.99)
