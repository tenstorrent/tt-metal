# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Decode RoPE recomposed from compute APIs that build for Quasar, through ttnn.experimental.generic_op.

tt-blaze's RoPE uses an SFPU kernel (rope_sfpu) that exists only for Blackhole. This rebuilds it from portable APIs:
for one decode token, x holds 32 heads by head_dim = 64 (two tiles), and

    out = x * cos + rotate_half(x) * sin,  rotate_half(x) = x @ T

with cos and sin the token's row, broadcast down the heads. Per output tile j: matmul_tiles accumulates x @ T over
both k tiles into the self-loop DFB rot; mul_tiles_bcast_rows gives rot * sin into the self-loop DFB rot_sin; then
mul_tiles_bcast_rows gives x * cos in DST and add_reuse_dest adds rot_sin. The result is compared with a torch RoPE.
On Quasar the packer is programmed for one output DFB at init, so every pack to a different DFB is preceded by
pack_init for it (LLK asserts catch a pack without it).
"""

import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc

E = ttnn.experimental

CORE = ttnn.CoreCoord(0, 0)
HEADS = 32
HEAD_DIM = 64
DIM_TILES = HEAD_DIM // 32
TILE_BYTES = 2 * 32 * 32

DM_SOURCE = """
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t dim_tiles = get_arg(args::dim_tiles);
    Noc noc;
    auto load = [&](auto& dfb, const auto& tensor, uint32_t num_tiles) {
        dfb.reserve_back(num_tiles);
        for (uint32_t t = 0; t < num_tiles; ++t) {
            noc.async_read(tensor, dfb, dfb.get_entry_size(), {.page_id = t}, {.offset_bytes = t * dfb.get_entry_size()});
        }
        noc.async_read_barrier();
        dfb.push_back(num_tiles);
    };
    DataflowBuffer x(dfb::x);
    DataflowBuffer trans(dfb::trans);
    DataflowBuffer cos(dfb::cos);
    DataflowBuffer sin(dfb::sin);
    DataflowBuffer out(dfb::out);
    load(x, TensorAccessor(tensor::x), dim_tiles);
    load(trans, TensorAccessor(tensor::trans), dim_tiles * dim_tiles);
    load(cos, TensorAccessor(tensor::cos), dim_tiles);
    load(sin, TensorAccessor(tensor::sin), dim_tiles);
    const auto result = TensorAccessor(tensor::result);
    for (uint32_t j = 0; j < dim_tiles; ++j) {
        out.wait_front(1);
        noc.async_write(out, result, out.get_entry_size(), {.offset_bytes = 0}, {.page_id = j});
        noc.async_write_barrier();
        out.pop_front(1);
    }
}
"""

COMPUTE_SOURCE = """
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t dim_tiles = get_arg(args::dim_tiles);
    DataflowBuffer x(dfb::x);
    DataflowBuffer trans(dfb::trans);
    DataflowBuffer cos(dfb::cos);
    DataflowBuffer sin(dfb::sin);
    DataflowBuffer rot(dfb::rot);
    DataflowBuffer rot_sin(dfb::rot_sin);
    DataflowBuffer out(dfb::out);
    compute_kernel_hw_startup(dfb::x, dfb::trans, dfb::out);
    x.wait_front(dim_tiles);
    trans.wait_front(dim_tiles * dim_tiles);
    cos.wait_front(dim_tiles);
    sin.wait_front(dim_tiles);
    for (uint32_t j = 0; j < dim_tiles; ++j) {
        // rot = (x @ T)[:, tile j]
        matmul_init(dfb::x, dfb::trans, /*transpose=*/0);
        tile_regs_acquire();
        for (uint32_t k = 0; k < dim_tiles; ++k) {
            matmul_tiles(dfb::x, dfb::trans, k, k * dim_tiles + j, 0);
        }
        tile_regs_commit();
        rot.reserve_back(1);
        tile_regs_wait();
        pack_init(dfb::rot);  // On Quasar a pack to another output DFB needs its own pack init.
        pack_tile(0, dfb::rot);
        tile_regs_release();
        rot.push_back(1);

        // rot_sin = rot * sin (the token's sin row, broadcast down the heads)
        rot.wait_front(1);
        mul_bcast_rows_init(dfb::rot, dfb::sin);
        tile_regs_acquire();
        mul_tiles_bcast_rows(dfb::rot, dfb::sin, 0, j, 0);
        tile_regs_commit();
        rot_sin.reserve_back(1);
        tile_regs_wait();
        pack_init(dfb::rot_sin);  // On Quasar a pack to another output DFB needs its own pack init.
        pack_tile(0, dfb::rot_sin);
        tile_regs_release();
        rot_sin.push_back(1);
        rot.pop_front(1);

        // out = x * cos + rot_sin
        rot_sin.wait_front(1);
        mul_bcast_rows_init(dfb::x, dfb::cos);
        tile_regs_acquire();
        mul_tiles_bcast_rows(dfb::x, dfb::cos, j, j, 0);
        add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(dfb::rot_sin);
        add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(dfb::rot_sin, 0, 0);
        tile_regs_commit();
        out.reserve_back(1);
        tile_regs_wait();
        pack_init(dfb::out);  // On Quasar a pack to another output DFB needs its own pack init.
        pack_tile(0, dfb::out);
        tile_regs_release();
        out.push_back(1);
        rot_sin.pop_front(1);
    }
    x.pop_front(dim_tiles);
    trans.pop_front(dim_tiles * dim_tiles);
    cos.pop_front(dim_tiles);
    sin.pop_front(dim_tiles);
}
"""


def rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def rotate_half_matrix(dim):
    """T with x @ T == rotate_half(x)."""
    t = torch.zeros(dim, dim)
    half = dim // 2
    for j in range(half):
        t[j + half, j] = -1
        t[j, j + half] = 1
    return t


def test_generic_spec_op_rope_recompose(device):
    torch.manual_seed(0)
    position = 17
    inv_freq = 1.0 / (10000 ** (torch.arange(0, HEAD_DIM, 2).float() / HEAD_DIM))
    angles = position * inv_freq
    cos_row = torch.cat((angles, angles)).cos()
    sin_row = torch.cat((angles, angles)).sin()
    torch_x = torch.randn(HEADS, HEAD_DIM).bfloat16()
    expected = torch_x.float() * cos_row + rotate_half(torch_x.float()) * sin_row

    def tile_tensor(t):
        return ttnn.from_torch(t.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # cos / sin: the token's row in row 0 of each tile, which mul_tiles_bcast_rows broadcasts down the heads.
    cos_rows = torch.zeros(32, HEAD_DIM)
    cos_rows[0] = cos_row
    sin_rows = torch.zeros(32, HEAD_DIM)
    sin_rows[0] = sin_row
    tensors = {
        "x": tile_tensor(torch_x),
        "trans": tile_tensor(rotate_half_matrix(HEAD_DIM)),
        "cos": tile_tensor(cos_rows),
        "sin": tile_tensor(sin_rows),
        "result": tile_tensor(torch.zeros(HEADS, HEAD_DIM)),
    }

    def dfb(name, num_entries):
        return E.DataflowBufferSpec(
            unique_id=name, entry_size=TILE_BYTES, num_entries=num_entries, data_format_metadata=ttnn.bfloat16
        )

    inputs = {"x": DIM_TILES, "trans": DIM_TILES * DIM_TILES, "cos": DIM_TILES, "sin": DIM_TILES}
    spec = E.ProgramSpec(
        name="rope_recompose",
        kernels=[
            E.KernelSpec(
                unique_id="dm",
                source=E.KernelSpec.SourceCode(code=DM_SOURCE),
                hw_config=E.create_reader_datamovement_config(disable_dfb_implicit_sync_for_all=True),
                dfb_bindings=[E.producer_of(n, n) for n in inputs] + [E.consumer_of("out", "out")],
                tensor_bindings=[E.TensorBinding(tensor_parameter_name=t, accessor_name=t) for t in tensors],
                compile_time_args={"dim_tiles": DIM_TILES},
            ),
            E.KernelSpec(
                unique_id="compute",
                source=E.KernelSpec.SourceCode(code=COMPUTE_SOURCE),
                hw_config=E.ComputeHardwareConfig(),
                dfb_bindings=[E.consumer_of(n, n) for n in inputs]
                + [E.producer_of(n, n) for n in ("rot", "rot_sin")]
                + [E.consumer_of(n, n) for n in ("rot", "rot_sin")]
                + [E.producer_of("out", "out")],
                compile_time_args={"dim_tiles": DIM_TILES},
            ),
        ],
        dataflow_buffers=[dfb(n, e) for n, e in inputs.items()] + [dfb("rot", 1), dfb("rot_sin", 1), dfb("out", 2)],
        tensor_parameters=[E.TensorParameter(unique_id=t, spec=v.spec) for t, v in tensors.items()],
        work_units=[E.WorkUnitSpec(kernels=["dm", "compute"], target_nodes=CORE)],
    )
    E.generic_op(list(tensors.values()), spec, E.ProgramRunArgs(tensor_args=tensors))

    assert_with_pcc(expected, ttnn.to_torch(tensors["result"]).float(), 0.999)
