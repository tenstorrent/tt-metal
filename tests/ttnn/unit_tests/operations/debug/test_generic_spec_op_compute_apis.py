# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Numerics of the compute APIs behind tt-blaze's next Quasar ports, through ttnn.experimental.generic_op.

Each case runs on one core with one bf16 tile per input: a data movement kernel feeds tile 0 of a (and of b) into
in0 (and in1) and writes the out tile back, and the compute kernel applies one API with the template arguments
tt-blaze uses, then packs DST 0 to out. The result is compared with a torch golden. A compile probe only shows an API
builds for Quasar; these show it computes the right thing.
"""

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc

E = ttnn.experimental

CORE = ttnn.CoreCoord(0, 0)
TILE_BYTES = 2 * 32 * 32

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
#ifdef USES_IN1
    DataflowBuffer in1(dfb::in1);
    const auto b = TensorAccessor(tensor::b);
    in1.reserve_back(1);
    noc.async_read(b, in1, in1.get_entry_size(), {.page_id = 0}, {.offset_bytes = 0});
#endif
    noc.async_read_barrier();
    in0.push_back(1);
#ifdef USES_IN1
    in1.push_back(1);
#endif
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
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/pack.h"
#include "api/compute/reduce.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    DataflowBuffer in0(dfb::in0);
    DataflowBuffer out(dfb::out);
#ifdef USES_IN1
    DataflowBuffer in1(dfb::in1);
    compute_kernel_hw_startup(dfb::in0, dfb::in1, dfb::out);
    in1.wait_front(1);
#else
    compute_kernel_hw_startup(dfb::in0, dfb::out);
#endif
    in0.wait_front(1);
    tile_regs_acquire();
    BODY
    tile_regs_commit();
    out.reserve_back(1);
    tile_regs_wait();
    pack_tile(0, dfb::out);
    tile_regs_release();
    out.push_back(1);
    in0.pop_front(1);
#ifdef USES_IN1
    in1.pop_front(1);
#endif
}
"""

TWO = "0x40000000u"  # 2.0f
COPY_A = "copy_init(dfb::in0); copy_tile(dfb::in0, 0, 0);"


def col0(t):
    return t[:, :1]


def row0(t):
    return t[:1, :]


# name: (body, uses b, golden(a, b) -> expected, compare(t) -> the part of the tile the API defines, b maker)
CASES = {
    "add_tiles": (
        "binary_tiles_init<true, EltwiseBinaryType::ELWADD>(dfb::in0, dfb::in1); add_tiles(dfb::in0, dfb::in1, 0, 0, 0);",
        True,
        lambda a, b: a + b,
        None,
        None,
    ),
    "mul_tiles": (
        "binary_tiles_init<true, EltwiseBinaryType::ELWMUL>(dfb::in0, dfb::in1); mul_tiles(dfb::in0, dfb::in1, 0, 0, 0);",
        True,
        lambda a, b: a * b,
        None,
        None,
    ),
    "add_reuse_dest(DEST_TO_SRCB)": (
        COPY_A + " add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(dfb::in1);"
        " add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(dfb::in1, 0, 0);",
        True,
        lambda a, b: a + b,
        None,
        None,
    ),
    "mul_unary_tile(binop_with_scalar)": (
        COPY_A + f" binop_with_scalar_tile_init(); mul_unary_tile(0, {TWO});",
        False,
        lambda a, b: a * 2,
        None,
        None,
    ),
    "rsqrt_tile": (
        COPY_A + " rsqrt_tile_init(); rsqrt_tile(0);",
        False,
        lambda a, b: torch.rsqrt(a),
        None,
        "positive",
    ),
    "silu_tile": (
        COPY_A + " silu_tile_init(); silu_tile(0);",
        False,
        lambda a, b: torch.nn.functional.silu(a),
        None,
        None,
    ),
    "mul_binary_tile(sfpu)": (
        COPY_A + " copy_init(dfb::in1); copy_tile(dfb::in1, 0, 1); mul_binary_tile_init(); mul_binary_tile(0, 1, 0);",
        True,
        lambda a, b: a * b,
        None,
        None,
    ),
    "reduce SUM ROW": (
        "reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(dfb::in0, dfb::in1, dfb::out);"
        " reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(dfb::in0, dfb::in1, 0, 0, 0); reduce_uninit();",
        True,
        lambda a, b: a.sum(dim=1, keepdim=True),
        col0,
        "ones",
    ),
    "reduce MAX ROW": (
        "reduce_init<PoolType::MAX, ReduceDim::REDUCE_ROW>(dfb::in0, dfb::in1, dfb::out);"
        " reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW>(dfb::in0, dfb::in1, 0, 0, 0); reduce_uninit();",
        True,
        lambda a, b: a.max(dim=1, keepdim=True).values,
        col0,
        "ones",
    ),
    "reduce SUM COL": (
        "reduce_init<PoolType::SUM, ReduceDim::REDUCE_COL>(dfb::in0, dfb::in1, dfb::out);"
        " reduce_tile<PoolType::SUM, ReduceDim::REDUCE_COL>(dfb::in0, dfb::in1, 0, 0, 0); reduce_uninit();",
        True,
        lambda a, b: a.sum(dim=0, keepdim=True),
        row0,
        "ones",
    ),
    "reduce MAX COL": (
        "reduce_init<PoolType::MAX, ReduceDim::REDUCE_COL>(dfb::in0, dfb::in1, dfb::out);"
        " reduce_tile<PoolType::MAX, ReduceDim::REDUCE_COL>(dfb::in0, dfb::in1, 0, 0, 0); reduce_uninit();",
        True,
        lambda a, b: a.max(dim=0, keepdim=True).values,
        row0,
        "ones",
    ),
    "mul_tiles_bcast SCALAR": (
        "mul_bcast_scalar_init(dfb::in0, dfb::in1);"
        " mul_tiles_bcast<BroadcastType::SCALAR>(dfb::in0, dfb::in1, 0, 0, 0);",
        True,
        lambda a, b: a * b[0, 0],
        None,
        None,
    ),
    "mul_tiles_bcast_cols": (
        "mul_bcast_cols_init(dfb::in0, dfb::in1); mul_tiles_bcast_cols(dfb::in0, dfb::in1, 0, 0, 0);",
        True,
        lambda a, b: a * b[:, :1],
        None,
        None,
    ),
    "mul_tiles_bcast_rows": (
        "mul_bcast_rows_init(dfb::in0, dfb::in1); mul_tiles_bcast_rows(dfb::in0, dfb::in1, 0, 0, 0);",
        True,
        lambda a, b: a * b[:1, :],
        None,
        None,
    ),
    "add_tiles_bcast_rows": (
        "add_bcast_rows_init(dfb::in0, dfb::in1); add_tiles_bcast_rows(dfb::in0, dfb::in1, 0, 0, 0);",
        True,
        lambda a, b: a + b[:1, :],
        None,
        None,
    ),
}


def tile_tensor(device, t):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)


def make_b(kind, shape):
    if kind == "ones":
        return torch.ones(shape).bfloat16()
    return torch.randn(shape).bfloat16()


@pytest.mark.parametrize("name", list(CASES))
def test_generic_spec_op_compute_api(device, name):
    body, uses_b, golden, part, kind = CASES[name]
    torch.manual_seed(0)
    torch_a = torch.randn(32, 32).bfloat16()
    if kind == "positive":
        torch_a = torch_a.abs() + 0.5
    torch_b = make_b(kind, (32, 32))
    tensors = {"a": tile_tensor(device, torch_a)}
    if uses_b:
        tensors["b"] = tile_tensor(device, torch_b)
    tensors["result"] = tile_tensor(device, torch.zeros(32, 32).bfloat16())

    defines = {"USES_IN1": "1"} if uses_b else {}
    names = ["in0", "in1"] if uses_b else ["in0"]
    dfbs = [
        E.DataflowBufferSpec(unique_id=n, entry_size=TILE_BYTES, num_entries=1, data_format_metadata=ttnn.bfloat16)
        for n in names + ["out"]
    ]
    spec = E.ProgramSpec(
        name="compute_api",
        kernels=[
            E.KernelSpec(
                unique_id="dm",
                source=E.KernelSpec.SourceCode(code=DM_SOURCE),
                hw_config=E.create_reader_datamovement_config(disable_dfb_implicit_sync_for_all=True),
                compiler_options=E.KernelSpec.CompilerOptions(defines=defines),
                dfb_bindings=[E.producer_of(n, n) for n in names] + [E.consumer_of("out", "out")],
                tensor_bindings=[E.TensorBinding(tensor_parameter_name=t, accessor_name=t) for t in tensors],
            ),
            E.KernelSpec(
                unique_id="compute",
                source=E.KernelSpec.SourceCode(code=COMPUTE_SOURCE.replace("BODY", body)),
                hw_config=E.ComputeHardwareConfig(),
                compiler_options=E.KernelSpec.CompilerOptions(defines=defines),
                dfb_bindings=[E.consumer_of(n, n) for n in names] + [E.producer_of("out", "out")],
            ),
        ],
        dataflow_buffers=dfbs,
        tensor_parameters=[E.TensorParameter(unique_id=t, spec=v.spec) for t, v in tensors.items()],
        work_units=[E.WorkUnitSpec(kernels=["dm", "compute"], target_nodes=CORE)],
    )
    E.generic_op(list(tensors.values()), spec, E.ProgramRunArgs(tensor_args=tensors))

    expected = golden(torch_a.float(), torch_b.float())
    result = ttnn.to_torch(tensors["result"]).float()
    if part is not None:
        result = part(result)
    assert_with_pcc(expected, result, 0.999)
