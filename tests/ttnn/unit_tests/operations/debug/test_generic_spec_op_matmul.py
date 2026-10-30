# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Decode-shaped matmul through ttnn.generic_op with a Metal 2.0 ProgramSpec.

out[M, N] = a[M, K] @ b[K, N], with a and out in tiny Tile([M, 32]) tiles and b in 32x32 tiles. Every tensor is an
L1 shard. b is width-sharded over the program's cores, and each core computes its own columns of out. b and out are
borrowed by their dataflow buffers, so no data is copied: one data movement kernel only announces the resident tiles,
the compute kernel reads them in place by index, and a second data movement kernel retires the output tiles, which
compute packed into the output shard. a lives on core (0, 0). It is borrowed there when the program runs on that core
alone; on more cores, each core reads it over the NoC, or core (0, 0) multicasts it to the others.
"""

import os

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from tt_py_test_utils_common.utility_functions import is_blackhole, is_quasar

# Announces the resident tiles of b, then brings in the tiles of a for each iteration.
PUSH_INPUTS_SOURCE = """
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/semaphore.h"
#include "api/tensor/local_tensor_accessor.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t Kt = get_arg(args::Kt);
    constexpr uint32_t Nt = get_arg(args::Nt);
    constexpr uint32_t Rt = get_arg(args::Rt);
    constexpr uint32_t num_iters = get_arg(args::num_iters);
    constexpr uint32_t in0_tiles = Rt * Kt;  // a's tiles per iteration
    DataflowBuffer in0(dfb::in0);
    DataflowBuffer in1(dfb::in1);

    in1.reserve_back(Kt * Nt);
    in1.push_back(Kt * Nt);
#if defined(READ_IN0)
    Noc noc;
    const auto a = TensorAccessor(tensor::in0);
    const uint32_t tile_size = in0.get_tile_size();
    for (uint32_t page = 0; page < num_iters * in0_tiles; ++page) {
        in0.reserve_back(1);
        noc.async_read(a, in0, tile_size, {.page_id = page, .offset_bytes = 0}, {.offset_bytes = 0});
        noc.async_read_barrier();
        in0.push_back(1);
    }
#elif defined(MULTICAST_IN0)
    // Core (0, 0) holds a. Every iteration, each receiver reserves room in its in0 and tells the sender (ready); the
    // sender copies the iteration's tiles into its own in0, multicasts them to the same in0 location on the
    // receivers, which every core allocated at the same address, then multicasts the valid flag.
    const uint32_t is_sender = get_arg(args::is_sender);
    const uint32_t sender_x = get_arg(args::sender_x);
    const uint32_t sender_y = get_arg(args::sender_y);
    const uint32_t num_receivers = get_arg(args::num_receivers);
    Noc noc;
    // NoC 1 runs the other way, so its multicast rectangle starts at the far corner.
    const bool noc0 = noc.get_noc_id() == 0;
    const uint32_t x_start = noc0 ? get_arg(args::receivers_x_min) : get_arg(args::receivers_x_max);
    const uint32_t y_start = noc0 ? get_arg(args::receivers_y_min) : get_arg(args::receivers_y_max);
    const uint32_t x_end = noc0 ? get_arg(args::receivers_x_max) : get_arg(args::receivers_x_min);
    const uint32_t y_end = noc0 ? get_arg(args::receivers_y_max) : get_arg(args::receivers_y_min);
    Semaphore<> ready(sem::ready);
    Semaphore<> valid(sem::valid);
    const uint32_t bytes = in0_tiles * in0.get_tile_size();
    for (uint32_t iter = 0; iter < num_iters; ++iter) {
        in0.reserve_back(in0_tiles);
        if (is_sender) {
            const LocalTensorAccessor<uint32_t> a(tensor::in0);
            noc.async_read(a, in0, bytes, {.offset_bytes = iter * bytes}, {.offset_bytes = 0});
            noc.async_read_barrier();
            ready.wait(num_receivers);
            ready.set(0);
            noc.async_write_multicast(
                a,
                in0,
                bytes,
                num_receivers,
                {.offset_bytes = iter * bytes},
                {.noc_x_start = x_start, .noc_y_start = y_start, .noc_x_end = x_end, .noc_y_end = y_end,
                 .offset_bytes = 0});
            noc.async_write_barrier();  // The data lands before the flag that says it is there.
            valid.set(1);
            valid.set_multicast(noc, x_start, y_start, x_end, y_end, num_receivers);
            noc.async_write_barrier();
        } else {
            valid.set(0);
            ready.up(noc, sender_x, sender_y, 1);
            valid.wait(1);
        }
        in0.push_back(in0_tiles);
    }
#else
    in0.reserve_back(num_iters * in0_tiles);
    in0.push_back(num_iters * in0_tiles);
#endif
}
"""

# Per iteration, Rt x Nt output tiles from Rt x Kt tiles of a. b stays at the front of in1 for every iteration.
MATMUL_SOURCE = """
#include <cstdint>
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t Kt = get_arg(args::Kt);
    constexpr uint32_t Nt = get_arg(args::Nt);
    constexpr uint32_t Rt = get_arg(args::Rt);
    constexpr uint32_t num_iters = get_arg(args::num_iters);
    DataflowBuffer in0(dfb::in0);
    DataflowBuffer in1(dfb::in1);
    DataflowBuffer out(dfb::out);

    compute_kernel_hw_startup<SrcOrder::Reverse>(dfb::in0, dfb::in1, dfb::out);
#ifdef MATMUL_BLOCK
    matmul_block_init(dfb::in0, dfb::in1, /*transpose=*/0, /*ct_dim=*/Nt, /*rt_dim=*/Rt, /*kt_dim=*/Kt);
#else
    matmul_init(dfb::in0, dfb::in1, /*transpose=*/0);
#endif

    in1.wait_front(Kt * Nt);
    for (uint32_t iter = 0; iter < num_iters; ++iter) {
        in0.wait_front(Rt * Kt);
#ifdef MATMUL_BLOCK
        // All Rt x Nt output tiles at once, row-major in DST. a's tiles are row-major, Kt per row, and the LLK steps
        // rows by kt_dim; b's row kt starts at tile kt * Nt.
        tile_regs_acquire();
        for (uint32_t kt = 0; kt < Kt; ++kt) {
            matmul_block(dfb::in0, dfb::in1, kt, kt * Nt, 0, /*transpose=*/0, Nt, Rt, Kt);
        }
        tile_regs_commit();
        out.reserve_back(Rt * Nt);
        tile_regs_wait();
        for (uint32_t t = 0; t < Rt * Nt; ++t) {
            pack_tile(t, dfb::out);
        }
        tile_regs_release();
        out.push_back(Rt * Nt);
#else
        for (uint32_t rt = 0; rt < Rt; ++rt) {
            for (uint32_t nt = 0; nt < Nt; ++nt) {
                tile_regs_acquire();
                for (uint32_t kt = 0; kt < Kt; ++kt) {
                    matmul_tiles(dfb::in0, dfb::in1, rt * Kt + kt, kt * Nt + nt, 0);
                }
                tile_regs_commit();
                out.reserve_back(1);
                tile_regs_wait();
                pack_tile(0, dfb::out);
                tile_regs_release();
                out.push_back(1);
            }
        }
#endif
        in0.pop_front(Rt * Kt);
    }
    in1.pop_front(Kt * Nt);
}
"""

# blaze's Matmul: custom_mm with the dense DST layout (two 1-row output tiles per DST tile), packed contiguously.
# The custom_mm and pack_block APIs exist only for Blackhole.
CUSTOM_MM_DENSE_SOURCE = """
#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/experimental/custom_mm.h"
#include "api/compute/experimental/pack_block.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t Kt = get_arg(args::Kt);
    constexpr uint32_t Nt = get_arg(args::Nt);
    constexpr uint32_t Rt = get_arg(args::Rt);
    constexpr uint32_t num_iters = get_arg(args::num_iters);
    static_assert(Rt == 1 && num_iters == 1, "custom_mm takes one row of activation tiles");
    constexpr auto layout = CustomMmDestLayout::DenseSplitAcc;
    DataflowBuffer in0(dfb::in0);
    DataflowBuffer in1(dfb::in1);
    DataflowBuffer out(dfb::out);

    compute_kernel_hw_startup<SrcOrder::Reverse>(dfb::in0, dfb::in1, dfb::out);
    custom_mm_block_init</*transpose=*/false, layout>(dfb::in0, dfb::in1, dfb::out, Nt);
    pack_block_contiguous_init(dfb::out);

    in1.wait_front(Kt * Nt);
    in0.wait_front(Kt);
    out.reserve_back(Nt);
    tile_regs_acquire();
    custom_mm_block<layout, /*finalize=*/true>(dfb::in0, dfb::in1, 0, 0, 0, Kt, Nt);
    tile_regs_commit();
    tile_regs_wait();
    pack_block_contiguous(0, dfb::out, Nt);
    tile_regs_release();
    out.push_back(Nt);
    custom_mm_block_uninit<layout>(dfb::out);
    in0.pop_front(Kt);
    in1.pop_front(Kt * Nt);
}
"""

# Retires the output tiles: compute already packed them into the output shard.
POP_OUTPUT_SOURCE = """
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t Nt = get_arg(args::Nt);
    constexpr uint32_t Rt = get_arg(args::Rt);
    constexpr uint32_t num_iters = get_arg(args::num_iters);
    DataflowBuffer out(dfb::out);
    for (uint32_t iter = 0; iter < num_iters; ++iter) {
        out.wait_front(Rt * Nt);
        out.pop_front(Rt * Nt);
    }
}
"""

FULL_TILE = ttnn.Tile([32, 32])


def core_grid(num_cores):
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores - 1, 0))})


def to_device(device, tensor, dtype, tile, num_cores=1):
    """Width-shards tensor over num_cores cores in row 0 (one shard on core (0, 0) for num_cores=1)."""
    height, width = tensor.shape
    memory_config = ttnn.create_sharded_memory_config(
        shape=(height, width // num_cores),
        core_grid=core_grid(num_cores),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    return ttnn.from_torch(
        tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT, tile=tile, device=device, memory_config=memory_config
    )


def make_weights(device, Kt, Nt, num_cores=1):
    """Returns (torch_b, b): b is [32 * Kt, 32 * Nt * num_cores], Nt tile columns per core."""
    torch_b = torch.randn((32 * Kt, 32 * Nt * num_cores), dtype=torch.bfloat16)
    return torch_b, to_device(device, torch_b, ttnn.bfloat16, FULL_TILE, num_cores)


def make_activation(device, M, Kt, num_row_blocks=1):
    """Returns (torch_a, a): a holds num_row_blocks [M, 32 * Kt] activations, stacked by rows, on core (0, 0)."""
    torch_a = torch.randn((num_row_blocks * M, 32 * Kt), dtype=torch.bfloat16)
    return torch_a, to_device(device, torch_a, ttnn.bfloat16, ttnn.Tile([M, 32]))


def make_output(device, M, b, num_row_blocks=1, dtype=ttnn.bfloat16):
    num_cores = b.memory_config().shard_spec.grid.num_cores()
    zeros = torch.zeros((num_row_blocks * M, b.shape[-1]), dtype=torch.float32)
    return to_device(device, zeros, dtype, ttnn.Tile([M, 32]), num_cores)


def run_decode_matmul(
    a,
    b,
    out,
    *,
    num_iters=1,
    Rt=1,
    matmul_block=False,
    in0_source="noc_read",
    enable_32_bit_dest=False,
    compute_source=MATMUL_SOURCE,
    math_fidelity=ttnn.MathFidelity.HiFi4,
):
    """Runs out = a @ b for each of a's M-row blocks, Rt blocks per matmul call and num_iters calls per program, and
    returns out as a float32 torch tensor. in0_source says how cores other than (0, 0) get a: "noc_read" or
    "multicast"."""
    device = a.device()
    grid = b.memory_config().shard_spec.grid
    num_cores = grid.num_cores()
    tiny_tile = a.tile
    Kt = a.shape[-1] // 32
    Nt = b.memory_config().shard_spec.shape[1] // 32
    in0_tiles = Rt * Kt
    # a is resident only when the program runs on core (0, 0) alone.
    if num_cores == 1:
        in0_source = "borrowed"

    def dfb(name, tensor, tile, num_entries, borrowed=True):
        return ttnn.DataflowBufferSpec(
            unique_id=name,
            entry_size=tile.get_tile_size(tensor.dtype),
            num_entries=num_entries,
            data_format=tensor.dtype,
            tile=tile,
            borrowed_from=name if borrowed else None,
        )

    def binding(name, endpoint_type):
        return ttnn.DFBBinding(dfb_spec_name=name, accessor_name=name, endpoint_type=endpoint_type)

    producer, consumer = ttnn.DFBEndpointType.PRODUCER, ttnn.DFBEndpointType.CONSUMER
    sizes = {"Kt": Kt, "Nt": Nt, "Rt": Rt, "num_iters": num_iters}
    push_inputs = dict(
        dfb_bindings=[binding("in0", producer), binding("in1", producer)],
        compile_time_args=sizes,
        defines={},
        tensor_bindings=[],
        semaphore_bindings=[],
        runtime_arg_names=[],
    )
    semaphores = []
    kernel_run_args = []
    if in0_source != "borrowed":
        push_inputs["tensor_bindings"] = [ttnn.TensorBinding(tensor_parameter_name="in0", accessor_name="in0")]
    if in0_source == "noc_read":
        push_inputs["defines"] = {"READ_IN0": "1"}
    elif in0_source == "multicast":
        push_inputs["defines"] = {"MULTICAST_IN0": "1"}
        semaphores = [ttnn.SemaphoreSpec(unique_id=name, target_nodes=grid) for name in ("ready", "valid")]
        push_inputs["semaphore_bindings"] = [
            ttnn.SemaphoreBinding(semaphore_spec_name=name, accessor_name=name) for name in ("ready", "valid")
        ]
        names = ["is_sender", "sender_x", "sender_y", "num_receivers"]
        names += ["receivers_x_min", "receivers_y_min", "receivers_x_max", "receivers_y_max"]
        push_inputs["runtime_arg_names"] = names
        cores = [ttnn.CoreCoord(x, 0) for x in range(num_cores)]
        noc = [device.worker_core_from_logical_core(core) for core in cores]
        receivers = noc[1:]
        run_args = ttnn.KernelRunArgs(kernel="push_inputs")
        for index, core in enumerate(cores):
            run_args.add_runtime_args_for_node(
                core,
                {
                    "is_sender": int(index == 0),
                    "sender_x": noc[0].x,
                    "sender_y": noc[0].y,
                    "num_receivers": len(receivers),
                    "receivers_x_min": min(c.x for c in receivers),
                    "receivers_y_min": min(c.y for c in receivers),
                    "receivers_x_max": max(c.x for c in receivers),
                    "receivers_y_max": max(c.y for c in receivers),
                },
            )
        kernel_run_args.append(run_args)

    spec = ttnn.ProgramSpec(
        name="decode_matmul",
        dataflow_buffers=[
            dfb(
                "in0",
                a,
                tiny_tile,
                num_iters * in0_tiles if in0_source == "borrowed" else in0_tiles,
                borrowed=in0_source == "borrowed",
            ),
            dfb("in1", b, FULL_TILE, Kt * Nt),
            dfb("out", out, tiny_tile, num_iters * Rt * Nt),
        ],
        semaphores=semaphores,
        tensor_parameters=[
            ttnn.TensorParameter(unique_id="in0", spec=a.spec),
            ttnn.TensorParameter(unique_id="in1", spec=b.spec),
            ttnn.TensorParameter(unique_id="out", spec=out.spec),
        ],
        kernels=[
            ttnn.KernelSpec(
                unique_id="push_inputs",
                source_code=PUSH_INPUTS_SOURCE,
                hw_config=ttnn.DataMovementHardwareConfig.reader(),
                **push_inputs,
            ),
            ttnn.KernelSpec(
                unique_id="matmul",
                source_code=compute_source,
                hw_config=ttnn.ComputeHardwareConfig(
                    math_fidelity=math_fidelity, enable_32_bit_dest=enable_32_bit_dest
                ),
                dfb_bindings=[binding("in0", consumer), binding("in1", consumer), binding("out", producer)],
                compile_time_args=sizes,
                defines={"MATMUL_BLOCK": "1"} if matmul_block else {},
            ),
            ttnn.KernelSpec(
                unique_id="pop_output",
                source_code=POP_OUTPUT_SOURCE,
                hw_config=ttnn.DataMovementHardwareConfig.writer(),
                dfb_bindings=[binding("out", consumer)],
                compile_time_args=sizes,
            ),
        ],
        work_units=[ttnn.WorkUnitSpec(name="main", kernels=["push_inputs", "matmul", "pop_output"], target_nodes=grid)],
    )
    run_args = ttnn.ProgramSpecRunArgs(kernel_run_args=kernel_run_args, tensor_args={"in0": a, "in1": b, "out": out})
    return ttnn.to_torch(ttnn.generic_op([a, b, out], spec, run_args)).float()


def expected_output(torch_a, torch_b, M):
    """Each M-row block of torch_a times torch_b, stacked like the device output."""
    return torch.cat([block.float() @ torch_b.float() for block in torch_a.split(M)])


@pytest.mark.parametrize("matmul_block", [False, True], ids=["matmul_tiles", "matmul_block"])
@pytest.mark.parametrize("M", [1, 8])
@pytest.mark.parametrize("Kt", [1, 4])
@pytest.mark.parametrize("Nt", [1, 2])
def test_generic_spec_op_decode_matmul_resident_weights(device, M, Kt, Nt, matmul_block):
    torch.manual_seed(0)
    torch_a, a = make_activation(device, M, Kt)
    torch_b, b = make_weights(device, Kt, Nt)
    result = run_decode_matmul(a, b, make_output(device, M, b), matmul_block=matmul_block)
    assert_with_pcc(expected_output(torch_a, torch_b, M), result, 0.999)


@pytest.mark.parametrize("matmul_block", [False, True], ids=["matmul_tiles", "matmul_block"])
@pytest.mark.parametrize("M", [1, 8])
def test_generic_spec_op_decode_matmul_fp32_accumulation(device, M, matmul_block):
    """Accumulates in a 32-bit DST and packs Float32 output tiles."""
    torch.manual_seed(0)
    torch_a, a = make_activation(device, M, Kt=8)
    torch_b, b = make_weights(device, Kt=8, Nt=2)
    out = make_output(device, M, b, dtype=ttnn.float32)
    result = run_decode_matmul(a, b, out, matmul_block=matmul_block, enable_32_bit_dest=True)
    assert_with_pcc(expected_output(torch_a, torch_b, M), result, 0.9999)


@pytest.mark.parametrize("matmul_block", [False, True], ids=["matmul_tiles", "matmul_block"])
@pytest.mark.parametrize("M", [1, 8])
@pytest.mark.parametrize("Rt,Nt", [(2, 1), (2, 2), (4, 2)])
def test_generic_spec_op_decode_matmul_row_blocks(device, M, Rt, Nt, matmul_block):
    """Rt row blocks of M activation rows (M * Rt rows in all) in one matmul call (rt_dim=Rt for matmul_block)."""
    torch.manual_seed(0)
    torch_a, a = make_activation(device, M, Kt=4, num_row_blocks=Rt)
    torch_b, b = make_weights(device, Kt=4, Nt=Nt)
    result = run_decode_matmul(a, b, make_output(device, M, b, Rt), Rt=Rt, matmul_block=matmul_block)
    assert_with_pcc(expected_output(torch_a, torch_b, M), result, 0.999)


@pytest.mark.parametrize("in0_source", ["noc_read", "multicast"])
@pytest.mark.parametrize("matmul_block", [False, True], ids=["matmul_tiles", "matmul_block"])
@pytest.mark.parametrize("M", [1, 8])
def test_generic_spec_op_decode_matmul_two_cores(device, M, matmul_block, in0_source):
    """b and out are split by columns over two cores; a is on core (0, 0). Two activations per program, so the
    multicast handshake runs twice."""
    torch.manual_seed(0)
    torch_a, a = make_activation(device, M, Kt=4, num_row_blocks=2)
    torch_b, b = make_weights(device, Kt=4, Nt=1, num_cores=2)
    out = make_output(device, M, b, num_row_blocks=2)
    result = run_decode_matmul(a, b, out, num_iters=2, matmul_block=matmul_block, in0_source=in0_source)
    assert_with_pcc(expected_output(torch_a, torch_b, M), result, 0.999)


@pytest.mark.parametrize("matmul_block", [False, True], ids=["matmul_tiles", "matmul_block"])
def test_generic_spec_op_decode_matmul_persistent_weights(device, matmul_block):
    """Several activations in one program run against weights pushed once and never re-read; then a second program
    run with new activations reuses the same weight tensor."""
    M, Kt, Nt, num_iters = 1, 4, 2, 3
    torch.manual_seed(0)
    torch_b, b = make_weights(device, Kt, Nt)
    for _ in range(2):
        torch_a, a = make_activation(device, M, Kt, num_iters)
        out = make_output(device, M, b, num_iters)
        result = run_decode_matmul(a, b, out, num_iters=num_iters, matmul_block=matmul_block)
        assert_with_pcc(expected_output(torch_a, torch_b, M), result, 0.999)


def test_generic_spec_op_decode_matmul_cache_hit_rebinds_borrowed_buffers(device):
    """A second run on new tensors hits the program cache and must read and write the new shards."""
    torch.manual_seed(0)
    runs = []
    for _ in range(2):
        torch_a, a = make_activation(device, 8, 4)
        torch_b, b = make_weights(device, 4, 2)
        runs.append((torch_a, torch_b, a, b, make_output(device, 8, b)))
    assert runs[1][3].buffer_address() != runs[0][3].buffer_address()

    for torch_a, torch_b, a, b, out in runs:
        assert_with_pcc(expected_output(torch_a, torch_b, 8), run_decode_matmul(a, b, out), 0.999)
    assert device.num_program_cache_entries() == 1


def run_custom_mm_dense(device, M, Nt):
    torch.manual_seed(0)
    torch_a, a = make_activation(device, M, Kt=4)
    torch_b, b = make_weights(device, Kt=4, Nt=Nt)
    result = run_decode_matmul(
        a,
        b,
        make_output(device, M, b),
        compute_source=CUSTOM_MM_DENSE_SOURCE,
        math_fidelity=ttnn.MathFidelity.LoFi,  # custom_mm is LoFi only
    )
    return expected_output(torch_a, torch_b, M), result


@pytest.mark.parametrize("M", [1, 8])
@pytest.mark.parametrize("Nt", [2, 3])
def test_generic_spec_op_decode_matmul_custom_mm_dense_on_blackhole(device, M, Nt):
    """blaze's dense-packed custom_mm runs through a ProgramSpec on Blackhole (odd Nt leaves half a DST tile)."""
    if not is_blackhole():
        pytest.skip("custom_mm and pack_block_contiguous exist only for Blackhole")
    if os.environ.get("TT_METAL_SIMULATOR"):
        # ttsim v1.15 aborts custom_mm: UnsupportedFunctionality tensix_unpacr_nop bank_clr_ctrl=1 (the SrcB clear at
        # init) and, with that clear skipped, tensix_unpacr src_b row=64.
        pytest.skip("ttsim does not model custom_mm's SrcB unpacks; run on Blackhole silicon")
    expected, result = run_custom_mm_dense(device, M, Nt)
    assert_with_pcc(expected, result, 0.99)


def test_generic_spec_op_decode_matmul_custom_mm_does_not_build_on_quasar(device, expect_error):
    """On Quasar the custom_mm API is compiled out, so the kernel build fails and the compiler error reaches Python."""
    if not is_quasar():
        pytest.skip("custom_mm builds on Blackhole")
    with expect_error(RuntimeError, "custom_mm_block_init"):
        run_custom_mm_dense(device, 1, 2)
