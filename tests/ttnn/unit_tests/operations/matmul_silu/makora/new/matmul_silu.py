import ttnn

# Kernel: matmul + SiLU fused
# Final accepted state (v82-equivalent baseline):
#   - HiFi2 + fp32_dest_acc_en
#   - Fused matmul+silu in single DST acquire (no cb_mm round-trip)
#   - Reader batches READ_BATCH=4 NoC reads per barrier
#   - cb_in depth 16, cb_out depth 8 for pipeline overlap
#   - Reader hoists base addresses out of K-loop
# Precision floor:
#   - K<=1024 (Kt<=32): 100% @ rtol=0.1
#   - K=2048 (Kt=64):   99.1% @ rtol=0.1 (structural bf16 LSB reduction-order
#     divergence from ttnn.matmul's matmul_block + transpose_wh_tile +
#     packer_l1_acc pipeline; per KB cannot be closed by precision tuning,
#     and all matmul_block replication attempts regressed correctness).
# Note: v103+ submissions to retry the matmul_block path destabilized infra.
#       Reverted to clean baseline.
# v127: Remove non-existent silu.h include (compile error). silu_tile is
#       exposed via eltwise_unary.h transitively in this TT-Metal build.
# v134: CRITICAL FIX per KB snippet batched_noc_read_offset_corruption.
#       Removed READ_BATCH=4 batched reads which used unsafe pointer
#       arithmetic (a_wp + i*2048) on the CB write pointer. CB pointer
#       only guarantees one tile of contiguous space; batching causes
#       wraparound corruption that may explain the timeouts (compute
#       reading garbage / dispatcher stall). One tile per reserve/push.

reader_kernel_src = """
// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t arg = 0;
    uint32_t a_addr            = get_common_arg_val<uint32_t>(arg++);
    uint32_t b_addr            = get_common_arg_val<uint32_t>(arg++);
    uint32_t Mt                = get_common_arg_val<uint32_t>(arg++);
    uint32_t Kt                = get_common_arg_val<uint32_t>(arg++);
    uint32_t Nt                = get_common_arg_val<uint32_t>(arg++);
    uint32_t base_per_core     = get_common_arg_val<uint32_t>(arg++);
    uint32_t extra_range       = get_common_arg_val<uint32_t>(arg++);
    uint32_t grid_x            = get_common_arg_val<uint32_t>(arg++);

    constexpr uint32_t cb_in0 = 0;
    constexpr uint32_t cb_in1 = 1;

    // Linearize 2D core grid to a 1D core id used for work distribution.
    uint32_t core_id = get_absolute_logical_y() * grid_x + get_absolute_logical_x();

    // Cores with id < extra_range get one extra output tile.
    uint32_t my_count;
    uint32_t my_start;
    if (core_id < extra_range) {
        my_count = base_per_core + 1;
        my_start = core_id * my_count;
    } else {
        my_count = base_per_core;
        my_start = extra_range * (base_per_core + 1) + (core_id - extra_range) * base_per_core;
    }
    if (my_count == 0) return;

    // bf16 tiles = 2048 bytes.
    constexpr auto a_args = TensorAccessorArgs<0>();
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    const auto a_acc = TensorAccessor(a_args, a_addr, 2048);
    const auto b_acc = TensorAccessor(b_args, b_addr, 2048);

    uint32_t MtKt = Mt * Kt;
    uint32_t KtNt = Kt * Nt;
    uint32_t MtNt = Mt * Nt;

    // Reader: one tile per reserve/push to keep CB write pointer valid.
    // Batching tile reads under a single cb_reserve_back is unsafe because
    // the CB write pointer wraps and only guarantees one tile of contiguous
    // space (per KB: batched_noc_read_offset_corruption).
    for (uint32_t out_idx = my_start; out_idx < my_start + my_count; ++out_idx) {
        uint32_t b = out_idx / MtNt;
        uint32_t rem = out_idx % MtNt;
        uint32_t mt = rem / Nt;
        uint32_t nt = rem % Nt;

        uint32_t a_base = b * MtKt + mt * Kt;
        uint32_t b_base = b * KtNt + nt;

        for (uint32_t kt = 0; kt < Kt; ++kt) {
            cb_reserve_back(cb_in0, 1);
            cb_reserve_back(cb_in1, 1);
            noc_async_read_tile(a_base + kt, a_acc, get_write_ptr(cb_in0));
            noc_async_read_tile(b_base + kt * Nt, b_acc, get_write_ptr(cb_in1));
            noc_async_read_barrier();
            cb_push_back(cb_in0, 1);
            cb_push_back(cb_in1, 1);
        }
    }
}
"""

compute_kernel_src = """
// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api.h"

namespace NAMESPACE {
void MAIN {
    uint32_t arg = 0;
    uint32_t Kt                = get_common_arg_val<uint32_t>(arg++);
    uint32_t base_per_core     = get_common_arg_val<uint32_t>(arg++);
    uint32_t extra_range       = get_common_arg_val<uint32_t>(arg++);
    uint32_t grid_x            = get_common_arg_val<uint32_t>(arg++);

    constexpr uint32_t cb_in0 = 0;
    constexpr uint32_t cb_in1 = 1;
    constexpr uint32_t cb_out = 16;

    uint32_t core_id = get_absolute_logical_y() * grid_x + get_absolute_logical_x();
    uint32_t my_count;
    if (core_id < extra_range) {
        my_count = base_per_core + 1;
    } else {
        my_count = base_per_core;
    }
    if (my_count == 0) return;

    // Fused matmul+silu in single DST acquire. dst[0] holds fp32 accumulator
    // throughout; silu_tile(0) reads/writes the fp32 value in place.
    mm_init(cb_in0, cb_in1, cb_out);

    for (uint32_t i = 0; i < my_count; ++i) {
        tile_regs_acquire();
        for (uint32_t kt = 0; kt < Kt; ++kt) {
            cb_wait_front(cb_in0, 1);
            cb_wait_front(cb_in1, 1);
            matmul_tiles(cb_in0, cb_in1, 0, 0, 0);
            cb_pop_front(cb_in0, 1);
            cb_pop_front(cb_in1, 1);
        }
        silu_tile_init();
        silu_tile(0);
        tile_regs_commit();

        cb_reserve_back(cb_out, 1);
        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();
        cb_push_back(cb_out, 1);
    }
}
} // namespace NAMESPACE
"""

writer_kernel_src = """
// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t arg = 0;
    uint32_t out_addr          = get_common_arg_val<uint32_t>(arg++);
    uint32_t base_per_core     = get_common_arg_val<uint32_t>(arg++);
    uint32_t extra_range       = get_common_arg_val<uint32_t>(arg++);
    uint32_t grid_x            = get_common_arg_val<uint32_t>(arg++);

    constexpr uint32_t cb_out = 16;

    uint32_t core_id = get_absolute_logical_y() * grid_x + get_absolute_logical_x();

    uint32_t my_count;
    uint32_t my_start;
    if (core_id < extra_range) {
        my_count = base_per_core + 1;
        my_start = core_id * my_count;
    } else {
        my_count = base_per_core;
        my_start = extra_range * (base_per_core + 1) + (core_id - extra_range) * base_per_core;
    }
    if (my_count == 0) return;

    constexpr auto out_args = TensorAccessorArgs<0>();
    const auto out_acc = TensorAccessor(out_args, out_addr, 2048);

    for (uint32_t out_idx = my_start; out_idx < my_start + my_count; ++out_idx) {
        cb_wait_front(cb_out, 1);
        noc_async_write_tile(out_idx, out_acc, get_read_ptr(cb_out));
        noc_async_write_barrier();
        cb_pop_front(cb_out, 1);
    }
}
"""


def host(a: ttnn.Tensor, b: ttnn.Tensor) -> ttnn.Tensor:
    device = a.device()

    a_shape = list(a.shape)
    b_shape = list(b.shape)

    M = a_shape[-2]
    K = a_shape[-1]
    N = b_shape[-1]

    a_batch_dims = a_shape[:-2] if len(a_shape) > 2 else [1]
    batch = 1
    for d in a_batch_dims:
        batch *= d

    out_shape = list(a_shape[:-2]) + [M, N]
    if len(out_shape) < 2:
        out_shape = [M, N]

    output = ttnn.allocate_tensor_on_device(
        ttnn.Shape(out_shape),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
    )

    Mt = M // 32
    Kt = K // 32
    Nt = N // 32

    total_out_tiles = batch * Mt * Nt

    grid = device.compute_with_storage_grid_size()
    grid_x = grid.x
    grid_y = grid.y
    num_cores = grid_x * grid_y

    base_per_core = total_out_tiles // num_cores
    extra_range = total_out_tiles % num_cores

    all_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, grid_y - 1))])

    tile_bytes = 2048

    cb_in_depth = 16

    cb_in0_fmt = ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16, page_size=tile_bytes)
    cb_in0 = ttnn.CBDescriptor(
        total_size=cb_in_depth * tile_bytes, core_ranges=all_cores, format_descriptors=[cb_in0_fmt]
    )

    cb_in1_fmt = ttnn.CBFormatDescriptor(buffer_index=1, data_format=ttnn.bfloat16, page_size=tile_bytes)
    cb_in1 = ttnn.CBDescriptor(
        total_size=cb_in_depth * tile_bytes, core_ranges=all_cores, format_descriptors=[cb_in1_fmt]
    )

    cb_out_fmt = ttnn.CBFormatDescriptor(buffer_index=16, data_format=ttnn.bfloat16, page_size=tile_bytes)
    cb_out = ttnn.CBDescriptor(total_size=8 * tile_bytes, core_ranges=all_cores, format_descriptors=[cb_out_fmt])

    a_cta = ttnn.TensorAccessorArgs(a).get_compile_time_args()
    b_cta = ttnn.TensorAccessorArgs(b).get_compile_time_args()
    out_cta = ttnn.TensorAccessorArgs(output).get_compile_time_args()

    reader_rt = [
        a.buffer_address(),
        b.buffer_address(),
        Mt,
        Kt,
        Nt,
        base_per_core,
        extra_range,
        grid_x,
    ]
    compute_rt = [
        Kt,
        base_per_core,
        extra_range,
        grid_x,
    ]
    writer_rt = [
        output.buffer_address(),
        base_per_core,
        extra_range,
        grid_x,
    ]

    reader_kd = ttnn.KernelDescriptor(
        kernel_source=reader_kernel_src,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=all_cores,
        compile_time_args=a_cta + b_cta,
        common_runtime_args=reader_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    compute_kd = ttnn.KernelDescriptor(
        kernel_source=compute_kernel_src,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=all_cores,
        compile_time_args=[],
        common_runtime_args=compute_rt,
        config=ttnn.ComputeConfigDescriptor(
            fp32_dest_acc_en=True,
            math_fidelity=ttnn.MathFidelity.HiFi2,
        ),
    )
    writer_kd = ttnn.KernelDescriptor(
        kernel_source=writer_kernel_src,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=all_cores,
        compile_time_args=out_cta,
        common_runtime_args=writer_rt,
        config=ttnn.WriterConfigDescriptor(),
    )

    program = ttnn.ProgramDescriptor(
        kernels=[reader_kd, compute_kd, writer_kd],
        cbs=[cb_in0, cb_in1, cb_out],
        semaphores=[],
    )

    return ttnn.generic_op([a, b, output], program)
