// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#ifndef QB2_ENTRY
#define QB2_ENTRY kernel_main
#endif
// Bind the current native Metal 2.0 RMSNorm arithmetic to fixed batch-one
// legacy CB indices. DataflowBuffer uses the same Blackhole CB interface.
#include "experimental/kernel_args.h"
#ifndef NORM_SUBBLOCK
#define NORM_SUBBLOCK 4
#endif
namespace args {
constexpr experimental::CtaVal<uint32_t> num_blocks_first_stage{8}, block_w{16}, block_h{1}, subblock_w{NORM_SUBBLOCK};
constexpr experimental::CtaVal<uint32_t> num_subblocks_w{16 / NORM_SUBBLOCK}, num_tiles_per_block{16}, float32_dtype{1};
constexpr experimental::CtaVal<uint32_t> legacy_rsqrt{0}, num_blocks_second_stage{1}, num_reduce_tiles_per_block_h{16};
constexpr experimental::CtaVal<uint32_t> num_rows_per_all_to_all_worker{1}, use_two_stage_reduce{0}, is_second_stage_reader{0};
}
namespace dfb {
constexpr uint32_t in0=0, scaler=2, eps=3, scaler_global=4, x=5, xmm=6;
constexpr uint32_t ex_partial2=7, ex2=8, ex_external2=9, ex_global=10, ex2pe=11, out=16;
}
#define RMSNORM
#define kernel_main native_norm_main
#include "ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/compute/layernorm_sharded.cpp"
#undef kernel_main
#include "tools/profiler/kernel_profiler.hpp"
void QB2_ENTRY() {
    DeviceZoneScopedN("MLP-NORM-MATH");
    // Native sharded input is already resident when dispatch starts. This
    // composed variant first gathers it through NCRISC, so wait explicitly.
    cb_wait_front(0, 16);
    cb_wait_front(2, 1);
    cb_wait_front(3, 1);
#ifdef IS_ALLGATHER_WORKER
    cb_wait_front(4, 1);
#endif
    native_norm_main();
}
