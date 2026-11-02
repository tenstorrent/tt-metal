// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Ternary decode matmul, step (a): codes only.
// deepseek_v3_b1's dram_streaming_matmul kernel trimmed to the plain matmul, with one
// fix: it never passed fp32_dest_acc_en to ComputeCTArgs (default 0), so custom_mm
// accumulated in a bf16 Dst whatever the compute config said. At K=17408 that costs
// PCC 0.998 on integer inputs. Here it is passed through.
// in1_tensor_addr is per core (bank shard base + this worker's column offset), so two
// workers can share one bank without any change to the reader.

#include "../../../../../deepseek_v3_b1/unified_kernels/kernel_op_api.hpp"
#include "../../../../../deepseek_v3_b1/unified_kernels/kernel_utils.hpp"
#include "../../../../../deepseek_v3_b1/unified_kernels/dram_streaming_matmul.hpp"

struct Core {
    static constexpr bool is_active_core = get_named_compile_time_arg_val("tm_is_active_core") == 1;
};

void kernel_main() {
#if defined(COMPILE_FOR_NCRISC)
    using CTArgs = deepseek_b1_ops::DRAMStreamingMatmul::ReaderCTArgs<
        get_named_compile_time_arg_val("tm_cb_in1"),
        get_named_compile_time_arg_val("tm_cb_out"),
        get_named_compile_time_arg_val("tm_in1_addr"),
        get_named_compile_time_arg_val("tm_in1_page_size"),
        get_named_compile_time_arg_val("tm_in1_num_pages"),
        get_named_compile_time_arg_val("tm_subblock_k"),
        get_named_compile_time_arg_val("tm_per_core_n"),
        get_named_compile_time_arg_val("tm_in1_block_bytes"),
        get_named_compile_time_arg_val("tm_per_core_n"),
        get_named_compile_time_arg_val("tm_num_subblocks_k"),
        get_named_compile_time_arg_val("tm_bank_id"),
        get_named_compile_time_arg_val("tm_vc")>;
    if constexpr (Core::is_active_core) {
        unified_kernels::setup_sharded_buffer(
            get_named_compile_time_arg_val("tm_cb_in0"), get_named_compile_time_arg_val("tm_num_tiles_k"));
    }
#elif defined(COMPILE_FOR_BRISC)
    using CTArgs = deepseek_b1_ops::DRAMStreamingMatmul::WriterCTArgs;
#elif defined(COMPILE_FOR_TRISC)
    using CTArgs = deepseek_b1_ops::DRAMStreamingMatmul::ComputeCTArgs<
        get_named_compile_time_arg_val("tm_cb_in0"),
        get_named_compile_time_arg_val("tm_cb_in1"),
        get_named_compile_time_arg_val("tm_cb_out"),
        get_named_compile_time_arg_val("tm_subblock_k"),
        get_named_compile_time_arg_val("tm_per_core_n"),
        get_named_compile_time_arg_val("tm_subblock_w"),
        get_named_compile_time_arg_val("tm_num_subblocks_k"),
        get_named_compile_time_arg_val("tm_tile_r_dim"),
        0,                         // fuse_silu
        DST_ACCUM_MODE ? 1 : 0>;  // the fix: accumulate in the Dst the compute config asked for
    deepseek_compute_kernel_init();
#endif
    deepseek_b1_ops::DRAMStreamingMatmul::Op<CTArgs, Core::is_active_core> mm;
    mm();
}
