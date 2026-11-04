// attention-front custom_mm throughput microbenchmark, TRISC only: N_CALLS calls of custom_mm_block(kt = 80, ct = CT)
// on resident data (no streaming), 1120 tile products in total. CT args: 0 cb_in0, 1 cb_w, 2 cb_out, 3 ct, 4 calls, 5
// fp32
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/experimental/custom_mm.h"
#include "api/compute/experimental/deepseek_compute_kernel_hw_startup.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/pack.h"
#include "api/compute/experimental/pack_block.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t cb_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);
    constexpr uint32_t CT = get_compile_time_arg_val(3);
    constexpr uint32_t CALLS = get_compile_time_arg_val(4);
    constexpr uint32_t fp32 = get_compile_time_arg_val(5);
    constexpr uint32_t KT = 80;
    deepseek_compute_kernel_init();
    CircularBuffer in0(cb_in0), w(cb_w), out(cb_out);
    in0.wait_front(KT);
    w.wait_front(KT * CT);
    reconfig_full_operand<SrcOrder::Reverse>(cb_in0, cb_w);
    pack_reconfig_data_format<true>(cb_out);
    custom_mm_block_init_short<false, true, false>(cb_in0, cb_w, cb_out, CT);
    pack_block_contiguous_init(cb_out);
    for (uint32_t c = 0; c < CALLS; ++c) {
        out.reserve_back(CT);
        tile_regs_acquire();
        custom_mm_block<true>(cb_in0, cb_w, 0, 0, 0, KT, CT);
        tile_regs_commit();
        tile_regs_wait();
        pack_block_contiguous(0, cb_out, CT);
        tile_regs_release();
        out.push_back(CT);
        out.wait_front(CT);
        out.pop_front(CT);
    }
}
