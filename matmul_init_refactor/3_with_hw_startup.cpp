// TESTED: compute_kernel_hw_startup + explicit reconfig + mm_init_short
// All 19 test_attn_matmul tests pass on Blackhole (branch ncvetkovic/22946_matmul_init_cleanup)

#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "api/compute/tilize.h"
#include "api/compute/untilize.h"
#include "experimental/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

using std::uint32_t;

// matmul C=A*B using dims MK*KN = MN (row major order)
//
void kernel_main() {
    constexpr uint32_t onetile = 1;

    constexpr uint32_t transpose_hw = get_compile_time_arg_val(0);
    uint32_t batch = get_arg_val<uint32_t>(0);
    uint32_t Mt = get_arg_val<uint32_t>(1);
    uint32_t Kt = get_arg_val<uint32_t>(2);
    uint32_t Nt = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_in1 = tt::CBIndex::c_1;
    constexpr uint32_t cb_intermed0 = tt::CBIndex::c_2;
    constexpr uint32_t cb_intermed1 = tt::CBIndex::c_3;
    constexpr uint32_t cb_intermed2 = tt::CBIndex::c_4;
    constexpr uint32_t out_cb_id = tt::CBIndex::c_5;

    experimental::CircularBuffer cb_in0_obj(cb_in0);
    experimental::CircularBuffer cb_in1_obj(cb_in1);
    experimental::CircularBuffer cb_intermed0_obj(cb_intermed0);
    experimental::CircularBuffer cb_intermed1_obj(cb_intermed1);
    experimental::CircularBuffer cb_intermed2_obj(cb_intermed2);
    experimental::CircularBuffer cb_out_obj(out_cb_id);

    constexpr uint32_t num_rows_in_one_tile = 32;

    // HW init once at kernel start. Matmul maps in0->srcB, in1->srcA,
    // so pass (in1, in0) to match compute_kernel_hw_startup's (srcA, srcB) convention.
    compute_kernel_hw_startup(cb_in1, cb_in0, cb_intermed0);
    mm_init_short(cb_in0, cb_in1, transpose_hw);

    for (uint32_t nb = 0; nb < batch; ++nb) {
        for (uint32_t mt_C = 0; mt_C < Mt; ++mt_C) {    // output tile of C
            for (uint32_t nt_C = 0; nt_C < Nt; ++nt_C)  // output tile index of C
            {
                for (uint32_t tile_row_id = 0; tile_row_id < num_rows_in_one_tile; ++tile_row_id) {
                    tile_regs_acquire();
                    for (uint32_t kt = 0; kt < Kt; ++kt) {
                        if (tile_row_id == 0) {
                            cb_in0_obj.wait_front(kt + 1);
                        }
                        cb_in1_obj.wait_front(onetile);

                        matmul_tiles(cb_in0, cb_in1, kt, 0, 0);

                        cb_in1_obj.pop_front(onetile);
                    }
                    tile_regs_commit();

                    cb_intermed0_obj.reserve_back(onetile);
                    tile_regs_wait();
                    pack_tile(0, cb_intermed0);
                    tile_regs_release();
                    cb_intermed0_obj.push_back(onetile);

                    compute_kernel_lib::untilize<
                        onetile,
                        cb_intermed0,
                        cb_intermed1,
                        compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit,
                        compute_kernel_lib::untilize_config::WaitMode::WaitBlock,
                        compute_kernel_lib::untilize_config::ReconfigureRegisterDatatypeMode::UnpackReconfigure>(1);

                    // Reconfigure srcA data format back to matmul mode after untilize changed it
                    reconfig_data_format_srca(cb_intermed0, cb_in1);
                    mm_init_short(cb_in0, cb_in1, transpose_hw);
                }
                cb_in0_obj.pop_front(Kt);

                compute_kernel_lib::tilize<onetile, cb_intermed2, out_cb_id>(1);

                // Reconfigure pack format and both src data formats back to matmul mode after tilize
                pack_reconfig_data_format(out_cb_id, cb_intermed0);
                reconfig_data_format(cb_intermed2, cb_in1, cb_intermed2, cb_in0);
                mm_init_short(cb_in0, cb_in1, transpose_hw);
            }
        }
    }
}
