// PROPOSED: Clean compute API — init/block/uninit triplet for all operations
// matmul_init, matmul_block, matmul_uninit from experimental shim (matmul_api.h)
// tilize/untilize already wrapped by compute_kernel_lib

#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/experimental/matmul_api.h"
#include "api/compute/tile_move_copy.h"
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

    // Op-agnostic HW startup — standard CB order
    compute_kernel_hw_startup(cb_in0, cb_in1, cb_intermed0);

    for (uint32_t nb = 0; nb < batch; ++nb) {
        for (uint32_t mt_C = 0; mt_C < Mt; ++mt_C) {
            for (uint32_t nt_C = 0; nt_C < Nt; ++nt_C) {
                for (uint32_t tile_row_id = 0; tile_row_id < num_rows_in_one_tile; ++tile_row_id) {
                    // --- matmul: init → block → uninit ---
                    matmul_init(cb_in0, cb_in1, cb_intermed0, transpose_hw);
                    tile_regs_acquire();
                    for (uint32_t kt = 0; kt < Kt; ++kt) {
                        if (tile_row_id == 0) {
                            cb_in0_obj.wait_front(kt + 1);
                        }
                        cb_in1_obj.wait_front(onetile);
                        matmul_block(cb_in0, cb_in1, kt, 0, 0);
                        cb_in1_obj.pop_front(onetile);
                    }
                    tile_regs_commit();
                    matmul_uninit();

                    // --- pack result ---
                    cb_intermed0_obj.reserve_back(onetile);
                    tile_regs_wait();
                    pack_tile(0, cb_intermed0);
                    tile_regs_release();
                    cb_intermed0_obj.push_back(onetile);

                    // --- untilize: init + block + uninit in one call ---
                    compute_kernel_lib::untilize<
                        onetile,
                        cb_intermed0,
                        cb_intermed1,
                        compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit,
                        compute_kernel_lib::untilize_config::WaitMode::WaitBlock,
                        compute_kernel_lib::untilize_config::ReconfigureRegisterDatatypeMode::UnpackReconfigure>(1);

                    // untilize changed srcA format — restore before next matmul_init
                    reconfig_operand_srca(cb_intermed0, cb_in1);
                }
                cb_in0_obj.pop_front(Kt);

                // --- tilize: init + block + uninit in one call ---
                compute_kernel_lib::tilize<onetile, cb_intermed2, out_cb_id>(1);

                // tilize changed both src formats + pack format — restore before next matmul_init
                reconfig_operand(cb_intermed2, cb_in1, cb_intermed2, cb_in0);
                reconfig_pack(out_cb_id, cb_intermed0);
            }
        }
    }
}
