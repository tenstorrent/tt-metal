#include <stdint.h>
#include "dataflow_api.h"

#include "debug_print.h"

void kernel_main() {
    // same arg indices as in reader_binary_diff_lenghts for compat
    uint32_t src0_addr  = get_arg_val<uint32_t>(0);
    uint32_t src1_addr  = get_arg_val<uint32_t>(1);
    uint32_t Mt         = get_arg_val<uint32_t>(2);
    uint32_t Kt         = get_arg_val<uint32_t>(3);
    uint32_t Nt         = get_arg_val<uint32_t>(4);
    uint32_t MtKt       = get_arg_val<uint32_t>(5); // if 0
    uint32_t KtNt       = get_arg_val<uint32_t>(6);
    uint32_t batch      = get_arg_val<uint32_t>(7);
    uint32_t bcast_B    = get_arg_val<uint32_t>(8); // if 1 we broadcast B to batch

    //DPRINT << "Mt=" << Mt << " Kt=" << Kt << " Nt=" << Nt << " MtKt=" << MtKt << "KtNt=" << KtNt << ENDL();
    //DPRINT << "src0=" << src0_addr << " src1=" << src1_addr << ENDL();
    //DPRINT << "batch=" << batch << ENDL();

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_in1 = 1;

    constexpr uint32_t onetile = 1;
    uint32_t tile_bytes = get_tile_size(cb_id_in0);

    uint32_t itileA_batch = 0;
    uint32_t itileB_batch = 0;

    const InterleavedPow2AddrGen s0 = {
        .bank_base_address = src0_addr,
        .num_used_banks = 8,
        .log_base_2_of_num_used_banks = 3,
        .log_base_2_of_bank_unit_size = 11
    };

    const InterleavedPow2AddrGen s1 = {
        .bank_base_address = src1_addr,
        .num_used_banks = 8,
        .log_base_2_of_num_used_banks = 3,
        .log_base_2_of_bank_unit_size = 11
    };

    for (uint32_t nb = 0; nb < batch; nb++) {
        uint32_t itileA = itileA_batch;
        for (uint32_t mt = 0; mt < Mt; mt++) {
            uint32_t itileB = itileB_batch;
            for (uint32_t nt = 0; nt < Nt; nt++) {
                for (uint32_t kt = 0; kt < Kt; kt++) {
                    { // Read A's tile at (mt, kt)
                        uint64_t src0_noc_addr = get_noc_addr(itileA, s0);
                        cb_reserve_back(cb_id_in0, onetile);
                        uint32_t l1_write_addr_in0 = get_write_ptr(cb_id_in0);
                        noc_async_read(src0_noc_addr, l1_write_addr_in0, tile_bytes);
                        noc_async_read_barrier();
                        cb_push_back(cb_id_in0, onetile);
                    }

                    { // Read B's tile at (kt, nt)
                        uint64_t src1_noc_addr = get_noc_addr(itileB, s1);
                        cb_reserve_back(cb_id_in1, onetile);
                        uint32_t l1_write_addr_in1 = get_write_ptr(cb_id_in1);
                        noc_async_read(src1_noc_addr, l1_write_addr_in1, tile_bytes);
                        noc_async_read_barrier();
                        cb_push_back(cb_id_in1, onetile);
                    }
                    //DPRINT << "Pushed itileA=" << itileA << " itileB=" << itileB << ENDL();

                    itileA += 1; // A is MK
                    itileB += Nt; // B is KN, so to get k++ we stride by Nt
                } // Kt loop
                itileB -= KtNt; // revert B to previous state before the K loop (to avoid multiplies)
                itileB += 1; // B is KN, so here in the end of Nt loop we increment N by 1
                itileA -= Kt; // resets tileA to kt=0, keep the same mt
            } // Nt loop
            itileA += Kt; // A is MK, advance to next M
        } // Mt loop
        itileA_batch += MtKt; // update batch strides
        if (bcast_B == 0) // don't increment batch if we broadcast matrix B
            itileB_batch += KtNt;
    } // batch loop
}
