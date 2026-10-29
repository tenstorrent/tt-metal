// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// row_reduce + gr_write, helper core of branch b (1..3): reader (NCRISC) reads the branch's dense residual tile (the
// valid-row chunks of its 10 residual tiles, chunk c at c * 64 B) and the injection tile, builds the coefficient tile
// (column 0 = the branch's injection value of the dense row's token row), then waits for the reduce core's block tile
// (the reduce core writes it into this core's CB_A and steps SEM_HELP) and pushes it; the compute kernel (compute3.cpp,
// HELPER) does the branch's gr_write math; the writer (BRISC) scatters the new residual tile's chunks to the residual
// output pages 10 b + j (face offsets).  Compile-time args: 0 CB_A, 1 CB_COEF, 2 CB_RES, 3 CB_NEW, 4 ROWS, 5 OUT_TILES,
// 6 SEM_HELP, 7.. accessors of the residual, the injection and the new residual.  Runtime args: 0 residual, 1
// injection, 2 new residual addresses, 3 branch.

#include <cstdint>

#include "api/compile_time_args.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/tensor/noc_traits.h"
#include "../../kernels/zones.h"

constexpr uint32_t CB_A = get_compile_time_arg_val(0);
constexpr uint32_t CB_COEF = get_compile_time_arg_val(1);
constexpr uint32_t CB_RES = get_compile_time_arg_val(2);
constexpr uint32_t CB_NEW = get_compile_time_arg_val(3);
constexpr uint32_t ROWS = get_compile_time_arg_val(4);
constexpr uint32_t OUT_TILES = get_compile_time_arg_val(5);
constexpr uint32_t SEM_HELP = get_compile_time_arg_val(6);
constexpr uint32_t BF_TILE = 2048, BF_FACE = 512, BF_CHUNK = ROWS * 32;

void kernel_main() {
    constexpr auto res_args = TensorAccessorArgs<7>();
    constexpr auto inj_args = TensorAccessorArgs<res_args.next_compile_time_args_offset()>();
    constexpr auto new_args = TensorAccessorArgs<inj_args.next_compile_time_args_offset()>();
    const auto residual = TensorAccessor(res_args, get_arg_val<uint32_t>(0));
    const auto injection = TensorAccessor(inj_args, get_arg_val<uint32_t>(1));
    const auto newres = TensorAccessor(new_args, get_arg_val<uint32_t>(2));
    const uint32_t b = get_arg_val<uint32_t>(3);
#if !defined(COMPILE_FOR_BRISC)
    {
        FUSED_ZONE("fz_rr_wh_read");
        Semaphore<> help(SEM_HELP);
        DataflowBuffer cres(CB_RES), ccoef(CB_COEF), ca(CB_A);
        cres.reserve_back(1);
        const uint32_t res_base = cres.get_write_ptr();
        ccoef.reserve_back(1);
        const uint32_t coef_base = ccoef.get_write_ptr();
        noc_async_read(injection.get_noc_addr(0, 0), coef_base, BF_TILE);
        for (uint32_t j = 0; j < OUT_TILES; ++j) {
            noc_async_read(residual.get_noc_addr(OUT_TILES * b + j, 0), res_base + (2 * j) * 64, BF_CHUNK);
            noc_async_read(residual.get_noc_addr(OUT_TILES * b + j, BF_FACE), res_base + (2 * j + 1) * 64, BF_CHUNK);
        }
        noc_async_read_barrier();
        volatile tt_l1_ptr uint16_t* coef = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(coef_base);
        const uint16_t v0 = coef[b], v1 = coef[16 + b];  // injection (token row 0 / 1, column b)
        for (uint32_t q = 0; q < 32; ++q) {
            coef[(q >> 4) * 512 + (q & 15) * 16] = (ROWS > 1 && (q & 1)) ? v1 : v0;
        }
        ccoef.push_back(1);
        cres.push_back(1);
        help.wait(1);  // the block tile is in CB_A
        help.set(0);
        ca.reserve_back(1);
        ca.push_back(1);
    }
#else
    {
        FUSED_ZONE("fz_rr_wh_write");
        DataflowBuffer cnew(CB_NEW);
        cnew.wait_front(1);
        const uint32_t nbase = cnew.get_read_ptr();
        for (uint32_t j = 0; j < OUT_TILES; ++j) {
            noc_async_write(nbase + (2 * j) * 64, newres.get_noc_addr(OUT_TILES * b + j, 0), BF_CHUNK);
            noc_async_write(nbase + (2 * j + 1) * 64, newres.get_noc_addr(OUT_TILES * b + j, BF_FACE), BF_CHUNK);
        }
        noc_async_write_barrier();
        cnew.pop_front(1);
    }
#endif
}
