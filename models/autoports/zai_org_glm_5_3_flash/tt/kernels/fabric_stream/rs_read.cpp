// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Stage 7a FabricStream RS with the lane reduce (tt/fabric_stream.FabricStreamRSR; prefill_megakernel.md §4b H5 -> H6),
// lane NCRISC. Interleaved per step i (the lane's lists, both in increasing block order):
//   send i:   wait LREADY + i >= g (its producer finished peer-owned block send[i]), stream its TPB tiles of the
//   partial
//             P (DRAM) into CB_SEND (NB-tile batches) for rs_send.cpp;
//   reduce i: wait the peer's copy of own block red[i] landed (ARR + q LANES + h: the peer's lane h sends this lane's
//             reduce list in this order, one fused +1 per block) and OWNRDY + i >= g (own partial complete), stream own
//             (P) and landed (r) tiles as matched batches into CB_A / CB_B for rs_reduce.cpp.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "fs_common.hpp"

#if !defined(MK_PHASED)  // standalone; a composite (tt/pf_stream.composite_source) defines both per phase namespace
constexpr uint32_t PH_CT0 = 0;
uint32_t ph_rt0 = 0;
#endif

constexpr uint32_t TPB = get_compile_time_arg_val(PH_CT0 + 0);
constexpr uint32_t NB = get_compile_time_arg_val(PH_CT0 + 1);
constexpr uint32_t CB_SEND = get_compile_time_arg_val(PH_CT0 + 2);
constexpr uint32_t CB_A = get_compile_time_arg_val(PH_CT0 + 3);
constexpr uint32_t CB_B = get_compile_time_arg_val(PH_CT0 + 4);
constexpr uint32_t TB = 2048;
static_assert(TPB % NB == 0, "a batch never straddles a block");
#if !defined(FS_RDY_MULT)
#define FS_RDY_MULT 1  // readiness increments per block per launch (P15 streamed: B's combine units per row block)
#endif

void kernel_main() {
    constexpr auto p_args = TensorAccessorArgs<PH_CT0 + 5>();
    constexpr auto r_args = TensorAccessorArgs<p_args.next_compile_time_args_offset()>();
    size_t a = ph_rt0;
    const uint32_t sync = get_arg_val<uint32_t>(a++);
    const uint32_t q = get_arg_val<uint32_t>(a++);  // the peer
    const uint32_t lanes = get_arg_val<uint32_t>(a++);
    const uint32_t h = get_arg_val<uint32_t>(a++);
    const uint32_t p_addr = get_arg_val<uint32_t>(a++);
    const uint32_t r_addr = get_arg_val<uint32_t>(a++);
    const uint32_t ready =
        get_arg_val<uint32_t>(a++);  // 0: the producers' readiness words are not used (unit: P prefilled)
    const uint32_t n_send = get_arg_val<uint32_t>(a++);
    const uint32_t n_red = get_arg_val<uint32_t>(a++);
    const uint32_t send_j = a;  // n_send global blocks (P row tiles)
    a += n_send;
    const uint32_t red_j = a;  // n_red: own P row tile, landed r row tile
    const auto pp = TensorAccessor(p_args, p_addr);
    const auto rr = TensorAccessor(r_args, r_addr);
    volatile tt_l1_ptr uint32_t* gen = fs::word(sync, fs::GEN_NC);
    const uint32_t g = *gen + 1;
    const uint32_t n = n_send > n_red ? n_send : n_red;
#if defined(FS_SEND_FIRST)
    // the streamed P15 (B's combine lands the half this chip sends first): every send, then every reduce -- an
    // interleaved send[i] / reduce[i] blocked the sends behind own blocks that land last (the landing buffer r holds
    // every block)
    for (uint32_t pass = 0; pass < 2; pass++) {
        for (uint32_t i = 0; i < n; i++) {
            if (pass == 0 && i < n_send) {
#else
    for (uint32_t i = 0; i < n; i++) {
        if (i < n_send) {
#endif
                if (ready) {
                    fs::wait_ge(fs::word(sync, fs::LREADY + i), g * FS_RDY_MULT);
                }
                const uint32_t j = get_arg_val<uint32_t>(send_j + i);
                for (uint32_t b = 0; b < TPB / NB; b++) {
                    cb_reserve_back(CB_SEND, NB);
                    const uint32_t wp = get_write_ptr(CB_SEND);
                    for (uint32_t t = 0; t < NB; t++) {
                        noc_async_read_page(j * TPB + b * NB + t, pp, wp + t * TB);
                    }
                    noc_async_read_barrier();
                    cb_push_back(CB_SEND, NB);
                }
            }
#if defined(FS_SEND_FIRST)
            if (pass == 1 && i < n_red) {
#else
        if (i < n_red) {
#endif
                if (ready) {
                    fs::wait_ge(fs::word(sync, fs::OWNRDY + i), g * FS_RDY_MULT);
                }
                fs::wait_ge(fs::word(sync, fs::ARR + q * lanes + h), (g - 1) * n_red + i + 1);
                const uint32_t jo = get_arg_val<uint32_t>(red_j + 2 * i), jl = get_arg_val<uint32_t>(red_j + 2 * i + 1);
                for (uint32_t b = 0; b < TPB / NB; b++) {
                    cb_reserve_back(CB_A, NB);
                    cb_reserve_back(CB_B, NB);
                    const uint32_t wa = get_write_ptr(CB_A), wb = get_write_ptr(CB_B);
                    for (uint32_t t = 0; t < NB; t++) {
                        noc_async_read_page(jo * TPB + b * NB + t, pp, wa + t * TB);
                        noc_async_read_page(jl * TPB + b * NB + t, rr, wb + t * TB);
                    }
                    noc_async_read_barrier();
                    cb_push_back(CB_A, NB);
                    cb_push_back(CB_B, NB);
                }
            }
        }
    }
    *gen = g;
}
