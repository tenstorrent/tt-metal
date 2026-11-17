// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Fused exchange + merge (fuse-sparse-merge A2), merge-core reader, V1 = full 32x32 tiles (the validated path: 13/13
// PCC cases, run_x2.log). Landing buffer CB_LAND = [8 source slots][TPC o tiles, ms, l] (aliases a persistent L1
// tensor; the 7 remote sources write their slot by fabric with a fused global-semaphore increment per packet). This
// core (user s, tile columns g*TPC..) reads its OWN source slot from DRAM (x = the local split-kernel output, owner
// block c), the sink tile, builds the ones tile, waits for 7 x (TPC/2 + 1) packets, re-arms the semaphore and releases
// the CB to compute.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t S = get_compile_time_arg_val(0);
    constexpr uint32_t TPC = get_compile_time_arg_val(1);
    constexpr uint32_t CB_ST = get_compile_time_arg_val(2);
    constexpr uint32_t CB_LAND = get_compile_time_arg_val(3);
    constexpr uint32_t CB_ONE = get_compile_time_arg_val(4);
    constexpr uint32_t DBG = get_compile_time_arg_val(5);
    constexpr uint32_t HALF = get_compile_time_arg_val(6);
    constexpr uint32_t DRAML = get_compile_time_arg_val(
        8);  // 1: remote slots are read from the DRAM landing (4 KiB pages (m * 8 + c) * 2 + p) after their semaphore
    constexpr uint32_t SLOTSEM =
        get_compile_time_arg_val(7);  // 1: one global semaphore per SOURCE slot, runtime args 3..10 = their L1
                                      // addresses; slot pages are pushed in slot order as they arrive
    constexpr uint32_t UNIT = HALF ? 1024 : 2048;  // landing pitch of a tile
    constexpr uint32_t NPK = HALF ? 1 : (TPC / 2 + 1);
    constexpr auto x_args = TensorAccessorArgs<9, 4>();
    constexpr auto s_args =
        TensorAccessorArgs<x_args.next_compile_time_args_offset(), x_args.next_common_runtime_args_offset()>();
    constexpr auto l_args =
        TensorAccessorArgs<s_args.next_compile_time_args_offset(), s_args.next_common_runtime_args_offset()>();
    constexpr uint32_t NC = 8, TILE_B = 2048, TPR = 18, LT = TPC + 2;
    const uint32_t x_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t s_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t gsem = get_common_arg_val<uint32_t>(2);
    const uint32_t s = get_arg_val<uint32_t>(0);
    const uint32_t g = get_arg_val<uint32_t>(1);
    const uint32_t c = get_arg_val<uint32_t>(2);  // ring position of this chip
    const auto xa = TensorAccessor(x_args, x_addr, TILE_B);
    const auto sa = TensorAccessor(s_args, s_addr, TILE_B);
    const auto la = TensorAccessor(l_args, DRAML ? get_common_arg_val<uint32_t>(3) : 0, 4096);
    constexpr uint32_t NSPLIT_R = 16 / TPC;
    CircularBuffer st(CB_ST), land(CB_LAND), one(CB_ONE);
    DeviceZoneScopedN("XR_ALL");
    land.reserve_back(NC * LT);
    const uint32_t p = land.get_write_ptr();
    const uint32_t base = (c * S + s) * TPR;
    for (uint32_t i = 0; i < TPC; ++i) {
        noc_async_read(xa.get_noc_addr(base + g * TPC + i), p + (c * LT + i) * UNIT, UNIT);
    }
    noc_async_read(xa.get_noc_addr(base + 16), p + (c * LT + TPC) * UNIT, UNIT);
    noc_async_read(xa.get_noc_addr(base + 17), p + (c * LT + TPC + 1) * UNIT, UNIT);
    st.reserve_back(1);
    noc_async_read(sa.get_noc_addr(s), st.get_write_ptr(), TILE_B);
    one.reserve_back(1);
    {
        volatile tt_l1_ptr uint32_t* po = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(one.get_write_ptr());
        for (uint32_t i = 0; i < TILE_B / 4; ++i) {
            po[i] = 0x3F803F80u;  // bf16 1.0 x 2
        }
    }
    one.push_back(1);
    noc_async_read_barrier();
    st.push_back(1);
    if constexpr (SLOTSEM) {
        DeviceZoneScopedN("XR_WAIT");
        if constexpr (DRAML) {
            // read-ahead: the DRAM reads of every slot whose semaphore is already set are issued together, one barrier,
            // then all those slots are pushed
            uint32_t issued = 0, pushed = 0;
            auto handle = [&](uint32_t cc) {
                if (cc != c) {
                    volatile tt_l1_ptr uint32_t* sm =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(3 + cc));
                    noc_semaphore_set(sm, 0);
                    const uint32_t m = s * NSPLIT_R + g;
                    noc_async_read(la.get_noc_addr((m * NC + cc) * 2), p + cc * LT * TILE_B, 4096);
                    noc_async_read(la.get_noc_addr((m * NC + cc) * 2 + 1), p + cc * LT * TILE_B + 4096, 4096);
                }
            };
            while (pushed < NC) {
                if (issued == pushed) {
                    if (issued != c) {
                        volatile tt_l1_ptr uint32_t* sm =
                            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(3 + issued));
                        noc_semaphore_wait_min(sm, NPK);
                    }
                    handle(issued);
                    ++issued;
                }
                while (issued < NC) {
                    if (issued != c) {
                        volatile tt_l1_ptr uint32_t* sm =
                            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(3 + issued));
                        if (*sm < NPK) {
                            break;
                        }
                    }
                    handle(issued);
                    ++issued;
                }
                noc_async_read_barrier();
                land.push_back((issued - pushed) * LT);
                pushed = issued;
            }
        } else {
            for (uint32_t cc = 0; cc < NC; ++cc) {
                if (cc != c) {
                    volatile tt_l1_ptr uint32_t* sem =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(3 + cc));
                    noc_semaphore_wait_min(sem, NPK);
                    noc_semaphore_set(sem, 0);
                }
                land.push_back(LT);
            }
        }
    } else {
        volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(gsem);
        if constexpr (DBG == 0) {
            DeviceZoneScopedN("XR_WAIT");
            noc_semaphore_wait_min(sem, 7 * NPK);
            noc_semaphore_set(sem, 0);
        }
        land.push_back(NC * LT);
    }
}
