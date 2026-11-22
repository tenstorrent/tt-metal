// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Copy of stage_linear_reader.cpp whose staged prefix comes from a tensor prefetcher pipe (tt/mk/pipe_stage.py):
// the DRAM cores stream this core's weight range into the pipe's L1 ring while other programs run, in entries of
// `entry` bytes, one slab of `n_entries` entries per linear call. A block is read from the ring once the entries
// holding it have landed (the slab starts at the pipe's read pointer and may wrap at the ring's end); the rest of
// the range (past `staged`) comes from DRAM as before. Same bytes into the same cb_w slots, so the output is
// unchanged. At the end the whole slab (padding included) is popped, which hands its entries back to the sender.
// Runtime args: weight address, bank, first tile slot, pipe config page address, ring address, staged bytes,
// number of passes, ncq per pass.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/prefetcher_pipe.h"

constexpr uint32_t cb_w = 1;
constexpr uint32_t kb = get_named_compile_time_arg_val("kb");
constexpr uint32_t nblk = get_named_compile_time_arg_val("nblk");
constexpr uint32_t nb = get_named_compile_time_arg_val("n_banks");
constexpr uint32_t max_ncq = get_named_compile_time_arg_val("max_ncq");
constexpr uint32_t ENTRY = get_named_compile_time_arg_val("entry");
constexpr uint32_t N_ENTRIES = get_named_compile_time_arg_val("n_entries");
constexpr uint32_t RING = get_named_compile_time_arg_val("ring_bytes");
constexpr uint32_t SLOT = kb * max_ncq;
constexpr uint32_t NSLOT = 4, IN_FLIGHT = 3;

void kernel_main() {
    uint32_t a = 0;
    const uint32_t w_addr = get_arg_val<uint32_t>(a++);
    const uint32_t bank = get_arg_val<uint32_t>(a++);
    uint32_t slot = get_arg_val<uint32_t>(a++);
    const uint32_t cfg = get_arg_val<uint32_t>(a++);
    const uint32_t ring = get_arg_val<uint32_t>(a++);
    const uint32_t staged = get_arg_val<uint32_t>(a++);
    const uint32_t npass = get_arg_val<uint32_t>(a++);
    uint32_t done = 0;  // bytes of this core's range already handed out

    constexpr auto w_args = TensorAccessorArgs<0>();
    const auto ws = TensorAccessor(w_args, w_addr);
    experimental::PrefetcherPipe pipe(experimental::ReceiverConfigPage{cfg}, ENTRY);
    const uint32_t start = pipe.get_read_ptr() - ring;  // slab offset in the ring
    uint32_t landed = 0;                                // entries known to have landed

    reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, noc_index);
    const uint32_t tb = get_tile_size(cb_w);
    const uint32_t base = get_write_ptr(cb_w);
    uint32_t issued = 0, retired = 0;  // blocks
    for (uint32_t q = 0; q < npass; ++q) {
        const uint32_t ncq = get_arg_val<uint32_t>(a + q);
        const uint32_t bytes = kb * ncq * tb;
        for (uint32_t p = 0; p < nblk; ++p) {
            cb_reserve_back(cb_w, SLOT * (issued - retired + 1));  // this block's slot plus the ones in flight
            noc_async_read_set_trid(issued % NSLOT + 1);
            const uint32_t dst = base + (issued % NSLOT) * SLOT * tb;
            const uint32_t s0 = done >= staged ? 0 : (done + bytes <= staged ? bytes : staged - done);
            if (s0) {
                const uint32_t need = (done + s0 + ENTRY - 1) / ENTRY;
                if (need > landed) {
                    pipe.wait_front(need);
                    landed = need;
                }
                uint32_t off = start + done;
                off = off >= RING ? off - RING : off;
                const uint32_t s1 = off + s0 > RING ? RING - off : s0;  // bytes before the ring's end
                noc_async_read(get_noc_addr(ring + off), dst, s1);
                if (s1 < s0) {
                    noc_async_read(get_noc_addr(ring), dst + s1, s0 - s1);
                }
            }
            if (s0 < bytes) {
                noc_async_read(ws.get_noc_addr(slot * nb + bank) + s0, dst + s0, bytes - s0);
            }
            done += bytes;
            slot += kb * ncq;
            ++issued;
            if (issued - retired == IN_FLIGHT) {
                noc_async_read_barrier_with_trid(retired % NSLOT + 1);
                cb_push_back(cb_w, SLOT);
                ++retired;
            }
        }
    }
    while (retired < issued) {
        noc_async_read_barrier_with_trid(retired % NSLOT + 1);
        cb_push_back(cb_w, SLOT);
        ++retired;
    }
    noc_async_read_set_trid(0);
    pipe.wait_front(N_ENTRIES);  // the padding entries too, then hand the whole slab back
    pipe.pop_front(N_ENTRIES);
}
