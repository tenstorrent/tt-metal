// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Test kernel: DFB producer with a configurable extra transfer, used to scope which uses of a
// Scratchpad disturb a DataflowBuffer the same kernel produces into.
//
// The DFB half of the loop is fixed and correct: reserve_back / async_read / async_read_barrier /
// push_back. Only the extra work varies, driven by runtime arguments, so one binary covers every
// case the tests sweep.
//
// Modes (arg 4):
//   0  nothing besides producing into the buffer
//   1  NoC-read into the scratchpad through its binding
//   2  touch the scratchpad from the CPU only, never a NoC destination
//   3  NoC-read into the scratchpad's own base address through a plain CoreLocalMem
//   4  NoC-read into the scratchpad's binding, at its far end rather than its base
//   5  NoC-read into a plain SRAM address well away from the scratchpad, scratchpad still bound
//   6  NoC-read into a plain SRAM address with NO scratchpad bound (compile with
//      NO_SCRATCHPAD_BINDING=1; the host must also omit the binding and the spec entry)
//
// Modes 3 and 5 separate three things a scratchpad NoC read bundles together. Mode 3 keeps the
// scratchpad bound and targets its own base address, taken from the scratchpad itself, but routes the
// transfer through a plain CoreLocalMem: same address, same size, same transfer, different route.
// Mode 5 keeps the scratchpad bound and targets an unrelated SRAM address instead. Together with
// mode 2 (bound, never a NoC destination) they say whether the trigger is the route, the region, or
// merely having a scratchpad bound while any NoC read is in flight.
//
// The start delay (arg 7) busy-waits before the DFB loop so the consumer reliably reaches its first
// wait_front first. A correct buffer is unaffected by that: it must hold the consumer until the entry
// is readable no matter how late the producer starts. It makes the outcome deterministic instead of
// racy.
//
// Runtime args:
//   arg 0: source DRAM address for the DFB entries
//   arg 1: DRAM bank ID
//   arg 2: number of DFB entries to push
//   arg 3: source DRAM address for the extra transfer
//   arg 4: mode, as above
//   arg 5: bytes for the extra transfer
//   arg 6: do the extra transfer every Nth entry, or 0 for once before the loop
//   arg 7: busy-wait iterations before the DFB loop
//   arg 8: plain SRAM address for mode 5

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/core_local_mem.h"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t bank_id = get_arg_val<uint32_t>(1);
    uint32_t num_entries = get_arg_val<uint32_t>(2);
    uint32_t extra_src_addr = get_arg_val<uint32_t>(3);
    uint32_t mode = get_arg_val<uint32_t>(4);
    uint32_t extra_bytes = get_arg_val<uint32_t>(5);
    uint32_t extra_period = get_arg_val<uint32_t>(6);
    uint32_t start_delay = get_arg_val<uint32_t>(7);
    uint32_t plain_addr = get_arg_val<uint32_t>(8);

    Noc noc;
    AllocatorBank<AllocatorBankType::DRAM> dram_src;
    DataflowBuffer buf(dfb::my_local_dfb_name);
    uint32_t entry_size = buf.get_entry_size();

    // The scratchpad handle has to be compiled out, not branched around: `scratch::staging` exists
    // only when the host declared the binding, and name lookup still runs inside a discarded
    // `if constexpr` branch. Mode 6 is the no-binding cell and needs this define.
#if !defined(NO_SCRATCHPAD_BINDING)
    Scratchpad<volatile uint32_t> pad(scratch::staging);
#endif

    // One lambda so every mode issues the same transfer, differing only in destination and route.
    auto extra_transfer = [&]() {
        switch (mode) {
#if !defined(NO_SCRATCHPAD_BINDING)
            case 1:
                noc.async_read(dram_src, pad, extra_bytes, {.bank_id = bank_id, .addr = extra_src_addr}, {});
                noc.async_read_barrier();
                break;
            case 2:
                // Bound and touched, but never a NoC destination.
                pad[0] = extra_src_addr;
                break;
            case 3: {
                // The scratchpad's own address, reached without the binding.
                CoreLocalMem<volatile uint32_t> plain{static_cast<uintptr_t>(pad.get_base_address())};
                noc.async_read(dram_src, plain, extra_bytes, {.bank_id = bank_id, .addr = extra_src_addr}, {});
                noc.async_read_barrier();
                break;
            }
#endif  // !NO_SCRATCHPAD_BINDING
            case 5:
            case 6: {
                // An unrelated SRAM address. Mode 5 has the scratchpad bound but untouched; mode 6 has
                // no scratchpad at all, which is the cell that says whether the binding is required.
                CoreLocalMem<volatile uint32_t> elsewhere{static_cast<uintptr_t>(plain_addr)};
                noc.async_read(dram_src, elsewhere, extra_bytes, {.bank_id = bank_id, .addr = extra_src_addr}, {});
                noc.async_read_barrier();
                break;
            }
#if !defined(NO_SCRATCHPAD_BINDING)
            case 4:
                noc.async_read(
                    dram_src,
                    pad,
                    extra_bytes,
                    {.bank_id = bank_id, .addr = extra_src_addr},
                    {.offset_bytes = pad.size_in_bytes() - extra_bytes});
                noc.async_read_barrier();
                break;
#endif
            default: break;
        }
    };

    if (mode != 0 && extra_period == 0) {
        extra_transfer();
    }

    // Empty asm with a memory clobber rather than a volatile counter: it cannot be optimized away and
    // avoids the deprecated increment of a volatile object.
    for (uint32_t d = 0; d < start_delay; d++) {
        asm volatile("" ::: "memory");
    }

    for (uint32_t i = 0; i < num_entries; i++) {
        if (mode != 0 && extra_period > 0 && (i % extra_period) == 0) {
            extra_transfer();
        }
        buf.reserve_back(1);
        noc.async_read(dram_src, buf, entry_size, {.bank_id = bank_id, .addr = src_addr}, {});
        noc.async_read_barrier();
        buf.push_back(1);
        src_addr += entry_size;
    }
}
