// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Times one data-movement core's NoC throughput. Mode 0 writes `total_bytes` from its scratchpad to the same
// scratchpad on a peer core, `tx_bytes` per write; mode 1 reads `total_bytes` from a DRAM bank into its scratchpad,
// `tx_bytes` per read. Every transfer is issued before any is waited on, and the transfers cycle through the
// scratchpad (and the DRAM buffer), so only the issue rate and the NoC bound the time. Records, in the last 64 B of
// the scratchpad: cycles to issue every transfer, cycles until every write left this core (mode 0) or every read
// landed (mode 1), and cycles until every write was acked (mode 0), and copies them to the start of the DRAM buffer.
// With several threads (KernelSpec::num_threads), each DM core moves an equal share and reports at 64 B * its id.
//
// Runtime args: role (0 times, anything else returns at once), peer NoC x, peer NoC y, DRAM buffer address,
// DRAM buffer bytes.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/scratchpad.h"
#include "experimental/kernel_args.h"
#include "internal/local_mem/address_access.h"

void kernel_main() {
    constexpr uint32_t total_bytes = get_arg(args::total_bytes);
    constexpr uint32_t tx_bytes = get_arg(args::tx_bytes);
    constexpr uint32_t mode = get_arg(args::mode);
    // Mode 0 writes cycle through request VCs [0, num_vc), one VC per write.
    constexpr uint32_t num_vc = get_arg(args::num_vc);
    if (get_arg_val<uint32_t>(0) != 0) {
        return;
    }
    const uint32_t peer_x = get_arg_val<uint32_t>(1);
    const uint32_t peer_y = get_arg_val<uint32_t>(2);
    const uint32_t dram_addr = get_arg_val<uint32_t>(3);
    const uint32_t dram_bytes = get_arg_val<uint32_t>(4);

    // With several threads (DM cores), each moves its share through its own part of the scratchpad, and reports at
    // its own 64 B of the result area and of the DRAM buffer.
    const uint32_t threads = get_num_threads();
    const uint32_t tid = get_my_thread_id();
    Scratchpad<uint32_t> scratchpad(scratch::buf);
    const uint32_t base = static_cast<uint32_t>(local_mem::detail::AddressAccess::address(scratchpad.local_mem()));
    const uint32_t results = base + scratchpad.size_in_bytes() - 64 * threads + 64 * tid;
    const uint32_t part = (scratchpad.size_in_bytes() - 64 * threads) / threads;
    const uint32_t buf = base + tid * part;
    const uint32_t span = part / tx_bytes * tx_bytes;
    const uint32_t dram_part = dram_bytes / threads / tx_bytes * tx_bytes;
    const uint32_t dram_base = dram_addr + tid * dram_part;

    Noc noc;
    UnicastEndpoint endpoint;
    const uint64_t t0 = get_timestamp();
    uint32_t off = 0;
    uint32_t dram_off = 0;
    uint32_t vc = 0;
    for (uint32_t sent = 0; sent < total_bytes / threads; sent += tx_bytes) {
        if constexpr (mode == 0) {
            noc.async_write<NocOptions::CUSTOM_VC>(
                endpoint,
                endpoint,
                tx_bytes,
                {.addr = buf + off},
                {.noc_x = peer_x, .noc_y = peer_y, .addr = buf + off},
                NocOptVals{.vc = num_vc == 1 ? NOC_UNICAST_WRITE_VC : vc});
            vc = vc + 1 == num_vc ? 0 : vc + 1;
        } else {
            noc_async_read(get_noc_addr_from_bank_id<true>(0, dram_base + dram_off), buf + off, tx_bytes);
            dram_off = dram_off + tx_bytes >= dram_part ? 0 : dram_off + tx_bytes;
        }
        off = off + tx_bytes >= span ? 0 : off + tx_bytes;
    }
    const uint64_t t1 = get_timestamp();
    uint64_t t2 = 0;
    uint64_t t3 = 0;
    if constexpr (mode == 0) {
        noc.async_writes_flushed();
        t2 = get_timestamp();
        noc.async_write_barrier();
        t3 = get_timestamp();
    } else {
        noc.async_read_barrier();
        t2 = get_timestamp();
        t3 = t2;
    }
#ifdef ARCH_QUASAR
    volatile tt_l1_ptr uint32_t* result =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(results + MEM_L1_UNCACHED_BASE);
#else
    volatile tt_l1_ptr uint32_t* result = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(results);
#endif
    result[0] = static_cast<uint32_t>(t1 - t0);
    result[1] = static_cast<uint32_t>(t2 - t0);
    result[2] = static_cast<uint32_t>(t3 - t0);
    // The host reads the counts from the start of the DRAM buffer.
    noc_async_write(results, get_noc_addr_from_bank_id<true>(0, dram_addr + 64 * tid), 64);
    noc_async_write_barrier();
}
