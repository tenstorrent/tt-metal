// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// One DRISC doing the whole job: NoC-read whole cores out of worker L1 straight into a socket page and
// push it to the host. No DRAM hop, no second DRISC.
//
// This is the direct alternative to the A -> DRAM -> B -> host pipeline. Since the host consumer is the
// wall in both, the question is whether the DRAM round-trip earns anything.
//
// One page is kCoresPerPage whole-core reads, so the socket page size and the read batch are the same
// object. Buffers are reused every kNumBuffers iterations; noc_async_write_barrier() frees one before
// refill (pcie_noc_utils passes update_counter=true and posted=false, so the counter tracks these
// writes and the barrier is meaningful).
//
// The NIU must already be in stream mode -- set by a prior program so the socket's config write can
// land in L1 -- so this kernel neither sets nor restores it.

#include <cstdint>

#include "api/compile_time_args.h"
#include "api/core_local_mem.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/socket_api.h"
#include "internal/tt-1xx/risc_common.h"
#include "pcie_noc_utils.h"

// DRISC firmware doesn't define cb_interface (no CB infra on DRAM cores).
CBInterface cb_interface[NUM_CIRCULAR_BUFFERS] __attribute__((used));

void kernel_main() {
    constexpr uint32_t kCoresPerPage = get_compile_time_arg_val(0);
    constexpr uint32_t kBytesPerCore = get_compile_time_arg_val(1);
    constexpr uint32_t kNumBuffers = get_compile_time_arg_val(2);
    constexpr uint32_t kL1Buf = get_compile_time_arg_val(3);
    constexpr uint32_t kSocketConfigAddr = get_compile_time_arg_val(4);
    constexpr uint32_t kResultsAddr = get_compile_time_arg_val(5);

    constexpr uint32_t kPageBytes = kCoresPerPage * kBytesPerCore;
    static_assert(kBytesPerCore <= NOC_MAX_BURST_SIZE, "per-core read must fit one NoC packet");
    static_assert(kNumBuffers == 1 || kNumBuffers == 2, "one or two page buffers");

    const uint32_t num_cores = get_arg_val<uint32_t>(0);
    const uint32_t num_sweeps = get_arg_val<uint32_t>(1);
    const uint32_t src_addr = get_arg_val<uint32_t>(2);
    volatile tt_l1_ptr uint32_t* coords = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_addr(3));

    Noc noc;
    UnicastEndpoint src;

    SocketSenderInterface sender = create_sender_socket_interface(kSocketConfigAddr);
    const uint32_t pcie_xy_enc = sender.d2h.pcie_xy_enc;
    const uint64_t pcie_base = (static_cast<uint64_t>(sender.d2h.data_addr_hi) << 32) | sender.downstream_fifo_addr;
    set_sender_socket_page_size(sender, kPageBytes);

    uint64_t t_wbar = 0;
    uint64_t t_read = 0;
    uint64_t t_reserve = 0;
    uint64_t t_push = 0;
    uint32_t pages = 0;
    uint32_t cur = 0;

    const uint64_t t_start = get_timestamp();
    for (uint32_t sweep = 0; sweep < num_sweeps; sweep++) {
        uint32_t core = 0;
        while (core < num_cores) {
            const uint32_t base = kL1Buf + cur * kPageBytes;

            const uint64_t w0 = get_timestamp();
            noc_async_write_barrier();  // the page written out of this buffer must have landed
            const uint64_t w1 = get_timestamp();

            for (uint32_t n = 0; n < kCoresPerPage; n++, core++) {
                const uint32_t xy = coords[core];
                CoreLocalMem<uint32_t> dst(base + n * kBytesPerCore);
                noc.async_read<NocOptions::DEFAULT, kBytesPerCore>(
                    src, dst, kBytesPerCore, {.noc_x = xy & 0xFFFFu, .noc_y = xy >> 16, .addr = src_addr}, {});
            }
            noc.async_read_barrier();
            const uint64_t w2 = get_timestamp();

            socket_reserve_pages(sender, 1);
            const uint64_t w3 = get_timestamp();

            noc_write_page_chunked(pcie_xy_enc, base, pcie_base + sender.write_ptr, kPageBytes);
            socket_push_pages(sender, 1);
            socket_notify_receiver(sender);
            const uint64_t w4 = get_timestamp();

            t_wbar += w1 - w0;
            t_read += w2 - w1;
            t_reserve += w3 - w2;
            t_push += w4 - w3;
            pages++;
            if constexpr (kNumBuffers == 2) {
                cur ^= 1;
            }
        }
    }
    socket_barrier(sender);
    const uint64_t t_end = get_timestamp();

    const uint64_t cycles = t_end - t_start;
    volatile tt_l1_ptr uint32_t* out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kResultsAddr);
    out[0] = static_cast<uint32_t>(cycles & 0xFFFFFFFFu);
    out[1] = static_cast<uint32_t>(cycles >> 32);
    out[2] = pages;
    out[3] = static_cast<uint32_t>(t_wbar & 0xFFFFFFFFu);
    out[4] = static_cast<uint32_t>(t_read & 0xFFFFFFFFu);
    out[5] = static_cast<uint32_t>(t_read >> 32);
    out[6] = static_cast<uint32_t>(t_reserve & 0xFFFFFFFFu);
    out[7] = static_cast<uint32_t>(t_reserve >> 32);
    out[8] = static_cast<uint32_t>(t_push & 0xFFFFFFFFu);
    out[9] = static_cast<uint32_t>(t_push >> 32);

    update_socket_config(sender);
}
