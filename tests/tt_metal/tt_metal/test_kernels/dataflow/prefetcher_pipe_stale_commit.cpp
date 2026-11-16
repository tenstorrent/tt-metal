// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Sender-only kernel that proves commit() rejects a stale entry_size epoch.
//
// Bindings:
//   pipe::out              — KernelAdvancedOptions::PrefetcherPipeBinding accessor (program slot id baked in)
// Args (named CTAs):
//   args::entry_size       - initial / dense-slot size (E1)
//   args::new_entry_size   - resize target (E2); must differ from E1
//   args::poison_wr_ptr    - value that must NOT land in word[4] on stale commit
// Args (named RTAs):
//   args::staging_addr     - sender-local L1 staging base
// Defines:
//   PREFETCHER_PIPE_TEST_HELPERS - exposes the test-only friend used below

#include "api/dataflow/prefetcher_pipe.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "experimental/kernel_args.h"

namespace experimental {

// This test-only friend is declared by PrefetcherPipe only when
// PREFETCHER_PIPE_TEST_HELPERS is defined. It is intentionally not part of the
// production object API.
FORCE_INLINE void test_stale_commit_after_resize(
    PrefetcherPipe& dfb, uint32_t new_entry_size, uint32_t stale_entry_size, uint32_t poison_wr_ptr) {
    CrossNodeSenderDFBInterface& iface = dfb.interface_.sender;
    ASSERT(
        static_cast<bool>(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(iface.config_ptr)[REMOTE_DFB_CFG_IS_SENDER]));

    // Persist the real post-push cursor first.
    dfb.commit();

    // Change the live epoch without touching peer credits; this test exercises
    // stale-epoch rejection only.
    dfb.resize_sender_interface<false>(new_entry_size, noc_index);
    volatile tt_l1_ptr uint32_t* config = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(iface.config_ptr);
    store_prefetcher_pipe_config_word(config, PREFETCHER_PIPE_CFG_APPLIED_ENTRY_SIZE, iface.fifo_page_size);

    // Move this receiver's stored cursor -- what commit() persists -- to a distinct, valid slot,
    // so a stale commit that got through would be visible in word[4].
    volatile tt_l1_ptr uint32_t* wr_offset_ptr = dfb.local_wr_offset_ptr(iface, 0);
    const uint32_t saved_wr_offset = *wr_offset_ptr;
    ASSERT(poison_wr_ptr >= iface.fifo_start_addr);
    ASSERT(poison_wr_ptr < iface.fifo_limit_page_aligned);
    ASSERT((poison_wr_ptr - iface.fifo_start_addr) % L1_ALIGNMENT == 0);
    *wr_offset_ptr = poison_wr_ptr - iface.fifo_start_addr;

    const uint32_t live_entry_size = iface.fifo_page_size;
    iface.fifo_page_size = stale_entry_size;
    dfb.commit();

    iface.fifo_page_size = live_entry_size;
    *wr_offset_ptr = saved_wr_offset;
}

}  // namespace experimental

void kernel_main() {
    constexpr uint32_t entry_size = get_arg(args::entry_size);
    constexpr uint32_t new_entry_size = get_arg(args::new_entry_size);
    constexpr uint32_t poison_wr_ptr = get_arg(args::poison_wr_ptr);
    const uint32_t staging_base = get_arg(args::staging_addr);
    const CoreLocalMem<uint8_t> staging(staging_base);

    static_assert(entry_size != new_entry_size, "stale-commit test requires distinct entry sizes");

    Noc noc;
    experimental::PrefetcherPipe dfb(pipe::out);

    // Advance one entry so the durable checkpoint is not fifo_start.
    dfb.reserve_back(1);
    dfb.write_to_receiver(noc, 0, staging, 1);
    dfb.flush_writes(noc);
    dfb.push_back(1, noc);

    // A stale entry-size epoch must not overwrite word[4] with poison_wr_ptr.
    experimental::test_stale_commit_after_resize(dfb, new_entry_size, entry_size, poison_wr_ptr);
    // ~PrefetcherPipe commits the cursor the helper restored, under the live E2 epoch.
}
