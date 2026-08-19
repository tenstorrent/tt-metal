// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Sender-only kernel that proves commit() rejects a stale entry_size epoch.
//
// Compile-time args:
//   [0] persistent_dfb_id
//   [1] entry_size          - initial / dense-slot size (E1)
//   [2] new_entry_size      - resize target (E2); must differ from E1
//   [3] poison_wr_ptr       - value that must NOT land in word[4] on stale commit
//
// Runtime args:
//   [0] l1_staging_addr

#include "api/dataflow/persistent_dfb.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"

namespace experimental {

// This test-only friend is declared by PersistentDFB only when
// PERSISTENT_DFB_TEST_HELPERS is defined. It is intentionally not part of the
// production object API.
FORCE_INLINE void test_stale_commit_after_resize(
    PersistentDFB& dfb, uint32_t new_entry_size, uint32_t stale_entry_size, uint32_t poison_wr_ptr) {
    CrossNodeSenderDFBInterface& iface = dfb.interface_.sender;
    ASSERT(
        static_cast<bool>(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(iface.config_ptr)[REMOTE_DFB_CFG_IS_SENDER]));

    // Persist the real post-push cursor first.
    dfb.commit();

    // Change the live epoch without touching peer credits; this test exercises
    // stale-epoch rejection only.
    iface.fifo_wr_ptr = iface.fifo_start_addr + dfb.derived_wr_offset(iface, 0);
    dfb.resize_sender_interface<false>(new_entry_size, noc_index);
    volatile tt_l1_ptr uint32_t* config = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(iface.config_ptr);
    config[PERSISTENT_DFB_CFG_APPLIED_ENTRY_SIZE] = iface.fifo_page_size;

    // Make the stale iface resolve to a distinct, valid credit-derived cursor.
    volatile tt_l1_ptr uint32_t* sent_ptr = dfb.local_sent_ptr(iface, 0);
    const uint32_t saved_sent = *sent_ptr;
    ASSERT(poison_wr_ptr >= iface.fifo_start_addr);
    ASSERT(poison_wr_ptr < iface.fifo_limit_page_aligned);
    ASSERT((poison_wr_ptr - iface.fifo_start_addr) % L1_ALIGNMENT == 0);
    *sent_ptr = (poison_wr_ptr - iface.fifo_start_addr) / L1_ALIGNMENT;

    const uint32_t live_entry_size = iface.fifo_page_size;
    iface.fifo_page_size = stale_entry_size;
    dfb.commit();

    iface.fifo_page_size = live_entry_size;
    *sent_ptr = saved_sent;
}

}  // namespace experimental

void kernel_main() {
    constexpr uint8_t persistent_dfb_id = get_compile_time_arg_val(0);
    constexpr uint32_t entry_size = get_compile_time_arg_val(1);
    constexpr uint32_t new_entry_size = get_compile_time_arg_val(2);
    constexpr uint32_t poison_wr_ptr = get_compile_time_arg_val(3);
    const uint32_t staging_base = get_arg_val<uint32_t>(0);

    static_assert(entry_size != new_entry_size, "stale-commit test requires distinct entry sizes");

    Noc noc;
    experimental::PersistentDFB dfb(persistent_dfb_id);

    // Advance one entry so the durable checkpoint is not fifo_start.
    dfb.reserve_back(1);
    dfb.write_to_receiver(0, staging_base, 1, noc);
    dfb.flush_writes(noc);
    dfb.push_back(1, noc);

    // A stale entry-size epoch must not overwrite word[4] with poison_wr_ptr.
    experimental::test_stale_commit_after_resize(dfb, new_entry_size, entry_size, poison_wr_ptr);
    // ~PersistentDFB commits the restored credit-derived cursor under the live E2 epoch.
}
