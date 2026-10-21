// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Quasar: leaves a word dirty in the cache line of this core's SENT credit block, then constructs a
// sender PrefetcherPipe over a host-stamped config page. The constructor drops the cached credit
// lines; the host checks afterwards that the dirty word reached L1 instead of being discarded.
//
// Args (named CTAs):
//   args::config_page_addr - sender config page (host-stamped, entry size already applied)
//   args::entry_size       - the page's applied entry size, so the constructor does not resize
//   args::probe_addr       - an unused word in the SENT block's cache line
//   args::probe_value      - the value left dirty there

#include "api/dataflow/prefetcher_pipe.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t config_page_addr = get_arg(args::config_page_addr);
    constexpr uint32_t entry_size = get_arg(args::entry_size);
    constexpr uint32_t probe_addr = get_arg(args::probe_addr);
    constexpr uint32_t probe_value = get_arg(args::probe_value);

    volatile tt_l1_ptr uint32_t* probe = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(probe_addr);
    // Load first, so the store hits the cached line and stays there, dirty.
    (void)*probe;
    *probe = probe_value;

    experimental::PrefetcherPipe pipe(experimental::DramSenderConfigPage{config_page_addr}, entry_size);
}
