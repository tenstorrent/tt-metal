// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Byte layout of a PrefetcherPipe config page, shared between the PrefetcherPipe itself and the
// DRAM-sender factory: the factory has to size the shared DRISC L1 block before any pipe exists.

#pragma once

#include <cstdint>

#include <tt_align.hpp>

#include "hostdev/remote_dfb_config_layout.h"

namespace tt::tt_metal::experimental {

struct PrefetcherPipeConfigPageLayout {
    uint32_t noc_xy_offset;
    uint32_t counters_offset;
    uint32_t page_size;
};

// A config page is the 9-word remote-DFB header, then a 2-word NOC XY entry per receiver, then an
// L1-aligned entries_sent/entries_acked pair per receiver (the pairs are NOC-atomic targets, hence
// the alignment).
inline PrefetcherPipeConfigPageLayout compute_prefetcher_pipe_config_page_layout(
    uint32_t num_receivers, uint32_t l1_alignment) {
    const uint32_t noc_xy_offset = prefetcher_pipe_noc_xy_byte_offset();
    const uint32_t counters_offset =
        tt::align(noc_xy_offset + 2 * num_receivers * static_cast<uint32_t>(sizeof(uint32_t)), l1_alignment);
    const uint32_t page_size = counters_offset + 2 * num_receivers * l1_alignment;
    return PrefetcherPipeConfigPageLayout{
        .noc_xy_offset = noc_xy_offset, .counters_offset = counters_offset, .page_size = page_size};
}

}  // namespace tt::tt_metal::experimental
