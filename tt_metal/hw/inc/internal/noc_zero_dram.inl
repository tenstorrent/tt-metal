// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/tensor/noc_traits.h"


// TODO: this overload zeros a single page per call, so bulk callers loop page-by-page.
// A page-range overload could zero a bank at a time instead of one write per page.
template <typename DSpecT, typename Scratch>
inline void Noc::async_write_zeros(
    const ::TensorAccessor<DSpecT>& accessor,
    uint32_t size_bytes,
    const dst_args_t<::TensorAccessor<DSpecT>>& args,
    const Scratch& scratch) const {
    static_assert(
        DSpecT::is_dram, "noc.async_write_zeros<TensorAccessor, Scratch> requires a DRAM-backed accessor.");
    static_assert(
        noc_zero_l1_endpoint_v<Scratch>,
        "noc.async_write_zeros scratch must be a local-L1 endpoint: CircularBuffer, DataflowBuffer, "
        "CoreLocalMem, Scratchpad or LocalTensorAccessor.");
    ASSERT(args.offset_bytes + size_bytes <= accessor.get_aligned_page_size());
    // Largest single noc_async_write the loop below will issue from scratch.
    // Caller's pre-zeroed prefix must cover at least this many bytes.
    const uint32_t max_chunk =
        (size_bytes > (uint32_t)NOC_MAX_BURST_SIZE) ? (uint32_t)NOC_MAX_BURST_SIZE : size_bytes;
    if constexpr (std::is_same_v<Scratch, DataflowBuffer>) {
        // DFB exposes get_entry_size(); assert one reserved entry is big enough. Scratchpad also has
        // an extent, checked in the branch below. CircularBuffer, CoreLocalMem and
        // LocalTensorAccessor expose no size at all, so for those three scratch capacity stays the
        // caller's responsibility.
        ASSERT(scratch.get_entry_size() >= max_chunk);
    } else if constexpr (is_scratchpad_v<Scratch>) {
        ASSERT(scratch.size_in_bytes() >= max_chunk);
    }
    uint32_t src_addr = get_src_ptr<AddressType::LOCAL_L1>(scratch, src_args_t<Scratch>{});
    uint64_t dst = accessor.get_noc_addr(args.page_id, args.offset_bytes, noc_id_);
    uint32_t remaining = size_bytes;
    while (remaining > 0) {
        uint32_t curr = (remaining > (uint32_t)NOC_MAX_BURST_SIZE) ? (uint32_t)NOC_MAX_BURST_SIZE : remaining;
        noc_async_write(src_addr, dst, curr, noc_id_);
        dst += curr;
        remaining -= curr;
    }
}

inline void Noc::write_zeros_dram_barrier() const { noc_async_write_barrier(noc_id_); }
