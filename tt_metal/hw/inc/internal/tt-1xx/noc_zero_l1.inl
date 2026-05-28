// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Chunked noc_async_read loopback from MEM_ZEROS_BASE, plus the matching
// write_zeros_l1_barrier (noc_async_read_barrier).

template <typename Dst>
inline void Noc::write_zeros(const Dst& dst, uint32_t size_bytes, const dst_args_t<Dst>& args) const {
    static_assert(
        std::is_same_v<Dst, CircularBuffer> || std::is_same_v<Dst, DataflowBuffer>,
        "noc.write_zeros local-L1 overload accepts CircularBuffer or DataflowBuffer only. "
        "Use the TensorAccessor overload for DRAM.");
    uint32_t local_addr = get_dst_ptr<AddressType::LOCAL_L1>(dst, args);
    uint64_t zeros_noc = ::get_noc_addr(NOC_X(my_x[noc_id_]), NOC_Y(my_y[noc_id_]), MEM_ZEROS_BASE);
    uint32_t remaining = size_bytes;
    while (remaining > 0) {
        uint32_t curr = (remaining > (uint32_t)MEM_ZEROS_SIZE) ? (uint32_t)MEM_ZEROS_SIZE : remaining;
        noc_async_read(zeros_noc, local_addr, curr, noc_id_);
        local_addr += curr;
        remaining -= curr;
    }
}

inline void Noc::write_zeros_l1_barrier() const { noc_async_read_barrier(noc_id_); }
