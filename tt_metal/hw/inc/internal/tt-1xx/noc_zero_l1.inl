// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// State-aware chunked noc_async_read loopback from MEM_ZEROS_BASE, plus the
// matching write_zeros_l1_barrier (noc_async_read_barrier).

#include "api/dataflow/endpoints.h"

template <typename Dst>
inline void Noc::async_write_zeros(const Dst& dst, uint32_t size_bytes, const dst_args_t<Dst>& args) const {
    static_assert(
        std::is_same_v<Dst, CircularBuffer> || std::is_same_v<Dst, DataflowBuffer>,
        "noc.async_write_zeros local-L1 overload accepts CircularBuffer or DataflowBuffer only. "
        "Use the TensorAccessor overload for DRAM.");

    UnicastEndpoint zeros_ep;
    const auto zeros_src = noc_traits_t<UnicastEndpoint>::src_args_type{
        .noc_x = my_x[noc_id_], .noc_y = my_y[noc_id_], .addr = MEM_ZEROS_BASE};

    auto chunk_args = args;
    uint32_t remaining = size_bytes;

    if (remaining >= (uint32_t)MEM_ZEROS_SIZE) {
        set_async_read_state<NocOptions::DEFAULT, MEM_ZEROS_SIZE>(zeros_ep, MEM_ZEROS_SIZE, zeros_src);

        do {
            async_read_with_state<NocOptions::DEFAULT, 1>(zeros_ep, dst, 0, zeros_src, chunk_args);
            chunk_args.offset_bytes += MEM_ZEROS_SIZE;
            remaining -= MEM_ZEROS_SIZE;
        } while (remaining >= (uint32_t)MEM_ZEROS_SIZE);
    }

    if (remaining > 0) {
        async_read(zeros_ep, dst, remaining, zeros_src, chunk_args);
    }
}

inline void Noc::write_zeros_l1_barrier() const { noc_async_read_barrier(noc_id_); }
