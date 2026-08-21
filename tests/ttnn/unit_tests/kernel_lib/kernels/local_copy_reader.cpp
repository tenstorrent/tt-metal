// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// local_copy_helpers_dataflow unit test: single reader kernel.
//
// Stages `payload_pages` pages of the DRAM input into cb_src, then copies cb_src -> cb_dst
// ENTIRELY through the local-L1-copy helpers (never through a plain noc.async_read of DRAM, and
// never through noc.async_write — which cannot take a CB/DFB as a unicast destination at all).
// The writer half of the program moves cb_dst back out to DRAM, so a bit-exact output proves the
// L1 -> L1 copy landed correctly.
//
// MODE selects which helper shape is under test:
//   0 GATHER_RAW    per-page set_read_state + read_with_state(raw dst addr)
//   1 GATHER_TYPED  per-page set_read_state + read_with_state(cb_dst, {.offset_bytes})
//   2 BCAST_RAW     ONE set_read_state, N reads of the same source page -> raw dst addrs
//   3 BCAST_TYPED   ONE set_read_state, N reads of the same source page -> cb_dst offsets
//   4 TRID          per-page set_read_state, reads tagged with a rotating trid and drained
//                   with async_read_barrier_with_trid
//   5 DIRECT        local_addr() alone, fed to a plain noc.async_read (no state) — the path a
//                   multi-packet copy has to use
//   6 TYPED_NOARGS  ONE set_read_state, ONE read_with_state(cb_dst, src) with no dst_args
//
// The BCAST_* and TYPED_NOARGS modes copy source page 0 into every destination page, so the host
// expects every output page to equal input page 0. Every other mode expects output == input.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp"

using namespace dataflow_kernel_lib;

constexpr uint32_t MODE_GATHER_RAW = 0;
constexpr uint32_t MODE_GATHER_TYPED = 1;
constexpr uint32_t MODE_BCAST_RAW = 2;
constexpr uint32_t MODE_BCAST_TYPED = 3;
constexpr uint32_t MODE_TRID = 4;
constexpr uint32_t MODE_DIRECT = 5;
constexpr uint32_t MODE_TYPED_NOARGS = 6;

void kernel_main() {
    constexpr uint32_t cb_src = get_compile_time_arg_val(0);
    constexpr uint32_t cb_dst = get_compile_time_arg_val(1);
    constexpr uint32_t payload_pages = get_compile_time_arg_val(2);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t mode = get_compile_time_arg_val(4);
    constexpr uint32_t num_trids = get_compile_time_arg_val(5);
    constexpr auto in_args = TensorAccessorArgs<6>();
    constexpr auto out_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();

    const uint32_t input_addr = get_arg_val<uint32_t>(0);
    const uint32_t output_addr = get_arg_val<uint32_t>(1);

    Noc noc;
    CircularBuffer cb_src_obj(cb_src);
    CircularBuffer cb_dst_obj(cb_dst);

    // ---- stage the payload into cb_src (DRAM -> L1) ----
    const auto in = TensorAccessor(in_args, input_addr);
    cb_src_obj.reserve_back(payload_pages);
    for (uint32_t i = 0; i < payload_pages; ++i) {
        noc.async_read(in, cb_src_obj, page_bytes, {.page_id = i}, {.offset_bytes = i * page_bytes});
    }
    noc.async_read_barrier();
    cb_src_obj.push_back(payload_pages);
    cb_src_obj.wait_front(payload_pages);

    // ---- cb_src -> cb_dst, purely through the local-copy helpers ----
    const uint32_t src_base = cb_src_obj.get_read_ptr();
    cb_dst_obj.reserve_back(payload_pages);
    const uint32_t dst_base = cb_dst_obj.get_write_ptr();

    if constexpr (mode == MODE_GATHER_RAW) {
        for (uint32_t i = 0; i < payload_pages; ++i) {
            // The source moves every iteration, so the state is reprogrammed every iteration.
            set_read_state<page_bytes>(noc, src_base + i * page_bytes);
            read_with_state<page_bytes>(noc, dst_base + i * page_bytes, src_base + i * page_bytes);
        }
        noc.async_read_barrier();
    } else if constexpr (mode == MODE_GATHER_TYPED) {
        for (uint32_t i = 0; i < payload_pages; ++i) {
            set_read_state<page_bytes>(noc, src_base + i * page_bytes);
            read_with_state(noc, cb_dst_obj, src_base + i * page_bytes, {.offset_bytes = i * page_bytes});
        }
        noc.async_read_barrier();
    } else if constexpr (mode == MODE_BCAST_RAW) {
        // The whole point of the state split: program source + size ONCE, issue N reads.
        set_read_state<page_bytes>(noc, src_base);
        for (uint32_t i = 0; i < payload_pages; ++i) {
            read_with_state(noc, dst_base + i * page_bytes, src_base);
        }
        noc.async_read_barrier();
    } else if constexpr (mode == MODE_BCAST_TYPED) {
        set_read_state<page_bytes>(noc, src_base);
        for (uint32_t i = 0; i < payload_pages; ++i) {
            read_with_state(noc, cb_dst_obj, src_base, {.offset_bytes = i * page_bytes});
        }
        noc.async_read_barrier();
    } else if constexpr (mode == MODE_TRID) {
        // Rotating trid: drain the batch that is `num_trids` behind before reusing its slot, so
        // the barrier is proven to wait on ONE batch and not on all outstanding reads.
        for (uint32_t i = 0; i < payload_pages; ++i) {
            if (i >= num_trids) {
                async_read_barrier_with_trid(noc, (i - num_trids) % num_trids + 1);
            }
            set_read_trid(noc, i % num_trids + 1);
            set_read_state<page_bytes>(noc, src_base + i * page_bytes);
            read_with_state(noc, cb_dst_obj, src_base + i * page_bytes, {.offset_bytes = i * page_bytes});
        }
        const uint32_t to_drain = payload_pages < num_trids ? payload_pages : num_trids;
        for (uint32_t d = 0; d < to_drain; ++d) {
            async_read_barrier_with_trid(noc, (payload_pages - to_drain + d) % num_trids + 1);
        }
        set_read_trid(noc, 0);  // restore untagged
    } else if constexpr (mode == MODE_DIRECT) {
        // local_addr() on its own: no state, so this shape also works past NOC_MAX_BURST_SIZE.
        UnicastEndpoint self_ep;
        for (uint32_t i = 0; i < payload_pages; ++i) {
            noc.async_read(
                self_ep,
                cb_dst_obj,
                page_bytes,
                local_addr(src_base + i * page_bytes, noc.get_noc_id()),
                {.offset_bytes = i * page_bytes});
        }
        noc.async_read_barrier();
    } else if constexpr (mode == MODE_TYPED_NOARGS) {
        // No-dst_args overload: lands at cb_dst's write pointer, i.e. page 0. Every remaining
        // destination page is filled from source page 0 too, via the dst_args overload, so the
        // host's "all pages == input page 0" expectation holds for this mode as well.
        set_read_state<page_bytes>(noc, src_base);
        read_with_state(noc, cb_dst_obj, src_base);
        for (uint32_t i = 1; i < payload_pages; ++i) {
            read_with_state(noc, cb_dst_obj, src_base, {.offset_bytes = i * page_bytes});
        }
        noc.async_read_barrier();
    }

    cb_dst_obj.push_back(payload_pages);

    // ---- cb_dst -> DRAM (L1 -> DRAM), for host verification ----
    const auto out = TensorAccessor(out_args, output_addr);
    cb_dst_obj.wait_front(payload_pages);
    for (uint32_t i = 0; i < payload_pages; ++i) {
        noc.async_write(cb_dst_obj, out, page_bytes, {.offset_bytes = i * page_bytes}, {.page_id = i});
    }
    noc.async_write_barrier();
    cb_dst_obj.pop_front(payload_pages);
    cb_src_obj.pop_front(payload_pages);
}
