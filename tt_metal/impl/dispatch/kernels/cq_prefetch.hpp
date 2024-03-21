// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Common prefetch code for use by _hd, _h, _d prefetch variants

#include "dataflow_api.h"
#include "debug/dprint.h"
#include "tt_metal/impl/dispatch/kernels/cq_common.hpp"

#define CQ_PREFETCH_DOWNSTREAM_PACKET_HEADER_SIZE L1_NOC_ALIGNMENT

template<uint32_t my_noc_xy,
         uint32_t my_sem_id,
         uint32_t downstream_noc_xy,
         uint32_t downstream_sem_id,
         uint32_t cb_base,
         uint32_t cb_end,
         uint32_t cb_log_page_size,
         uint32_t cb_page_size>
FORCE_INLINE
void write_downstream(uint32_t& data_ptr,
                      uint32_t& downstream_data_ptr,
                      uint32_t length) {

    uint32_t npages = (length + cb_page_size - 1) >> cb_log_page_size;

    // Assume the downstream buffer is big relative to cmddat command size that we can
    // grab what we need in one chunk
    downstream_cb_acquire_pages<my_noc_xy, my_sem_id>(npages);
    uint32_t downstream_pages_left = (cb_end - downstream_data_ptr) / cb_page_size;
    if (downstream_pages_left >= npages) {
        noc_async_write(data_ptr, get_noc_addr_helper(downstream_noc_xy, downstream_data_ptr), length);
        downstream_data_ptr += npages * cb_page_size;
    } else {
        uint32_t tail_pages = npages - downstream_pages_left;
        uint32_t available = downstream_pages_left * cb_page_size;
        if (available > 0) {
            noc_async_write(data_ptr, get_noc_addr_helper(downstream_noc_xy, downstream_data_ptr), available);
            data_ptr += available;
            length -= available;
        }

        noc_async_write(data_ptr, get_noc_addr_helper(downstream_noc_xy, cb_base), length);
        downstream_data_ptr = cb_base + tail_pages * cb_page_size;
    }

    // XXXXX - painful syncing right now?  move this into get_cmds
    noc_async_writes_flushed();
    downstream_cb_release_pages<downstream_noc_xy, downstream_sem_id>(npages);
}

template<uint32_t cmddat_q_base,
         uint32_t cmddat_q_size,
         uint32_t pcie_base,
         uint32_t pcie_size,
         uint32_t prefetch_q_base,
         uint32_t prefetch_q_end,
         uint32_t prefetch_q_rd_ptr_addr,
         uint32_t preamble_size>
FORCE_INLINE
void read_from_pcie(volatile tt_l1_ptr uint16_t *& prefetch_q_rd_ptr,
                    uint32_t& pending_read_size,
                    uint32_t& fence,
                    uint32_t& pcie_read_ptr,
                    uint32_t cmd_ptr,
                    uint32_t size) {

    // Wrap cmddat_q
    if (fence + size + preamble_size > cmddat_q_base + cmddat_q_size) {
        // only wrap if there are no commands ready, otherwise we'll leave some on the floor
        // TODO: does this matter for perf?
        if (cmd_ptr != fence) {
            return;
        }
        fence = cmddat_q_base;
    }
    fence += preamble_size;

    // Wrap pcie/hugepage
    if (pcie_read_ptr + size > pcie_size) {
        pcie_read_ptr = pcie_base;
    }

    uint64_t host_src_addr = get_noc_addr_helper(NOC_XY_ENCODING(PCIE_NOC_X, PCIE_NOC_Y), pcie_read_ptr);
    noc_async_read(host_src_addr, fence, size);
    pending_read_size = size;
    pcie_read_ptr += size;

    *prefetch_q_rd_ptr = 0;

    // Tell host we read
    *(volatile tt_l1_ptr uint32_t *) prefetch_q_rd_ptr_addr = (uint32_t)prefetch_q_rd_ptr;

    prefetch_q_rd_ptr++;

    // Wrap prefetch_q
    if ((uint32_t)prefetch_q_rd_ptr == prefetch_q_end) {
        prefetch_q_rd_ptr = (volatile tt_l1_ptr uint16_t*)prefetch_q_base;
    }
}

// This routine can be called in 8 states based on the boolean values cmd_ready, prefetch_q_ready, read_pending:
//  - !cmd_ready, !prefetch_q_ready, !read_pending: stall on prefetch_q, issue read, read barrier
//  - !cmd_ready, !prefetch_q_ready,  read pending: read barrier (and re-evaluate prefetch_q_ready)
//  - !cmd_ready,  prefetch_q_ready, !read_pending: issue read, read barrier (XXXX +issue read after?)
//  - !cmd_ready,  prefetch_q_ready,  read_pending: read barrier, issue read
//  -  cmd_ready, !prefetch_q_ready, !read_pending: exit
//  -  cmd_ready, !prefetch_q_ready,  read_pending: exit (no barrier yet)
//  -  cmd_ready,  prefetch_q_ready, !read_pending: issue read
//  -  cmd_ready,  prefetch_q_ready,  read_pending: exit (don't add latency to the in flight request)
//
// With WH tagging of reads:
// open question: should fetcher loop on prefetch_q_ready issuing reads until !prefetch_q_ready
//  - !cmd_ready, !prefetch_q_ready, !read_pending: stall on prefetch_q, issue read, read barrier
//  - !cmd_ready, !prefetch_q_ready,  read pending: read barrier on oldest tag
//  - !cmd_ready,  prefetch_q_ready, !read_pending: issue read, read barrier (XXXX +retry after?)
//  - !cmd_ready,  prefetch_q_ready,  read_pending: issue read, read barrier on oldest tag
//  -  cmd_ready, !prefetch_q_ready, !read_pending: exit
//  -  cmd_ready, !prefetch_q_ready,  read_pending: exit (no barrier yet)
//  -  cmd_ready,  prefetch_q_ready, !read_pending: issue and tag read
//  -  cmd_ready,  prefetch_q_ready,  read_pending: issue and tag read
template<uint32_t cmddat_q_base,
         uint32_t cmddat_q_size,
         uint32_t pcie_base,
         uint32_t pcie_size,
         uint32_t prefetch_q_base,
         uint32_t prefetch_q_end,
         uint32_t prefetch_q_log_minsize,
         uint32_t prefetch_q_rd_ptr_addr,
         uint32_t preamble_size>
inline void fetch_q_get_cmds(uint32_t& fence, uint32_t& cmd_ptr, uint32_t& pcie_read_ptr) {

    static uint32_t pending_read_size = 0;
    static volatile tt_l1_ptr uint16_t* prefetch_q_rd_ptr = (volatile tt_l1_ptr uint16_t*)prefetch_q_base;

    if (fence < cmd_ptr) {
        DPRINT << "wrap cmd ptr1 " << fence << " " << cmd_ptr << ENDL();
        cmd_ptr = fence;
    }

    bool cmd_ready = (cmd_ptr != fence);
    uint32_t fetch_size = (uint32_t)*prefetch_q_rd_ptr << prefetch_q_log_minsize;

    if (fetch_size != 0 && pending_read_size == 0) {
        DPRINT << "read1: " << (uint32_t)prefetch_q_rd_ptr << " " << " " << fence << " " << fetch_size << ENDL();
        read_from_pcie<cmddat_q_base,
                       cmddat_q_size,
                       pcie_base,
                       pcie_size,
                       prefetch_q_base,
                       prefetch_q_end,
                       prefetch_q_rd_ptr_addr,
                       preamble_size>
            (prefetch_q_rd_ptr, pending_read_size, fence, pcie_read_ptr, cmd_ptr, fetch_size);
    }
    if (!cmd_ready) {
        if (pending_read_size != 0) {
            DPRINT << "barrier" << ENDL();
            noc_async_read_barrier();

            // wrap the cmddat_q
            if (fence < cmd_ptr) {
                cmd_ptr = fence;
            }

            fence += pending_read_size;
            pending_read_size = 0;
            // After the stall, re-check the host
            fetch_size = (uint32_t)*prefetch_q_rd_ptr << prefetch_q_log_minsize;
            if (fetch_size != 0) {
                DPRINT << "read2: " << (uint32_t)prefetch_q_rd_ptr << " " << fetch_size << ENDL();
                read_from_pcie<cmddat_q_base,
                               cmddat_q_size,
                               pcie_base,
                               pcie_size,
                               prefetch_q_base,
                               prefetch_q_end,
                               prefetch_q_rd_ptr_addr,
                               preamble_size>
                    (prefetch_q_rd_ptr, pending_read_size, fence, pcie_read_ptr, cmd_ptr, fetch_size);
            }
        } else {
            // By here, prefetch_q_ready must be false
            // Nothing to fetch, nothing pending, nothing available, stall on host
            DEBUG_STATUS('H', 'Q', 'W');
            DPRINT << "prefetcher stall" << ENDL();
            while ((fetch_size = *prefetch_q_rd_ptr) == 0);
            DPRINT << "recurse" << ENDL();
            fetch_q_get_cmds<cmddat_q_base,
                             cmddat_q_size,
                             pcie_base,
                             pcie_size,
                             prefetch_q_base,
                             prefetch_q_end,
                             prefetch_q_log_minsize,
                             prefetch_q_rd_ptr_addr,
                             preamble_size>(fence, cmd_ptr, pcie_read_ptr);
            DEBUG_STATUS('H', 'Q', 'D');
        }
    }
}
