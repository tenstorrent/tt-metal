// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/debug/assert.h"
#include "api/debug/kernel_profiler.hpp"
#include "api/debug/waypoint.h"
#include "internal/debug/sanitize.h"

// Implementations behind api/dataflow/dataflow_api.h's public functions that use raw addresses.
// The functions emit tt_buf_rw::note_opaque() (address not attributable to any bound accessor) and forward here.
// Already-attributed callers (the Noc class) and AddrGen wrappers call _impl functions directly,
// skipping the public wrapper's note_opaque().

template <bool enable_noc_tracing = true>
FORCE_INLINE void noc_async_read_one_packet_impl(
    uint64_t src_noc_addr,
    uint32_t dst_local_l1_addr,
    uint32_t size,
    uint8_t noc = noc_index,
    uint32_t read_req_vc = NOC_UNICAST_WRITE_VC) {
#ifdef ARCH_BLACKHOLE
    // A PCIe-routed src_noc_addr would be sent on-chip, since this does not write NOC_TARG_ADDR_MID.
    // Use noc_async_read_pcie, or noc_async_read_set_pcie_state for a batch.
    ASSERT(((src_noc_addr >> 32) & NOC_PCIE_MASK) == 0);
#endif
    /*
        Read requests - use static VC
        Read responses - assigned VCs dynamically
    */
    if constexpr (enable_noc_tracing) {
        RECORD_NOC_EVENT_WITH_ADDR(NocEventType::READ, dst_local_l1_addr, src_noc_addr, size, -1, false, noc);
    }

    WAYPOINT("RP2W");
    while (!noc_cmd_buf_ready(noc, read_cmd_buf));
    WAYPOINT("RP2D");

    WAYPOINT("NAOW");
    DEBUG_SANITIZE_NOC_READ_TRANSACTION(noc, src_noc_addr, dst_local_l1_addr, size);
    ncrisc_noc_fast_read<noc_mode>(noc, read_cmd_buf, src_noc_addr, dst_local_l1_addr, size, read_req_vc);
    WAYPOINT("NAOD");
}

template <uint32_t max_page_size = NOC_MAX_BURST_SIZE + 1, bool enable_noc_tracing = true>
inline void noc_async_read_impl(
    uint64_t src_noc_addr,
    uint32_t dst_local_l1_addr,
    uint32_t size,
    uint8_t noc = noc_index,
    uint32_t read_req_vc = NOC_UNICAST_WRITE_VC) {
    /*
        Read requests - use static VC
        Read responses - assigned VCs dynamically
    */
    if constexpr (enable_noc_tracing) {
        RECORD_NOC_EVENT_WITH_ADDR(NocEventType::READ, dst_local_l1_addr, src_noc_addr, size, -1, false, noc);
    }

    if constexpr (max_page_size <= NOC_MAX_BURST_SIZE) {
        // noc_async_read_one_packet runs this assert itself, so it is not repeated here.
        noc_async_read_one_packet_impl<false>(src_noc_addr, dst_local_l1_addr, size, noc, read_req_vc);
    } else {
#ifdef ARCH_BLACKHOLE
        // A PCIe-routed src_noc_addr would be sent on-chip, since this does not write NOC_TARG_ADDR_MID.
        // Use noc_async_read_pcie, or noc_async_read_set_pcie_state for a batch.
        ASSERT(((src_noc_addr >> 32) & NOC_PCIE_MASK) == 0);
#endif
        WAYPOINT("NARW");
        DEBUG_SANITIZE_NOC_READ_TRANSACTION(noc, src_noc_addr, dst_local_l1_addr, size);
        ncrisc_noc_fast_read_any_len<noc_mode>(noc, read_cmd_buf, src_noc_addr, dst_local_l1_addr, size, read_req_vc);
        WAYPOINT("NARD");
    }
}

template <bool use_vc = false>
FORCE_INLINE void noc_async_read_one_packet_set_state_impl(
    uint64_t src_noc_addr, uint32_t size, const uint32_t vc = 0, uint8_t noc = noc_index) {
    /*
        Read requests - use static VC
        Read responses - assigned VCs dynamically
    */
    DEBUG_SANITIZE_NO_LINKED_TRANSACTION(noc, DEBUG_SANITIZE_NOC_UNICAST);
    RECORD_NOC_EVENT_WITH_ADDR(
        NocEventType::READ_SET_STATE, 0, src_noc_addr, size, (use_vc) ? static_cast<int8_t>(vc) : -1, false, noc);

    WAYPOINT("NASW");
    ncrisc_noc_read_set_state<noc_mode, true /* one_packet */, use_vc>(noc, read_cmd_buf, src_noc_addr, size, vc);
    WAYPOINT("NASD");
}

FORCE_INLINE
void noc_async_read_set_state_impl(uint64_t src_noc_addr, uint8_t noc = noc_index) {
    /*
        Read requests - use static VC
        Read responses - assigned VCs dynamically
    */
    DEBUG_SANITIZE_NO_LINKED_TRANSACTION(noc, DEBUG_SANITIZE_NOC_UNICAST);
    RECORD_NOC_EVENT_WITH_ADDR(NocEventType::READ_SET_STATE, 0, src_noc_addr, 0, -1, false, noc);

    WAYPOINT("NAUW");
    ncrisc_noc_read_set_state<noc_mode>(noc, read_cmd_buf, src_noc_addr);
    WAYPOINT("NAUD");
}

inline void noc_async_read_set_pcie_state_impl(uint64_t src_noc_addr, uint8_t noc = noc_index) {
    noc_async_read_set_state_impl(src_noc_addr, noc);
}

template <bool enable_noc_tracing = true, bool posted = false>
FORCE_INLINE void noc_async_write_one_packet_impl(
    std::uint32_t src_local_l1_addr,
    std::uint64_t dst_noc_addr,
    std::uint32_t size,
    uint8_t noc = noc_index,
    uint32_t vc = NOC_UNICAST_WRITE_VC) {
#ifdef ARCH_BLACKHOLE
    // A PCIe-routed dst_noc_addr would be sent on-chip, since this does not write NOC_RET_ADDR_MID.
    // Use noc_async_write_pcie, or noc_async_write_set_pcie_state for a batch.
    ASSERT(((dst_noc_addr >> 32) & NOC_PCIE_MASK) == 0);
#endif
    if constexpr (enable_noc_tracing) {
        RECORD_NOC_EVENT_WITH_ADDR(NocEventType::WRITE_, src_local_l1_addr, dst_noc_addr, size, vc, posted, noc);
    }

    WAYPOINT("NWPW");
    DEBUG_SANITIZE_NOC_WRITE_TRANSACTION(noc, dst_noc_addr, src_local_l1_addr, size);
    while (!noc_cmd_buf_ready(noc, write_cmd_buf));
    WAYPOINT("NWPD");

    ncrisc_noc_fast_write<noc_mode>(
        noc,
        write_cmd_buf,
        src_local_l1_addr,
        dst_noc_addr,
        size,
        vc,
        false /* mcast */,
        false /* linked */,
        1 /* num_dests */,
        true /* multicast_path_reserve */,
        posted);
}

template <uint32_t max_page_size = NOC_MAX_BURST_SIZE + 1, bool enable_noc_tracing = true, bool posted = false>
inline void noc_async_write_impl(
    uint32_t src_local_l1_addr,
    uint64_t dst_noc_addr,
    uint32_t size,
    uint8_t noc = noc_index,
    uint32_t vc = NOC_UNICAST_WRITE_VC) {
    if constexpr (enable_noc_tracing) {
        RECORD_NOC_EVENT_WITH_ADDR(NocEventType::WRITE_, src_local_l1_addr, dst_noc_addr, size, vc, posted, noc);
    }

    if constexpr (max_page_size <= NOC_MAX_BURST_SIZE) {
        // noc_async_write_one_packet runs this assert itself, so it is not repeated here.
        noc_async_write_one_packet_impl<false, posted>(src_local_l1_addr, dst_noc_addr, size, noc, vc);
    } else {
#ifdef ARCH_BLACKHOLE
        // A PCIe-routed dst_noc_addr would be sent on-chip, since this does not write NOC_RET_ADDR_MID.
        // Use noc_async_write_pcie, or noc_async_write_set_pcie_state for a batch.
        ASSERT(((dst_noc_addr >> 32) & NOC_PCIE_MASK) == 0);
#endif
        WAYPOINT("NAWW");
        DEBUG_SANITIZE_NOC_WRITE_TRANSACTION(noc, dst_noc_addr, src_local_l1_addr, size);
        ncrisc_noc_fast_write_any_len<noc_mode>(
            noc, write_cmd_buf, src_local_l1_addr, dst_noc_addr, size, vc, false, false, 1, true, posted);
        WAYPOINT("NAWD");
    }
}

template <bool enable_noc_tracing = true>
FORCE_INLINE void noc_async_write_multicast_one_packet_impl(
    std::uint32_t src_local_l1_addr,
    std::uint64_t dst_noc_addr_multicast,
    std::uint32_t size,
    std::uint32_t num_dests,
    bool linked = false,
    uint8_t noc = noc_index) {
    if constexpr (enable_noc_tracing) {
        NOC_TRACE_QUICK_PUSH_IF_LINKED(write_cmd_buf, linked);
        RECORD_NOC_EVENT_WITH_ADDR(
            NocEventType::WRITE_MULTICAST,
            src_local_l1_addr,
            dst_noc_addr_multicast,
            size,
            NOC_MULTICAST_WRITE_VC,
            false,
            noc);
    }
    DEBUG_SANITIZE_NOC_MULTI_WRITE_TRANSACTION(noc, dst_noc_addr_multicast, src_local_l1_addr, size);
    while (!noc_cmd_buf_ready(noc, write_cmd_buf));
    WAYPOINT("NWPD");

    ncrisc_noc_fast_write<noc_mode>(
        noc,
        write_cmd_buf,
        src_local_l1_addr,
        dst_noc_addr_multicast,
        size,
        NOC_MULTICAST_WRITE_VC,
        true /* mcast */,
        linked,
        num_dests,
        true /* multicast_path_reserve */);
}

template <uint32_t max_page_size = NOC_MAX_BURST_SIZE + 1>
inline void noc_async_write_multicast_impl(
    uint32_t src_local_l1_addr,
    uint64_t dst_noc_addr_multicast,
    uint32_t size,
    uint32_t num_dests,
    bool linked = false,
    uint8_t noc = noc_index,
    uint8_t vc = NOC_MULTICAST_WRITE_VC) {
    RECORD_NOC_EVENT_WITH_ADDR(
        NocEventType::WRITE_MULTICAST, src_local_l1_addr, dst_noc_addr_multicast, size, vc, false, noc);

    if constexpr (max_page_size <= NOC_MAX_BURST_SIZE) {
        noc_async_write_multicast_one_packet_impl<false>(
            src_local_l1_addr, dst_noc_addr_multicast, size, num_dests, linked, noc);
    } else {
        WAYPOINT("NMWW");
        NOC_TRACE_QUICK_PUSH_IF_LINKED(write_cmd_buf, linked);
        DEBUG_SANITIZE_NOC_MULTI_WRITE_TRANSACTION(noc, dst_noc_addr_multicast, src_local_l1_addr, size);
        ncrisc_noc_fast_write_any_len<noc_mode>(
            noc,
            write_cmd_buf,
            src_local_l1_addr,
            dst_noc_addr_multicast,
            size,
            vc,
            true /* mcast */,
            linked,
            num_dests,
            true /* multicast_path_reserve */);
        WAYPOINT("NMWD");
    }
}

template <bool posted = false>
FORCE_INLINE void noc_async_write_one_packet_set_state_impl(
    uint64_t dst_noc_addr, uint32_t size, uint8_t noc = noc_index, uint8_t vc = NOC_UNICAST_WRITE_VC) {
    DEBUG_SANITIZE_NO_LINKED_TRANSACTION(noc, DEBUG_SANITIZE_NOC_UNICAST);
    RECORD_NOC_EVENT_WITH_ADDR(NocEventType::WRITE_SET_STATE, 0, dst_noc_addr, size, vc, posted, noc);

    WAYPOINT("NWPW");
    ncrisc_noc_write_set_state<posted, true /* one_packet */>(noc, write_cmd_buf, dst_noc_addr, size, vc);
    WAYPOINT("NWPD");
}

template <bool posted = false>
FORCE_INLINE void noc_async_write_set_state_impl(
    uint64_t dst_noc_addr, uint8_t noc = noc_index, uint8_t vc = NOC_UNICAST_WRITE_VC) {
    DEBUG_SANITIZE_NO_LINKED_TRANSACTION(noc, DEBUG_SANITIZE_NOC_UNICAST);
    RECORD_NOC_EVENT_WITH_ADDR(NocEventType::WRITE_SET_STATE, 0, dst_noc_addr, 0, vc, posted, noc);

    WAYPOINT("NWSW");
    ncrisc_noc_write_set_state<posted, false /* one_packet */>(noc, write_cmd_buf, dst_noc_addr, 0, vc);
    WAYPOINT("NWSD");
}

inline void noc_async_write_set_pcie_state_impl(
    uint64_t dst_noc_addr, uint8_t noc = noc_index, uint8_t vc = NOC_UNICAST_WRITE_VC) {
    noc_async_write_set_state_impl(dst_noc_addr, noc, vc);
}

inline void noc_async_write_multicast_loopback_src_impl(
    std::uint32_t src_local_l1_addr,
    std::uint64_t dst_noc_addr_multicast,
    std::uint32_t size,
    std::uint32_t num_dests,
    bool linked = false,
    uint8_t noc = noc_index) {
    constexpr bool multicast_path_reserve = true;

    NOC_TRACE_QUICK_PUSH_IF_LINKED(write_cmd_buf, linked);
    RECORD_NOC_EVENT_WITH_ADDR(
        NocEventType::WRITE_MULTICAST,
        src_local_l1_addr,
        dst_noc_addr_multicast,
        size,
        NOC_MULTICAST_WRITE_VC,
        false,
        noc);

    WAYPOINT("NMLW");
    DEBUG_SANITIZE_NOC_MULTI_WRITE_TRANSACTION(noc, dst_noc_addr_multicast, src_local_l1_addr, size);
    ncrisc_noc_fast_write_any_len_loopback_src<noc_mode>(
        noc,
        write_cmd_buf,
        src_local_l1_addr,
        dst_noc_addr_multicast,
        size,
        NOC_MULTICAST_WRITE_VC,
        true,
        linked,
        num_dests,
        multicast_path_reserve);
    WAYPOINT("NMLD");
}

template <bool update_counter = true, bool posted = false>
FORCE_INLINE void noc_async_write_one_packet_with_trid_impl(
    uint32_t src_local_l1_addr,
    uint64_t dst_noc_addr,
    uint32_t size,
    uint32_t trid,
    uint8_t cmd_buf = write_cmd_buf,
    uint8_t noc = noc_index,
    uint8_t vc = NOC_UNICAST_WRITE_VC) {
    WAYPOINT("NAWW");
    RECORD_NOC_EVENT_WITH_ADDR(NocEventType::WRITE_WITH_TRID, src_local_l1_addr, dst_noc_addr, size, -1, posted, noc);
    DEBUG_SANITIZE_NOC_WRITE_TRANSACTION(noc, dst_noc_addr, src_local_l1_addr, size);
    while (!noc_cmd_buf_ready(noc, cmd_buf));

#ifdef ARCH_BLACKHOLE
    // Issue https://github.com/tenstorrent/tt-metal/issues/28758: always update counter for blackhole as a temporary
    // workaround for avoiding hangs in fabric router, as counters will be checked inside the
    // noc_fast_spoof_write_dw_inline, will remove this restriction once all inline write change to stream reg write.
    constexpr bool update_counter_in_callee = true;
#else
    constexpr bool update_counter_in_callee = update_counter;
#endif

    ncrisc_noc_fast_write<noc_mode, true /* use_trid */, update_counter_in_callee>(
        noc,
        cmd_buf,
        src_local_l1_addr,
        dst_noc_addr,
        size,
        vc,
        false,  // mcast
        false,  // linked
        1,      // num_dests
        true,   // multicast_path_reserve
        posted,
        trid);
    WAYPOINT("NWPD");
}

template <bool posted = false>
FORCE_INLINE void noc_async_write_one_packet_with_trid_set_state_impl(
    uint64_t dst_noc_addr,
    uint8_t cmd_buf = write_cmd_buf,
    uint8_t noc = noc_index,
    uint8_t vc = NOC_UNICAST_WRITE_VC) {
    WAYPOINT("NAWW");
    DEBUG_SANITIZE_NO_LINKED_TRANSACTION(noc, DEBUG_SANITIZE_NOC_UNICAST);
    RECORD_NOC_EVENT_WITH_ADDR(NocEventType::WRITE_WITH_TRID_SET_STATE, 0, dst_noc_addr, 0, vc, posted, noc);

    ncrisc_noc_write_set_state<posted, false /* one_packet */>(noc, cmd_buf, dst_noc_addr, 0 /* len_bytes */, vc);
    WAYPOINT("NAWD");
}

template <InlineWriteDst dst_type = InlineWriteDst::DEFAULT, bool posted = false, bool flush = true>
FORCE_INLINE void noc_inline_dw_write_impl(
    uint64_t addr,
    uint32_t val,
    uint8_t be = 0xF,
    uint8_t noc = noc_index,
    uint8_t vc = NOC_UNICAST_WRITE_VC,
    uint32_t customized_src_addr = 0) {
    WAYPOINT("NWIW");
    // Inline dword write: 4-byte immediate value.
    RECORD_NOC_EVENT_WITH_ADDR(NocEventType::WRITE_INLINE, 0, addr, 4, vc, posted, noc);
    DEBUG_SANITIZE_NOC_ADDR(noc, addr, 4);
    DEBUG_SANITIZE_NO_DRAM_ADDR(noc, addr, 4);
#if defined(ARCH_BLACKHOLE) && defined(WATCHER_ENABLED)
    if constexpr (dst_type == InlineWriteDst::L1) {
        if constexpr (!flush) {
            ASSERT(customized_src_addr != 0);
            DEBUG_SANITIZE_NOC_WRITE_TRANSACTION(noc, addr, customized_src_addr, 4);
        } else {
            uint32_t src_addr = noc_get_interim_inline_value_addr(noc, addr);
            DEBUG_SANITIZE_NOC_WRITE_TRANSACTION(noc, addr, src_addr, 4);
        }
    }
#endif

    noc_fast_write_dw_inline<noc_mode, dst_type, flush>(
        noc,
        write_at_cmd_buf,
        val,
        addr,
        be,  // byte-enable
        vc,
        false,   // mcast
        posted,  // posted
        customized_src_addr);
    WAYPOINT("NWID");
}

template <InlineWriteDst dst_type = InlineWriteDst::DEFAULT, bool posted = false, bool flush = true>
FORCE_INLINE void noc_inline_mcast_dw_write_impl(
    uint64_t addr,
    uint32_t val,
    uint8_t be = 0xF,
    uint8_t noc = noc_index,
    uint8_t vc = NOC_MULTICAST_WRITE_VC,
    uint32_t customized_src_addr = 0,
    uint32_t num_dest = 1) {
    WAYPOINT("NWIW");
    DEBUG_SANITIZE_NOC_MULTI_ADDR(noc, addr, 4);
    DEBUG_SANITIZE_NO_DRAM_ADDR(noc, addr, 4);
#if defined(ARCH_BLACKHOLE) && defined(WATCHER_ENABLED)
    if constexpr (dst_type == InlineWriteDst::L1) {
        if constexpr (!flush) {
            ASSERT(customized_src_addr != 0);
            DEBUG_SANITIZE_NOC_WRITE_TRANSACTION(noc, addr, customized_src_addr, 4);
        } else {
            uint32_t src_addr = noc_get_interim_inline_value_addr(noc, addr);
            DEBUG_SANITIZE_NOC_WRITE_TRANSACTION(noc, addr, src_addr, 4);
        }
    }
#endif

    noc_fast_write_dw_inline_multicast<noc_mode, dst_type, flush>(
        noc,
        write_at_cmd_buf,
        val,
        addr,
        be,  // byte-enable
        vc,
        true,    // mcast
        posted,  // posted
        customized_src_addr,
        num_dest);

    WAYPOINT("NWID");
}

template <bool posted = false, bool set_val = false>
FORCE_INLINE void noc_inline_dw_write_set_state_impl(
    uint64_t addr,
    uint32_t val = 0,
    uint8_t be = 0xF,
    uint8_t cmd_buf = write_at_cmd_buf,
    uint8_t noc = noc_index,
    uint8_t vc = NOC_UNICAST_WRITE_VC) {
    WAYPOINT("NWIW");
    noc_fast_write_dw_inline_set_state<posted, set_val>(noc, cmd_buf, addr, be, vc, val);
    WAYPOINT("NWID");
}
