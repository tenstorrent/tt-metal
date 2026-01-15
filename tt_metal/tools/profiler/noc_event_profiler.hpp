// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if defined(PROFILE_NOC_EVENTS) && (defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC))

#include <utility>
#include <tuple>
#include <algorithm>
#include "event_metadata.hpp"
#include "internal/risc_attribs.h"
#include "kernel_profiler.hpp"

namespace noc_event_profiler {

template <bool DRAM = false>
FORCE_INLINE std::pair<uint32_t, uint32_t> decode_noc_coord_reg_to_coord(uint16_t noc_xy_bits) {
    constexpr uint32_t NOC_COORD_MASK = 0x3F;
    uint32_t x = noc_xy_bits & NOC_COORD_MASK;
    uint32_t y = (noc_xy_bits >> NOC_ADDR_NODE_ID_BITS) & NOC_COORD_MASK;
    return {x, y};
}

FORCE_INLINE
std::pair<uint32_t, uint32_t> decode_noc_addr_to_coord(uint64_t noc_addr) {
    return decode_noc_coord_reg_to_coord(noc_addr >> NOC_ADDR_LOCAL_BITS);
}

FORCE_INLINE
uint32_t decode_noc_addr_to_local_addr(uint64_t noc_addr) { return NOC_LOCAL_ADDR_OFFSET(noc_addr); }

FORCE_INLINE
std::tuple<uint32_t, uint32_t, uint32_t, uint32_t> decode_noc_addr_to_multicast_coord(uint64_t noc_addr) {
    // coordinates are stored as two packed pairs. End coordinate is in lower
    // bits like normal noc address; Start coordinate is in higher bits
    auto [xend, yend] = decode_noc_coord_reg_to_coord(noc_addr >> NOC_ADDR_LOCAL_BITS);
    auto [xstart, ystart] =
        decode_noc_coord_reg_to_coord(noc_addr >> (NOC_ADDR_LOCAL_BITS + 2 * NOC_ADDR_NODE_ID_BITS));

    return {xstart, ystart, xend, yend};
}

template <bool DRAM>
FORCE_INLINE std::pair<uint32_t, uint32_t> decode_noc_id_into_coord(uint32_t id, uint8_t noc = noc_index) {
    uint32_t bank_offset_index = interleaved_addr_gen::get_bank_offset_index<DRAM>(id);
    uint32_t bank_index = interleaved_addr_gen::get_bank_index<DRAM>(id, bank_offset_index);
    // shift so that coordinate is in LSB
    return decode_noc_coord_reg_to_coord<DRAM>(
        interleaved_addr_gen::get_noc_xy<DRAM>(bank_index, noc) >> NOC_COORD_REG_OFFSET);
}

template <KernelProfilerNocEventMetadata::NocEventType noc_event_type, bool posted>
FORCE_INLINE KernelProfilerNocEventMetadata createNocEventDstTrailer(uint32_t src_addr, uint32_t dst_addr) {
    KernelProfilerNocEventMetadata ev_md;
    ev_md.data.local_event_dst_trailer.setSrcAddr(src_addr);
    ev_md.data.local_event_dst_trailer.setDstAddr(dst_addr);
    if constexpr (noc_event_type == KernelProfilerNocEventMetadata::NocEventType::WRITE_) {
        ev_md.data.local_event_dst_trailer.counter_value = get_noc_counter_for_debug<true, posted>(noc_index);
    } else if constexpr (noc_event_type == KernelProfilerNocEventMetadata::NocEventType::READ) {
        ev_md.data.local_event_dst_trailer.counter_value = get_noc_counter_for_debug<false, posted>(noc_index);
    } else {
        ev_md.data.local_event_dst_trailer.counter_value = 0;
    }
    return ev_md;
}

template <KernelProfilerNocEventMetadata::NocEventType noc_event_type, bool posted, uint32_t STATIC_ID = 12345>
FORCE_INLINE void recordNocEvent(
    int32_t dst_x = -1,
    int32_t dst_y = -1,
    uint32_t num_bytes = 0,
    int8_t vc = -1,
    uint8_t noc = noc_index,
    uint32_t local_addr = 0,
    uint32_t dst_local_addr = 0) {
    KernelProfilerNocEventMetadata ev_md;

    auto& local_noc_event = ev_md.data.local_event;
    local_noc_event.noc_xfer_type = noc_event_type;
    local_noc_event.dst_x = dst_x;
    local_noc_event.dst_y = dst_y;
    local_noc_event.setAttributes(num_bytes, posted);
    local_noc_event.noc_vc = vc;
    local_noc_event.noc_type =
        (noc == 1) ? KernelProfilerNocEventMetadata::NocType::NOC_1 : KernelProfilerNocEventMetadata::NocType::NOC_0;

    if constexpr (kernel_profiler::NON_DROPPING) {
        KernelProfilerNocEventMetadata dst_data =
            createNocEventDstTrailer<noc_event_type, posted>(local_addr, dst_local_addr);

        kernel_profiler::flush_to_dram_if_full<kernel_profiler::DoingDispatch::DISPATCH>(
            kernel_profiler::PROFILER_L1_MARKER_UINT32_SIZE * 3);

        kernel_profiler::timeStampedData<
            STATIC_ID,
            kernel_profiler::DoingDispatch::DISPATCH,
            kernel_profiler::PacketTypes::TS_DATA_16B>(ev_md.asU64(), dst_data.asU64());
    } else {
        kernel_profiler::flush_to_dram_if_full<kernel_profiler::DoingDispatch::DISPATCH>();
        kernel_profiler::timeStampedData<STATIC_ID, kernel_profiler::DoingDispatch::DISPATCH>(ev_md.asU64());
    }
}

template <uint32_t STATIC_ID = 12345>
FORCE_INLINE void recordMulticastNocEvent(
    KernelProfilerNocEventMetadata::NocEventType noc_event_type,
    int32_t mcast_dst_start_x,
    int32_t mcast_dst_start_y,
    int32_t mcast_dst_end_x,
    int32_t mcast_dst_end_y,
    uint32_t num_bytes,
    int8_t vc = -1,
    uint8_t noc = noc_index) {
    KernelProfilerNocEventMetadata ev_md;

    auto& local_noc_event = ev_md.data.local_event;
    local_noc_event.noc_xfer_type = noc_event_type;
    local_noc_event.dst_x = mcast_dst_start_x;
    local_noc_event.dst_y = mcast_dst_start_y;
    local_noc_event.mcast_end_dst_x = mcast_dst_end_x;
    local_noc_event.mcast_end_dst_y = mcast_dst_end_y;
    local_noc_event.setAttributes(num_bytes, /*posted=*/false);
    local_noc_event.noc_vc = vc;
    local_noc_event.noc_type =
        (noc == 1) ? KernelProfilerNocEventMetadata::NocType::NOC_1 : KernelProfilerNocEventMetadata::NocType::NOC_0;

    kernel_profiler::flush_to_dram_if_full<kernel_profiler::DoingDispatch::DISPATCH>();
    kernel_profiler::timeStampedData<STATIC_ID, kernel_profiler::DoingDispatch::DISPATCH>(ev_md.asU64());
}

template <KernelProfilerNocEventMetadata::NocEventType noc_event_type, bool posted, typename AddrGen, typename NocIDU32>
FORCE_INLINE void recordNocEventWithID(
    uint32_t local_addr, NocIDU32 noc_id, AddrGen addrgen, uint32_t num_bytes, uint32_t offset, int8_t vc) {
    static_assert(std::is_same_v<NocIDU32, uint32_t>);
    static_assert(
        has_required_addrgen_traits_v<AddrGen>,
        "AddrGen must have get_noc_addr() and either page_size or log_base_2_of_page_size member variable");
    auto [decoded_x, decoded_y] = decode_noc_id_into_coord<addrgen.is_dram>(noc_id);
    if constexpr (kernel_profiler::NON_DROPPING) {
        auto noc_addr_local =
            decode_noc_addr_to_local_addr(get_noc_addr_from_bank_id<addrgen.is_dram>(noc_id, offset, noc_index));
        recordNocEvent<noc_event_type, posted>(
            decoded_x, decoded_y, num_bytes, vc, noc_index, local_addr, noc_addr_local);
    } else {
        recordNocEvent<noc_event_type, posted>(decoded_x, decoded_y, num_bytes, vc, noc_index, 0, 0);
    }
}

template <KernelProfilerNocEventMetadata::NocEventType noc_event_type, bool posted, typename NocAddrU64>
FORCE_INLINE void recordNocEventWithAddr(uint32_t local_addr, NocAddrU64 noc_addr, uint32_t num_bytes, int8_t vc) {
    static_assert(std::is_same_v<NocAddrU64, uint64_t>);
    auto [decoded_x, decoded_y] = decode_noc_addr_to_coord(noc_addr);
    if constexpr (kernel_profiler::NON_DROPPING) {
        auto noc_addr_local = decode_noc_addr_to_local_addr(noc_addr);
        recordNocEvent<noc_event_type, posted>(
            decoded_x, decoded_y, num_bytes, vc, noc_index, local_addr, noc_addr_local);
    } else {
        recordNocEvent<noc_event_type, posted>(decoded_x, decoded_y, num_bytes, vc, noc_index, 0, 0);
    }
}
}  // namespace noc_event_profiler

#define RECORD_NOC_EVENT_WITH_ADDR(event_type, local_addr, noc_addr, num_bytes, vc, posted)                         \
    {                                                                                                               \
        using NocEventType = KernelProfilerNocEventMetadata::NocEventType;                                          \
        if constexpr (event_type != NocEventType::WRITE_MULTICAST) {                                                \
            noc_event_profiler::recordNocEventWithAddr<event_type, posted>(local_addr, noc_addr, num_bytes, vc);    \
        } else {                                                                                                    \
            auto [mcast_dst_start_x, mcast_dst_start_y, mcast_dst_end_x, mcast_dst_end_y] =                         \
                noc_event_profiler::decode_noc_addr_to_multicast_coord(noc_addr);                                   \
            noc_event_profiler::recordMulticastNocEvent(                                                            \
                event_type, mcast_dst_start_x, mcast_dst_start_y, mcast_dst_end_x, mcast_dst_end_y, num_bytes, vc); \
        }                                                                                                           \
    }

#define RECORD_NOC_EVENT_WITH_ID(event_type, local_addr, noc_id, addrgen, offset, num_bytes, vc, posted) \
    {                                                                                                    \
        using NocEventType = KernelProfilerNocEventMetadata::NocEventType;                               \
        noc_event_profiler::recordNocEventWithID<event_type, posted>(                                    \
            local_addr, noc_id, addrgen, offset, num_bytes, vc);                                         \
    }

#define RECORD_NOC_EVENT(event_type, posted)                               \
    {                                                                      \
        using NocEventType = KernelProfilerNocEventMetadata::NocEventType; \
        noc_event_profiler::recordNocEvent<event_type, posted>();          \
    }

// preemptive quick push if transitioning from unlinked state to linked state
#define NOC_TRACE_QUICK_PUSH_IF_LINKED(cmd_buf, linked)         \
    {                                                           \
        kernel_profiler::quick_push_if_linked(cmd_buf, linked); \
    }

#else

// null macros when noc tracing is disabled
#define RECORD_NOC_EVENT_WITH_ADDR(type, local_addr, noc_addr, num_bytes, vc, posted)
#define RECORD_NOC_EVENT_WITH_ID(type, local_addr, noc_id, addrgen, offset, num_bytes, vc, posted)
#define RECORD_NOC_EVENT(type, posted)
#define NOC_TRACE_QUICK_PUSH_IF_LINKED(cmd_buf, linked)

#endif
