// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if defined(PROFILE_NOC_EVENTS) && (defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC) || \
                                    defined(COMPILE_FOR_ERISC) || defined(COMPILE_FOR_IDLE_ERISC))

#include <utility>
#include <tuple>
#include "event_metadata.hpp"
#include "risc_attribs.h"
#include <algorithm>

namespace noc_event_profiler {

FORCE_INLINE
std::pair<uint32_t, uint32_t> decode_noc_coord_reg_to_coord(uint16_t noc_xy_bits) {
    constexpr uint32_t NOC_COORD_MASK = 0x3F;
    uint32_t x = noc_xy_bits & NOC_COORD_MASK;
    uint32_t y = (noc_xy_bits >> NOC_ADDR_NODE_ID_BITS) & NOC_COORD_MASK;
    return {x, y};
}

FORCE_INLINE
std::pair<uint32_t, uint32_t> decode_noc_xy_to_coord(uint32_t noc_xy) {
    // shift so that coordinate is in LSB
    return decode_noc_coord_reg_to_coord(noc_xy >> NOC_COORD_REG_OFFSET);
}

FORCE_INLINE
std::pair<uint32_t, uint32_t> decode_noc_addr_to_coord(uint64_t noc_addr) {
    return decode_noc_coord_reg_to_coord(noc_addr >> NOC_ADDR_LOCAL_BITS);
}

FORCE_INLINE
std::tuple<uint32_t, uint32_t, uint32_t, uint32_t> decode_noc_addr_to_multicast_coord(uint64_t noc_addr) {
    // coordinates are stored as two packed pairs. End coordinate is in lower
    // bits like normal noc address; Start coordinate is in higher bits
    auto [xend, yend] = decode_noc_coord_reg_to_coord(noc_addr >> NOC_ADDR_LOCAL_BITS);
    auto [xstart, ystart] =
        decode_noc_coord_reg_to_coord(noc_addr >> NOC_ADDR_LOCAL_BITS + (2 * NOC_ADDR_NODE_ID_BITS));

    return {xstart, ystart, xend, yend};
}

template <bool DRAM>
FORCE_INLINE std::pair<uint32_t, uint32_t> decode_noc_id_into_coord(uint32_t id, uint8_t noc = noc_index) {
    uint32_t bank_offset_index = interleaved_addr_gen::get_bank_offset_index<DRAM>(id);
    uint32_t bank_index = interleaved_addr_gen::get_bank_index<DRAM>(id, bank_offset_index);
    return decode_noc_xy_to_coord(interleaved_addr_gen::get_noc_xy<DRAM>(bank_index, noc));
}

template <uint32_t STATIC_ID = 12345>
FORCE_INLINE void recordNocEvent(
    KernelProfilerNocEventMetadata::NocEventType noc_event_type,
    int32_t dst_x = -1,
    int32_t dst_y = -1,
    uint32_t num_bytes = 0,
    int8_t vc = -1,
    uint8_t noc = noc_index) {
    KernelProfilerNocEventMetadata ev_md;
    ev_md.dst_x = dst_x;
    ev_md.dst_y = dst_y;
    ev_md.noc_xfer_type = noc_event_type;
    ev_md.num_bytes = std::min(std::numeric_limits<std::uint32_t>::max(), num_bytes);
    ev_md.noc_vc = vc;
    ev_md.noc_type =
        (noc == 1) ? KernelProfilerNocEventMetadata::NocType::NOC_1 : KernelProfilerNocEventMetadata::NocType::NOC_0;

    kernel_profiler::flush_to_dram_if_full<kernel_profiler::DoingDispatch::DISPATCH>();
    kernel_profiler::timeStampedData<STATIC_ID, kernel_profiler::DoingDispatch::DISPATCH>(ev_md.asU64());
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
    ev_md.dst_x = mcast_dst_start_x;
    ev_md.dst_y = mcast_dst_start_y;
    ev_md.mcast_end_dst_x = mcast_dst_end_x;
    ev_md.mcast_end_dst_y = mcast_dst_end_y;

    ev_md.noc_xfer_type = noc_event_type;
    ev_md.num_bytes = std::min(std::numeric_limits<std::uint32_t>::max(), num_bytes);

    ev_md.noc_vc = vc;
    ev_md.noc_type =
        (noc == 1) ? KernelProfilerNocEventMetadata::NocType::NOC_1 : KernelProfilerNocEventMetadata::NocType::NOC_0;

    kernel_profiler::flush_to_dram_if_full<kernel_profiler::DoingDispatch::DISPATCH>();
    kernel_profiler::timeStampedData<STATIC_ID, kernel_profiler::DoingDispatch::DISPATCH>(ev_md.asU64());
}

template <bool DRAM, typename NocIDU32>
void recordNocEventWithID(
    KernelProfilerNocEventMetadata::NocEventType noc_event_type, NocIDU32 noc_id, uint32_t num_bytes, int8_t vc) {
    static_assert(std::is_same_v<NocIDU32, uint32_t>);
    auto [decoded_x, decoded_y] = decode_noc_id_into_coord<DRAM>(noc_id);
    recordNocEvent(noc_event_type, decoded_x, decoded_y, num_bytes, vc);
}

template <typename NocAddrU64>
void recordNocEventWithAddr(
    KernelProfilerNocEventMetadata::NocEventType noc_event_type, NocAddrU64 noc_addr, uint32_t num_bytes, int8_t vc) {
    static_assert(std::is_same_v<NocAddrU64, uint64_t>);
    auto [decoded_x, decoded_y] = decode_noc_addr_to_coord(noc_addr);
    recordNocEvent(noc_event_type, decoded_x, decoded_y, num_bytes, vc);
}

}  // namespace noc_event_profiler

#define RECORD_NOC_EVENT_WITH_ADDR(event_type, noc_addr, num_bytes, vc)                                             \
    {                                                                                                               \
        using NocEventType = KernelProfilerNocEventMetadata::NocEventType;                                          \
        if constexpr (event_type != NocEventType::WRITE_MULTICAST) {                                                \
            noc_event_profiler::recordNocEventWithAddr(event_type, noc_addr, num_bytes, vc);                        \
        } else {                                                                                                    \
            auto [mcast_dst_start_x, mcast_dst_start_y, mcast_dst_end_x, mcast_dst_end_y] =                         \
                noc_event_profiler::decode_noc_addr_to_multicast_coord(noc_addr);                                   \
            noc_event_profiler::recordMulticastNocEvent(                                                            \
                event_type, mcast_dst_start_x, mcast_dst_start_y, mcast_dst_end_x, mcast_dst_end_y, num_bytes, vc); \
        }                                                                                                           \
    }

#define RECORD_NOC_EVENT_WITH_ID(event_type, noc_id, num_bytes, vc)                        \
    {                                                                                      \
        using NocEventType = KernelProfilerNocEventMetadata::NocEventType;                 \
        noc_event_profiler::recordNocEventWithID<DRAM>(event_type, noc_id, num_bytes, vc); \
    }

#define RECORD_NOC_EVENT(event_type)                                       \
    {                                                                      \
        using NocEventType = KernelProfilerNocEventMetadata::NocEventType; \
        noc_event_profiler::recordNocEvent(event_type);                    \
    }

#else

// null macros when noc tracing is disabled
#define RECORD_NOC_EVENT_WITH_ADDR(type, noc_addr, num_bytes, vc)
#define RECORD_NOC_EVENT_WITH_ID(type, noc_id, num_bytes, vc)
#define RECORD_NOC_EVENT(type)

#endif
