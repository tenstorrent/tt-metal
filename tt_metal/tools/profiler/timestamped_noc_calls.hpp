#include "dataflow_api.h"
#include "event_metadata.hpp"
#include "tools/profiler/kernel_profiler.hpp"

inline std::pair<uint32_t, uint32_t> decode_noc_xy_to_coord(uint32_t noc_xy) {
    // shift so that coordinate is in LSB
    noc_xy = noc_xy >> NOC_COORD_REG_OFFSET;

    constexpr uint32_t NOC_COORD_MASK = 0x3F;

    uint32_t encoded_x = (noc_xy)&NOC_COORD_MASK;
    uint32_t decoded_x = (noc_index == 1) ? (noc_size_x - 1 - encoded_x) : encoded_x;

    uint32_t encoded_y = (noc_xy >> (NOC_ADDR_NODE_ID_BITS)) & NOC_COORD_MASK;
    uint32_t decoded_y = (noc_index == 1) ? (noc_size_y - 1 - encoded_y) : encoded_y;

    return {decoded_x, decoded_y};
}

inline std::pair<uint32_t, uint32_t> decode_noc_addr_to_coord(uint64_t noc_addr) {
    // See noc_parameters.h for definition of NOC address
    constexpr int NOC_COORD_MASK = 0x3F;

    uint32_t encoded_x = (noc_addr >> NOC_ADDR_LOCAL_BITS) & NOC_COORD_MASK;
    uint32_t decoded_x = (noc_index == 1) ? (noc_size_x - 1 - encoded_x) : encoded_x;

    uint32_t encoded_y = (noc_addr >> (NOC_ADDR_LOCAL_BITS + NOC_ADDR_NODE_ID_BITS)) & NOC_COORD_MASK;
    uint32_t decoded_y = (noc_index == 1) ? (noc_size_y - 1 - encoded_y) : encoded_y;

    return {decoded_x, decoded_y};
}

template <bool DRAM>
inline std::pair<uint32_t, uint32_t> decode_noc_id_into_coord(uint32_t id, uint8_t noc = noc_index) {
    uint32_t bank_offset_index = interleaved_addr_gen::get_bank_offset_index<DRAM>(id);
    uint32_t bank_index = interleaved_addr_gen::get_bank_index<DRAM>(id, bank_offset_index);
    return decode_noc_xy_to_coord(interleaved_addr_gen::get_noc_xy<DRAM>(bank_index, noc));
}

template <uint16_t STATIC_ID = 12345>
inline void recordTimestampedEvent(
    KernelProfilerEventMetadata::NocXferType noc_xfer_type,
    uint32_t dst_x = 0,
    uint32_t dst_y = 0,
    uint32_t num_bytes = 0,
    uint8_t noc = noc_index) {
    KernelProfilerEventMetadata ev_md;
    ev_md.dst_x = dst_x;
    ev_md.dst_y = dst_y;
    ev_md.noc_xfer_type = noc_xfer_type;
    ev_md.noc_type =
        (noc == 1) ? KernelProfilerEventMetadata::NocType::NOC_1 : KernelProfilerEventMetadata::NocType::NOC_0;
    ev_md.num_bytes = num_bytes;

    // not actually using the const id here for now
    DeviceTimestampedData(STATIC_ID, ev_md.asU64());
}

/* -------------------------------------------------------------------------- */
/*                   tile-sized reads with dram address gen                   */
/* -------------------------------------------------------------------------- */
template <bool DRAM, uint32_t tile_hw>
void noc_async_write_tile_ts(
    const uint32_t id,
    const InterleavedAddrGenFast<DRAM, tile_hw>& s,
    std::uint32_t dst_local_l1_addr,
    uint8_t noc = noc_index) {
    constexpr uint32_t NUM_BYTES_IN_TILE = 2048;
    auto [decoded_x, decoded_y] = decode_noc_id_into_coord<DRAM>(id);
    recordTimestampedEvent(KernelProfilerEventMetadata::NocXferType::WRITE, decoded_x, decoded_y, NUM_BYTES_IN_TILE);

    noc_async_write_tile(id, s, dst_local_l1_addr, noc);
}

template <bool DRAM, uint32_t tile_hw>
void noc_async_read_tile_ts(
    const uint32_t id,
    const InterleavedAddrGenFast<DRAM, tile_hw>& s,
    std::uint32_t dst_local_l1_addr,
    uint32_t offset = 0,
    uint8_t noc = noc_index) {
    constexpr uint32_t NUM_BYTES_IN_TILE = 2048;
    auto [decoded_x, decoded_y] = decode_noc_id_into_coord<DRAM>(id);
    recordTimestampedEvent(KernelProfilerEventMetadata::NocXferType::READ, decoded_x, decoded_y, NUM_BYTES_IN_TILE);

    noc_async_read_tile(id, s, dst_local_l1_addr, offset, noc);
}

/* -------------------------------------------------------------------------- */
/*                       arbitrary size read/write calls                      */
/* -------------------------------------------------------------------------- */
void noc_async_read_ts(std::uint64_t noc_addr, uint32_t local_l1_addr, uint32_t num_bytes, int noc = noc_index) {
    auto [decoded_x, decoded_y] = decode_noc_addr_to_coord(noc_addr);
    recordTimestampedEvent(KernelProfilerEventMetadata::NocXferType::READ, decoded_x, decoded_y, num_bytes);

    noc_async_read(noc_addr, local_l1_addr, num_bytes, noc);
}

void noc_async_write_ts(uint32_t local_l1_addr, std::uint64_t noc_addr, uint32_t num_bytes, int noc = noc_index) {
    auto [decoded_x, decoded_y] = decode_noc_addr_to_coord(noc_addr);
    recordTimestampedEvent(KernelProfilerEventMetadata::NocXferType::WRITE, decoded_x, decoded_y, num_bytes);

    noc_async_write(local_l1_addr, noc_addr, num_bytes, noc);
}

/* -------------------------------------------------------------------------- */
/*                                flush checks                                */
/* -------------------------------------------------------------------------- */
void noc_async_writes_flushed_ts(uint8_t noc = noc_index) {
    recordTimestampedEvent(KernelProfilerEventMetadata::NocXferType::WRITE_FLUSH);
    noc_async_writes_flushed(noc);
}

/* -------------------------------------------------------------------------- */
/*                                  barriers                                  */
/* -------------------------------------------------------------------------- */
void noc_async_read_barrier_ts(uint8_t noc = noc_index) {
    recordTimestampedEvent(KernelProfilerEventMetadata::NocXferType::READ_BARRIER);
    noc_async_read_barrier(noc);
}
void noc_async_write_barrier_ts(uint8_t noc = noc_index) {
    recordTimestampedEvent(KernelProfilerEventMetadata::NocXferType::WRITE_BARRIER);
    noc_async_write_barrier(noc);
}