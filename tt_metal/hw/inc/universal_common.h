#pragma once

#include <cstdint>

#ifdef COMPILE_FOR_TRISC
#define NOC_INDEX 0
#define NOC_MODE 0
#include "debug/assert.h"
#include "dataflow_api_common.h"
#include "accessor/tensor_accessor.h"
#endif

#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api.h"

using namespace ckernel;

#ifdef COMPILE_FOR_TRISC
#define KERNEL_MAIN       \
    namespace NAMESPACE { \
    void MAIN;            \
    }                     \
    void NAMESPACE::MAIN
#else
#define KERNEL_MAIN void kernel_main()
#endif

namespace universal_kernel::detail {

#ifdef COMPILE_FOR_TRISC
uint32_t read_pages[TOTAL_NUM_CIRCULAR_BUFFERS];
uint32_t popped_pages[TOTAL_NUM_CIRCULAR_BUFFERS];
#endif
#ifdef COMPILE_FOR_BRISC
uint32_t offset_pages[TOTAL_NUM_CIRCULAR_BUFFERS];
#endif

FORCE_INLINE void release_read_tile(uint32_t cb_id) {
#ifdef COMPILE_FOR_TRISC
    cb_pop_front(cb_id, 1);
    popped_pages[cb_id] += 1;
#endif
#ifdef COMPILE_FOR_BRISC
    noc_async_read_barrier();
    cb_push_back(cb_id, 1);
    offset_pages[cb_id] -= 1;
#endif
}

struct ReadTile {
    constexpr ReadTile(uint32_t cb_id, uint32_t id) : cb_id_(cb_id), id_(id) {}
    ReadTile(const ReadTile&) = delete;
    ReadTile& operator=(const ReadTile&) = delete;
    ~ReadTile() { release_read_tile(cb_id_); }
    uint32_t local_id() const {
#ifdef COMPILE_FOR_TRISC
        return id_ - popped_pages[cb_id_];
#else
        return id_;
#endif
    }
    uint32_t cb_id() const { return cb_id_; }

private:
    uint32_t cb_id_ = 0;
    uint32_t id_ = 0;
};
struct ConstantTile {
    constexpr ConstantTile(uint32_t cb_id, uint32_t id) : cb_id_(cb_id), id_(id) {}
    uint32_t local_id() const { return id_; }
    uint32_t cb_id() const { return cb_id_; }

private:
    uint32_t cb_id_ = 0;
    uint32_t id_ = 0;
};

template <typename Accessor>
FORCE_INLINE auto read_tile_impl(
    const uint32_t cb_id, const uint32_t id, const Accessor& addrgen, const uint32_t tile_size_bytes) {
#ifdef COMPILE_FOR_TRISC
    cb_wait_front(cb_id, 1);
    read_pages[cb_id] += 1;
    return ReadTile(cb_id, read_pages[cb_id] - 1);
#elif COMPILE_FOR_BRISC
    cb_reserve_back(cb_id, 1);
    noc_async_read_tile(id, addrgen, get_write_ptr(cb_id) + tile_size_bytes * offset_pages[cb_id]);
    offset_pages[cb_id] += 1;
    return ReadTile(cb_id, offset_pages[cb_id] - 1);
#else
    return ConstantTile(0, 0);
#endif
}

template <typename Accessor>
FORCE_INLINE void write_tile_impl(
    const uint32_t from_dst_idx,
    const uint32_t cb_id,
    const uint32_t into_page_id,
    const Accessor& addrgen,
    const uint32_t tile_size_bytes) {
#ifdef COMPILE_FOR_TRISC
    cb_reserve_back(cb_id, 1);
    pack_tile(from_dst_idx, cb_id);
    cb_push_back(cb_id, 1);
#elif COMPILE_FOR_NCRISC
    cb_wait_front(cb_id, 1);
    noc_async_write_tile(into_page_id, addrgen, get_read_ptr(cb_id));
    noc_async_write_barrier();
    cb_pop_front(cb_id, 1);
#endif
}

}  // namespace universal_kernel::detail

#define read_tile(tensor, id) \
    universal_kernel::detail::read_tile_impl(tensor##_cb, id, tensor, tensor##_page_size_bytes)
#define write_tile(from_dst_idx, tensor, into_page_id) \
    universal_kernel::detail::write_tile_impl(from_dst_idx, tensor##_cb, into_page_id, tensor, tensor##_page_size_bytes)
using ReadTile = universal_kernel::detail::ReadTile;
using ConstantTile = universal_kernel::detail::ConstantTile;

template <typename TileA, typename TileB>
FORCE_INLINE void add_tiles(const TileA& in0, const TileB& in1, uint32_t dst_idx) {
    add_tiles(in0.cb_id(), in1.cb_id(), in0.local_id(), in1.local_id(), dst_idx);
}

template <typename TileA, typename TileB>
FORCE_INLINE void reduce_tile(const TileA& in0, const TileB& in1, uint32_t dst_idx) {
    reduce_tile(in0.cb_id(), in1.cb_id(), in0.local_id(), in1.local_id(), dst_idx);
}
