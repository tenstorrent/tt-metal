#include <cstdint>

#ifdef COMPILE_FOR_TRISC
#define NOC_INDEX 0
#define NOC_MODE 0
#include "debug/assert.h"
#include "dataflow_api_common.h"
#include "accessor/tensor_accessor.h"
#endif

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
uint32_t offset_pages[TOTAL_NUM_CIRCULAR_BUFFERS];
}

template <typename Accessor>
FORCE_INLINE void read_tile(
    const uint32_t cb_id, const uint32_t id, const Accessor& addrgen, const uint32_t tile_size_bytes) {
#ifdef COMPILE_FOR_TRISC
    cb_wait_front(cb_id, 1);
#endif
#ifdef COMPILE_FOR_BRISC
    cb_reserve_back(cb_id, 1);
    noc_async_read_tile(
        id, addrgen, get_write_ptr(cb_id) + tile_size_bytes * universal_kernel::detail::offset_pages[cb_id]);
    universal_kernel::detail::offset_pages[cb_id] += 1;
#endif
}

FORCE_INLINE void release_read_tiles(uint32_t cb_id, uint32_t num_tiles) {
#ifdef COMPILE_FOR_TRISC
    cb_pop_front(cb_id, num_tiles);
#endif
#ifdef COMPILE_FOR_BRISC
    noc_async_read_barrier();
    cb_push_back(cb_id, num_tiles);
    universal_kernel::detail::offset_pages[cb_id] -= num_tiles;
#endif
}

template <typename Accessor>
FORCE_INLINE void write_packed_tile(
    const uint32_t from_dst_idx,
    const uint32_t cb_id,
    const uint32_t into_page_id,
    const Accessor& addrgen,
    const uint32_t tile_size_bytes) {
#ifdef COMPILE_FOR_TRISC
    cb_reserve_back(cb_id, 1);
    pack_tile(from_dst_idx, cb_id);
#endif
#ifdef COMPILE_FOR_NCRISC
    cb_wait_front(cb_id, 1);
    noc_async_write_tile(
        into_page_id, addrgen, get_read_ptr(cb_id) + tile_size_bytes * universal_kernel::detail::offset_pages[cb_id]);
    universal_kernel::detail::offset_pages[cb_id] += 1;
#endif
}

FORCE_INLINE void release_write_tiles(const uint32_t cb_id, const uint32_t num_tiles) {
#ifdef COMPILE_FOR_TRISC
    cb_push_back(cb_id, num_tiles);
#endif
#ifdef COMPILE_FOR_NCRISC
    noc_async_write_barrier();
    cb_pop_front(cb_id, num_tiles);
    universal_kernel::detail::offset_pages[cb_id] -= num_tiles;
#endif
}
