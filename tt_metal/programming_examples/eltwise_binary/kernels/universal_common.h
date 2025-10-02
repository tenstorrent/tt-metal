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

constexpr auto cb_in = tt::CBIndex::c_0;
constexpr auto cb_out = tt::CBIndex::c_1;

uint32_t offset_pages = 0;

template <typename Accessor>
FORCE_INLINE void read_tile(const uint32_t id, const Accessor& addrgen, const uint32_t tile_size_bytes) {
#ifdef COMPILE_FOR_TRISC
    cb_wait_front(cb_in, 1);
#endif
#ifdef COMPILE_FOR_BRISC
    cb_reserve_back(cb_in, 1);
    noc_async_read_tile(id, addrgen, get_write_ptr(cb_in) + tile_size_bytes * offset_pages);
    offset_pages += 1;
#endif
}

FORCE_INLINE void release_read_tiles(uint32_t num_tiles) {
#ifdef COMPILE_FOR_TRISC
    cb_pop_front(cb_in, num_tiles);
#endif
#ifdef COMPILE_FOR_BRISC
    noc_async_read_barrier();
    cb_push_back(cb_in, num_tiles);
    offset_pages -= num_tiles;
#endif
}

template <typename Accessor>
FORCE_INLINE void write_packed_tile(
    const uint32_t from_dst_idx, const uint32_t into_page_id, const Accessor& addrgen, const uint32_t tile_size_bytes) {
#ifdef COMPILE_FOR_TRISC
    cb_reserve_back(cb_out, 1);
    pack_tile(from_dst_idx, cb_out);
#endif
#ifdef COMPILE_FOR_NCRISC
    cb_wait_front(cb_out, 1);
    noc_async_write_tile(into_page_id, addrgen, get_read_ptr(cb_out) + tile_size_bytes * offset_pages);
    offset_pages += 1;
#endif
}

FORCE_INLINE void release_write_tiles(uint32_t num_tiles) {
#ifdef COMPILE_FOR_TRISC
    cb_push_back(cb_out, num_tiles);
#endif
#ifdef COMPILE_FOR_NCRISC
    noc_async_write_barrier();
    cb_pop_front(cb_out, num_tiles);
    offset_pages -= num_tiles;
#endif
}
