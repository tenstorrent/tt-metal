// SPDX-License-Identifier: Apache-2.0

#pragma once

#if defined(COMPILE_FOR_BRISC)
#define IS_DM_THREAD 1
#define TT_DM_THREAD_ID 0
#elif defined(COMPILE_FOR_NCRISC)
#define IS_DM_THREAD 1
#define TT_DM_THREAD_ID 1
#elif defined(UCK_CHLKC_UNPACK) || defined(UCK_CHLKC_MATH) || defined(UCK_CHLKC_PACK)
#define IS_COMPUTE_THREAD 1
#else
#error "unified_metal.hpp: no metal thread-identity define present"
#endif

#include <cstdint>
#include <type_traits>

#include "api/dataflow/dataflow_buffer.h"

#if defined(IS_COMPUTE_THREAD) && IS_COMPUTE_THREAD
#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/eltwise_unary/relu.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/sqrt.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/bcast.h"
#include "api/compute/binary_max_min.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reduce.h"
#include "api/compute/tile_move_copy.h"
#include "api/tensor/tensor_accessor_args.h"
#else
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/tensor/tensor_accessor.h"
#include "api/tensor/tensor_accessor_args.h"
#endif

#if defined(IS_COMPUTE_THREAD) && IS_COMPUTE_THREAD
struct TensorAccessor {
    template <typename Args>
    constexpr TensorAccessor(Args, uint32_t) {}

    template <typename Token, typename = std::enable_if_t<!std::is_same<std::decay_t<Token>, TensorAccessor>::value>>
    constexpr explicit TensorAccessor(Token) {}

    std::uint64_t get_noc_addr(uint32_t, uint32_t = 0, uint8_t = 0) const {
        ASSERT(false);
        return 0;
    }
};

inline void noc_async_read(std::uint64_t, uint32_t, uint32_t, uint8_t = 0) { ASSERT(false); }
inline void noc_async_write(uint32_t, std::uint64_t, uint32_t, uint8_t = 0) { ASSERT(false); }
inline void noc_async_write_multicast(uint32_t, std::uint64_t, uint32_t, uint32_t, bool = false, uint8_t = 0) {
    ASSERT(false);
}
inline std::uint64_t get_noc_addr(uint32_t, uint32_t, uint32_t, uint8_t = 0) {
    ASSERT(false);
    return 0;
}
inline std::uint64_t get_noc_addr(uint32_t) {
    ASSERT(false);
    return 0;
}
inline uint32_t get_write_ptr(uint32_t) {
    ASSERT(false);
    return 0;
}
inline uint32_t get_read_ptr(uint32_t) {
    ASSERT(false);
    return 0;
}

inline constexpr uint8_t noc_index = 0;

inline void noc_async_read_barrier(uint8_t = 0) { ASSERT(false); }
inline void noc_async_write_barrier(uint8_t = 0) { ASSERT(false); }
inline void noc_async_writes_flushed(uint8_t = 0) { ASSERT(false); }
inline void noc_async_atomic_barrier(uint8_t = 0) { ASSERT(false); }

struct Noc {
    Noc() = default;
    explicit Noc(uint8_t) {}
    void async_read_barrier() const { ASSERT(false); }
    void async_write_barrier() const { ASSERT(false); }
    void async_writes_flushed() const { ASSERT(false); }
    void async_atomic_barrier() const { ASSERT(false); }
};
#endif

namespace tt {
namespace unified {

inline DataflowBuffer buffer(uint32_t dfb) { return DataflowBuffer(static_cast<uint16_t>(dfb)); }

inline uint32_t dfb_entry_bytes(uint32_t dfb) { return buffer(dfb).get_entry_size(); }

inline uint32_t dfb_num_entries(uint32_t dfb) { return buffer(dfb).get_total_num_entries(); }

#if defined(IS_COMPUTE_THREAD) && IS_COMPUTE_THREAD && !defined(UCK_CHLKC_PACK)
#define TT_U_HAVE_DFB_TILE_GEOMETRY 1

inline constexpr uint32_t dfb_tile_rows(uint32_t dfb) {
    return static_cast<uint32_t>(unpack_tile_face_r_dim[dfb]) * static_cast<uint32_t>(unpack_num_faces_r_dim[dfb]);
}

inline constexpr uint32_t dfb_tile_cols(uint32_t dfb) {
    return static_cast<uint32_t>(unpack_num_faces_c_dim[dfb]) * static_cast<uint32_t>(ckernel::FACE_C_DIM);
}

inline constexpr uint32_t unpack_tile_geometry(uint32_t dfb) {
    return (static_cast<uint32_t>(unpack_tile_face_r_dim[dfb]) << 24) |
           (static_cast<uint32_t>(unpack_tile_num_faces[dfb]) << 16) |
           (static_cast<uint32_t>(unpack_partial_face[dfb]) << 8) | static_cast<uint32_t>(unpack_narrow_tile[dfb]);
}
#endif

#if defined(IS_COMPUTE_THREAD) && IS_COMPUTE_THREAD && defined(UCK_CHLKC_PACK)
#define TT_U_HAVE_PACK_TILE_GEOMETRY 1

inline constexpr uint32_t pack_tile_geometry(uint32_t dfb) {
    return (static_cast<uint32_t>(pack_tile_face_r_dim[dfb]) << 24) |
           (static_cast<uint32_t>(pack_tile_num_faces[dfb]) << 16) |
           (static_cast<uint32_t>(pack_partial_face[dfb]) << 8) | static_cast<uint32_t>(pack_narrow_tile[dfb]);
}
#endif

}  // namespace unified
}  // namespace tt
