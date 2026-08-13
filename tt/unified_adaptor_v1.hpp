// SPDX-License-Identifier: Apache-2.0
//
// Metal (v1) binding for the unified programming model.
//
// <tt/unified> includes this automatically. To bind the model to something else
// -- the host trace harness, say -- define TT_UNIFIED_CUSTOM_BINDING before
// including <tt/unified> and provide the same names yourself.
//
// Two jobs:
//
//   1. Derive the model's thread identity from the defines metal already emits
//      for every kernel build (tt_metal/llrt/hal/tt-1xx/hal_1xx_common.cpp).
//      Nothing is passed from the host -- pointing three KernelDescriptors at
//      one source is enough, because each build already knows what it is.
//
//        COMPILE_FOR_BRISC   -> DM thread 0   (writer, by metal convention)
//        COMPILE_FOR_NCRISC  -> DM thread 1   (reader)
//        UCK_CHLKC_{UNPACK,MATH,PACK} -> compute
//
//   2. Map the model's intrinsics onto real metal APIs, where metal does not
//      already provide a thread-polymorphic name. The CB protocol needs no
//      binding at all -- metal's cb_reserve_back / cb_push_back / cb_wait_front
//      / cb_pop_front already resolve per projection.
//
// Note on what is NOT here: there are almost no cross-projection no-ops. Every
// compute intrinsic is only ever called from inside a `#if IS_COMPUTE_THREAD`
// region (in fusion.hpp's Strategy and op guards), and every data-movement
// intrinsic only from inside a `#if IS_DM_THREAD` region (in unified.hpp's
// noc_*). So each projection declares only its own half, and there are no
// cross-projection no-op functions left. `TensorAccessor` is named on a shared
// path, but it is a *type*, and compute simply gets an empty one under the same
// name. The hardware-startup entry points live in tt/unified_math.hpp, where
// they self-guard.

#pragma once

// ---------------------------------------------------------------------------
// Thread identity
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Metal headers, per projection
//
// tensor_accessor_args.h works on both (it only needs get_compile_time_arg_val),
// which is what lets a shared source chain CT-arg offsets. The full
// tensor_accessor.h does NOT compile on a TRISC -- it wants NOC_INDEX and
// redeclares get_common_arg_addr -- so accessors are constructed only inside
// data-movement regions.
// ---------------------------------------------------------------------------

#include <cstdint>

#if defined(IS_COMPUTE_THREAD) && IS_COMPUTE_THREAD
#include "api/compute/common.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/relu.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/tensor/tensor_accessor_args.h"
#else
#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"
#include "api/tensor/tensor_accessor_args.h"
#endif

#if defined(IS_COMPUTE_THREAD) && IS_COMPUTE_THREAD
// api/tensor/tensor_accessor.h does not compile on a TRISC: it wants NOC_INDEX
// and redeclares get_common_arg_addr against api/compute/common.h. A compute
// kernel never dereferences an accessor -- it only carries one through a
// statement shared with the data-movement projections -- so an empty stand-in
// under metal's own name is enough, and kernels spell it identically either way.
struct TensorAccessor {
    template <typename Args>
    constexpr TensorAccessor(Args, uint32_t) {}

    // Present so a custom load/store routine compiles here; see below.
    std::uint64_t get_noc_addr(uint32_t, uint32_t = 0, uint8_t = 0) const {
        ASSERT(false);
        return 0;
    }
};

// The data-movement intrinsics, as no-ops.
//
// Needed because a custom routine's body is compiled on EVERY projection. The
// harness only *calls* it from inside a `#if IS_DM_THREAD` region, but the
// closure lives in the shared kernel source, and C++ compiles a non-generic
// lambda's body where it is written -- so
//
//     noc_load<1>(storage, [&](uint32_t l1, uint32_t bytes) {
//         noc_async_read(acc.get_noc_addr(p), l1, bytes);   // <-- compiled on TRISC too
//     });
//
// fails to build on a compute projection unless these names resolve. (A generic
// `[](auto l1, ...)` lambda happens to survive, because a dependent argument
// defers lookup to an instantiation that never happens on this projection -- but
// that is too fragile to ask users to rely on.)
//
// They are unreachable rather than merely empty: nothing on a compute thread has
// any business touching the NOC, so each one asserts if it is ever really
// called, and the dead bodies strip out of the TRISC binary.
inline void noc_async_read(std::uint64_t, uint32_t, uint32_t, uint8_t = 0) { ASSERT(false); }
inline void noc_async_write(uint32_t, std::uint64_t, uint32_t, uint8_t = 0) { ASSERT(false); }
inline std::uint64_t get_noc_addr(uint32_t, uint32_t, uint32_t, uint8_t = 0) {
    ASSERT(false);
    return 0;
}
inline std::uint64_t get_noc_addr(uint32_t) {
    ASSERT(false);
    return 0;
}
inline void noc_async_read_barrier(uint8_t = 0) { ASSERT(false); }
inline void noc_async_write_barrier(uint8_t = 0) { ASSERT(false); }
inline void noc_async_writes_flushed(uint8_t = 0) { ASSERT(false); }
#endif

namespace tt {
namespace unified {

// ---------------------------------------------------------------------------
// Circular-buffer protocol
//
// Nothing to bind: metal already exposes cb_reserve_back / cb_push_back /
// cb_wait_front / cb_pop_front under the same names on every projection --
// dataflow_api.h on a DM core, api/compute/cb_api.h on a TRISC (where they
// resolve to PACK(llk_push_tiles) / UNPACK(llk_wait_tiles) and friends). The
// model calls them directly.
// ---------------------------------------------------------------------------

#if defined(IS_COMPUTE_THREAD) && IS_COMPUTE_THREAD

// ---------------------------------------------------------------------------
// Compute
// ---------------------------------------------------------------------------

#else

// ---------------------------------------------------------------------------
// Data movement
// ---------------------------------------------------------------------------

// The CB's *configured* page size, not the data format's tile size --
// get_tile_size() is derived from unpack_tile_size[] and only coincides with the
// page size when a page happens to hold exactly one tile.
//
// fifo_page_size is stored pre-shifted by cb_addr_shift, which is 0 on a
// data-movement build (bytes) and CIRCULAR_BUFFER_COMPUTE_ADDR_SHIFT on a TRISC
// (16B words). The shift is written out so this stays right if it ever moves.
inline uint32_t cb_page_bytes(uint32_t cb) { return get_local_cb_interface(cb).fifo_page_size << cb_addr_shift; }

#endif

}  // namespace unified
}  // namespace tt
