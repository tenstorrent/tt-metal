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
};
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
