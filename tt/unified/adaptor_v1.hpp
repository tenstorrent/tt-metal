// SPDX-License-Identifier: Apache-2.0
//
// Metal (v1) binding for the unified programming model.
//
// <tt/unified/core> includes this automatically. To bind the model to something else
// -- the host trace harness, say -- define TT_UNIFIED_CUSTOM_BINDING before
// including <tt/unified/core> and provide the same names yourself.
//
// Two jobs:
//
//   1. Derive thread identity from the defines metal already emits for every
//      kernel build (tt_metal/llrt/hal/tt-1xx/hal_1xx_common.cpp). Nothing is
//      passed from the host -- pointing three KernelDescriptors at one source is
//      enough, because each build already knows what it is.
//
//        COMPILE_FOR_BRISC            -> DM thread 0  (writer, by convention)
//        COMPILE_FOR_NCRISC           -> DM thread 1  (reader)
//        UCK_CHLKC_{UNPACK,MATH,PACK} -> compute
//
//   2. Map the model's intrinsics onto metal APIs where metal has no
//      thread-polymorphic name. The CB protocol needs no binding at all:
//      cb_reserve_back / cb_push_back / cb_wait_front / cb_pop_front already
//      resolve per projection -- dataflow_api.h on a DM core, api/compute/cb_api.h
//      on a TRISC (where they become PACK(llk_push_tiles) and friends).

#pragma once

// --- Thread identity ---

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

// --- Metal headers, per projection ---
//
// tensor_accessor_args.h works on both (it only needs get_compile_time_arg_val),
// which is what lets a shared source chain CT-arg offsets.

#include <cstdint>
#include <type_traits>

#if defined(IS_COMPUTE_THREAD) && IS_COMPUTE_THREAD
#include "api/compute/common.h"
// Both binary headers, because add/sub/mul exist on both units: _sfpu.h has the
// forms that take two DST slots, eltwise_binary.h the ones that read two circular
// buffers. Which is cheaper is what FpuEltwiseFusion is for.
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/eltwise_unary/relu.h"
// silu has no per-op header of its own: it is declared only in metal's umbrella
// compute_kernel_api.h, so that is where SwiGLU's activation has to come from.
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
#include "api/dataflow/noc_semaphore.h"
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

    // The Metal 2.0 spelling: TensorAccessor(tensor::name), built from a binding token
    // rather than an args block and a base address. A compute projection still never
    // dereferences one -- it only carries it through a statement shared with the
    // data-movement projections -- so this stand-in takes the token and ignores it, the
    // same way the two-argument form above ignores its arguments.
    //
    // Constrained so it cannot out-compete the copy constructor for a non-const
    // TensorAccessor lvalue, which is the standard hazard of a one-argument template
    // constructor: the template would deduce an exact match where the copy constructor
    // needs a qualification conversion.
    template <typename Token, typename = std::enable_if_t<!std::is_same<std::decay_t<Token>, TensorAccessor>::value>>
    constexpr explicit TensorAccessor(Token) {}

    // Present so a custom load/store routine compiles here; see below.
    std::uint64_t get_noc_addr(uint32_t, uint32_t = 0, uint8_t = 0) const {
        ASSERT(false);
        return 0;
    }
};

// The data-movement intrinsics, as unreachable no-ops.
//
// A custom routine's body is compiled on EVERY projection: the harness only
// *calls* it from inside a `#if IS_DM_THREAD` region, but the closure lives in
// the shared kernel source, and C++ compiles a non-generic lambda's body where
// it is written. So
//
//     noc_load<1>(storage, [&](uint32_t l1, uint32_t bytes) {
//         noc_async_read(acc.get_noc_addr(p), l1, bytes);   // <-- compiled on TRISC too
//     });
//
// fails to build on compute unless these names resolve. (A generic `[](auto l1,
// ...)` lambda happens to survive, because a dependent argument defers lookup to
// an instantiation that never happens here -- too fragile to rely on.)
//
// They assert rather than sit empty: nothing on a compute thread has any
// business touching the NOC, and the dead bodies strip out of the TRISC binary.
inline void noc_async_read(std::uint64_t, uint32_t, uint32_t, uint8_t = 0) { ASSERT(false); }
inline void noc_async_write(uint32_t, std::uint64_t, uint32_t, uint8_t = 0) { ASSERT(false); }
// Named by the multicast noc_load, which is itself written as a custom routine --
// so its body reaches here the same way a user's would.
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
// A custom routine targeting a peer's buffer reaches for these: it addresses the
// peer by its own copy of the same circular buffer.
inline uint32_t get_write_ptr(uint32_t) {
    ASSERT(false);
    return 0;
}
inline uint32_t get_read_ptr(uint32_t) {
    ASSERT(false);
    return 0;
}

// The thread's NOC, for a routine that passes it explicitly.
inline constexpr uint8_t noc_index = 0;

inline void noc_async_read_barrier(uint8_t = 0) { ASSERT(false); }
inline void noc_async_write_barrier(uint8_t = 0) { ASSERT(false); }
inline void noc_async_writes_flushed(uint8_t = 0) { ASSERT(false); }
#endif

namespace tt {
namespace unified {

// The CB's *configured* page size, not the data format's tile size --
// get_tile_size() is derived from unpack_tile_size[] and only coincides with the
// page size when a page happens to hold exactly one tile.
//
// fifo_page_size is stored pre-shifted by cb_addr_shift, which is 0 on a
// data-movement build (bytes) and CIRCULAR_BUFFER_COMPUTE_ADDR_SHIFT on a TRISC
// (16B words). The shift is written out so this stays right if it ever moves.
//
// Defined on every projection, unlike the NOC intrinsics above: both names it
// needs resolve on a TRISC too (api/compute/cb_api.h uses get_local_cb_interface,
// and cb_addr_shift has a compute variant). A kernel converting a tile count to a
// byte offset needs the answer in code shared by all five threads.
inline uint32_t cb_page_bytes(uint32_t cb) { return get_local_cb_interface(cb).fifo_page_size << cb_addr_shift; }

// How many pages the HOST configured this circular buffer with.
//
// DATA MOVEMENT ONLY, unlike cb_page_bytes above, and the difference is a LINK one rather
// than anything about the value. `cb_interface` has no definition in a TRISC link -- a
// live reference from a compute projection fails with "undefined reference to
// cb_interface" out of the LLK headers. cb_page_bytes gets away with appearing in shared
// code only because its result is invariably dead on compute and LTO deletes the call
// before the linker sees it; a use that compute genuinely evaluates would fail the same
// way. So anything reading this must sit behind a data-movement guard.
//
// The value is the same fact on every projection regardless, since the host configures
// one circular buffer for the core, which is what makes checking it on one thread enough.
inline uint32_t cb_num_pages(uint32_t cb) { return get_local_cb_interface(cb).fifo_num_pages; }

}  // namespace unified
}  // namespace tt
