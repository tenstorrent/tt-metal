// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/compute/common_globals.h"                            // chlkc format/geometry arrays, MAX_FACE_C_DIM, ALWI
#include "api/compute/experimental/2_0/internal/llk_descriptor.h"  // LLKMemDescriptor
#include "internal/circular_buffer_interface.h"                    // get_local_cb_interface (universal, all TRISCs)

// =====================================================================================================
// TEST-COMMON (device-side) CB helpers. These are the CB-specific "source -> id-free operand" glue used by
// the 2.0 test kernels; they are deliberately NOT part of the public compute API (llk_operand.h). A shipping
// kernel resolves an operand from its real source; the unit tests resolve it from a classic CircularBuffer.
//
//   * cb_read_address / cb_write_address -- runtime "where": absolute L1 tile base pointer from a CB.
//   * Cb<CbId>                            -- compile-time CB accessor carrying its id in its TYPE (foldable).
//   * to_llk_mem_descriptor(Cb<CbId>)     -- compile-time "what": LLKMemDescriptor from the chlkc arrays.
// =====================================================================================================

namespace ckernel {
namespace experimental {

// -----------------------------------------------------------------------------------------------------
// Address seam (runtime "where"). Resolves an absolute L1 tile base pointer from a CB, with NO data format /
// geometry and NO side effects: get_operand_id / get_output_id are identity on Blackhole (interface index ==
// cb_id), and these are pure reads of the local CB interface (valid on every TRISC), so a kernel can build the
// base pointer on any thread and hand it to an id-free op. Absolute (out-of-order) addressing: the op
// packs/unpacks at exactly this address.
// -----------------------------------------------------------------------------------------------------
ALWI std::uint32_t cb_read_address(std::uint32_t cb_id, std::uint32_t tile_index = 0) {
    const auto& cb = get_local_cb_interface(cb_id);
    return cb.fifo_rd_ptr - 1 + cb.fifo_page_size * tile_index;
}

ALWI std::uint32_t cb_write_address(std::uint32_t cb_id, std::uint32_t out_tile_index = 0) {
    const auto& cb = get_local_cb_interface(cb_id);
    return cb.fifo_wr_ptr - 1 + cb.fifo_page_size * out_tile_index;
}

// -----------------------------------------------------------------------------------------------------
// Source -> descriptor translator (compile-time). The compute op only ever consumes an LLKMemDescriptor NTTP;
// the SOURCE is known ONLY here. Folding needs COMPILE-TIME source identity, so the accessor carries its
// identity in its TYPE. `Cb<CbId>` is the compile-time CB accessor used for the id-free path.
// -----------------------------------------------------------------------------------------------------
template <std::uint32_t CbId>
struct Cb {
    static constexpr std::uint32_t id = CbId;
    ALWI std::uint32_t read_address(std::uint32_t tile_index = 0) const { return cb_read_address(CbId, tile_index); }
    ALWI std::uint32_t write_address(std::uint32_t tile_index = 0) const { return cb_write_address(CbId, tile_index); }
};

// CB source -> LLKMemDescriptor. chlkc format/geometry is indexed by the (compile-time) CB id. The arrays are
// thread-partitioned (unpack_* on UNPACK/MATH, pack_* on PACK), so this reads whichever exist on the calling
// thread; the host equalizes both sides to the same L1 format + geometry per CB, so the result is identical on
// every thread and safe to call anywhere (incl. the kernel body). Folds to a constant.
template <std::uint32_t CbId>
constexpr LLKMemDescriptor to_llk_mem_descriptor(Cb<CbId> /*cb*/) {
#if defined(UCK_CHLKC_PACK)
    return LLKMemDescriptor{
        static_cast<DataFormat>(pack_dst_format[CbId]),
        TensorShape{
            pack_tile_face_r_dim[CbId], MAX_FACE_C_DIM, pack_num_faces_r_dim[CbId], pack_num_faces_c_dim[CbId]}};
#else
    return LLKMemDescriptor{
        static_cast<DataFormat>(unpack_src_format[CbId]),
        TensorShape{
            unpack_tile_face_r_dim[CbId], MAX_FACE_C_DIM, unpack_num_faces_r_dim[CbId], unpack_num_faces_c_dim[CbId]}};
#endif
}

}  // namespace experimental
}  // namespace ckernel
