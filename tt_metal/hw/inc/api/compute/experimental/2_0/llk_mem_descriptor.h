// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "tensor_shape.h"                        // ckernel::TensorShape + geometry helpers
#include "api/compute/common_globals.h"          // DataFormat enum, DST_ACCUM_MODE, ALWI
#include "internal/circular_buffer_interface.h"  // get_local_cb_interface (universal, all TRISCs)

// =====================================================================================================
// Two id-free types:
//   * LLKMemDescriptor -- the compile-time "sticky note" the LLK ops consume as an NTTP (buffer L1 data
//     format + tile geometry). Passed as a non-type template parameter (-ftt-nttp) so the per-format
//     switches / register writes / asserts fold and DCE away.
//   * LLKOperand<Format, Shape> -- the public operand the COMPUTE ops consume. It bundles the descriptor
//     NTTPs (Format/Shape -> ::descriptor) with the single runtime member l1_address (the "where").
//
// The register-side format is NOT carried (derived inside the LLK from `format`); there is NO CB id and NO
// knowledge of the source (CB / DataflowBuffer / Scratchpad / LocalTensorAccessor). Source -> operand goes
// through to_llk_mem_descriptor(accessor) (compile-time) + the address seam (runtime).
// =====================================================================================================

namespace ckernel {
namespace experimental {

// The compile-time descriptor the LLK APIs accept as an NTTP: buffer L1 format + tile geometry.
struct LLKMemDescriptor {
    std::uint8_t format;  // buffer L1 format (what the unpacker reads / the packer writes)
    TensorShape shape;    // tile geometry; derive num_faces / tile dims via TensorShape helpers
};

// The public, id-free operand the compute ops consume. It bundles the two halves of "an L1 tile" split by
// compile-time vs runtime:
//   * Format + Shape are NON-TYPE TEMPLATE PARAMETERS (-ftt-nttp) -- the compile-time "what". They build
//     ::descriptor (an LLKMemDescriptor), forwarded to the LLK as an NTTP so the per-format switches /
//     register writes / asserts fold and DCE away.
//   * l1_address is the ONLY runtime member -- the "where", resolved from the address seam
//     (cb_read_address / cb_write_address). A runtime value cannot be an NTTP, so the split lives INSIDE
//     the type (NTTP vs member).
// Bundling keeps an address welded to its own descriptor (a wrong pairing is unrepresentable), and lets an
// op derive per-tile addresses internally from the compile-time geometry (see tilize_block + SCALE_DATUM_SIZE).
template <DataFormat Format, TensorShape Shape>
struct LLKOperand {
    std::uint32_t l1_address;  // runtime "where"; Format/Shape are the compile-time "what"
    constexpr explicit LLKOperand(std::uint32_t addr) : l1_address(addr) {}

    // The descriptor the LLK APIs accept (buffer L1 format + geometry).
    static constexpr LLKMemDescriptor descriptor = LLKMemDescriptor{static_cast<std::uint8_t>(Format), Shape};
};

// -----------------------------------------------------------------------------------------------------
// Address seam (runtime "where"). Resolves an absolute L1 tile base pointer from a CB, with NO data
// format / geometry and NO side effects: get_operand_id / get_output_id are identity on Blackhole
// (interface index == cb_id), and these are pure reads of the local CB interface (valid on every TRISC),
// so a kernel can build the base pointer on any thread and hand it to an id-free op. Absolute
// (out-of-order) addressing: the op packs/unpacks at exactly this address. In Phase 1/2 these become the
// CB specialization of the get_llk_meminfo(source) translators.
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
// Source -> descriptor translator (compile-time). The compute op only ever consumes an LLKMemDescriptor
// NTTP; the SOURCE (CB / DataflowBuffer / Scratchpad / LocalTensorAccessor) is known ONLY here. Each
// source is a named accessor OBJECT, and each gets one `to_llk_mem_descriptor(accessor)` overload -- so
// adding a source is purely additive; the op and every call site stay unchanged. Folding needs COMPILE-
// TIME source identity, so the accessor carries its identity in its TYPE. `Cb<CbId>` is the compile-time
// CB accessor used for the id-free path (the dataflow CircularBuffer holds a RUNTIME id and cannot fold).
// -----------------------------------------------------------------------------------------------------
template <std::uint32_t CbId>
struct Cb {
    static constexpr std::uint32_t id = CbId;
    ALWI std::uint32_t read_address(std::uint32_t tile_index = 0) const { return cb_read_address(CbId, tile_index); }
    ALWI std::uint32_t write_address(std::uint32_t tile_index = 0) const { return cb_write_address(CbId, tile_index); }
};

// CB source -> LLKMemDescriptor. chlkc format/geometry is indexed by the (compile-time) CB id. The arrays
// are thread-partitioned (unpack_* on UNPACK/MATH, pack_* on PACK), so this reads whichever exist on the
// calling thread; the host equalizes both sides to the same L1 format + geometry per CB, so the result is
// identical on every thread and safe to call anywhere (incl. the kernel body). Folds to a constant.
template <std::uint32_t CbId>
constexpr LLKMemDescriptor to_llk_mem_descriptor(Cb<CbId> /*cb*/) {
#if defined(UCK_CHLKC_PACK)
    return LLKMemDescriptor{
        pack_dst_format[CbId],
        TensorShape{
            pack_tile_face_r_dim[CbId], MAX_FACE_C_DIM, pack_num_faces_r_dim[CbId], pack_num_faces_c_dim[CbId]}};
#else
    return LLKMemDescriptor{
        unpack_src_format[CbId],
        TensorShape{
            unpack_tile_face_r_dim[CbId], MAX_FACE_C_DIM, unpack_num_faces_r_dim[CbId], unpack_num_faces_c_dim[CbId]}};
#endif
}

}  // namespace experimental
}  // namespace ckernel

// The Metal 2.0 BindingToken overloads of to_llk_mem_descriptor. Included last (they need the complete
// LLKMemDescriptor above) so a kernel including only this header still picks them up.
#include "api/compute/experimental/2_0/binding_token_llk.h"
