// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// sst::tensor::FaceShape
// ----------------------------------------------------------------------------
// Type-level tag for the Face layer of the Tensor / Tile / Face stack. A Face
// is the smallest addressable rectangle the Tensix datapath operates on; it is
// never named directly by the kernel author. The Resolver (`resolver.h`)
// computes the per-face HW configuration at compile time.
//
// The shape exposes only `static constexpr` members; there are no data members.
// ----------------------------------------------------------------------------

#include <cstdint>

#include "format_traits.h"

namespace sst::tensor {

template <uint8_t FaceRDim, uint8_t FaceCDim, DataFormat F>
struct FaceShape {
    static constexpr uint8_t face_r_dim = FaceRDim;
    static constexpr uint8_t face_c_dim = FaceCDim;
    static constexpr DataFormat data_format = F;
};

}  // namespace sst::tensor
