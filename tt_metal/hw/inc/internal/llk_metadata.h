// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Constructs within this namespace are reserved for metal 2.0 binding infrastructure and is subject to change without
// notice.
namespace binding_details {

// The LLK operand metadata a Metal 2.0 BindingToken carries, as baked on by headergen.
//
// This should never be used directly, kernel authors should interact with these metadata via LLKOperandFrom.
//
// This struct is explicitly separated from llk constructs to avoid:
// 1. Coupling of LLK metadata with the Metal 2.0 binding infrastructure.
// 2. Dealing with conditional includes of LLK among different kernels.
struct LLKMetadata {
    static constexpr uint8_t kNoFormat = 0xFF;

    uint8_t format = kNoFormat;

    // Needed to construct ckernel::TensorShape
    uint8_t face_r_dim = 16;
    uint8_t face_c_dim = 16;
    uint8_t num_faces_r_dim = 2;
    uint8_t num_faces_c_dim = 2;
};

}  // namespace binding_details
