// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Host tensor tests include device side headers and provide their own ASSERT macro overload.
// See: #62337
#ifndef ASSERT
#include "api/debug/assert.h"
#endif

// Device HW DataFormat from tensix_types.h. Declared here so get_dataformat(token) can return it
// without including the arch-specific header. LLKMetadata::kNoFormat is DataFormat::Invalid (0xFF).
enum class DataFormat : uint8_t;

// Constructs within this namespace are reserved for metal 2.0 binding infrastructure and is subject to change without
// notice.
namespace binding_details {

// The LLK operand metadata a Metal 2.0 BindingToken carries, as baked on by headergen.
//
// Kernel authors read this through get_dataformat(token) / get_face_r_dim(token) / ...
// (defined in each token header), or build an LLKOperand with LLKOperandFrom<token>
// (api/llk_operand_from_tokens.h).
//
// This struct is explicitly separated from llk constructs to avoid:
// 1. Coupling of LLK metadata with the Metal 2.0 binding infrastructure.
// 2. Dealing with conditional includes of LLK among different kernels.
struct LLKMetadata {
    // Should be the same as DataFormat::Invalid
    static constexpr uint8_t kNoFormat = 0xFF;

    uint8_t format = kNoFormat;

    // Needed to construct ckernel::TensorShape
    uint8_t face_r_dim = 16;
    uint8_t face_c_dim = 16;
    uint8_t num_faces_r_dim = 2;
    uint8_t num_faces_c_dim = 2;
};

// Deliberately non-constexpr: a missing format fails constant evaluation even when ASSERT is disabled.
// Runtime callers retain the configured device assertion behavior.
inline void data_format_metadata_not_set() { ASSERT(false); }

}  // namespace binding_details
