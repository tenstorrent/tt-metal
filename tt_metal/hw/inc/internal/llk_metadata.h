// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Constructs within this namespace are reserved for metal 2.0 binding infrastructure and is subject to change without
// notice.
namespace binding_details {

// The LLK operand metadata a Metal 2.0 BindingToken carries, as baked on by headergen. Kernels never name
// this type: headergen emits it as a nested braced-initializer on the token's constructor, and the token
// exposes it only through to_llk_mem_descriptor().
//
// This separation is explicitly designed to avoid:
// 1. Coupling of LLK metadata with the Metal 2.0 binding infrastructure.
// 2. Conditionally injecting metadata between different types of kernels (some LLK metadata definitions may only be
// available for compute kernels).
//
// Layout mirrors LLKMemDescriptor (format + a 4-byte TensorShape).
struct LLKMetadata {
    static constexpr uint8_t kNoFormat = 0xFF;

    uint8_t format = kNoFormat;
    uint8_t face_r_dim = 16;
    uint8_t face_c_dim = 16;
    uint8_t num_faces_r_dim = 2;
    uint8_t num_faces_c_dim = 2;
};

}  // namespace binding_details
