// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

inline constexpr uint8_t kNoLlkFormat = 0xFF;

struct LlkOperandMembers {
    uint8_t format = kNoLlkFormat;
    uint8_t face_r_dim = 16;
    uint8_t face_c_dim = 16;
    uint8_t num_faces_r_dim = 2;
    uint8_t num_faces_c_dim = 2;
};
