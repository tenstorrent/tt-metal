// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "internal/risc_attribs.h"

namespace deepseek_b1_ops {

struct DeepseekMetadata {
    // Output fields
    uint32_t tok0_id;
    uint32_t tok0_type;
    uint32_t tok0_pos;
    uint32_t tok1_id;
    uint32_t tok1_type;
    uint32_t tok1_pos;
    // Input fields
    uint32_t slot_id;
    uint32_t token_id;
    uint32_t position_id;
};

}  // namespace deepseek_b1_ops
