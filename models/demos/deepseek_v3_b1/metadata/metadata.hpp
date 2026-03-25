// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "internal/risc_attribs.h"

namespace deepseek_b1_ops {

struct DeepseekMetadata {
    uint32_t position_id;
    uint32_t slot_id;
};

}  // namespace deepseek_b1_ops
