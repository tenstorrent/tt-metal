// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace tt::tt_metal {

using ContextId = int;

// The default context ID is for a silicon cluster
static constexpr ContextId SILICON_CONTEXT_ID = 0;

// Max number of MetalContext instances allowed
static constexpr uint8_t MAX_CONTEXT_COUNT = 16;

}  // namespace tt::tt_metal
