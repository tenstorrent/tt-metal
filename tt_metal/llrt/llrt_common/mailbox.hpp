// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dev_msgs.h"

namespace tt::llrt_common {

template <typename ProcessorType>
constexpr uint32_t k_SingleProcessorMailboxSize =
    sizeof(mailboxes_t) - sizeof(profiler_msg_t::buffer) +
    sizeof(profiler_msg_t::buffer) / PROFILER_RISC_COUNT * static_cast<uint32_t>(ProcessorType::COUNT);

}  // namespace tt::llrt_common
