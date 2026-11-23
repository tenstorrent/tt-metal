// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"

// HAL gap: the Blackhole HAL has no thread mailboxes. These wrap ckernel's.
namespace llk2::platform::mailbox
{

/** @brief Select the TRISC a mailbox word goes to or comes from. */
enum class Thread : std::uint8_t
{
    Unpack = ckernel::ThreadId::UnpackThreadId,
    Math   = ckernel::ThreadId::MathThreadId,
    Pack   = ckernel::ThreadId::PackThreadId
};

/**
 * @brief Send one word to another thread's mailbox with an MMIO store.
 * @note Each ordered pair of threads has its own FIFO. Nothing here checks that it has room:
 * keep the words in flight on one pair within the FIFO's depth.
 */
inline __attribute__((always_inline)) void write(const Thread to, const std::uint32_t value)
{
    ckernel::mailbox_write(static_cast<std::uint8_t>(to), value);
}

/** @brief Take the oldest word another thread sent this one; blocks until one has arrived. */
[[nodiscard]] inline __attribute__((always_inline)) std::uint32_t read(const Thread from)
{
    return ckernel::mailbox_read(static_cast<std::uint8_t>(from));
}

} // namespace llk2::platform::mailbox
