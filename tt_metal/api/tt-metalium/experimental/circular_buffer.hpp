// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <variant>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/kernel_types.hpp>

namespace tt::tt_metal {
class Program;

namespace experimental {

/**
 * @brief Get the address of a circular buffer. Circular buffers need to be allocated before this API is called (throws
 * otherwise).
 *
 * @param program The program which contains the circular buffer
 * @param cb_handle Handle of the circular buffer
 * @return Address of the circular buffer
 */
uint32_t GetCircularBufferAddress(Program& program, CBHandle cb_handle);

}  // namespace experimental
}  // namespace tt::tt_metal
