// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// TODO: This will no longer be a constant, swap this out with a getter ASAP.
// TODO: This should be in the tt::tt_metal namespace.
// TODO: This is used by both host and device.
constexpr static std::uint32_t NUM_CIRCULAR_BUFFERS = 32;
