// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include <unordered_set>

#include <tt-metalium/core_coord.hpp>

// Intentional header leakage for convience
#include <tt-metalium/circular_buffer_constants.h>

namespace tt::tt_metal {

using CBHandle = uintptr_t;

class CircularBufferImpl;
class CircularBuffer {
public:
    explicit CircularBuffer(const CircularBufferImpl* impl);
    CBHandle id() const;
    const CoreRangeSet& core_ranges() const;
    std::size_t size() const;
    bool globally_allocated() const;
    const std::unordered_set<uint8_t>& buffer_indices() const;

private:
    const CircularBufferImpl* impl_;
};

}  // namespace tt::tt_metal
