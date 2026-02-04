// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include <unordered_set>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>

// Intentional header leakage for convenience
#include <tt-metalium/circular_buffer_constants.h>

namespace tt::tt_metal {
struct Tile;
}  // namespace tt::tt_metal

namespace tt::tt_metal {

using CBHandle = uintptr_t;

class CircularBufferImpl;
// Note: this class shares the lifetime of it's associated program.
class CircularBuffer {
public:
    explicit CircularBuffer(const CircularBufferImpl* impl);
    CBHandle id() const;
    const CoreRangeSet& core_ranges() const;
    std::size_t size() const;
    bool globally_allocated() const;
    const std::unordered_set<uint8_t>& buffer_indices() const;

    uint32_t page_size(uint32_t buffer_index) const;
    uint32_t num_pages(uint32_t buffer_index) const;
    DataFormat data_format(uint32_t buffer_index) const;

private:
    const CircularBufferImpl* impl_;
};

}  // namespace tt::tt_metal
