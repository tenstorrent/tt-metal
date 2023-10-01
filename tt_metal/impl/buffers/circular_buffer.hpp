/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "common/tt_backend_api_types.hpp"
#include "common/core_coord.h"
#include "tt_metal/impl/device/device.hpp"
#include "tt_metal/impl/buffers/buffer_types.hpp"

namespace tt {

namespace tt_metal {

class CircularBuffer {
   public:
    CircularBuffer(const CoreRangeSet &core_range_set, const CircularBufferConfig &config);

    const CircularBufferID id() const { return id_; }

    const CoreRangeSet &core_ranges() const { return core_ranges_; }

    const CircularBufferConfig &config() const { return config_; }

    CircularBufferConfig &config();

    const std::unordered_set<uint32_t> &buffer_indices() const { return buffer_indices_; }

    uint32_t page_size(uint32_t buffer_index) const;

    bool globally_allocated() const;

    uint32_t size() const;

    uint32_t num_pages(uint32_t buffer_index) const;

    DataFormat data_format(uint32_t buffer_index) const;

    uint32_t address() const;

    bool is_on_logical_corerange( const CoreRange & logical_cr ) const;

    bool is_on_logical_core(const CoreCoord &logical_core) const;

    void invalidate_allocation() { this->is_allocated_ = false; }

    void set_address(uint32_t address);

   private:
    bool uses_buffer_index(uint32_t buffer_index) const;

    const uintptr_t id_;
    const CoreRangeSet core_ranges_;
    CircularBufferConfig config_;
    std::unordered_set<uint32_t> buffer_indices_;
    bool is_allocated_;
    uint32_t address_;
};

}  // namespace tt_metal

}  // namespace tt
