// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>

#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/core_coord.hpp>

#include "dev/dataflow_buffer.h"

namespace tt::tt_metal {

class DataflowBufferConfig {
public:
    // Access pattern configuration for a specific buffer index
    struct AccessPattern {
        DataflowBufferAccessPattern write_pattern = DataflowBufferAccessPattern::NONE;
        DataflowBufferAccessPattern read_pattern = DataflowBufferAccessPattern::NONE;
        uint8_t num_writer_threads = 0;
        uint8_t num_reader_threads = 0;
    };

    // Static circular buffer spec
    DataflowBufferConfig(uint32_t total_size, const tt::DataFormat& data_format);

    // User is expected to use the builder here.
    DataflowBufferConfig(uint32_t total_size);

    DataflowBufferConfig& set_page_size(uint32_t page_size);

    DataflowBufferConfig& set_total_size(uint32_t total_size);

    DataflowBufferConfig& set_access_pattern(const AccessPattern& pattern);

    uint8_t buffer_index() const;

    uint32_t total_size() const;

    tt::DataFormat data_format() const;

    uint32_t page_size() const;

    const AccessPattern& access_pattern() const;

    class Builder {
    public:
        static Builder LocalBuilder(DataflowBufferConfig& parent, uint8_t buffer_index);

        const Builder& set_data_format(tt::DataFormat data_format) const;

        const Builder& set_total_size(uint32_t total_size) const;

        const Builder& set_page_size(uint32_t page_size) const;

        // Access pattern methods
        const Builder& set_access_pattern(const AccessPattern& pattern) const;

    private:
        Builder(DataflowBufferConfig& parent, uint8_t buffer_index);

        DataflowBufferConfig& parent_;
        uint8_t buffer_index_;
    };

    Builder index(uint8_t buffer_index);

    friend bool operator==(const DataflowBufferConfig& lhs, const DataflowBufferConfig& rhs);
    friend bool operator!=(const DataflowBufferConfig& lhs, const DataflowBufferConfig& rhs);

private:
    void set_config(const tt::DataFormat& data_format);

    uint8_t buffer_index_ = 0;  // need to separate out remote and local buffer indices
    uint32_t total_size_ = 0;
    tt::DataFormat data_format_;
    uint32_t page_size_;
    uint32_t max_size_ = 0;
    AccessPattern access_patterns_;
};

bool operator==(const DataflowBufferConfig::AccessPattern& lhs, const DataflowBufferConfig::AccessPattern& rhs);
bool operator!=(const DataflowBufferConfig::AccessPattern& lhs, const DataflowBufferConfig::AccessPattern& rhs);
bool operator==(const DataflowBufferConfig& lhs, const DataflowBufferConfig& rhs);
bool operator!=(const DataflowBufferConfig& lhs, const DataflowBufferConfig& rhs);

// in this sim environment, this will set overlay_cluster_dfb_access_pattern_tracker and dfb_to_register_allocation
void CreateDataflowBuffer(
    const DataflowBufferConfig& config, const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec);

}  // namespace tt::tt_metal
