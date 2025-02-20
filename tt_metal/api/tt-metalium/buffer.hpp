// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <tuple>
#include <unordered_map>
#include <variant>
#include <vector>

#include "buffer_config.hpp"
#include "sub_device_types.hpp"
#include "umd/device/tt_soc_descriptor.h"
#include "umd/device/types/xy_pair.h"
#include "concepts.hpp"
#include "assert.hpp"

namespace tt::tt_metal {
inline namespace v0 {

class Buffer {
public:
    Buffer() = default;
    virtual ~Buffer() = default;
    virtual IDevice* device() const = 0;
    virtual Allocator* allocator() const = 0;
    virtual DeviceAddr size() const = 0;
    virtual bool is_allocated() const = 0;

    // Returns address of buffer in the first bank
    virtual uint32_t address() const = 0;

    virtual DeviceAddr page_size() const = 0;
    virtual void set_page_size(DeviceAddr page_size) = 0;

    virtual uint32_t num_pages() const = 0;
    virtual uint32_t num_dev_pages() const = 0;

    virtual BufferType buffer_type() const = 0;
    virtual CoreType core_type() const = 0;

    virtual bool is_l1() const = 0;
    virtual bool is_dram() const = 0;
    virtual bool is_trace() const = 0;

    virtual bool is_valid_region(const BufferRegion& region) const = 0;
    virtual bool is_valid_partial_region(const BufferRegion& region) const = 0;

    virtual TensorMemoryLayout buffer_layout() const = 0;

    virtual bool bottom_up() const = 0;

    virtual uint32_t dram_channel_from_bank_id(uint32_t bank_id) const = 0;

    virtual CoreCoord logical_core_from_bank_id(uint32_t bank_id) const = 0;

    virtual DeviceAddr page_address(uint32_t bank_id, uint32_t page_index) const = 0;

    virtual DeviceAddr bank_local_page_address(uint32_t bank_id, uint32_t page_index) const = 0;
    virtual uint32_t alignment() const = 0;
    virtual DeviceAddr aligned_page_size() const = 0;
    virtual DeviceAddr aligned_size() const = 0;
    virtual DeviceAddr aligned_size_per_bank() const = 0;

    // SHARDED API STARTS HERE
    // TODO: WILL SEPARATE INTO SHARDED BUFFER CLASS

    virtual DeviceAddr sharded_page_address(uint32_t bank_id, uint32_t page_index) const = 0;

    virtual ShardSpecBuffer shard_spec() const = 0;
    virtual void set_shard_spec(const ShardSpecBuffer& shard_spec) = 0;

    virtual std::optional<uint32_t> num_cores() const = 0;

    virtual const std::shared_ptr<const BufferPageMapping>& get_buffer_page_mapping() = 0;

    virtual std::optional<SubDeviceId> sub_device_id() const = 0;
    virtual std::optional<SubDeviceManagerId> sub_device_manager_id() const = 0;

    virtual size_t unique_id() const = 0;

    virtual void deallocate() = 0;
};
}  // namespace v0
}  // namespace tt::tt_metal
