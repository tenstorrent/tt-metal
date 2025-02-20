// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "buffer.hpp"

namespace tt::tt_metal {

inline namespace v0 {

class SingleBuffer : public Buffer {
    struct Private {
        explicit Private() = default;
    };

public:
    static std::shared_ptr<SingleBuffer> create(
        IDevice* device,
        DeviceAddr size,
        DeviceAddr page_size,
        BufferType buffer_type,
        TensorMemoryLayout buffer_layout = TensorMemoryLayout::INTERLEAVED,
        const std::optional<ShardSpecBuffer>& shard_parameter = std::nullopt,
        std::optional<bool> bottom_up = std::nullopt,
        std::optional<SubDeviceId> sub_device_id = std::nullopt);
    static std::shared_ptr<SingleBuffer> create(
        IDevice* device,
        DeviceAddr address,
        DeviceAddr size,
        DeviceAddr page_size,
        BufferType buffer_type,
        TensorMemoryLayout buffer_layout = TensorMemoryLayout::INTERLEAVED,
        const std::optional<ShardSpecBuffer>& shard_parameter = std::nullopt,
        std::optional<bool> bottom_up = std::nullopt,
        std::optional<SubDeviceId> sub_device_id = std::nullopt);

    SingleBuffer(const SingleBuffer& other) = delete;
    SingleBuffer& operator=(const SingleBuffer& other) = delete;
    SingleBuffer(SingleBuffer&& other) = delete;
    SingleBuffer& operator=(SingleBuffer&& other) = delete;

    IDevice* device() const override { return device_; }
    Allocator* allocator() const override { return allocator_; }
    DeviceAddr size() const override { return size_; }
    bool is_allocated() const override;

    // Returns address of buffer in the first bank
    uint32_t address() const override;

    DeviceAddr page_size() const override;
    void set_page_size(DeviceAddr page_size) override;

    uint32_t num_pages() const override;
    uint32_t num_dev_pages() const override;

    BufferType buffer_type() const override { return buffer_type_; }
    CoreType core_type() const override;

    bool is_l1() const override;
    bool is_dram() const override;
    bool is_trace() const override;

    bool is_valid_region(const BufferRegion& region) const override;
    bool is_valid_partial_region(const BufferRegion& region) const override;

    TensorMemoryLayout buffer_layout() const override { return buffer_layout_; }

    bool bottom_up() const override { return bottom_up_; }

    uint32_t dram_channel_from_bank_id(uint32_t bank_id) const override;

    CoreCoord logical_core_from_bank_id(uint32_t bank_id) const override;

    DeviceAddr page_address(uint32_t bank_id, uint32_t page_index) const override;

    DeviceAddr bank_local_page_address(uint32_t bank_id, uint32_t page_index) const override;
    uint32_t alignment() const override;
    DeviceAddr aligned_page_size() const override;
    DeviceAddr aligned_size() const override;
    DeviceAddr aligned_size_per_bank() const override;

    // SHARDED API STARTS HERE
    // TODO: WILL SEPARATE INTO SHARDED BUFFER CLASS

    DeviceAddr sharded_page_address(uint32_t bank_id, uint32_t page_index) const override;

    ShardSpecBuffer shard_spec() const override;
    void set_shard_spec(const ShardSpecBuffer& shard_spec) override;

    std::optional<uint32_t> num_cores() const override;

    const std::shared_ptr<const BufferPageMapping>& get_buffer_page_mapping() override;

    std::optional<SubDeviceId> sub_device_id() const override { return sub_device_id_; }
    std::optional<SubDeviceManagerId> sub_device_manager_id() const override { return sub_device_manager_id_; }

    size_t unique_id() const override { return unique_id_; }

    // Deallocate is allowed to be called multiple times on the same buffer
    void deallocate() override;

    SingleBuffer(
        IDevice* device,
        DeviceAddr size,
        DeviceAddr page_size,
        BufferType buffer_type,
        TensorMemoryLayout buffer_layout,
        const std::optional<ShardSpecBuffer>& shard_parameter,
        std::optional<bool> bottom_up,
        std::optional<SubDeviceId> sub_device_id,
        bool owns_data,
        Private);

private:
    enum class AllocationStatus : uint8_t {
        ALLOCATION_REQUESTED,
        ALLOCATION_FAILED,
        ALLOCATED,
        DEALLOCATED,
    };

    void allocate_impl();

    static void deleter(SingleBuffer* buffer);
    void deallocate_impl();
    friend void DeallocateBuffer(SingleBuffer& buffer);

    DeviceAddr translate_page_address(uint64_t offset, uint32_t bank_id) const;

    IDevice* const device_;
    const DeviceAddr size_;  // Size in bytes
    const BufferType buffer_type_;
    const TensorMemoryLayout buffer_layout_;
    const bool bottom_up_;
    const std::optional<SubDeviceId> sub_device_id_;
    const bool owns_data_;

    std::optional<SubDeviceManagerId> sub_device_manager_id_;
    Allocator* allocator_;

    std::atomic<AllocationStatus> allocation_status_ = AllocationStatus::ALLOCATION_REQUESTED;
    DeviceAddr address_ = 0;
    mutable std::mutex allocation_mutex_;
    mutable std::condition_variable allocation_cv_;
    // Used exclusively for is_allocated() method
    std::atomic<bool> deallocation_requested_ = false;

    // These members must be only accessed on the device worker thread
    DeviceAddr page_size_;  // Size of unit being interleaved. For non-interleaved buffers: size == page_size
    std::optional<ShardSpecBuffer> shard_parameters_;
    std::shared_ptr<const BufferPageMapping> buffer_page_mapping_;

    std::weak_ptr<SingleBuffer> weak_self;
    size_t unique_id_ = 0;
    static std::atomic<size_t> next_unique_id;
};

}  // namespace v0
}  // namespace tt::tt_metal
