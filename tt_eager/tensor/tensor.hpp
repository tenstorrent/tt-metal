// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <random>
#include <tuple>
#include <variant>
#include <vector>

#include "common/bfloat16.hpp"
#include "common/bfloat8.hpp"
#include "common/bfloat4.hpp"
#include "common/test_tiles.hpp"
#include "common/tt_backend_api_types.hpp"
#include "tensor/types.hpp"
#include "tt_metal/impl/buffers/buffer.hpp"
#include "tt_metal/impl/device/device.hpp"
#include "tt_metal/impl/device/multi_device.hpp"
#include "tt_metal/tt_stl/reflection.hpp"

namespace tt {

namespace tt_metal {

struct Tensor {
    struct TensorAttributes {
        Storage storage;
        ttnn::Shape shape;
        DataType dtype;
        Layout layout;

        TensorAttributes(const Storage storage, const ttnn::Shape shape, DataType dtype, Layout layout) : storage(storage), shape(shape), dtype(dtype), layout(layout) {}
        TensorAttributes() : shape({0xff, 0xff, 0xff, 0xff}), dtype(DataType::INVALID), layout(Layout::INVALID) {}
        ~TensorAttributes() = default;
    };

    // Shared pointer to all attributes associated with this tensor
    // Can be safely passed between threads when the tensor is copied
    std::shared_ptr<TensorAttributes> tensor_attributes;

    // ======================================================================================
    //                                  Hi Level APIs
    // ======================================================================================
    Tensor(const Storage storage, const ttnn::Shape shape, DataType dtype, Layout layout);
    Tensor(const Storage storage, const Shape shape, DataType dtype, Layout layout);

    // Default constructor to initialize empty tensor
    Tensor() : tensor_attributes(std::make_shared<TensorAttributes>()) {}

    Tensor(const Tensor &other) = default;
    Tensor &operator=(const Tensor &other) = default;

    Tensor(Tensor &&other) = default;
    Tensor &operator=(Tensor &&other) = default;

    ~Tensor();

    void deallocate(bool force = false);

    Tensor to(
        Device *target_device,
        const MemoryConfig &mem_config = {.memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED}) const;

    Tensor to(
        DeviceMesh *device_mesh,
        const MemoryConfig &mem_config = {.memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED}) const;

    Tensor to(
        CommandQueue &queue,
        const MemoryConfig &mem_config = {.memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED}) const;
    Tensor to(Layout target_layout) const;

    Tensor pad(const Shape &output_tensor_shape, const Shape &input_tensor_start, float pad_value) const;

    Tensor cpu(CommandQueue &queue, bool blocking = true) const;
    Tensor cpu(bool blocking = true) const;

    Tensor cpu_sharded() const;

    Tensor unpad(const Shape &output_tensor_start, const Shape &output_tensor_end) const;

    Tensor pad_to_tile(float pad_value) const;

    Tensor unpad_from_tile(const Shape &output_tensor_shape) const;

    const std::string write_to_string() const;
    void print() const;

    Tensor extract_shard(const CoreCoord &core) const;
    Tensor extract_shard(const uint32_t &core_id) const;

    // ======================================================================================
    //                                  Low Level APIs
    // ======================================================================================
    Tensor reshape(int N, int C, int H, int W) const;
    Tensor reshape(const Shape &new_shape) const;

    // ======================================================================================
    //                                      Getters
    // ======================================================================================
    const TensorAttributes& get_attr() const { return *tensor_attributes; }
    const Storage &get_storage() const;
    const Shape &get_legacy_shape() const { return this->get_attr().shape.value(); }
    const ttnn::Shape &get_shape() const { return this->get_attr().shape; }
    const DataType& get_dtype() const { return this->get_attr().dtype; }
    const Layout& get_layout() const { return this->get_attr().layout; }


    // ======================================================================================
    //                                      Setters
    // ======================================================================================
    void set_storage(const Storage& storage) { this->tensor_attributes->storage = storage; }
    void set_shape (const ttnn::Shape& shape) { this->tensor_attributes->shape = shape; }
    void set_dtype(const DataType& dtype) { this->tensor_attributes->dtype = dtype; }
    void set_layout(const Layout& layout) { this->tensor_attributes->layout = layout; }

    // ======================================================================================
    //                                      Extra Helper Functions
    // ======================================================================================

    StorageType storage_type() const;
    const Shape strides() const;
    uint32_t volume() const;

    bool is_allocated() const;

    bool is_contiguous() const {
        if (this->get_layout() == tt::tt_metal::Layout::ROW_MAJOR) {
            return this->get_legacy_shape() == this->get_legacy_shape().without_padding();
        } else {
            return false;
        }
    }

    // TODO(arakhmati): clean up the methods below
    Buffer *buffer() const { return std::get<DeviceStorage>(this->get_storage()).buffer.get(); }
    DeviceBuffer device_buffer() const { return std::get<DeviceStorage>(this->get_storage()).buffer; }

    Device *device() const {
        if (this->storage_type() == tt::tt_metal::StorageType::DEVICE) {
            auto buffer = this->buffer();
            if (buffer == nullptr)
                TT_THROW("Cannot get the device from a tensor without an allocated buffer");
            return buffer->device();
        } else {
            TT_THROW("Cannot get the device from a tensor with host storage");
        }
    }
    const MemoryConfig memory_config() const {
        return std::visit(
            [](const auto &storage) -> MemoryConfig {
                using T = std::decay_t<decltype(storage)>;
                if constexpr (std::is_same_v<T, DeviceStorage>) {
                    return storage.memory_config();
                } else {
                    TT_THROW("MemoryConfig can only be obtained for a tensor with DeviceStorage");
                }
            },
            this->get_storage());
    }
    const std::optional<ShardSpec> shard_spec() const { return this->memory_config().shard_spec; }

    const bool is_sharded() const {
        return this->storage_type() == StorageType::DEVICE ? this->memory_config().is_sharded() : false;
    }

    // Size in bytes of a single element held in tensor
    uint32_t element_size() const;

    static constexpr auto attribute_names = std::make_tuple("storage", "shape", "dtype", "layout");
    const auto attribute_values() const {
        return std::make_tuple(
            std::cref(this->get_storage()), std::cref(this->get_shape()), std::cref(this->get_dtype()), std::cref(this->get_layout()));
    }

    std::vector<uint32_t> host_page_ordering();

    private:
        void log_deallocate(const Buffer &, const bool force);
};

Tensor create_device_tensor(const Shape& shape, DataType dtype, Layout layout, Device *device, const MemoryConfig& memory_config = {.memory_layout=tt::tt_metal::TensorMemoryLayout::INTERLEAVED});

Tensor create_sharded_device_tensor(const Shape& shape, DataType data_type, Layout layout, Device *device, const MemoryConfig& memory_config, bool pad_to_same_shard_size=false);

// template<typename Buffer>
// void *get_host_buffer(const Tensor &tensor);
void *get_raw_host_data_ptr(const Tensor &tensor);

void memcpy(
    CommandQueue &queue, void *dst, const Tensor &src, const std::optional<std::size_t> transfer_size = std::nullopt);
void memcpy(
    CommandQueue &queue, Tensor &dst, const void *src, const std::optional<std::size_t> transfer_size = std::nullopt);
void memcpy(
    CommandQueue &queue, Tensor &dst, const Tensor &src, const std::optional<std::size_t> transfer_size = std::nullopt);

void memcpy(void *dst, const Tensor &src, const std::optional<std::size_t> transfer_size = std::nullopt);
void memcpy(Tensor &dst, const void *src, const std::optional<std::size_t> transfer_size = std::nullopt);
void memcpy(Tensor &dst, const Tensor &src, const std::optional<std::size_t> transfer_size = std::nullopt);

}  // namespace tt_metal

}  // namespace tt

namespace ttnn {

using Tensor = tt::tt_metal::Tensor;

}  // namespace ttnn
