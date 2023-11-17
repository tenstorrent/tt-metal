// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <array>
#include <random>
#include <tuple>

#include "tensor/types.hpp"
#include "tt_metal/impl/device/device.hpp"
#include "tt_metal/impl/buffers/buffer.hpp"
#include "common/test_tiles.hpp"
#include "common/tt_backend_api_types.hpp"
#include "common/bfloat16.hpp"
#include "common/bfloat8.hpp"

#include "tt_stl/reflection.hpp"

namespace tt {

namespace tt_metal {

class Tensor {

    public:
        // ======================================================================================
        //                                  Hi Level APIs
        // ======================================================================================
        Tensor(const Storage& storage, const Shape& shape, DataType dtype, Layout layout, std::optional<ShardSpec> shard_spec);
        Tensor(const Storage& storage, const Shape& shape, DataType dtype, Layout layout);

        Tensor(const Tensor &other) = default;
        Tensor& operator=(const Tensor &other) = default;

        Tensor(Tensor &&other) = default;
        Tensor& operator=(Tensor &&other) = default;

        ~Tensor();

        void deallocate(bool force=false);

        Tensor to(Device *target_device, const MemoryConfig &mem_config={.memory_layout=tt::tt_metal::TensorMemoryLayout::INTERLEAVED}) const;

        Tensor to(Layout target_layout) const;

        Tensor pad(const Shape &output_tensor_shape, const Shape &input_tensor_start, float pad_value) const;

        Tensor cpu() const;

        Tensor unpad(const Shape &output_tensor_start, const Shape &output_tensor_end) const;

        Tensor pad_to_tile(float pad_value) const;

        Tensor unpad_from_tile(const Shape &output_tensor_shape) const;

        std::string to_string(Layout print_layout=Layout::ROW_MAJOR, bool pretty_print=false) const;
        void print(Layout print_layout=Layout::ROW_MAJOR, bool pretty_print=false) const;

        // ======================================================================================
        //                                  Low Level APIs
        // ======================================================================================
        Tensor reshape(int N, int C, int H, int W) const;
        Tensor reshape(const Shape& new_shape) const;

        // ======================================================================================
        //                                      Getters
        // ======================================================================================
        const Storage& storage() const;
        const Shape& shape() const { return this->shape_; }
        DataType dtype() const { return this->dtype_; }
        Layout layout() const { return this->layout_; }
        const std::optional<ShardSpec>& shard_spec() const { return this->shard_spec_; }

        // ======================================================================================
        //                                      Extra Helper Functions
        // ======================================================================================

        StorageType storage_type() const;
        const Shape strides() const;
        uint32_t volume() const;

        bool is_allocated() const;

        // TODO(arakhmati): clean up the methods below
        Buffer* buffer() const { return std::get<DeviceStorage>(this->storage_).buffer.get(); }
        Device* device() const { return std::get<DeviceStorage>(this->storage_).device; }
        const MemoryConfig memory_config() const { return std::get<DeviceStorage>(this->storage_).memory_config; }

        const bool is_sharded() const { return this->memory_config().is_sharded(); }

        // Size in bytes of a single element held in tensor
        uint32_t element_size() const;

        static constexpr auto attribute_names = std::make_tuple("storage", "shape", "dtype", "layout", "shard_spec");
        const auto attribute_values() const {
            return std::make_tuple(
                std::cref(this->storage_),
                std::cref(this->shape_),
                std::cref(this->dtype_),
                std::cref(this->layout_),
                std::cref(this->shard_spec_));
        }

    private:
        Storage storage_;
        Shape shape_;
        DataType dtype_;
        Layout layout_;
        std::optional<ShardSpec> shard_spec_;

};

Tensor create_device_tensor(const Shape& shape, DataType dtype, Layout layout, Device *device, const MemoryConfig& memory_config = {.memory_layout=tt::tt_metal::TensorMemoryLayout::INTERLEAVED});

Tensor create_sharded_device_tensor(const Shape& shape, DataType data_type, Layout layout, Device *device, const MemoryConfig& memory_config, ShardSpec shard_spec);

}  // namespace tt_metal

}  // namespace tt
