#pragma once

#include <vector>
#include <array>
#include <random>
#include <tuple>

#include "tensor/types.hpp"
#include "tt_metal/impl/device/device.hpp"
#include "tt_metal/impl/device/host.hpp"
#include "tt_metal/impl/buffers/buffer.hpp"
#include "common/test_tiles.hpp"
#include "common/tt_backend_api_types.hpp"
#include "common/bfloat16.hpp"
#include "common/bfloat8.hpp"

namespace tt {

namespace tt_metal {

enum class Layout {
    ROW_MAJOR = 0,
    TILE = 1,
    CHANNELS_LAST = 2
};

enum class DataType {
    BFLOAT16 = 0,
    FLOAT32 = 1,
    UINT32 = 2,
    BFLOAT8_B = 3
};

struct MemoryConfig {
    bool interleaved = true;    // Interleave the data across multiple DRAM banks
    BufferType buffer_type = BufferType::DRAM; // Can be either DRAM or L1
};

class Tensor {

    public:
        // ======================================================================================
        //                                  Hi Level APIs
        // ======================================================================================
        Tensor(const HostBuffer& host_buffer, const Shape& shape, DataType dtype, Layout layout);

        Tensor(const DeviceBuffer& device_buffer, const Shape& shape, DataType dtype, Layout layout, Device *device, const MemoryConfig &mem_config = {.interleaved=true});

        Tensor(const Tensor &other) = default;
        Tensor& operator=(const Tensor &other) = default;

        Tensor(Tensor &&other) = default;
        Tensor& operator=(Tensor &&other) = default;

        ~Tensor();

        void deallocate();

        Tensor to(Device *target_device, const MemoryConfig &mem_config={.interleaved=true}) const;

        Tensor to(Host *host) const;

        Tensor to(Layout target_layout) const;

        Tensor pad(const std::array<uint32_t, 4> &output_tensor_shape, const std::array<uint32_t, 4> &input_tensor_start, float pad_value) const;

        Tensor unpad(const std::array<uint32_t, 4> &output_tensor_start, const std::array<uint32_t, 4> &output_tensor_end) const;

        Tensor pad_to_tile(float pad_value) const;

        Tensor unpad_from_tile(const std::array<uint32_t, 4> &output_tensor_shape) const;

        void print(Layout print_layout=Layout::ROW_MAJOR, bool pretty_print=false) const;

        // Prints like numpy print function to make it more readable. Only supports row major layout.
        void pretty_print() const;

        // ======================================================================================
        //                                  Low Level APIs
        // ======================================================================================
        Tensor reshape(int N, int C, int H, int W);
        Tensor reshape(const Shape& new_shape) const;

        // ======================================================================================
        //                                      Getters
        // ======================================================================================
        const Shape& shape() const { return this->shape_; }

        const std::array<uint32_t, 4> strides() const;

        uint32_t volume() const;

        DataType dtype() const { return dtype_; }

        Layout layout() const { return layout_; }

        Device *device() const { return device_; }

        Buffer *buffer() const { return device_buffer_.get(); }

        const HostBuffer& host_buffer() const { return this->host_buffer_; }

        bool is_allocated_on_host() const { return bool(this->host_buffer_); }
        bool is_allocated_on_device() const { return bool(this->device_buffer_); }

        bool on_host() const { return device_ == nullptr; }

        const MemoryConfig& memory_config() const { return this->memory_config_; };

        // Size in bytes of a single element held in tensor
        uint32_t element_size() const;

    private:

        std::array<uint32_t, 4> shape_;             // Outer-most dimension first
        DataType dtype_;
        Layout layout_;

        // TODO(arakhmati): use std::variant to for storage

        // Host Storage
        HostBuffer host_buffer_{};                 // Unpopulated if tensor is on device

        // Device Storage
        Device *device_ = nullptr;                  // Set if tensor is allocated on device
        std::shared_ptr<Buffer> device_buffer_{};  // Tensors on device are backed by an underlying buffer
        MemoryConfig memory_config_;
};

Tensor create_device_tensor(const Shape& shape, DataType dtype, Layout layout, Device *device, const MemoryConfig& memory_config = {.interleaved=true});

}  // namespace tt_metal

}  // namespace tt
