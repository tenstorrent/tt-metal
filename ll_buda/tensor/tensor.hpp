#pragma once
#include <vector>
#include <array>
#include <random>

#include "ll_buda/impl/device/device.hpp"
#include "ll_buda/impl/device/host.hpp"
#include "ll_buda/impl/buffers/buffer.hpp"
#include "common/test_tiles.hpp"
#include "common/tt_backend_api_types.hpp"
#include "common/bfloat16.hpp"

using SHAPE = std::array<std::uint32_t, 4>;

namespace tt {

namespace ll_buda {

// TODO: this is duplicated
enum class Initialize 
{    
    ZEROS = 0, 
    ONES = 1,
    INCREMENT = 2,
    RANDOM = 3
};

enum class Layout {
    ROW_MAJOR = 0,
    TILE = 1,
    CHANNELS_LAST = 2 // TODO(abhullar): Rename this to STICKS?
};

class Tensor
{
    public:
        Tensor(std::vector<uint32_t> data, const std::array<uint32_t, 4> &shape, DataFormat data_type, Layout layout);

        Tensor(std::vector<uint32_t> data, const std::array<uint32_t, 4> &shape, DataFormat data_type, Layout layout, Device *device);

        Tensor(std::vector<float> &data, const std::array<uint32_t, 4> &shape, DataFormat data_type, Layout layout);

        Tensor(std::vector<float> &data, const std::array<uint32_t, 4> &shape, DataFormat data_type, Layout layout, Device *device);

        Tensor(const std::array<uint32_t, 4> &shape, Initialize init_type, DataFormat data_type, Layout layout);

        Tensor(const std::array<uint32_t, 4> &shape, Initialize init_type, DataFormat data_type, Layout layout, Device *device);

        Tensor(const std::array<uint32_t, 4> &shape, DataFormat data_type, Layout layout, Device *device);

        const std::vector<bfloat16>& to_vec() const { return this->data_; }

        std::vector<float> data() const {
            std::vector<float> d;
            for (bfloat16 v: this->to_vec()) {
                d.push_back(v.to_float());
            }
            return d;
        }

        const std::array<uint32_t, 4>& reshape(int N, int C, int H, int W);

        const std::array<uint32_t, 4>& shape() const { return this->shape_; }

        const std::array<uint32_t, 4>& strides() const { return this->strides_; }

        uint32_t volume() const { return shape_[0] * shape_[1] * shape_[2] * shape_[3]; }

        DataFormat dtype() const { return data_type_; }
    
        Layout layout() const { return layout_; }

        Device *device() const { return device_; }

        // Returns the first buffer that holds the tensor data
        DramBuffer *buffer() const { return buffers_.at(0); }

        bool on_host() const { return device_ == nullptr; }

        Tensor to(Device *target_device) const;

        Tensor to(Host *host) const;

        void print(Layout print_layout = Layout::ROW_MAJOR) const;

        // Prints like numpy print function to make it more readable. Only supports row major layout.
        void pretty_print(Layout print_layout = Layout::ROW_MAJOR) const;

    private:
        const std::array<uint32_t, 4> compute_strides() const {
            return {shape_[1]*shape_[2]*shape_[3], shape_[2]*shape_[3], shape_[3], 1};
        }

        void allocate_buffers_on_device(uint32_t buffer_size_bytes);

        Tensor copy_to_host() const;

        Tensor copy_to_device(Device *device) const;

        std::vector<bfloat16> initialize_data(const std::array<uint32_t, 4> &shape, Initialize init_type, Layout layout);

        template <class T>
        std::vector<T> convert_layout_row_major_to_tile(const std::vector<T>& data_to_convert) const {
            std::vector<uint32_t> shape_vec = {shape_[0], shape_[1], shape_[2], shape_[3]};
            return convert_layout(data_to_convert, shape_vec, TensorLayout::LIN_ROW_MAJOR, TensorLayout::TILED32_4FACES);
        }

        template <class T>
        std::vector<T> convert_layout_tile_to_row_major(const std::vector<T>& data_to_convert) const {
            std::vector<uint32_t> shape_vec = {shape_[0], shape_[1], shape_[2], shape_[3]};
            return convert_layout(data_to_convert, shape_vec, TensorLayout::TILED32_4FACES, TensorLayout::LIN_ROW_MAJOR);
        }

        std::vector<bfloat16> data_;                                // Unpopulated if tensor is on device
        std::array<uint32_t, 4> shape_;             // Outer-most dimension first
        std::array<uint32_t, 4> strides_;           // Outer-most dimension first
        DataFormat data_type_;
        Layout layout_;
        Device *device_ = nullptr;             // Set if tensor is allocated on device
        // Tensor can be stored in multiple DRAM buffers across multiple banks
        // Order corresponds to order in which tensor was split up and written
        std::vector<DramBuffer *> buffers_;
};

}  // namespace ll_buda

}  // namespace tt
