#pragma once
#include <vector>
#include <array>
#include <random>
#include <tuple>

#include "ll_buda/impl/device/device.hpp"
#include "ll_buda/impl/device/host.hpp"
#include "ll_buda/impl/buffers/interleaved_buffer.hpp"
#include "common/test_tiles.hpp"
#include "common/tt_backend_api_types.hpp"
#include "common/bfloat16.hpp"

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

enum class DataType {
    BFLOAT16 = 0,
    FLOAT32 = 1,
    UINT32 = 2
};

// Forward declarations
class Tensor;

namespace tensor_impl {
    void allocate_interleaved_buffer_on_device(Tensor &tensor, uint32_t buffer_size_bytes);

    template <typename T>
    void initialize_data_helper(Tensor &tensor, Initialize init_type);
}

class Tensor {
    public:
        Tensor(std::vector<bfloat16> data, const std::array<uint32_t, 4> &shape, DataType data_type, Layout layout)
            : data_(static_cast<void *>(new std::vector<bfloat16>(std::move(data)))), shape_(shape), strides_(compute_strides()), data_type_(data_type), layout_(layout) {}

        Tensor(std::vector<bfloat16> data, const std::array<uint32_t, 4> &shape, DataType data_type, Layout layout, Device *device);

        Tensor(std::vector<uint32_t> data, const std::array<uint32_t, 4> &shape, DataType data_type, Layout layout)
            : data_(static_cast<void *>(new std::vector<uint32_t>(std::move(data)))), shape_(shape), strides_(compute_strides()), data_type_(data_type), layout_(layout) {}

        Tensor(std::vector<uint32_t> data, const std::array<uint32_t, 4> &shape, DataType data_type, Layout layout, Device *device);

        Tensor(std::vector<float> data, const std::array<uint32_t, 4> &shape, DataType data_type, Layout layout)
            : data_(static_cast<void *>(new std::vector<float>(std::move(data)))), shape_(shape), strides_(compute_strides()), data_type_(data_type), layout_(layout) {}

        Tensor(std::vector<float> data, const std::array<uint32_t, 4> &shape, DataType data_type, Layout layout, Device *device);

        Tensor(const std::array<uint32_t, 4> &shape, Initialize init_type, DataType data_type, Layout layout);

        Tensor(const std::array<uint32_t, 4> &shape, Initialize init_type, DataType data_type, Layout layout, Device *device);

        Tensor(const std::array<uint32_t, 4> &shape, DataType data_type, Layout layout, Device *device);

        const std::array<uint32_t, 4>& reshape(int N, int C, int H, int W);

        const std::array<uint32_t, 4>& shape() const { return this->shape_; }

        const std::array<uint32_t, 4>& strides() const { return this->strides_; }

        uint32_t volume() const { return shape_[0] * shape_[1] * shape_[2] * shape_[3]; }

        DataType dtype() const { return data_type_; }

        Layout layout() const { return layout_; }

        Device *device() const { return device_; }

        InterleavedDramBuffer *buffer() const { return interleaved_buffer_; }

        void *data_ptr() const { return data_; }

        bool on_host() const { return device_ == nullptr; }

        Tensor to(Device *target_device) const;

        Tensor to(Host *host) const;

        void print(Layout print_layout=Layout::ROW_MAJOR, bool pretty_print=false) const;

        // Prints like numpy print function to make it more readable. Only supports row major layout.
        void pretty_print(Layout print_layout = Layout::ROW_MAJOR) const;

    private:
        const std::array<uint32_t, 4> compute_strides() const {
            return {shape_[1]*shape_[2]*shape_[3], shape_[2]*shape_[3], shape_[3], 1};
        }

        friend void tensor_impl::allocate_interleaved_buffer_on_device(Tensor &tensor, uint32_t buffer_size_bytes);

        template <typename T>
        friend void tensor_impl::initialize_data_helper(Tensor &tensor, Initialize init_type);

        void *data_ = nullptr;                      // Unpopulated if tensor is on device
        std::array<uint32_t, 4> shape_;             // Outer-most dimension first
        std::array<uint32_t, 4> strides_;           // Outer-most dimension first
        DataType data_type_;
        Layout layout_;
        Device *device_ = nullptr;                                  // Set if tensor is allocated on device
        InterleavedDramBuffer *interleaved_buffer_ = nullptr;       // Tensor is stored in multiple DRAM buffers across multiple banks
};

}  // namespace ll_buda

}  // namespace tt
