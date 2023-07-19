#pragma once

#include "tensor/borrowed_buffer.hpp"
#include "tensor/owned_buffer.hpp"

#include "common/bfloat16.hpp"
#include "tt_metal/impl/buffers/buffer.hpp"

#include "tt_stl/reflection.hpp"

#include <memory>
#include <variant>
#include <vector>


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

enum class StorageType {
    OWNED,
    DEVICE,
    BORROWED,  // for storing torch/numpy/etc tensors
};

tt::DataFormat datatype_to_dataformat_converter(DataType datatype);

class Shape {
    std::vector<uint32_t> data;

  public:
    Shape(const std::initializer_list<uint32_t> data);
    Shape(const std::array<uint32_t, 4>& data);
    Shape(const std::vector<uint32_t>& data);

    uint32_t rank() const;

    uint32_t& operator[](const std::size_t index);
    const uint32_t& operator[](const std::size_t index) const;

    uint32_t& back();
    const uint32_t& back() const;

    const uint32_t* begin() const;
    const uint32_t* end() const;
};

bool operator==(const Shape& shape_a, const Shape& shape_b);
bool operator!=(const Shape& shape_a, const Shape& shape_b);
std::ostream& operator<<(std::ostream& os, const Shape& shape);

struct MemoryConfig {
    bool interleaved = true;    // Interleave the data across multiple DRAM banks
    BufferType buffer_type = BufferType::DRAM; // Can be either DRAM or L1
    tt::stl::reflection::Attributes attributes() const;
};

using OwnedBuffer = std::variant<
    owned_buffer::Buffer<uint32_t>,
    owned_buffer::Buffer<float>,
    owned_buffer::Buffer<bfloat16>
>;
struct OwnedStorage {
    OwnedBuffer buffer;
    tt::stl::reflection::Attributes attributes() const;
};

using DeviceBuffer = std::shared_ptr<Buffer>;
struct DeviceStorage {
    DeviceBuffer buffer;
    Device* device;
    MemoryConfig memory_config;
    tt::stl::reflection::Attributes attributes() const;
};

using BorrowedBuffer = std::variant<
    borrowed_buffer::Buffer<uint32_t>,
    borrowed_buffer::Buffer<float>,
    borrowed_buffer::Buffer<bfloat16>
>;
struct BorrowedStorage {
    BorrowedBuffer buffer;
    std::function<void()> on_creation_callback = []{};
    std::function<void()> on_destruction_callback = []{};


    explicit BorrowedStorage(const BorrowedBuffer& buffer, const std::function<void()>& on_creation_callback, const std::function<void()>& on_destruction_callback)
    : buffer(buffer), on_creation_callback(on_creation_callback), on_destruction_callback(on_destruction_callback) {
        this->on_creation_callback();
    }

    BorrowedStorage(const BorrowedStorage& other)
    : buffer(other.buffer), on_creation_callback(other.on_creation_callback), on_destruction_callback(other.on_destruction_callback) {
        this->on_creation_callback();
    }

    BorrowedStorage operator=(const BorrowedStorage& other) {
        this->buffer = other.buffer;
        this->on_creation_callback = other.on_creation_callback;
        this->on_destruction_callback = other.on_destruction_callback;
        this->on_creation_callback();
        return *this;
    }

    BorrowedStorage(BorrowedStorage&& other)
    : buffer(other.buffer), on_creation_callback(other.on_creation_callback), on_destruction_callback(other.on_destruction_callback) {
        other.on_creation_callback = []{};
        other.on_destruction_callback = []{};
    }

    BorrowedStorage operator=(BorrowedStorage&& other) {
        this->buffer = other.buffer;
        this->on_creation_callback = other.on_creation_callback;
        this->on_destruction_callback = other.on_destruction_callback;
        other.on_creation_callback = []{};
        other.on_destruction_callback = []{};
        return *this;
    }

    ~BorrowedStorage() {
        this->on_destruction_callback();
    }

    tt::stl::reflection::Attributes attributes() const;
};

using Storage = std::variant<
    OwnedStorage,
    DeviceStorage,
    BorrowedStorage
>;

namespace detail {
template<typename>
inline constexpr bool unsupported_storage = false;
}

template<typename T>
constexpr void raise_unsupported_storage() {
    static_assert(detail::unsupported_storage<T>, "Unsupported Storage");
}

}  // namespace tt_metal

}  // namespace tt
