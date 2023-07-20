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

using Shape = std::array<uint32_t, 4>;

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
    std::shared_ptr<Buffer> buffer;
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
