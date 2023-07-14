#pragma once

#include "tensor/host_buffer.hpp"

#include "tt_metal/impl/buffers/buffer.hpp"
#include "common/bfloat16.hpp"
#include "third_party/magic_enum/magic_enum.hpp"

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
    HOST = 0,
    DEVICE = 1,
};

tt::DataFormat datatype_to_dataformat_converter(DataType datatype);

using Shape = std::array<uint32_t, 4>;

struct MemoryConfig {
    bool interleaved = true;    // Interleave the data across multiple DRAM banks
    BufferType buffer_type = BufferType::DRAM; // Can be either DRAM or L1
    tt::stl::reflection::Attributes attributes() const;
};

using HostBuffer = std::variant<
    host_buffer::HostBufferForDataType<uint32_t>,
    host_buffer::HostBufferForDataType<float>,
    host_buffer::HostBufferForDataType<bfloat16>
>;
struct HostStorage {
    HostBuffer buffer;
    tt::stl::reflection::Attributes attributes() const;
};

using DeviceBuffer = std::shared_ptr<Buffer>;
struct DeviceStorage {
    std::shared_ptr<Buffer> buffer;
    Device* device;
    MemoryConfig memory_config;
    tt::stl::reflection::Attributes attributes() const;
};

using Storage = std::variant<
    HostStorage,
    DeviceStorage
>;

}  // namespace tt_metal

}  // namespace tt
