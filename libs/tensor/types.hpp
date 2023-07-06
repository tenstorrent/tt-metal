#pragma once

#include "tt_metal/impl/buffers/buffer.hpp"
#include "tt_metal/impl/buffers/buffer.hpp"

#include <memory>
#include <variant>
#include <vector>

namespace tt {

namespace tt_metal {

using Shape = std::array<uint32_t, 4>;

struct MemoryConfig {
    bool interleaved = true;    // Interleave the data across multiple DRAM banks
    BufferType buffer_type = BufferType::DRAM; // Can be either DRAM or L1
};

using HostBufferDataType = uint8_t;
using HostBufferContainer = std::vector<HostBufferDataType>;
using HostBuffer = std::shared_ptr<HostBufferContainer>;
struct HostStorage {
    HostBuffer buffer;
};

using DeviceBuffer = std::shared_ptr<Buffer>;
struct DeviceStorage {
    std::shared_ptr<Buffer> buffer;
    Device* device;
    MemoryConfig memory_config;
};

using Storage = std::variant<HostStorage, DeviceStorage>;

}  // namespace tt_metal

}  // namespace tt
