#pragma once

#include "tt_metal/impl/buffers/buffer.hpp"

#include <memory>
#include <vector>

namespace tt {

namespace tt_metal {

using Shape = std::array<uint32_t, 4>;

using HostBufferDataType = uint8_t;
using HostBufferContainer = std::vector<HostBufferDataType>;
using HostBuffer = std::shared_ptr<HostBufferContainer>;

using DeviceBuffer = std::shared_ptr<Buffer>;

}  // namespace tt_metal

}  // namespace tt
