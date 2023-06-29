#pragma once

#include <memory>
#include <vector>

namespace tt {

namespace tt_metal {

using HostBufferDataType = uint8_t;
using HostBufferContainer = std::vector<HostBufferDataType>;
using HostBuffer = std::shared_ptr<HostBufferContainer>;

}  // namespace tt_metal

}  // namespace tt
