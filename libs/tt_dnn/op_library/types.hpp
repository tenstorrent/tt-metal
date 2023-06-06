#pragma once

#include <array>
#include <string>

namespace tt::tt_metal {
using Shape = std::array<uint32_t, 4>;
using ProgramHash = std::string; // TODO(arakhmati): switch to a an integral type?
}
