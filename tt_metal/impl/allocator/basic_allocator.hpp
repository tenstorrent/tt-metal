#pragma once

#include <cstdint>
#include <vector>
#include <unordered_map>
#include <memory>

#include "tt_metal/impl/allocator/allocator.hpp"

namespace tt {

namespace tt_metal {

struct BasicAllocator : Allocator {
    BasicAllocator(const tt_SocDescriptor &soc_desc);
};

}  // namespace tt_metal

}  // namespace tt
