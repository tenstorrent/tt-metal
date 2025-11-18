// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <hostdevcommon/common_values.hpp>
#include <hostdevcommon/kernel_structs.h>  // Not used here, but leaked to programming examples
#include <tt-metalium/data_types.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/sub_device_types.hpp>
#include <tt-metalium/core_coord.hpp>

#include <umd/device/types/cluster_descriptor_types.hpp>
#include <umd/device/types/core_coordinates.hpp>

#include <tt_stl/span.hpp>

namespace tt {

namespace tt_metal {

namespace program_cache::detail {
struct ProgramCache;
}
/*
MemoryBlockTable is a list of memory blocks in the following format:
[{"blockID": "0", "address": "0", "size": "0", "prevID": "0", "nextID": "0", "allocated": true}]
address: bytes
size: bytes
*/
using MemoryBlockTable = std::vector<std::unordered_map<std::string, std::string>>;
enum class BufferType;

class Allocator;
class Buffer;
class Program;
class SubDevice;

class CommandQueue;
class SystemMemoryManager;
struct TraceBuffer;
struct TraceDescriptor;

namespace distributed {
class MeshDevice;
}

}  // namespace tt_metal

}  // namespace tt
