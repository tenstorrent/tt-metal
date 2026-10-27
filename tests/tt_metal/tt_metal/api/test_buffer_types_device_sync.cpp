// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The device header api/buffer_types.hpp mirrors a subset of tt-metalium/buffer_types.hpp. Enumerator values
// travel from host to device as compile-time args, so the two must agree. Compile-time check only.

#include <cstdint>

#include <tt-metalium/buffer_types.hpp>

// Wrapped so the device enums do not collide with the host ones. <cstdint> is already included above.
namespace device_copy {
#include "../../../../tt_metal/hw/inc/api/buffer_types.hpp"
}  // namespace device_copy

namespace {

namespace host = tt::tt_metal;
namespace dev = device_copy::tt::tt_metal;

#define CHECK_SAME(Enum, Name) \
    static_assert(static_cast<int>(host::Enum::Name) == static_cast<int>(dev::Enum::Name), #Enum "::" #Name)

CHECK_SAME(TensorMemoryLayout, INTERLEAVED);
CHECK_SAME(TensorMemoryLayout, HEIGHT_SHARDED);
CHECK_SAME(TensorMemoryLayout, WIDTH_SHARDED);
CHECK_SAME(TensorMemoryLayout, BLOCK_SHARDED);
CHECK_SAME(TensorMemoryLayout, ND_SHARDED);
CHECK_SAME(BufferType, DRAM);
CHECK_SAME(BufferType, L1);
CHECK_SAME(BufferType, SYSTEM_MEMORY);
CHECK_SAME(BufferType, L1_SMALL);
CHECK_SAME(BufferType, TRACE);

}  // namespace
