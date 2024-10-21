// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_metal/impl/dispatch/dispatch_core_manager.hpp"

namespace tt::tt_metal {

namespace v1 {

// Opaque classes
class CommandQueueHandle;
class DeviceHandle;
class KernelHandle;
class ProgramHandle;
class TraceHandle;

// Ideally these would be opaque but this work requires
// completion of the prototype of the runtime args.
class CircularBuffer;
class Buffer;

// Not likely going to be opaque, but pending review of
// completion of the prototype of the runtime args.
class Event;
class RuntimeArgs;
class RuntimeArgsData;

struct DeviceConfig {
    // L1 small space to reserve (default: DEFAULT_L1_SMALL_SIZE).
    std::size_t l1_small_size = DEFAULT_L1_SMALL_SIZE;
    // Trace region size to reserve (default: DEFAULT_TRACE_REGION_SIZE).
    std::size_t trace_region_size = DEFAULT_TRACE_REGION_SIZE;
    // Dispatch core type to use (default: DispatchCoreType::WORKER).
    DispatchCoreType dispatch_core_type = DispatchCoreType::WORKER;
    // Number of hardware command queues (default: 1, valid range: 1 to 2).
    std::uint8_t num_hw_cqs = 1;
};

enum class AllocationOrder : bool {
    TopDown,
    BottomUp,
};

enum class ShardOrder : bool {
    Mapped,
    Unmapped,
};

enum class CommandQueueMode : bool {
    Immediate,
    Lazy,
};

}  // namespace v1

}  // namespace tt::tt_metal
