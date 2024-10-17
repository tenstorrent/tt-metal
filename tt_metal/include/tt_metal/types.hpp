// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device/tt_cluster_descriptor_types.h"
#include "hostdevcommon/common_values.hpp"
#include "tt_metal/impl/dispatch/dispatch_core_manager.hpp"

namespace tt::tt_metal{

namespace v1 {

// Opaque classes
class Program;
class Device;
class CommandQueue;
class Trace;

// Ideally these would be opaque but this work requires
// completion of the prototype of the runtime args.
class CircularBuffer;
class Buffer;

// Not likely going to be opaque, but pending review of
// completion of the prototype of the runtime args.
class Event;
class RuntimeArgs;
class RuntimeArgsData;

}



}
