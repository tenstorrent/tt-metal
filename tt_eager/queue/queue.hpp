// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>
#include "tt_eager/tensor/tensor.hpp"
#include "tt_metal/impl/dispatch/command_queue.hpp"

namespace tt::tt_metal {

class Event;
namespace operation{
   class DeviceOperation;
}

void EnqueueHostToDeviceTransfer(
    CommandQueue&, Tensor& dst, const void* src, const std::optional<std::size_t> transfer_size = std::nullopt);

void EnqueueDeviceToHostTransfer(
    CommandQueue&,
    Tensor& src,
    void* dst,
    const std::optional<std::size_t> transfer_size = std::nullopt,
    size_t src_offset = 0);

void EnqueueQueueRecordEvent(CommandQueue&, Event&);
void EnqueueQueueWaitForEvent(CommandQueue&, Event&);
void EventSynchronize(Event&);
void QueueSynchronize(CommandQueue&);

std::vector<Tensor> EnqueueOperation(
    CommandQueue&,
    operation::DeviceOperation&,
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors = {},
    const std::vector<std::optional<Tensor>>& optional_output_tensors = {});
}

// Future APIs to be added later
// void EnqueueAllocateDeviceBuffer(CommandQueue&, DeviceBuffer&); // Part 2
// void EnqueueAllocateHostBuffer(CommandQueue&, HostBuffer&); // Part 2


// void EnqueueDeallocate(CommandQueue&, Tensor&); // Part 2
// void EnqueueReallocate(CommandQueue&, Tensor&); // TBD
// void EnqueueAllocateHostMemory(CommandQueue& , owned_buffer::Buffer& ); // TBD
