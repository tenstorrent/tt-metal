// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>
#include "tt_eager/tensor/tensor.hpp"
#include "tt_metal/impl/dispatch/command_queue.hpp"

namespace tt::tt_metal {

namespace operation{
   class DeviceOperation;
}

//Dummy placeholder, definition will be moved to metal-level when it's ready
struct Event {
   uint32_t id;
};

void EnqueueHostToDeviceTransfer(CommandQueue&, Tensor& dst, const void* src, size_t transfer_size);

void EnqueueDeviceToHostTransfer(CommandQueue&, Tensor& src, void* dst, size_t transfer_size, size_t src_offset = 0);


void QueueRecordEvent(CommandQueue&, Event&);
void QueueWaitForEvent(CommandQueue&, Event&);
void EventSynchronize(Event&);
void QueueSynchronize(CommandQueue&);

void EnqueueOperation(CommandQueue&,
                      operation::DeviceOperation&,
                      const std::vector<Tensor>& input_tensors,
                      std::vector<Tensor>& output_tensors,
                      const std::vector<std::optional<const Tensor>>& optional_input_tensors = {},
                      const std::vector<std::optional<Tensor>>& optional_output_tensors = {});
}

// Future APIs to be added later
// void EnqueueAllocateDeviceBuffer(CommandQueue&, DeviceBuffer&); // Part 2
// void EnqueueAllocateHostBuffer(CommandQueue&, HostBuffer&); // Part 2


// void EnqueueDeallocate(CommandQueue&, Tensor&); // Part 2
// void EnqueueReallocate(CommandQueue&, Tensor&); // TBD
// void EnqueueAllocateHostMemory(CommandQueue& , owned_buffer::Buffer& ); // TBD

// Example
// auto Sqrt =
//    tt::tt_metal::EltwiseUnary{{tt::tt_metal::UnaryWithParam{tt::tt_metal::UnaryOpType::SQRT, std::nullopt}}};
// void sqrt(CommandQueue& queue, Tensor& input, Tensor& output) { EnqueueOperation(queue, Sqrt, {input, output}); }
// void sqrt(Tensor& input, Tensor& output) { EnqueueOperation(GetDefaultQueue(), Sqrt, {input, output}); }

// void example() {


//    Tensor host_input_tensor = ...;


//    Queue data_queue = GetDefaultQueue();
//    Queue math_queue = CreateNewQueue();
//    Queue third_queue = CreateNewQueue(); // throw error because only 2 queues are supported
//    size_t num_queues = GetNumQueues();


//    Event data_transfer_event;
//    Event math_event;


//    std::shared_ptr<Buffer> device_input_buffer = create_device_buffer(device, size);
//    Tensor device_input_tensor = Tensor{DeviceStorage{device_input_buffer, ...}, ...};


//    EnqueueAllocateDeviceBuffer(data_queue, device_input_buffer);
//    EnqueueHostToDeviceTransfer(data_queue, host_input_tensor, device_input_tensor);


//    std::shared<Buffer> device_output_buffer{device};
//    EnqueueAllocateDeviceBuffer(data_queue, device_output_buffer);
//    Tensor device_output_tensor = Tensor{DeviceStorage{device_output_buffer, ...}, ...};


//    RecordEvent(data_queue, data_transfer_event);
//    WaitForEvent(math_queue, data_transfer_event);


//    EnqueueOperation(math_queue, Sqrt, {device_input_tensor}, {device_output_tensor});
//    // OR to run on default_queue
//    sqrt(device_input_tensor, device_output_tensor);


//    RecordEvent(math_queue, math_event);


//    owned_buffer::Buffer host_output_buffer;
//    EnqueueAllocateHostMemory(data_queue, host_output_buffer); // ???
//    Tensor host_output_tensor = Tensor{OwnedStorage{host_buffer, ...}, ...};


//    WaitForEvent(data_queue, math_event);
//    EnqueueDeviceToHostTransfer(data_queue, device_output_tensor, host_output_tensor);
// }
