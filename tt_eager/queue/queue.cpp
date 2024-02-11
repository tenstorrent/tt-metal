
// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_eager/queue/queue.hpp"
#include "tt_eager/tt_dnn/op_library/operation.hpp"

namespace tt::tt_metal
{
    void EnqueueHostToDeviceTransfer(CommandQueue& q, Tensor& dst, const void* src, size_t transfer_size)
    {
        dst.to(q, MemoryConfig{});
    }

    void EnqueueHostToDeviceTransfer(CommandQueue& q, Tensor& dst, const void* src, size_t transfer_size, const MemoryConfig& mem_config)
    {
        dst.to(q, mem_config);
    }

    void EnqueueDeviceToHostTransfer(CommandQueue& q, Tensor& src, void* dst, size_t transfer_size, size_t src_offset)
    {
    }

   void QueueSynchronize(CommandQueue& q){
        Finish(q);
    }

    void QueueRecordEvent(CommandQueue& q, Event&e){}
    void QueueWaitForEvent(CommandQueue& q, Event&e){}
    void EventSynchronize(Event&e){}

    void EnqueueOperation(CommandQueue& q, operation::DeviceOperation& devop, const std::vector<Tensor>& input_tensors, const std::vector<Tensor>& output_tensors){}
    void EnqueueOperation(CommandQueue& q, operation::DeviceOperation& devop, const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& optional_input_tensors, const std::vector<Tensor>& output_tensors){}

}
