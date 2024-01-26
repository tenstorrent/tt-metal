#include "tt_eager/queue/queue.hpp"
#include "tt_eager/tt_dnn/op_library/operation.hpp"
#include "tt_metal/impl/dispatch/command_queue.hpp"

namespace tt::tt_metal
{
    void EnqueueHostToDeviceTransfer(Queue&q, Tensor& dst, void* src, size_t transfer_size)
    {
    }

    void EnqueueDeviceToHostTransfer(Queue&q, Tensor& src, void* dst, size_t transfer_size, size_t src_offset)
    {
    }


    void QueueRecordEvent(Queue&q, Event&e){}
    void QueueWaitForEvent(Queue&q, Event&e){}
    void EventSynchronize(Event&e){}
    void QueueSynchronize(Queue&q){}


    void EnqueueOperation(Queue&q, operation::DeviceOperation& devop, const std::vector<Tensor>& input_tensors, const std::vector<Tensor>& output_tensors){}
    void EnqueueOperation(Queue&q, operation::DeviceOperation& devop, const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& optional_input_tensors, const std::vector<Tensor>& output_tensors){}

}
