void EnqueueAllocateDeviceBuffer(Queue&, DeviceBuffer&);
void EnqueueAllocateHostBuffer(Queue&, HostBuffer&);


void EnqueueDeallocate(Queue&, Tensor&);
void EnqueueReallocate(Queue&, Tensor&);


void EnqueueHostToDeviceTransfer(Queue&, Tensor& src, Tensor& dst);
void EnqueueDeviceToHostTransfer(Queue&, Tensor& src, Tensor& dst);


void QueueRecordEvent(Queue&, Event&);
void QueueWaitForEvent(Queue&, Event&);
void EventSynchronize(Event&);


void EnqueueOperation(Queue&, DeviceOperation&, const std::vector<Tensor>& input_tensors, const std::vector<Tensor>& output_tensors);
void EnqueueOperation(Queue&, DeviceOperation&, const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& optional_input_tensors, const std::vector<Tensor>& output_tensors);





// Example
auto Sqrt =
   tt::tt_metal::EltwiseUnary{{tt::tt_metal::UnaryWithParam{tt::tt_metal::UnaryOpType::SQRT, std::nullopt}}};
void sqrt(Queue& queue, Tensor& input, Tensor& output) { EnqueueOperation(queue, Sqrt, {input, output}); }
void sqrt(Tensor& input, Tensor& output) { EnqueueOperation(GetDefaultQueue(), Sqrt, {input, output}); }

void example() {


   Tensor host_input_tensor = ...;


   Queue data_queue = GetDefaultQueue();
   Queue math_queue = CreateNewQueue();
   Queue third_queue = CreateNewQueue(); // throw error because only 2 queues are supported
   size_t num_queues = GetNumQueues();


   Event data_transfer_event;
   Event math_event;


   std::shared_ptr<Buffer> device_input_buffer = create_device_buffer(device, size);
   Tensor device_input_tensor = Tensor{DeviceStorage{device_input_buffer, ...}, ...};


   EnqueueAllocateDeviceBuffer(data_queue, device_input_buffer);
   EnqueueHostToDeviceTransfer(data_queue, host_input_tensor, device_input_tensor);


   std::shared<Buffer> device_output_buffer{device};
   EnqueueAllocateDeviceBuffer(data_queue, device_output_buffer);
   Tensor device_output_tensor = Tensor{DeviceStorage{device_output_buffer, ...}, ...};


   RecordEvent(data_queue, data_transfer_event);
   WaitForEvent(math_queue, data_transfer_event);


   EnqueueOperation(math_queue, Sqrt, {device_input_tensor}, {device_output_tensor});
   // OR to run on default_queue
   sqrt(device_input_tensor, device_output_tensor);


   RecordEvent(math_queue, math_event);


   owned_buffer::Buffer host_output_buffer;
   EnqueueAllocateHostMemory(data_queue, host_output_buffer);
   Tensor host_output_tensor = Tensor{OwnedStorage{host_buffer, ...}, ...};


   WaitForEvent(data_queue, math_event);
   EnqueueDeviceToHostTransfer(data_queue, device_output_tensor, host_output_tensor);
}
