# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from typing import Callable

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


def create_input_tensors(input_shape, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG, device=None):
    torch_input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)
    ttnn_input_tensor = ttnn.from_torch(
        torch_input_tensor, dtype=ttnn.bfloat16, layout=layout, memory_config=memory_config, device=device
    )
    return ttnn_input_tensor, torch_input_tensor


def create_test_model(input_shape):
    def run(l1_input_tensor):
        assert l1_input_tensor.storage_type() == ttnn.StorageType.DEVICE, "Model expects input tensor to be on device"
        assert (
            l1_input_tensor.memory_config().buffer_type == ttnn.BufferType.L1
        ), "Model expects input tensor to be in L1"
        assert input_shape == l1_input_tensor.shape, "Unexpected input shape"
        x = ttnn.tilize(l1_input_tensor)
        x = ttnn.add(x, x)
        x = ttnn.relu(x)
        return x

    def run_reference(torch_input_tensor):
        assert input_shape == torch_input_tensor.shape, "Unexpected input shape"
        return torch.nn.functional.relu(torch_input_tensor + torch_input_tensor)

    return run, run_reference


from abc import ABC, abstractmethod


class Executor(ABC):
    @abstractmethod
    def compile(self, sample_input):
        pass

    @abstractmethod
    def execute(self, input_tensor):
        pass

    @abstractmethod
    def cleanup(self):
        pass


class ModelExecutor(Executor):
    def __init__(self, model: Callable):
        self.model = model

    def compile(self, sample_input):
        return self.model(sample_input)

    def execute(self, input_tensor):
        return self.model(input_tensor)

    def cleanup(self):
        pass


class TracedModelExecutor(Executor):
    def __init__(self, child, device, dram_input_memory_config, l1_input_memory_config, cq_id=0):
        self.child = child
        self.device = device
        self.cq_id = cq_id
        self.dram_input_memory_config = dram_input_memory_config
        self.l1_input_memory_config = l1_input_memory_config

        self.output_tensor = None

    def compile(self, host_input):
        # Compile the model
        self.dram_input_tensor = ttnn.allocate_tensor_on_device(
            host_input.shape, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, self.device, self.dram_input_memory_config
        )
        ttnn.copy_host_to_device_tensor(host_input, self.dram_input_tensor, cq_id=0)
        l1_input_tensor = ttnn.reshard(self.dram_input_tensor, self.l1_input_memory_config)
        output_tensor = self.child.execute(l1_input_tensor)
        ttnn.deallocate(l1_input_tensor)
        ttnn.deallocate(output_tensor)

        # Capture trace
        ttnn.copy_host_to_device_tensor(host_input, self.dram_input_tensor, cq_id=0)
        l1_input_tensor = ttnn.reshard(self.dram_input_tensor, self.l1_input_memory_config)

        input_trace_addr = l1_input_tensor.buffer_address()
        spec = l1_input_tensor.spec
        output_tensor.deallocate(force=True)

        self.trace_id = ttnn.begin_trace_capture(self.device, cq_id=0)
        self.output_tensor = self.child.execute(l1_input_tensor)
        ttnn.deallocate(l1_input_tensor)

        # Try allocating our persistent input tensor here and verifying it matches the address that trace captured
        l1_input_tensor = ttnn.allocate_tensor_on_device(spec, self.device)
        assert input_trace_addr == l1_input_tensor.buffer_address()
        ttnn.end_trace_capture(self.device, self.trace_id, cq_id=0)

        ttnn.synchronize_device(self.device)

    def execute(self, input_tensor):
        ttnn.copy_host_to_device_tensor(input_tensor, self.dram_input_tensor, cq_id=self.cq_id)
        l1_input_tensor = ttnn.reshard(self.dram_input_tensor, self.l1_input_memory_config)
        ttnn.execute_trace(self.device, self.trace_id, cq_id=self.cq_id, blocking=False)
        return self.output_tensor.cpu(blocking=False)

    def cleanup(self):
        ttnn.release_trace(self.device, self.trace_id)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_base_model_executor(device):
    input_shape = (1, 1, 1024, 1024)

    model, reference_model = create_test_model(input_shape)
    input_tensor, reference_input_tensor = create_input_tensors(
        input_shape, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )

    torch_output_tensor = reference_model(reference_input_tensor)

    executor = ModelExecutor(model)
    executor.compile(input_tensor)
    for _ in range(32):
        output_tensor = executor.execute(input_tensor)

    assert_with_pcc(ttnn.to_torch(output_tensor), torch_output_tensor, 0.99999)


@pytest.mark.parametrize(
    "device_params",
    [
        {
            "l1_small_size": 16384,
            "trace_region_size": 32768,
            "num_command_queues": 1,
        }
    ],
    indirect=True,
)
def test_traced_model_executor(device):
    input_shape = (1, 1, 1024, 32)

    model, reference_model = create_test_model(input_shape)
    input_tensor, _ = create_input_tensors(input_shape, device=None, memory_config=None)

    dram_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
        [512, 32],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    dram_memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, dram_shard_spec
    )

    l1_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
        [256, 32],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    l1_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, l1_shard_spec)

    executor = TracedModelExecutor(
        ModelExecutor(model),
        device=device,
        dram_input_memory_config=dram_memory_config,
        l1_input_memory_config=l1_memory_config,
    )
    executor.compile(input_tensor)

    outputs = []
    reference_outputs = []
    for _ in range(8):
        input_tensor, reference_input_tensor = create_input_tensors(input_shape, device=None, memory_config=None)

        torch_output_tensor = reference_model(reference_input_tensor)
        reference_outputs.append(torch_output_tensor)

        output_tensor = executor.execute(input_tensor)
        outputs.append(output_tensor)

    ttnn.synchronize_device(device)
    executor.cleanup()

    assert len(outputs) == len(reference_outputs), "Expected same number of outputs"
    [
        assert_with_pcc(ttnn.to_torch(output_tensor), torch_output_tensor, 0.99999)
        for output_tensor, torch_output_tensor in zip(outputs, reference_outputs)
    ]


@pytest.mark.parametrize(
    "device_params",
    [
        {
            "l1_small_size": 16384,
            "trace_region_size": 32768,
            "num_command_queues": 2,
        }
    ],
    indirect=True,
)
def test_traced_model_executor2(device):
    input_shape = (1, 1, 1024, 32)

    model, reference_model = create_test_model(input_shape)
    host_tensor, reference_input_tensor = create_input_tensors(input_shape, device=None, memory_config=None)

    torch_output_tensor = reference_model(reference_input_tensor)

    dram_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
        [512, 32],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    dram_memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, dram_shard_spec
    )

    l1_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
        [256, 32],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    l1_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, l1_shard_spec)

    output_sharded_dram_mem_config = dram_memory_config

    ############## EXAMPLE BEGINS

    input_dram_tensor = ttnn.allocate_tensor_on_device(
        host_tensor.shape, host_tensor.dtype, host_tensor.layout, device, dram_memory_config
    )

    # Dummy record an op event on CQ 0 since we wait on this first in the loop
    first_op_event = ttnn.record_event(device, 0)
    # Dummy record a read event on CQ 1 since we wait on this first in the loop
    read_event = ttnn.record_event(device, 1)

    # First run to compile the model
    # Stall CQ 1 for the input tensor consumer (CQ 0) to signal it has finished so we can start overwriting the previous input tensor with the new one
    ttnn.wait_for_event(1, first_op_event)
    # Write the next input tensor on CQ 1
    ttnn.copy_host_to_device_tensor(host_tensor, input_dram_tensor, cq_id=1)
    # Signal that the write has finished on CQ 1
    write_event = ttnn.record_event(device, 1)
    # Make CQ 0 stall until CQ 1 has signalled that the write has finished
    ttnn.wait_for_event(0, write_event)
    # Run the first operation of the model on CQ 0
    input_l1_tensor = ttnn.to_memory_config(input_dram_tensor, l1_memory_config)
    # Signal to the producer (CQ 1) that CQ 0 is finished with the input and it can be overwritten
    first_op_event = ttnn.record_event(device, 0)
    # Run the rest of the model on the default CQ (0)
    output_tensor = model(input_l1_tensor)
    ttnn.deallocate(input_l1_tensor)  # NEEDED TO ADD THIS

    # Make CQ 0 stall until CQ 1 has signalled that the read has finished
    ttnn.wait_for_event(0, read_event)
    # Run the last operation of the model on CQ 0
    output_dram_tensor = ttnn.reshard(output_tensor, output_sharded_dram_mem_config)
    # Signal that the model has finished on CQ 0
    last_op_event = ttnn.record_event(device, 0)

    # Capture the trace of the model
    ttnn.wait_for_event(1, last_op_event)
    ttnn.copy_host_to_device_tensor(host_tensor, input_dram_tensor, cq_id=1)
    write_event = ttnn.record_event(device, 1)
    ttnn.wait_for_event(0, write_event)
    input_l1_tensor = ttnn.to_memory_config(input_dram_tensor, l1_memory_config)
    op_event = ttnn.record_event(device, 0)
    # Record the address of the input tensor to trace so that we can validate we allocated our input tensor at the right address
    input_trace_addr = input_l1_tensor.buffer_address()
    spec = input_l1_tensor.spec
    # Deallocate the previous output tensor here so that we will allocate our input tensor at the right address afterwards
    output_tensor.deallocate(force=True)

    tid = ttnn.begin_trace_capture(device, cq_id=0)
    # It is important that we keep the output tensor on device returned here, so that we have the output tensor and associated address to read from after executing trace
    output_tensor = model(input_l1_tensor)
    ttnn.deallocate(input_l1_tensor)

    # Try allocating our persistent input tensor here and verifying it matches the address that trace captured
    input_l1_tensor = ttnn.allocate_tensor_on_device(spec, device)
    assert input_trace_addr == input_l1_tensor.buffer_address()

    ttnn.end_trace_capture(device, tid, cq_id=0)

    outputs = []

    # Stall CQ 1 for the input tensor consumer (CQ 0) to signal it has finished so we can start overwriting the previous input tensor with the new one
    ttnn.wait_for_event(1, first_op_event)
    # Write the next input tensor on CQ 1
    ttnn.copy_host_to_device_tensor(host_tensor, input_dram_tensor, cq_id=1)
    # Signal that the write has finished on CQ 1
    write_event = ttnn.record_event(device, 1)

    for iter in range(0, 2):
        # Make CQ 0 stall until CQ 1 has signalled that the write has finished
        ttnn.wait_for_event(0, write_event)
        # Run the first operation of the model on CQ 0
        # Note here that we are writing to our persisent input tensor in place to reuse the address
        input_l1_tensor = ttnn.reshard(input_dram_tensor, l1_memory_config, input_l1_tensor)
        # Signal to the producer (CQ 1) that CQ 0 is finished with the input and it can be overwritten
        first_op_event = ttnn.record_event(device, 0)
        # Run the rest of the model on the default CQ (0)
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
        # Make CQ 0 stall until CQ 1 has signalled that the read has finished
        ttnn.wait_for_event(0, read_event)
        # Run the last operation of the model on CQ 0
        output_dram_tensor = ttnn.reshard(output_tensor, output_sharded_dram_mem_config, output_dram_tensor)
        # Signal that the model has finished on CQ 0
        last_op_event = ttnn.record_event(device, 0)

        # Stall CQ 1 for the input tensor consumer (CQ 0) to signal it has finished so we can start overwriting the previous input tensor with the new one
        ttnn.wait_for_event(1, first_op_event)
        # Write the next input tensor on CQ 1
        ttnn.copy_host_to_device_tensor(host_tensor, input_dram_tensor, cq_id=1)
        # Signal that the write has finished on CQ 1
        write_event = ttnn.record_event(device, 1)

        # Make CQ 1 stall until CQ 0 has signalled that the model has finished
        ttnn.wait_for_event(1, last_op_event)
        outputs.append(output_dram_tensor.cpu(blocking=False, cq_id=1))
        # Signal that the read has finished on CQ 1
        read_event = ttnn.record_event(device, 1)

    # Make CQ 0 stall until CQ 1 has signalled that the write has finished
    ttnn.wait_for_event(0, write_event)
    # Run the first operation of the model on CQ 0
    # Note here that we are writing to our persisent input tensor in place to reuse the address
    input_l1_tensor = ttnn.reshard(input_dram_tensor, l1_memory_config, input_l1_tensor)
    # Signal to the producer (CQ 1) that CQ 0 is finished with the input and it can be overwritten
    first_op_event = ttnn.record_event(device, 0)
    # Run the rest of the model on the default CQ (0)
    ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
    # Make CQ 0 stall until CQ 1 has signalled that the read has finished
    ttnn.wait_for_event(0, read_event)
    # Run the last operation of the model on CQ 0
    output_dram_tensor = ttnn.reshard(output_tensor, output_sharded_dram_mem_config, output_dram_tensor)
    # Signal that the model has finished on CQ 0
    last_op_event = ttnn.record_event(device, 0)

    # Make CQ 1 stall until CQ 0 has signalled that the model has finished
    ttnn.wait_for_event(1, last_op_event)
    outputs.append(output_dram_tensor.cpu(blocking=False, cq_id=1))
    # Signal that the read has finished on CQ 1
    read_event = ttnn.record_event(device, 1)

    # Final synchronize to wait for all outputs to be read to host since we used non-blocking reads
    ttnn.synchronize_device(device)


@pytest.mark.parametrize(
    "device_params",
    [
        {
            "l1_small_size": 16384,
            "trace_region_size": 32768,
            "num_command_queues": 2,
        }
    ],
    indirect=True,
)
def test_traced_model_executor3(device):
    input_shape = (1, 1, 1024, 32)

    model, reference_model = create_test_model(input_shape)
    host_tensor, reference_input_tensor = create_input_tensors(input_shape, device=None, memory_config=None)

    torch_output_tensor = reference_model(reference_input_tensor)

    dram_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
        [512, 32],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    dram_memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, dram_shard_spec
    )

    l1_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
        [256, 32],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    l1_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, l1_shard_spec)

    output_sharded_dram_mem_config = dram_memory_config

    # This example uses 1 CQ for only writing inputs (CQ 1), and one CQ for executing programs/reading back the output (CQ 0)

    # `op_event` signals when the first operation is completed. This is the consumer of the input tensor so once this is completed, we can issue the next write

    # `write_event` signals when input write is completed. This is used to signal that the input tensor can be read/consumed

    # Allocate our persistent input tensor
    input_dram_tensor = ttnn.allocate_tensor_on_device(
        host_tensor.shape, host_tensor.dtype, host_tensor.layout, device, dram_memory_config
    )

    # Dummy record an op event on CQ 0 since we wait on this first in the loop
    op_event = ttnn.record_event(device, 0)

    # First run to compile the model
    # Stall CQ 1 for the input tensor consumer (CQ 0) to signal it has finished so we can start overwriting the previous input tensor with the new one
    ttnn.wait_for_event(1, op_event)
    # Write the next input tensor on CQ 1
    ttnn.copy_host_to_device_tensor(host_tensor, input_dram_tensor, cq_id=1)
    # Signal that the write has finished on CQ 1
    write_event = ttnn.record_event(device, 1)
    # Make CQ 0 stall until CQ 1 has signalled that the write has finished
    ttnn.wait_for_event(0, write_event)
    # Run the first operation of the model on CQ 0
    input_l1_tensor = ttnn.to_memory_config(input_dram_tensor, l1_memory_config)
    # Signal to the producer (CQ 1) that CQ 0 is finished with the input and it can be overwritten
    op_event = ttnn.record_event(device, 0)
    # Run the rest of the model and issue output readback on the default CQ (0)
    output_tensor = model(input_l1_tensor)

    ttnn.deallocate(input_l1_tensor)

    # Capture the trace of the model
    ttnn.wait_for_event(1, op_event)
    ttnn.copy_host_to_device_tensor(host_tensor, input_dram_tensor, cq_id=1)
    write_event = ttnn.record_event(device, 1)
    ttnn.wait_for_event(0, write_event)
    input_l1_tensor = ttnn.to_memory_config(input_dram_tensor, l1_memory_config)
    op_event = ttnn.record_event(device, 0)
    # Record the address of the input tensor to trace so that we can validate we allocated our input tensor at the right address
    input_trace_addr = input_l1_tensor.buffer_address()
    spec = input_l1_tensor.spec
    # Deallocate the previous output tensor here so that we will allocate our input tensor at the right address afterwards
    output_tensor.deallocate(force=True)
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    # It is important that we keep the output tensor on device returned here, so that we have the output tensor and associated address to read from after executing trace
    output_tensor = model(input_l1_tensor)

    ttnn.deallocate(input_l1_tensor)

    # Try allocating our persistent input tensor here and verifying it matches the address that trace captured
    input_l1_tensor = ttnn.allocate_tensor_on_device(spec, device)
    assert input_trace_addr == input_l1_tensor.buffer_address()

    ttnn.end_trace_capture(device, tid, cq_id=0)

    outputs = []

    for iter in range(0, 2):
        # Stall CQ 1 for the input tensor consumer (CQ 0) to signal it has finished so we can start overwriting the previous input tensor with the new one
        ttnn.wait_for_event(1, op_event)
        # Write the next input tensor on CQ 1
        ttnn.copy_host_to_device_tensor(host_tensor, input_dram_tensor, cq_id=1)
        # Signal that the write has finished on CQ 1
        write_event = ttnn.record_event(device, 1)
        # Make CQ 0 stall until CQ 1 has signalled that the write has finished
        ttnn.wait_for_event(0, write_event)
        # Run the first operation of the model on CQ 0
        # Note here that we are writing to our persisent input tensor in place to reuse the address
        input_l1_tensor = ttnn.reshard(input_dram_tensor, l1_memory_config, input_l1_tensor)
        # Signal to the producer (CQ 1) that CQ 0 is finished with the input and it can be overwritten
        op_event = ttnn.record_event(device, 0)
        # Run the rest of the model and issue output readback on the default CQ (0)
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
        outputs.append(output_tensor.cpu(blocking=False))

    # Final synchronize to wait for all outputs to be read to host since we used non-blocking reads
    ttnn.synchronize_device(device)
