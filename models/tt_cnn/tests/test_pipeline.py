# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Callable

import pytest
import torch

import ttnn
from models.tt_cnn.tt.executor import (
    ModelExecutor,
    MultiCQModelOverlappedInputExecutor,
    MultiCQTracedModelOverlappedInputExecutor,
    MultiCQTracedModelPipelinedIOExecutor,
    TracedModelExecutor,
)
from models.tt_cnn.tt.pipeline import (
    PipelineConfig,
    create_pipeline_from_config,
    get_memory_config_for_persistent_dram_tensor,
)
from tests.ttnn.utils_for_testing import assert_equal


@dataclass
class TestShapeConfig:
    input_shape: tuple
    dram_cores: int
    l1_cores: int


def create_input_tensors(input_shape, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG, device=None):
    torch_input_tensor = torch.rand(input_shape, dtype=torch.bfloat16)
    ttnn_input_tensor = ttnn.from_torch(
        torch_input_tensor, dtype=ttnn.bfloat16, layout=layout, memory_config=memory_config, device=device
    )
    return ttnn_input_tensor, torch_input_tensor


def create_identity_test_model(input_shape, should_deallocate_input_tensor=True):
    def run(l1_input_tensor):
        assert l1_input_tensor.storage_type() == ttnn.StorageType.DEVICE, "Model expects input tensor to be on device"
        assert (
            l1_input_tensor.memory_config().buffer_type == ttnn.BufferType.L1
        ), "Model expects input tensor to be in L1"
        assert input_shape == l1_input_tensor.shape, "Unexpected input shape"
        return ttnn.identity(l1_input_tensor)

    def run_reference(torch_input_tensor):
        assert input_shape == torch_input_tensor.shape, "Unexpected input shape"
        return torch_input_tensor

    return run, run_reference


def create_test_model(input_shape, should_deallocate_input_tensor=True):
    def run(l1_input_tensor):
        assert l1_input_tensor.storage_type() == ttnn.StorageType.DEVICE, "Model expects input tensor to be on device"
        assert (
            l1_input_tensor.memory_config().buffer_type == ttnn.BufferType.L1
        ), "Model expects input tensor to be in L1"
        assert input_shape == l1_input_tensor.shape, "Unexpected input shape"
        x = ttnn.tilize(l1_input_tensor)
        if should_deallocate_input_tensor:
            ttnn.deallocate(l1_input_tensor)
        x = ttnn.relu(x)
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        return x

    def run_reference(torch_input_tensor):
        assert input_shape == torch_input_tensor.shape, "Unexpected input shape"
        return torch.nn.functional.relu(torch_input_tensor)

    return run, run_reference


def create_multi_output_test_model(input_shape, should_deallocate_input_tensor=True):
    def run(l1_input_tensor):
        assert l1_input_tensor.storage_type() == ttnn.StorageType.DEVICE, "Model expects input tensor to be on device"
        assert (
            l1_input_tensor.memory_config().buffer_type == ttnn.BufferType.L1
        ), "Model expects input tensor to be in L1"
        assert input_shape == l1_input_tensor.shape, "Unexpected input shape"

        identity_output = ttnn.to(l1_input_tensor, memory_config=l1_input_tensor.memory_config())
        relu_output = ttnn.relu(l1_input_tensor)

        if should_deallocate_input_tensor:
            ttnn.deallocate(l1_input_tensor)

        return [identity_output, relu_output]

    def run_reference(torch_input_tensor):
        assert input_shape == torch_input_tensor.shape, "Unexpected input shape"
        return [torch_input_tensor, torch.nn.functional.relu(torch_input_tensor)]

    return run, run_reference


@dataclass
class ExecutorTestConfig:
    name: str
    use_trace: bool
    num_command_queues: int
    all_transfers_on_separate_command_queue: bool = False
    requires_output_config: bool = False
    requires_dram_output_config: bool = False
    requires_minimum_inputs: int = 1
    expected_executor_type: type = None


EXECUTOR_CONFIGS = [
    ExecutorTestConfig(
        name="ModelExecutor",
        use_trace=False,
        num_command_queues=1,
        all_transfers_on_separate_command_queue=False,
        requires_minimum_inputs=1,
        expected_executor_type=ModelExecutor,
    ),
    ExecutorTestConfig(
        name="TracedModelExecutor",
        use_trace=True,
        num_command_queues=1,
        all_transfers_on_separate_command_queue=False,
        requires_minimum_inputs=1,
        expected_executor_type=TracedModelExecutor,
    ),
    ExecutorTestConfig(
        name="MultiCQModelOverlappedInputExecutor",
        use_trace=False,
        num_command_queues=2,
        all_transfers_on_separate_command_queue=False,
        requires_minimum_inputs=1,
        expected_executor_type=MultiCQModelOverlappedInputExecutor,
    ),
    ExecutorTestConfig(
        name="MultiCQTracedModelOverlappedInputExecutor",
        use_trace=True,
        num_command_queues=2,
        all_transfers_on_separate_command_queue=False,
        requires_minimum_inputs=1,
        expected_executor_type=MultiCQTracedModelOverlappedInputExecutor,
    ),
    ExecutorTestConfig(
        name="MultiCQTracedModelPipelinedIOExecutor",
        use_trace=True,
        num_command_queues=2,
        all_transfers_on_separate_command_queue=True,
        requires_output_config=True,
        requires_minimum_inputs=2,  # Pipelined I/O requires at least 2 inputs to enable overlapping of input transfer with model execution
        expected_executor_type=MultiCQTracedModelPipelinedIOExecutor,
    ),
]


SHAPE_CONFIGS = [
    TestShapeConfig(
        input_shape=(1, 1, 32, 32),
        dram_cores=1,
        l1_cores=1,
    ),
    TestShapeConfig(
        input_shape=(1, 1, 512, 64),
        dram_cores=4,
        l1_cores=8,
    ),
]


def create_memory_configs(input_shape, dram_cores, l1_cores):
    assert all([shape == 1 for shape in input_shape[:-2]])
    dram_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_cores - 1, 0))}),
        [input_shape[-2] // dram_cores, input_shape[-1]],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    dram_memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, dram_shard_spec
    )

    l1_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(l1_cores - 1, 0))}),
        [input_shape[-2] // l1_cores, input_shape[-1]],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    l1_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, l1_shard_spec)

    return dram_memory_config, l1_memory_config


def create_pipeline_for_executor_config(
    executor_config: ExecutorTestConfig,
    model,
    device,
    input_shape,
    dram_cores,
    l1_cores,
):
    dram_memory_config, l1_memory_config = create_memory_configs(input_shape, dram_cores, l1_cores)

    config = PipelineConfig(
        use_trace=executor_config.use_trace,
        num_command_queues=executor_config.num_command_queues,
        all_transfers_on_separate_command_queue=executor_config.all_transfers_on_separate_command_queue,
    )

    pipeline_args = {
        "config": config,
        "model": model,
        "device": device,
        "dram_input_memory_config": dram_memory_config,
        "l1_input_memory_config": l1_memory_config,
        "dram_output_memory_config": dram_memory_config,
    }

    return create_pipeline_from_config(**pipeline_args)


@pytest.mark.parametrize("executor_config", EXECUTOR_CONFIGS, ids=lambda cfg: cfg.name)
@pytest.mark.parametrize("shape_config", SHAPE_CONFIGS, ids=lambda cfg: f"shape_{cfg.input_shape}")
@pytest.mark.parametrize(
    "create_model", [create_test_model, create_identity_test_model], ids=["relu_model", "identity_model"]
)
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 16384, "trace_region_size": 32768, "num_command_queues": 2}],
    indirect=True,
)
def test_executor_single_runs(
    device, executor_config: ExecutorTestConfig, shape_config: TestShapeConfig, create_model: Callable
):
    """Test basic functionality across all executor types with parameterized configurations"""
    input_shape = shape_config.input_shape
    dram_cores = shape_config.dram_cores
    l1_cores = shape_config.l1_cores

    model, reference_model = create_model(input_shape)

    pipe = create_pipeline_for_executor_config(executor_config, model, device, input_shape, dram_cores, l1_cores)
    executor = pipe.executor

    assert isinstance(
        executor, executor_config.expected_executor_type
    ), f"Expected {executor_config.expected_executor_type.__name__}, got {type(executor).__name__}"

    num_inputs = 32
    host_inputs = []
    reference_outputs = []
    for _ in range(num_inputs):
        input_tensor, reference_input = create_input_tensors(input_shape, device=None, memory_config=None)
        host_inputs.append(input_tensor)
        reference_outputs.append(reference_model(reference_input))

    pipe.compile(host_inputs[0])
    outputs = pipe.enqueue(host_inputs).pop_all()
    pipe.cleanup()

    assert len(outputs) == len(reference_outputs), f"Expected {len(reference_outputs)} outputs, got {len(outputs)}"
    for i, (output, reference_output) in enumerate(zip(outputs, reference_outputs)):
        assert (
            output.storage_type() == ttnn.StorageType.HOST
        ), f"Output {i} should be on host for {executor_config.name}"
        assert_equal(ttnn.to_torch(output), reference_output)


@pytest.mark.parametrize("executor_config", EXECUTOR_CONFIGS, ids=lambda cfg: cfg.name)
@pytest.mark.parametrize("shape_config", SHAPE_CONFIGS, ids=lambda cfg: f"shape_{cfg.input_shape}")
@pytest.mark.parametrize(
    "create_model", [create_test_model, create_identity_test_model], ids=["relu_model", "identity_model"]
)
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 16384, "trace_region_size": 32768, "num_command_queues": 2}],
    indirect=True,
)
def test_executor_multiple_runs(
    device, executor_config: ExecutorTestConfig, shape_config: TestShapeConfig, create_model: Callable
):
    """Test multiple execution rounds across all executor types"""
    input_shape = shape_config.input_shape
    dram_cores = shape_config.dram_cores
    l1_cores = shape_config.l1_cores
    model, reference_model = create_model(input_shape)

    pipe = create_pipeline_for_executor_config(executor_config, model, device, input_shape, dram_cores, l1_cores)

    sample_input, _ = create_input_tensors(input_shape, device=None, memory_config=None)
    pipe.compile(sample_input)

    total_outputs = []
    total_references = []

    for _ in range(3):
        round_size = 32
        round_inputs = []
        round_references = []

        for _ in range(round_size):
            input_tensor, reference_input = create_input_tensors(input_shape, device=None, memory_config=None)
            round_inputs.append(input_tensor)
            round_references.append(reference_model(reference_input))

        outputs = pipe.enqueue(round_inputs).pop_all()
        total_outputs.extend(outputs)
        total_references.extend(round_references)

    pipe.cleanup()

    assert len(total_outputs) == len(
        total_references
    ), f"Expected {len(total_references)} total outputs for {executor_config.name}"

    for i, (output, reference_output) in enumerate(zip(total_outputs, total_references)):
        assert_equal(ttnn.to_torch(output), reference_output)


@pytest.mark.parametrize("executor_config", EXECUTOR_CONFIGS, ids=lambda cfg: cfg.name)
@pytest.mark.parametrize("shape_config", SHAPE_CONFIGS, ids=lambda cfg: f"shape_{cfg.input_shape}")
@pytest.mark.parametrize(
    "create_model", [create_test_model, create_identity_test_model], ids=["relu_model", "identity_model"]
)
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 16384, "trace_region_size": 32768, "num_command_queues": 2}],
    indirect=True,
)
def test_executor_with_preallocated_outputs(
    device, executor_config: ExecutorTestConfig, shape_config: TestShapeConfig, create_model: Callable
):
    input_shape = shape_config.input_shape
    dram_cores = shape_config.dram_cores
    l1_cores = shape_config.l1_cores
    model, reference_model = create_model(input_shape)

    pipe = create_pipeline_for_executor_config(executor_config, model, device, input_shape, dram_cores, l1_cores)

    sample_input, _ = create_input_tensors(input_shape, device=None, memory_config=None)
    pipe.compile(sample_input)

    num_inputs = 32
    pipe.preallocate_output_tensors_on_host(num_inputs)

    assert (
        pipe.preallocated_output_tensors == True
    ), f"Pipeline should be in preallocated mode for {executor_config.name}"
    assert (
        len(pipe.output_tensors) == num_inputs
    ), f"Expected {num_inputs} preallocated tensors for {executor_config.name}"

    host_inputs = []
    reference_outputs = []
    for _ in range(num_inputs):
        input_tensor, reference_input = create_input_tensors(input_shape, device=None, memory_config=None)
        host_inputs.append(input_tensor)
        reference_outputs.append(reference_model(reference_input))

    outputs = pipe.enqueue(host_inputs).pop_all()
    pipe.cleanup()

    assert outputs is pipe.output_tensors, f"Should return preallocated tensors for {executor_config.name}"
    assert len(outputs) == num_inputs, f"Expected {num_inputs} outputs for {executor_config.name}"

    for i, (output, reference_output) in enumerate(zip(outputs, reference_outputs)):
        assert (
            output.storage_type() == ttnn.StorageType.HOST
        ), f"Preallocated output {i} should be on host for {executor_config.name}"
        assert_equal(
            ttnn.to_torch(output),
            reference_output,
        )


@pytest.mark.parametrize(
    "shard_strategy", [ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.TensorMemoryLayout.HEIGHT_SHARDED]
)
def test_get_memory_config_for_persistent_dram_tensor(shard_strategy):
    dram_grid_size = ttnn.CoreCoord(2, 1)
    shape = (1, 1, 64, 64)
    dram_memory_config = get_memory_config_for_persistent_dram_tensor(shape, shard_strategy, dram_grid_size)
    assert dram_memory_config.buffer_type == ttnn.BufferType.DRAM
    assert (
        dram_memory_config.shard_spec.shape == [64, 32]
        if shard_strategy == ttnn.TensorMemoryLayout.WIDTH_SHARDED
        else [32, 64]
    )


def test_get_dram_sharded_memory_config_for_tensor_invalid_grid_size():
    dram_grid_size = ttnn.CoreCoord(2, 2)
    shape = (1, 1, 64, 64)
    with pytest.raises(ValueError):
        get_memory_config_for_persistent_dram_tensor(shape, ttnn.TensorMemoryLayout.WIDTH_SHARDED, dram_grid_size)
