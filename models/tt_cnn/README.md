# TT-CNN

The TT-CNN library comprises several modules with reusable functions designed to facilitate the development of high-performance vision models in TT-NN. The goal of the library is to minimize code duplication and enhance maintainability, as well as enabling quicker model development with superior initial model performance.

The following sections will document usage for each of these modules.

## Pipeline

This module provides a high-performance pipeline framework for executing CNN models on Tenstorrent devices. It includes various executor implementations optimized for different use cases and hardware configurations.

### Overview

The TT-CNN Pipeline API provides:
- **Traced execution** for optimal performance through kernel compilation and command recording.
- **Multi-command queue support** for parallel I/O and compute operations.

### Getting Started

```python
import ttnn
from models.tt_cnn.tt.pipeline import PipelineConfig, create_pipeline_from_config

# Define your model as a Callable function
def my_model(l1_input_tensor):
    # Your model receives L1 device tensors - the Pipeline handles host-to-device transfers automatically
    assert l1_input_tensor.storage_type() == ttnn.StorageType.DEVICE
    assert l1_input_tensor.memory_config().buffer_type == ttnn.BufferType.L1

    x = ttnn.relu(l1_input_tensor)
    x = ttnn.tilize(x)
    x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)

    return x

# Configure the pipeline settings
config = PipelineConfig(
    use_trace=True,  # Enable tracing
    num_command_queues=2,  # Use two command queues to pipeline subsequent model runs
    all_transfers_on_separate_command_queue=False  # Use a separate command queue for all data transfers
)

# Create the pipeline with the specified configuration
pipeline = create_pipeline_from_config(
    config=config,
    model=my_model,
    device=device,
    dram_input_memory_config=dram_memory_config,  # DRAM memory configuration for input
    l1_input_memory_config=l1_memory_config  # L1 memory configuration for input
)

# Compile the pipeline with a sample input to warmup the program cache and record traces (if needed)
# Note: sample_input must be a host tensor (StorageType.HOST)
pipeline.compile(sample_input)
# Enqueue TTNN host tensors for processing - pipeline handles all device transfers automatically
outputs = pipeline.enqueue(inputs).pop_all()
# Clean up and release resources used by the pipeline
pipeline.cleanup()
```

### User Guide

The TT-CNN Pipeline API provides end-to-end performance optimization techniques to improve throughput and lower latency for CNN models. This guide helps you understand and configure these optimizations effectively.

### Performance Optimization Techniques

#### Trace
 Tracing eliminates host overhead by recording operation commands into device memory and replaying them during execution. This is particularly effective for **host-bound** models where the host cannot dispatch commands fast enough to keep the device busy.

**Benefits:**
- Removes gaps between operations on the device.
- Eliminates host-device communication delays during execution.

**Best for:**
- Models with static input/output shapes (image classification, object detection).
- Host-bound scenarios where command dispatch is the bottleneck.

**Not suitable for:**
- Generative models with changing sequence lengths (without advanced techniques).

#### Multiple Command Queues
Uses two independent command queues to overlap I/O operations with computation, ideal for models where input/output transfer time is significant.

**Benefits:**
- Overlaps host-to-device and/or device-to-host transfers with model execution.
- Enables host to enqueue commands ahead of device execution.
- Maximizes device utilization, improving throughput.

**Best for:**
- Models where I/O transfer time is significant compared to computation time.
- Large input tensors that take significant time to transfer.
- Scenarios where host can benefit from enqueueing commands ahead of device execution.

### Choosing the Right Configuration

#### Single Command Queue (Basic)
```python
config = PipelineConfig(
    use_trace=False,
    num_command_queues=1
)
```
- **Use when**: Simple functional models, prototyping, or debugging

#### Single Command Queue + Trace
```python
config = PipelineConfig(
    use_trace=True,
    num_command_queues=1
)
```
- **Use when**: Host-bound models with minimal I/O overhead
- **Executor**: `TracedModelExecutor`
- **Performance**: Reduces host dispatch overhead, improving latency.
- **Requirements**: `dram_input_memory_config` for persistent DRAM tensors

#### Multi-Command Queue + Trace (Recommended)
```python
config = PipelineConfig(
    use_trace=True,
    num_command_queues=2,
    all_transfers_on_separate_command_queue=False
)
```
- **Use when**: Host-bound models with significant host-to-device transfer overhead
- **Performance**: Combines trace benefits with overlapped input transfers during execution
- **Queue Usage**: CQ0 for operations + output reads, CQ1 dedicated for input writes
- **Requirements**: `dram_input_memory_config` - pipeline handles persistent DRAM tensor allocation and event synchronization

#### Multi-Command Queue + Trace + Dedicated CQ for I/O
```python
config = PipelineConfig(
    use_trace=True,
    num_command_queues=2,
    all_transfers_on_separate_command_queue=True
)
```
- **Use when**: Host-bound models with significant bidirectional I/O transfer overhead
- **Executor**: `MultiCQTracedModelPipelinedIOExecutor`
- **Performance**: Fully overlaps both input writes and output reads with model execution
- **Queue Usage**: CQ0 for operations only, CQ1 for both input writes and output reads
- **Requirements**: `dram_output_memory_config` for persistent output tensors, requires a minimum of two inputs for effective pipelining, complex event synchronization

#### DRAM Memory Configuration
Many pipeline configurations require memory configs for the persistent tensors that the pipeline automatically allocates and manages during execution. To automatically select optimal DRAM configs given your input shape, use the following utility:

```python
from models.tt_cnn.tt.pipeline import get_memory_config_for_persistent_dram_tensor

dram_memory_config = get_memory_config_for_persistent_dram_tensor(
    shape=input_shape,
    shard_strategy=ttnn.TensorMemoryLayout.WIDTH_SHARDED,  # or HEIGHT_SHARDED
    dram_grid_size=device.dram_grid_size()
)
```

The L1 memory configs are also required by the Pipeline API and are typically model-specific. The pipeline automatically reshards tensors from DRAM to L1 using these configurations before passing them to your model function.

#### Preallocating Host Memory for Output Tensors

Some workloads can become bottlenecked by host memory allocations. To pre-allocate host memory for output tensors, configure your pipeline with the following call:

```python
pipe.preallocate_output_tensors_on_host(number_of_inputs)
```

### Troubleshooting

#### Common Issues
- **Input tensor validation errors**: Ensure all input tensors are host tensors (`StorageType.HOST`) - the pipeline will reject device tensors
- **Trace buffer size errors**: Increase `trace_region_size` in device parameters. The error message will indicate the required size: `Creating trace buffers of size XB on device 0, but only 0B is allocated for trace region`
- **L1 memory allocation failures**: Verify DRAM and L1 memory configurations are correct for your input shapes

### Examples

See the following model implementations for examples of real-world usage:

- **[YOLOv8x](../demos/yolov8x/tests/perf/test_e2e_performant.py)** - Object detection with trace+2CQ
- **[ResNet50](../demos/ttnn_resnet/tests/perf_e2e_resnet50.py)** - Image classification pipeline
- **[YOLOv4](../demos/yolov4/demo.py)** - Demo with pipeline integration
