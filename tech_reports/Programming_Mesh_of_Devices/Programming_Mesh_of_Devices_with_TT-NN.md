# Programming Mesh of Devices with TT-NN

Authors: Scale-Out Team

## Contents

- [Programming Mesh of Devices with TT-NN](#programming-mesh-of-devices-with-tt-nn)
- [1. Overview](#1-overview)
- [2. MeshDevice](#2-meshdevice)
  - [2.1 System Topology](#21-system-topology)
  - [2.2 MeshDevice Management](#22-meshdevice-management)
    - [2.2.1 MeshDevice Initialization/Close](#221-meshdevice-initializationclose)
    - [2.2.2 MeshDevice Visualization](#222-meshdevice-visualization)
  - [2.3 Controlling Device Visibility](#23-controlling-device-visibility)
- [3. Distributing Tensor to MeshDevice](#3-distributing-tensor-to-meshdevice)
  - [3.1 Distribution Strategies](#31-distribution-strategies)
  - [3.2 Programming Example: Sharding](#32-programming-example-sharding)
- [4. Single-Program Multiple Device](#4-single-program-multiple-device)
  - [4.1 Execution Model](#41-execution-model)
  - [4.2 Single Device to Multiple Device Execution](#42-single-device-to-multiple-device-execution)
    - [4.2.1 Single Device Execution](#421-single-device-execution)
    - [4.2.2 Mesh Device Execution](#422-mesh-device-execution)
- [5. MeshDevice and Collective Communication Library (CCL)](#5-meshdevice-and-collective-communication-library-ccl)
  - [5.1 CCL Operations](#51-ccl-operations)
  - [5.2 All-Gather](#52-all-gather)
    - [5.2.1 Programming Example: All-Gather (Ring)](#521-programming-example-all-gather-ring)
    - [5.2.2 Programming Example: All-Gather (Line)](#522-programming-example-all-gather-line)
- [6. Programming Mesh of Devices Using Data Parallel](#6-programming-mesh-of-devices-using-data-parallel)
  - [6.1 Data Parallel Programming Example](#61-data-parallel-programming-example)
- [7. Programming Mesh of Devices Using Tensor Parallel](#7-programming-mesh-of-devices-using-tensor-parallel)
  - [7.1 Tensor Parallel Programming Example](#71-tensor-parallel-programming-example)
- [8. Programming Mesh of Devices Using Hybrid Tensor and Data Parallel](#8-programming-mesh-of-devices-using-hybrid-tensor-and-data-parallel)
  - [8.1 Llama-3.1 70B Hybrid Tensor and Data Parallel](#81-llama-31-70b-hybrid-tensor-and-data-parallel)
  - [8.2 Llama-3.1 70B Performance Scaling](#82-llama-31-70b-performance-scaling)
  - [8.3 Hybrid Tensor and Data Parallel Programming Example](#83-hybrid-tensor-and-data-parallel-programming-example)
    - [8.3.1 Overview of Changes](#831-overview-of-changes)
    - [8.3.2 Key Components](#832-key-components)
- [9. MeshDevice vs PyTorch Multi-Device Programming](#9-meshdevice-vs-pytorch-multi-device-programming)
  - [9.1 Overview of Multi-Device Handling](#91-overview-of-multi-device-handling)
  - [9.2 Code Comparison: Matrix Multiplication with All-Gather](#92-code-comparison-matrix-multiplication-with-all-gather)

## 1. Overview

TT-NN library natively supports multi-device operations, enabling users to scale their single-device application code to multiple devices seamlessly. TT-NN employs a Single-Program Multiple-Device (SPMD) technique to parallelize a computation across a set of connected devices operating on different input data. This is achieved through a few key components:

- **MeshDevice**: This "virtual device" abstraction defines a logical 2-D mesh of connected physical devices. Operations that "run on device" are distributed through SPMD across all devices captured in the mesh.

- **Input Data Distribution**: Defines how input data resident in host-memory is distributed to MeshDevice on-device memory. When operations are distributed to MeshDevice, the operation within a single-device scope works on its local input data.

- **Tensor**: Defines a N-dimensional matrix containing elements of a single data type. In a MeshDevice context, a Tensor, or colloquially referred to as MeshTensor, represents a collection of tensor shards distributed across devices in a 2D Mesh.


These concepts are key to understanding how we scale models using **Data-Parallel**, **Tensor-Parallel**, and **Hybrid Data + Tensor Parallel.**

## 2. MeshDevice

MeshDevice is a virtual device abstraction that bundles together multiple physical devices to enable efficient parallel execution across a mesh topology. This abstraction is natively supported at the runtime level, allowing for deep integration with the hardware and dispatch mechanisms. When operations are dispatched to a MeshDevice, command queues are utilized to distribute work across all constituent devices in parallel, significantly reducing dispatch overhead compared to sequential device-by-device execution.

To optimize performance, MeshDevice implements several key optimizations:
- **Kernel Compilation Broadcasting**: When kernels are compiled for execution on a MeshDevice, the compilation artifacts are automatically broadcasted to all devices in the mesh where applicable, avoiding redundant per-device compilation.
- **Data Broadcasting**: For replicated tensors (where the same data needs to exist on multiple devices), MeshDevice leverages the mesh topology to efficiently broadcast data across devices rather than performing individual writes to each device.
- **Unified Command Dispatch**: Operations are dispatched through mesh-aware command queues that coordinate execution across all devices, ensuring lock-step parallel execution.

While MeshDevice provides these performance optimizations, it maintains explicit control over data distribution and communication patterns. MeshDevice does not hide the distributed nature of the computation - users must explicitly specify:
- **Data Distribution**: How input tensors are distributed across devices (sharded or replicated) using mesh mappers
- **Collective Communication**: When and how devices need to communicate using CCL operations like all-gather, reduce-scatter, etc.

This explicit control allows users to optimize their applications for specific hardware topologies and workload characteristics. For detailed information on data distribution strategies, refer to [Section 3 (Distributing Tensor to MeshDevice)](#3-distributing-tensor-to-meshdevice). For collective communication patterns, see [Section 5 (MeshDevice and Collective Communication Library)](#5-meshdevice-and-collective-communication-library-ccl).

### 2.1 System Topology

A MeshDevice can be instantiated over a collection of physically connected devices. Examples of the supported configurations are: N300 (1x2), QuietBox (Wormhole) (2x4), Galaxy (8x4).

The N300 form-factor houses two wormhole chips. The host is connected to the "left" chip via PCIe and the "left" chip is connected to the "right" chip via two ethernet links. Each ethernet link has a 200 Gbps bi-directional bandwidth. For N300, one of the ethernet links connecting the "left" chip to the "right" chip is reserved for fast-dispatch. At the user-level, this means only a single ethernet link is made available for use. The N300 represents the smallest multi-device configuration that we can instantiate a MeshDevice over.

<!-- ![image1](images/image1.png){width=15 height=15} -->
<img src="../EthernetMultichip/images/t3000.png" style="width:500px;"/>

*Figure 1: QuietBox (Wormhole) System Topology. QuietBox (Wormhole) is composed of 4x N300 wormhole cards, totalling 8 wormhole chips, connected in a 2x4 mesh configuration. Each pair of wormhole-chips are connected via two ethernet links.*


<img src="../EthernetMultichip/images/TG.png" style="width:500px;"/>

*Figure 2: Galaxy System Topology. Galaxy is composed of 32x galaxy wormhole cards, totalling 32 wormhole chips, connected in a 8x4 mesh configuration. Each pair of wormhole-chips are connected via four ethernet links.*


[tt-topology](https://github.com/tenstorrent/tt-topology) can be used to flash multiple wormhole cards on a system to a specific ethernet routing configuration (linear, ring, mesh) and used to visualize the organization of the chip layout.

<img src="images/image3.png" style="width:500px;"/>

*Figure 3: QuietBox (Wormhole) Chip Layout dumped from tt-topology*

#### 2.1.1 SystemMesh Visualization

```py
ttnn.visualize_system_mesh()
>
SystemMesh Global Shape: MeshShape([1, 2])

SystemMesh Local Shape: MeshShape([1, 2])

   SystemMesh Global Shape: (1, 2) | Local Shape: (1, 2)
┌──────────────────────────────┬──────────────────────────────┐
│          Dev. ID: 0          │          Dev. ID: 1          │
│            (0, 0)            │            (0, 1)            │
└──────────────────────────────┴──────────────────────────────┘
```


### 2.2 MeshDevice Management

#### 2.2.1 MeshDevice Initialization/Close

Using an N300, we can instantiate a MeshDevice over 1x2 Wormhole devices:

```py
> import ttnn
> mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1,2))
```

```py
> mesh_device
> <MeshDevice: 1x2 grid, 2 devices>
```

####

#### 2.2.1 MeshDevice Visualization

```py
ttnn.visualize_mesh_device(mesh_device)
>
   MeshDevice(rows=1, cols=2):
┌───────────────┬───────────────┐
│  Dev. ID: 0   │  Dev. ID: 1   │
│    (0, 0)     │    (0, 1)     │
│               │               │
└───────────────┴───────────────┘
```

##

### 2.3 Controlling Device Visibility

In multi-device systems, the set of *PCIe*-visible devices can be narrowed using the `TT_VISIBLE_DEVICES` environment variable.

Set `TT_VISIBLE_DEVICES` to a comma-separated list of device IDs (matching `/dev/tenstorrent/<id>`) to restrict which devices are visible to your process. If unset, all devices are visible; if set, only the listed devices are available. This is useful for:

- Partitioning devices to run independent jobs so that they do not collide.
- Prototyping a smaller mesh inside a larger topology (for example, emulating an N300 within a T3000).
- Emulating a multi-host configuration by simultaneously launching multiple processes working on independent parts of the available system mesh.

#### Usage Examples

1. Expose a single PCIe device. For N300, this exposes both the PCIe and the remote / ethernet-connected device.
```bash
TT_VISIBLE_DEVICES="0" python your_script.py
```

2. Expose two PCIe devices. On a T3000, using `TT_VISIBLE_DEVICES="0,1"` exposes two PCIe devices and their associated remote / ethernet-connected device. If PCIe device {0,1} is connected, then this effectively exposes a 2x2 mesh.
```bash
TT_VISIBLE_DEVICES="0,1" ./your_cpp_program
```


For more examples, please see `tests/tt_metal/distributed/multiprocess/run_visible_devices_mp_tests.sh`.

### 2.4 Distributed Process Launch with tt-run

#### 2.4.1 Overview and Design Philosophy

`tt-run` is a distributed process launcher for TT-Metal/TTNN workloads supporting both **single-host multi-process** and **multi-host distributed** configurations. It provides a declarative YAML-based interface to orchestrate MPI-based distributed execution, abstracting the complexity of environment management, device partitioning, and mesh topology configuration.

**Design Patterns Supported**

`tt-run` enables two primary distributed execution patterns:

1. **SPMD "Big-Mesh" Pattern**: Multiple MPI ranks coordinate to present a single, unified logical mesh spanning multiple hosts. Each rank manages a local sub-mesh while executing identical program code. This pattern is ideal for:
   - Tensor Parallelism (TP) across large meshes
   - Data Parallelism (DP) with uniform data distribution
   - Hybrid TP+DP workloads that scale uniformly across the mesh
   - Described in detail: [TT-Distributed: Multi-Host Runtime](../../tech_reports/TT-Distributed/MultiHostMeshRuntime.md)

2. **Multi-Mesh Pattern**: Different MPI ranks manage independent meshes, potentially running different workload stages. This pattern supports:
   - Pipeline Parallelism where each rank handles different model layers
   - Multi-model inference where independent models run on separate meshes
   - Heterogeneous workloads requiring different mesh configurations per stage

**Core Abstraction: Rank Bindings**

The central concept in `tt-run` is the **rank binding**, which maps each MPI rank to:
- A mesh identifier (`mesh_id`) - which logical mesh this rank belongs to
- A mesh host rank (`mesh_host_rank`) - position within a multi-host mesh (for SPMD Big-Mesh)
- Environment overrides - global and rank-specific environment variables

**Mesh Graph Descriptors (MGD 2.0)**

The Mesh Graph Descriptor defines the topology of the distributed system, including host topology for multi-host meshes and inter-mesh connections for multi-mesh scenarios.

MGD 2.0 uses Protobuf text format (`.textproto` extension). For detailed schema documentation and examples, see: [`tt_metal/fabric/MGD_README.md`](../../tt_metal/fabric/MGD_README.md)

**Automatic Environment Isolation**

To ensure safe multi-process execution, `tt-run` automatically manages per-rank environments:
- **TT_METAL_CACHE**: Unique cache directory per rank (default: `~/.cache/{hostname}_rank{N}`) prevents kernel compilation conflicts
- **TT_VISIBLE_DEVICES**: Controls PCIe device visibility per rank (see [Section 2.3](#23-controlling-device-visibility)). By default, all devices are visible.
- **TT_MESH_GRAPH_DESC_PATH**: Path to topology descriptor
- **TT_MESH_ID** & **TT_MESH_HOST_RANK**: Mesh identification for runtime coordination

#### 2.4.2 Configuration and Usage

**Basic Configuration Example**

```yaml
rank_bindings:
  - rank: 0
    mesh_id: 0                 # Mesh identifier
    mesh_host_rank: 0          # Position within multi-host mesh
    env_overrides:
      TT_VISIBLE_DEVICES: "0,1"  # Devices visible to this rank

  - rank: 1
    mesh_id: 0
    mesh_host_rank: 1
    env_overrides:
      TT_VISIBLE_DEVICES: "2,3"

mesh_graph_desc_path: "mesh_descriptor.textproto"  # MGD 2.0 topology definition

global_env:                    # Environment variables for all ranks
  TT_METAL_LOGGER_LEVEL: "INFO"
```

**Command-Line Invocation**

```bash
tt-run --rank-binding config.yaml [--mpi-args "<mpi_args>"] <program> [args...]
```

Common options:
- `--dry-run`: Preview generated MPI command without execution
- `--verbose`: Enable detailed logging
- `--mpi-args`: Pass additional MPI arguments (rankfiles, network options, etc.)

#### 2.4.3 Usage Patterns

**Pattern 1: SPMD Big-Mesh**

Multiple ranks collaborate to form a single logical mesh spanning hosts. All ranks share the same `mesh_id` but have different `mesh_host_rank` values.

```yaml
rank_bindings:
  - rank: 0
    mesh_id: 0
    mesh_host_rank: 0
    env_overrides: {TT_VISIBLE_DEVICES: "0,1"}
  - rank: 1
    mesh_id: 0
    mesh_host_rank: 1
    env_overrides: {TT_VISIBLE_DEVICES: "2,3"}

mesh_graph_desc_path: "dual_host_mesh.textproto"  # Multi-host topology
```

**Pattern 2: Multi-Mesh (Independent Meshes)**

Each rank manages an independent mesh, enabling pipeline parallelism or multi-model scenarios. Ranks have different `mesh_id` values.

```yaml
rank_bindings:
  - rank: 0
    mesh_id: 0      # First pipeline stage
    env_overrides: {TT_VISIBLE_DEVICES: "0"}
  - rank: 1
    mesh_id: 1      # Second pipeline stage
    env_overrides: {TT_VISIBLE_DEVICES: "1"}

mesh_graph_desc_path: "multi_mesh.textproto"
```

**Pattern 3: Single-Host Emulation**

Emulate multi-host workloads on a single host by partitioning devices across processes:

```yaml
rank_bindings:
  - {rank: 0, mesh_id: 0, mesh_host_rank: 0, env_overrides: {TT_VISIBLE_DEVICES: "0,1"}}
  - {rank: 1, mesh_id: 0, mesh_host_rank: 1, env_overrides: {TT_VISIBLE_DEVICES: "2,3"}}
mesh_graph_desc_path: "emulated_dual_host.textproto"
```

**Pattern 4: Multi-Host Cluster Deployment**

For multi-host clusters, combine rank bindings with MPI rankfiles to specify physical host assignments:

```bash
# Rankfile specifies physical host assignment
# rank 0=host1 slot=0
# rank 1=host2 slot=0

tt-run --rank-binding config.yaml \
       --mpi-args "--rankfile hosts.txt --mca btl tcp" \
       python distributed_workload.py
```


**Example Files and Tests**

Configuration examples:
- `tests/tt_metal/distributed/config/2x2_multiprocess_rank_bindings.yaml` - Single-host multi-process configuration
- `tt_metal/fabric/mesh_graph_descriptors/t3k_mesh_graph_descriptor.textproto` - T3000 mesh topology (MGD 2.0)
- `tt_metal/fabric/mesh_graph_descriptors/tg_mesh_graph_descriptor.textproto` - Galaxy mesh topology (MGD 2.0)

Test scripts demonstrating `tt-run` usage:
- `tests/scripts/run_t3000_unit_tests.sh` - T3000 multi-process test launcher
- `tests/scripts/run_dual_galaxy_tests.sh` - Multi-host Galaxy test launcher
- `tests/tt_metal/distributed/multiprocess/run_visible_devices_mp_tests.sh` - Device visibility examples

**Related Documentation**

- Section 2.3: [Controlling Device Visibility](#23-controlling-device-visibility) - for `TT_VISIBLE_DEVICES` details
- Section 2.1: [System Topology](#21-system-topology) - for understanding mesh configurations
- [`tt_metal/fabric/MGD_README.md`](../../tt_metal/fabric/MGD_README.md) - MGD 2.0 schema and examples
- [TT-Distributed: Multi-Host Runtime](../../tech_reports/TT-Distributed/MultiHostMeshRuntime.md) - SPMD architecture details

## 3. Distributing Tensor to MeshDevice

### 3.1 Distribution Strategies

MeshDevice in TT-NN provides a flexible way to distribute data across multiple devices. The distribution is primarily handled through the use of "mesh mappers" when creating tensors.

There are two main types of distribution strategies:

1. **Sharding**: This distribution strategy splits the tensor along specified dimension(s) and distributes the parts across devices in the mesh. This is useful for cases where the model-parameters cannot fit on a single-device and instead each device stores a slice of the model weights.

2. **Replication**: This distribution strategy copies the entire tensor to all devices in the mesh. This is useful for parameters that need to be available on all devices, such as model weights.

###

### 3.2 Programming Example: Sharding

Let's see how to split our data across two devices:

```py
import ttnn
import torch

# Open our 1x2 MeshDevice
mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 2))

# Initialize a torch tensor
torch_tensor = torch.zeros(1, 1, 32, 64)
torch_tensor[..., 0:32] = 1.0
torch_tensor[..., 32:64] = 2.0

# Create a mesh mapper. Given a mesh device shape, placements specify replication or sharding of data per each dimension of the mesh shape.
mesh_mapper = ttnn.create_mesh_mapper(
    mesh_device,
    ttnn.MeshMapperConfig(
        placements=[
            # Replicate data across first dimension of the mesh
            ttnn.PlacementReplicate(),
            # Shard dimension 3 of tensor across second dimension of the mesh
            ttnn.PlacementShard(3),
        ],
    ),
)

# Convert to ttnn.Tensor; MeshTensor holds buffers to two shards in host-memory
mesh_tensor = ttnn.from_torch(
    torch_tensor,
    mesh_mapper=mesh_mapper,
    layout=ttnn.TILE_LAYOUT,
)
```

Let's inspect our ttnn.Tensor object. At this point, the data still resides in host-memory.

```py
> mesh_tensor
ttnn.Tensor([[[[ 1.00000,  1.00000,  ...,  1.00000,  1.00000],
               [ 1.00000,  1.00000,  ...,  1.00000,  1.00000],
               ...,
               [ 1.00000,  1.00000,  ...,  1.00000,  1.00000],
               [ 1.00000,  1.00000,  ...,  1.00000,  1.00000]]]], shape=Shape([1, 1, 32, 32]), dtype=DataType::FLOAT32, layout=Layout::TILE)
ttnn.Tensor([[[[ 2.00000,  2.00000,  ...,  2.00000,  2.00000],
               [ 2.00000,  2.00000,  ...,  2.00000,  2.00000],
               ...,
               [ 2.00000,  2.00000,  ...,  2.00000,  2.00000],
               [ 2.00000,  2.00000,  ...,  2.00000,  2.00000]]]], shape=Shape([1, 1, 32, 32]), dtype=DataType::FLOAT32, layout=Layout::TILE)

```

We can visualize this tensor to see how it's stored in host-memory.

```py
ttnn.visualize_tensor(mesh_tensor)
```

<img src="images/image6_host_tensor_vis.png" style="width:500px;"/>

Let's now transfer to device:

```py
> mesh_tensor = ttnn.to_device(mesh_tensor, mesh_device)
> mesh_tensor

device_id:0
ttnn.Tensor([[[[ 1.00000,  1.00000,  ...,  1.00000,  1.00000],
               [ 1.00000,  1.00000,  ...,  1.00000,  1.00000],
               ...,
               [ 1.00000,  1.00000,  ...,  1.00000,  1.00000],
               [ 1.00000,  1.00000,  ...,  1.00000,  1.00000]]]], shape=Shape([1, 1, 32, 32]), dtype=DataType::FLOAT32, layout=Layout::TILE)
device_id:1
ttnn.Tensor([[[[ 2.00000,  2.00000,  ...,  2.00000,  2.00000],
               [ 2.00000,  2.00000,  ...,  2.00000,  2.00000],
               ...,
               [ 2.00000,  2.00000,  ...,  2.00000,  2.00000],
               [ 2.00000,  2.00000,  ...,  2.00000,  2.00000]]]], shape=Shape([1, 1, 32, 32]), dtype=DataType::FLOAT32, layout=Layout::TILE)

```

We now see that the following:

- 32x32 chunk with elements of 1.0 is residing in Device 0 DRAM
- 32x32 chunk with elements of 2.0 is residing in Device 1 DRAM

We can also visualize this tensor distributed across our MeshDevice. The visualization will color devices that have shards resident to the device. For a strategy that uses replication to distribute a tensor across our MeshDevice, replicated shards will have the same color mapped to them.

```py
ttnn.visualize_tensor(mesh_tensor)
```

<img src="images/image7_device_tensor_vis.png" style="width:500px;"/>

## 4. Single-Program Multiple Device

### 4.1 Execution Model

TT-NN uses a Single-Program Multiple-Device (SPMD) technique to parallelize computations across multiple devices. This approach allows the same computation to run on multiple devices simultaneously, each operating on different portions of the input data.

### 4.2 Single Device to Multiple Device Execution

#### 4.2.1 Single Device Execution

Let's run a simple gelu operation on a single-device:

```py
# Open a single device
device = ttnn.open_device(0)

# Create test tensor of data
torch_tensor = torch.rand((1,1,32,32), dtype=torch.bfloat16)

# Convert to ttnn.Tensor, tilize and move onto device DRAM
ttnn_tensor = ttnn.from_torch(
    torch_input_tensor,
    layout=ttnn.TILE_LAYOUT,
    device=device,
)

# Execute operation on-device
output_tensor = ttnn.gelu(ttnn_tensor)
```

####

#### 4.2.2 Mesh Device Execution

```py
# Open MeshDevice
mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1,4))

# Create test tensor of data; 4 chunks of 32x32
torch_tensor = torch.rand((1,1,32,128), dtype=torch.bfloat16)

# Convert to ttnn.Tensor, tilize and move onto devices across mesh DRAM
ttnn_tensor = ttnn.from_torch(
    torch_input_tensor,
    layout=ttnn.TILE_LAYOUT,
    device=mesh_device,
)

# Invoke ttnn.gelu on each of the devices in the mesh
output_tensor = ttnn.gelu(ttnn_tensor)

```

<!-- ![image1](images/image2.png){width=10 height=10} -->
<img src="images/image2.png" style="width:500px;"/>

*Figure 4: Parallel execution of gelu operation on 4 devices*


## 5. MeshDevice and Collective Communication Library (CCL)

The Collective Communication Library (CCL) provides a set of operations for efficient device-to-device communication in a MeshDevice. These operations are used as building blocks for implementing tensor-parallel and other distributed computing strategies.

### 5.1 CCL Operations

CCL supports several collective operations, including:

1. All-Gather (Ring, Line)
2. Reduce-Scatter (Ring)
3. All-Reduce (planned)
4. Send/Receive (planned)

### 5.2 All-Gather


The All-Gather operation is a fundamental collective communication primitive used to aggregate data from all participating devices and makes the aggregated result available on each device.

Each device in the MeshDevice begins with a local tensor. The all-gather operation effectively "gathers" these local tensors along some specified dimension(s), with each device receiving a full copy of the fully gathered/concatenated tensor. The concatenation order reflects the position of the device in the ring all-gather.


#### 5.2.1 Programming Example: All-Gather (Ring)

Let's see an example of how to use the Ring All-Gather operation:


<img src="images/image4_ring_all_gather.png" style="width:500px;"/>

*Figure 5: Ring All-Gather execution on 2x4 MeshDevice*

```py
import ttnn

mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(2, 4))

# Construct test tensor of data; 8 chunks of 32x32
torch_tensor = torch.rand((1,1,32,256), dtype=torch.bfloat16)

# Convert to ttnn.Tensor, tilize and move onto devices across mesh DRAM
mesh_tensor = ttnn.from_torch(
    torch_tensor,
    layout=ttnn.TILE_LAYOUT,
    device=mesh_device,
    mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=3),
)

# Execute All-Gather on the tensor; `num_links=1` specifies the number of ethernet links to use
output_tensor = ttnn.all_gather(mesh_tensor, dim=3, num_links=1)
```


#### 5.2.2 Programming Example: All-Gather (Line)

Here we issue a Line All-Gather operation along the cluster-axis 0 (y-dimension), where the y-dimension is the height of the cluster.
This kicks off four parallel CCL Line All-Gather operations, one for each column in the cluster. Each "line" is a list of two devices.

<img src="images/image5_line_all_gather.png" style="width:500px;"/>

*Figure 6: Line All-Gather execution on 2x4 MeshDevice*

The result tensor for each device in the column is the concatenation in `dim=3` for each device in the column. The per-device tensor shape is `[1, 1, 32, 32]` before the operation and `[1, 1, 32, 64]` after the operation.

```py
import ttnn

mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(2, 4))

# Construct test tensor of data; 8 chunks of 32x32
torch_tensor = torch.rand((1,1,32,256), dtype=torch.bfloat16)

# Convert to ttnn.Tensor, tilize and move onto devices across mesh DRAM
mesh_tensor = ttnn.from_torch(
    torch_input_tensor,
    layout=ttnn.TILE_LAYOUT,
    device=mesh_device,
    mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=3),
)

# Execute Line All-Gather on the tensor
output_tensor = ttnn.all_gather(
    mesh_tensor,
    dim=3,
    cluster_axis=0,
    mesh_device=mesh_device,
    topology=ttnn.Topology.Linear,
)

ttnn.close_mesh_device(mesh_device)
```


## 6. Programming Mesh of Devices Using Data Parallel

This tutorial demonstrates how to convert a model running on a single-device to multiple devices using data-parallel strategy. Using data parallel can be a good strategy to scale to multiple devices when your model fits on a single-device.

At a high-level, using Data Parallel to scale performance to N-devices involves:

1. Shard the input activation data along the batch dimension into N-chunks
2. Replicate the model weights for each of the N-devices

Effectively, each device contains a replica of the model and is responsible for computing a shard of the final output tensor.

###

### 6.1 Data Parallel Programming Example:

Let's start by creating a simple MLP model in TT-NN on a single-device and scale to multiple devices by using data-parallel. We'll use pretrained weights and compare against torch for validation:

1. **Create a TT-NN Falcon-7B MLP Module implementation**

```py
import ttnn

class TtFalconMLP:
    def __init__(self, parameters):
        super().__init__()
        self.dense_h_to_4h_weights = parameters.dense_h_to_4h.weight
        self.dense_4h_to_h_weights = parameters.dense_4h_to_h.weight

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        ff1_linear: ttnn.Tensor = ttnn.linear(x, self.dense_h_to_4h_weights)
        gelu = ttnn.gelu(ff1_linear)
        ff2_linear: ttnn.Tensor = ttnn.linear(gelu, self.dense_4h_to_h_weights)

        return ff2_linear
```

2. **Instantiate torch model for comparison**

```py
# Load Falcon MLP model from huggingface
config = transformers.FalconConfig.from_pretrained("tiiuae/falcon-7b-instruct")
model = transformers.models.falcon.modeling_falcon.FalconMLP(config).eval()
```

3. **Execute TT-NN Falcon-7B MLP Module on a single Tenstorrent Device**

```py

# Device Initialization
device = ttnn.open_device(0)

# Convert torch input activations to ttnn.Tensor and move to device

# Initialize hidden states
batch_size, sequence_length = 1, 128
torch_hidden_states = (torch.rand(batch_size, 1, sequence_length, config.hidden_size, dtype=torch.float32) * 2) - 1

hidden_states = ttnn.from_torch(torch_hidden_states, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

# Preprocess model parameters: loop through model parameters, convert to ttnn tensors, and move to device
parameters = ttnn.model_preprocessing.preprocess_model_parameters(
    initialize_model=lambda: model,
    device=device,
)

# Initialize Model
ttnn_model = TtFalconMLP(parameters) # Initialize Model

# Run Model
ttnn_output = ttnn_model(hidden_states)

assert_with_pcc(
    torch_model.forward(torch_hidden_states),
    ttnn.to_torch(ttnn_output),
    0.98
)

ttnn.close_device(device)
```

4. **Executing TT-NN Falcon-7B MLP Module on MeshDevice with Data Parallel**

Full code example can be found in `tests/ttnn/distributed/test_data_parallel_example_TG.py`

```py
# Load Falcon MLP model from huggingface
config = transformers.FalconConfig.from_pretrained("tiiuae/falcon-7b-instruct")
model = transformers.models.falcon.modeling_falcon.FalconMLP(config).eval()

# Initialize hidden states
batch_size, sequence_length = 4, 128
torch_hidden_states = (torch.rand(batch_size, 1, sequence_length, config.hidden_size, dtype=torch.float32) * 2) - 1
torch_output = model.forward(torch_hidden_states)

# Device Initialization
mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(y=1, x=4))

# Shard input activations on batch dimension to devices in the mesh
with ttnn.distribute(ttnn.ShardTensorToMesh(mesh_device, dim=0)):
    hidden_states = ttnn.from_torch(
        torch_hidden_states,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
    )

# Replicate model parameters to devices in the mesh
with ttnn.distribute(ttnn.ReplicateTensorToMesh(mesh_device)):
    parameters = ttnn.model_preprocessing.preprocess_model_parameters(
        initialize_model=lambda: model,
        device=mesh_device,
    )

# Initialize Model
ttnn_model = TtFalconMLP(parameters)
ttnn_output = ttnn_model(hidden_states)

with ttnn.distribute(ttnn.ConcatMeshToTensor(mesh_device, dim=0)):
    assert_with_pcc(torch_output, ttnn.to_torch(ttnn_output), 0.98)
```


## 7. Programming Mesh of Devices Using Tensor Parallel

When your model is too large to fit on a single device, tensor parallelism provides a solution by sharding the model parameters across the distributed SRAM/DRAM of multiple devices. Each device then performs computations on its portion of the data, with communication between devices occurring as needed to aggregate results via CCL primitives.

Key benefits of tensor parallelism include:
1. Ability to run larger models that exceed single-device memory capacity
2. Potential for increased computational throughput
3. Efficient utilization of multi-device systems

###

### 7.1 Tensor Parallel Programming Example:

Let's re-use the same example as the data-parallel example above, but this time we'll run it with tensor-parallel. In this example, we'll implement a simple tensor-parallel where we shard all model parameters on the width dimension.


1. **Create a TT-NN Falcon-7B MLP Module implementation**

```py
import ttnn

class TtFalconMLP:
    def __init__(self, parameters):
        super().__init__()
        self.dense_h_to_4h_weights = parameters.dense_h_to_4h.weight
        self.dense_4h_to_h_weights = parameters.dense_4h_to_h.weight

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        ff1_linear: ttnn.Tensor = ttnn.linear(x, self.dense_h_to_4h_weights)
        gelu = ttnn.gelu(ff1_linear)

        # Invoke CCL Ring All-Gather on gelu before passing to ff2_linear
        gelu = ttnn.all_gather(gelu, dim=3, num_links=1)

        ff2_linear: ttnn.Tensor = ttnn.linear(gelu, self.dense_4h_to_h_weights)

        return ff2_linear
```

2. **Instantiate torch model for comparison**

```py
# Load Falcon MLP model from huggingface
config = transformers.FalconConfig.from_pretrained("tiiuae/falcon-7b-instruct")
model = transformers.models.falcon.modeling_falcon.FalconMLP(config).eval()
```

3. **Executing TT-NN Falcon-7B MLP Module on MeshDevice with Tensor Parallel**

See full code example in `tests/ttnn/distributed/test_tensor_parallel_example_T3000.py`

```py
# Initialize hidden states
batch_size, sequence_length = 1, 256
torch_hidden_states = (torch.rand(batch_size, 1, sequence_length, config.hidden_size, dtype=torch.float32) * 2) - 1
torch_output = model.forward(torch_hidden_states)

# Device Initialization
mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(2,4))

# Initialize input activations on all devices in the mesh
# Alternatively, we can shard the input activations on the height dimension and
# subsequently invoke all-gather on the height dimension to form a complete tensor per device.
with ttnn.distribute(ttnn.ReplicateTensorToMesh(mesh_device)):
    hidden_states = ttnn.from_torch(
        torch_hidden_states,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
    )

# Shard model parameters on width dimension to devices in the mesh
with ttnn.distribute(ttnn.ShardTensorToMesh(t3k_mesh_device, dim=-1)):
    parameters = ttnn.model_preprocessing.preprocess_model_parameters(
        initialize_model=lambda: model,
        device=t3k_mesh_device,
    )

# Initialize Model
ttnn_model = TtFalconMLP(parameters)

# Run Model
ttnn_output = ttnn_model(hidden_states)

with ttnn.distribute(ttnn.ConcatMeshToTensor(mesh_device, dim=3)):
    assert_with_pcc(torch_output, ttnn.to_torch(ttnn_output), 0.98)
```

## 8. Programming Mesh of Devices Using Hybrid Tensor and Data Parallel

### 8.1 Llama-3.1 70B Hybrid Tensor and Data Parallel

<img src="images/llama-3.1-70b-hybrid-dp-tp.png" style="width:500px;"/>

*Figure 7: Llama-3.1 70B model mapped onto QuietBox (Wormhole) and Galaxy systems.*


### 8.2 Llama-3.1 70B Performance Scaling

| System                | Batch Size  | tok/s/u | tok/s  |
|-----------------------|-------------|---------|--------|
| QuietBox (Wormhole)   | 32          | 15.1    | 483.2  |
| Galaxy                | 128         | 14.3    | 1835.5 |

*Table 1: Llama-3.1 70B model scaling from QuietBox (Wormhole) to Galaxy. Tokens per second (toks/s) throughput scales near-linear (3.8x) as we tile our model replicas across the Galaxy mesh.*


### 8.3 Hybrid Tensor and Data Parallel Programming Example

This sections explains how to employ hybrid tensor and data parallelism by tiling a submesh across a larger mesh.

#### 8.3.1 Overview of Changes

The main changes involve:

1. Creating multiple submeshes from the main mesh
2. Running the model on each submesh
3. Capturing and replaying a trace across all submeshes in parallel

#### 8.3.2 Key Components

These three components are used to achieve linear scaling of performance as we tile our model replicas across the mesh.
See `models/demos/t3000/llama2_70b/tests/test_llama_perf_decode.py::test_Llama_perf_hybrid_data_tensor_parallel` for full example.

1. Submesh Creation

```py
    submesh_devices: List[ttnn.MeshDevice] = mesh_device.create_submeshes(ttnn.MeshShape(2, 4))
```

2. Compile & Run the Model on Each Submesh

```python
    for submesh_device in submesh_devices:
        model.forward(activations, device=submesh_device, ...)
```

3. Capture Model Trace: See [Advanced Performance Optimizations For Models](../AdvancedPerformanceOptimizationsForModels/AdvancedPerformanceOptimizationsForModels.md) guide for more details on how to capture and replay a trace.

```python
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)

    # Run the existing model on each submesh
    for submesh_device in submesh_devices:
        model.forward(activations, device=submesh_device, ...)

    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
```

4. Execute Model Trace:

```python
    # Execute Model Trace across all submeshes in parallel
    ttnn.execute_trace(mesh_device, trace_id, blocking=False)
    ttnn.release_trace(mesh_device, trace_id)
```

APIs will be further refined in future releases. A proposal for refined set of APIs can be found [here](https://github.com/tenstorrent/tt-metal/issues/13852).


## 9. TT-NN MeshDevice vs PyTorch Multi-Device Programming

### 9.1 Overview of Multi-Device Handling

This section compares TT-NN's MeshDevice with PyTorch's approach to **single-node multi-device** programming. Note that PyTorch Distributed (torch.distributed) is designed for multi-node systems and is not the appropriate comparison here. Similarly, TT-NN's multi-node MeshDevice support is currently under development and not covered in this comparison.

**MeshDevice in TT-NN (Single-Node):**

MeshDevice provides a unified abstraction for managing multiple devices within a single node as a single logical entity. It bundles devices together for coordinated execution, enabling:
- Automatic kernel compilation broadcasting across devices
- Efficient data distribution and replication
- Native runtime-level support for mesh topology-aware command dispatch
- Explicit control over data distribution and collective communication operations

**PyTorch Single-Node Multi-GPU Approach:**

In PyTorch, single-node multi-GPU programming requires developers to either:
- **Manual Management**: Explicitly place tensors on specific devices (`cuda:0`, `cuda:1`, etc.) and manually orchestrate data movement and synchronization between devices
- **DataParallel**: Use `torch.nn.DataParallel` which automatically splits data across GPUs but operates in a single process with potential GIL bottlenecks
- **DistributedDataParallel**: While primarily designed for multi-node, DDP can be used on single-node with multiple processes (one per GPU) for better performance than DataParallel

For manual multi-GPU management in a single process, PyTorch provides no built-in abstractions for:
- Coordinated command dispatch across devices
- Efficient collective communication operations
- Automatic kernel or data broadcasting

### 9.2 Code Comparison: Matrix Multiplication with All-Gather

The following examples demonstrate how to perform matrix multiplication followed by an all-gather operation across multiple devices.

<table>
<tr>
<th>TT-NN MeshDevice</th>
<th>PyTorch Single-Node Multi-GPU</th>
</tr>
<tr>
<td>


```python
import ttnn
import torch

# Open a 1x2 MeshDevice
mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 2))

# Create input tensors
# Note that ttnn.rand can be used here, however this example
# demonstrates the use of mesh distribution APIs for generic inputs
torch_input_a = torch.randn(1, 1, 128, 128, dtype=torch.bfloat16)
torch_input_b = torch.randn(1, 1, 128, 256, dtype=torch.bfloat16)

# Replicate input A and shard input B (each device gets 128x128)
# Supplying `mesh_device` argument implicitly transfers the tensor to device
input_a = ttnn.from_torch(
    torch_input_a,
    layout=ttnn.TILE_LAYOUT,
    device=mesh_device,
    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
)

input_b = ttnn.from_torch(
    torch_input_b,
    layout=ttnn.TILE_LAYOUT,
    device=mesh_device,
    mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=3),
)

# Perform matrix multiplication - done in parallel on 2 devices
output = ttnn.matmul(input_a, input_b)

# All-gather to collect results from all devices
# The gathered tensor stores a copy of the concatenated tensor on each device
gathered = ttnn.all_gather(output, dim=3, cluster_axis=0)

```

</td>
<td>


```python
import torch

# Check available GPUs
num_gpus = torch.cuda.device_count()  # Assumes 2 GPUs
devices = [torch.device(f'cuda:{i}') for i in range(num_gpus)]

# Create input tensors
torch_input_a = torch.randn(1, 1, 128, 128, dtype=torch.bfloat16)
torch_input_b = torch.randn(1, 1, 128, 256, dtype=torch.bfloat16)

# Replicate input A on all devices
replicated_a = [torch_input_a.to(device) for device in devices]

# Manually shard input B across devices (columns)
shard_size = 256 // num_gpus
sharded_b = []
for i in range(num_gpus):
    start_idx = i * shard_size
    end_idx = start_idx + shard_size
    shard = torch_input_b[:, :, :, start_idx:end_idx].to(devices[i])
    sharded_b.append(shard)

# Perform matrix multiplication on each device
outputs = []
for i in range(num_gpus):
    with torch.cuda.device(devices[i]):
        output = torch.matmul(replicated_a[i], sharded_b[i])
        outputs.append(output)

# Manual all-gather: copy all outputs to first device
gathered = []
for output in outputs:
    gathered.append(output.to(devices[0]))

# Concatenate gathered outputs along the column dimension to complete all-gather
final_result = torch.cat(gathered, dim=3)
```

</td>
</tr>
</table>
