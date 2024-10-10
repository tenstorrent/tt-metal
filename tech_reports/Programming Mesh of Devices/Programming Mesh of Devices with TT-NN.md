# Programming Mesh of Devices with TT-NN

Author: Joseph Chu

## Contents

- [Programming Mesh of Devices with TT-NN](#programming-mesh-of-devices-with-tt-nn)
- [1. Overview](#1-overview)
- [2. MeshDevice](#2-meshdevice)
  - [2.1 System Topology](#21-system-topology)
  - [2.2 MeshDevice Management](#22-meshdevice-management)
    - [2.2.1 MeshDevice Initialization/Close](#221-meshdevice-initializationclose)
    - [2.2.2 MeshDevice Visualization](#222-meshdevice-visualization)
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

## 1. Overview

TT-NN library natively supports multi-device operations, enabling users to scale their single-device application code to multiple devices seamlessly. TT-NN employs a Single-Program Multiple-Device (SPMD) technique to parallelize a computation across a set of connected devices operating on different input data. This is achieved through a few key components:

- **MeshDevice**: This "virtual device" abstraction defines a logical 2-D mesh of connected physical devices. Operations that "run on device" are distributed through SPMD across all devices captured in the mesh.

- **Input Data Distribution**: Defines how input data resident in host-memory is distributed to DeviceMesh on-device memory. When operations are distributed to MeshDevice, the operation within a single-device scope works on its local input data.

- **Tensor**: Defines a N-dimensional matrix containing elements of a single data type. In a MeshDevice context, a Tensor, or colloquially referred to as MeshTensor, represents a collection of tensor shards distributed across devices in a 2D Mesh.


These concepts are key to understanding how we scale models using **Data-Parallel**, **Tensor-Parallel**, and **Hybrid Data + Tensor Parallel.**

## 2. MeshDevice

### 2.1 System Topology

A MeshDevice can be instantiated over a collection of physically connected devices. The supported configurations are N300 (1x2), T3000 (2x4), Galaxy (8x4).

With the N300 form-factor, it houses two wormhole chips. The host is connected to the "left" chip via PCIe and the "left" chip is connected to the "right" chip via two ethernet links. Each ethernet link has a 200 Gbps bi-directional bandwidth. For N300, one of the ethernet links connecting the "left" chip to the "right" chip is reserved for fast-dispatch. At the user-level, this means only a single ethernet link is made available for use. The N300 represents the smallest multi-device configuration that we can instantiate a MeshDevice over.

<!-- ![image1](images/image1.png){width=15 height=15} -->
<img src="../EthernetMultichip/images/t3000.png" style="width:500px;"/>

*Figure 1: T3000 System Topology. T3000 is composed of 4x N300 wormhole cards, totalling 8 wormhole chips, connected in a 2x4 mesh configuration. Each pair of wormhole-chips are connected via two ethernet links.*


<img src="../EthernetMultichip/images/TG.png" style="width:500px;"/>

*Figure 2: TG System Topology. TG is composed of 16x N300 wormhole cards, totalling 32 wormhole chips, connected in a 8x4 mesh configuration. Each pair of wormhole-chips are connected via four ethernet links.*


[tt-topology](https://github.com/tenstorrent/tt-topology) can be used to flash multiple wormhole cards on a system to a specific ethernet routing configuration (linear, ring, mesh) and used to visualize the organization of the chip layout.

<img src="images/image3.png" style="width:500px;"/>

*Figure 3: T3000 Chip Layout dumped from tt-topology*


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

# Open our 1x2 MeshDevice
mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 2))

# Initialize a torch tensor
torch_tensor = torch.zeros(1, 1, 32, 64)
torch_tensor[..., 0:32] = 1.0
torch_tensor[..., 32:64] = 2.0

# Convert to ttnn.Tensor; MeshTensor holds buffers to two shards in host-memory
mesh_tensor: ttnn.Tensor = ttnn.from_torch(
    torch_tensor,
    mesh_mapper=ttnn.ShardTensorToMesh(device_mesh, dim=3),
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

Let's now transfer to device:

```py
> mesh_tensor = ttnn.to_device(mesh_tensor, device_mesh)
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

- 32x32 chunk with elements of 1.0 is residing in Device 11 DRAM
- 32x32 chunk with elements of 2.0 is residing in Device 10 DRAM

We can also visualize this tensor distributed across our MeshDevice. The visualization will color devices that have shards resident to the device.

```py
ttnn.visualize_mesh_device(mesh_device, tensor=mesh_tensor)

>
                  DeviceMesh(rows=1, cols=2):
┌──────────────────────────────┬──────────────────────────────┐
│         Dev. ID: 11          │         Dev. ID: 10          │
│            (0, 0)            │            (0, 1)            │
│  ttnn.Shape([1, 1, 32, 32])  │  ttnn.Shape([1, 1, 32, 32])  │
└──────────────────────────────┴──────────────────────────────┘
```

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

The Collective Communication Library (CCL) provides a set of operations for efficient device-to-device communication in a MeshDevice. See the [CCL Developer Guide](../EthernetMultichip/CclDeveloperGuide.md) for more comprehensive coverage. These operations are used as building blocks for implementing tensor-parallel and other distributed computing strategies.

### 5.1 CCL Operations

CCL supports several collective operations, including:

1. All-Gather (Ring, Line)
2. Reduce-Scatter (Ring)
3. All-Reduce (planned)
4. Send/Receive (planned)

Our library of supported operations can be found [here](../EthernetMultichip/CclDeveloperGuide.md#op-list-op-list).

### 5.2 All-Gather


The All-Gather operation is a fundamental collective communication primitive used to aggregate data from all participating devices and makes the aggregated result available on each device.

Each device in the MeshDevice begins with a local tensor. The all-gather operation effectively "gathers" these local tensors along some specified dimension(s), with each device receiving a full copy of the fully gathered/concatenated tensor. The concatenation order reflects the position of the device in the ring all-gather.


#### 5.2.1 Programming Example: All-Gather (Ring)

Let's see an example of how to use the Ring All-Gather operation:


<img src="images/image4_ring_all_gather.png" style="width:500px;"/>

*Figure 5: Ring All-Gather execution on 2x4 MeshDevice*

```py
import ttnn

mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(2, 4), mesh_type=ttnn.MeshType.Ring)

# Construct test tensor of data; 8 chunks of 32x32
torch_tensor = torch.rand((1,1,32,128), dtype=torch.bfloat16)

# Convert to ttnn.Tensor, tilize and move onto devices across mesh DRAM
mesh_tensor = ttnn.from_torch(
    torch_input_tensor,
    layout=ttnn.TILE_LAYOUT,
    device=mesh_device,
    mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=3),
)

# Execute All-Gather on the tensor; `num_links=1` specifies the number of ethernet links to use
output_tensor = ttnn.all_gather(mesh_tensor, dim=3, num_links=1)
```


#### 5.2.2 Programming Example: All-Gather (Line)

This time, we'll issue the CCL Line All-Gather operation along the cluster y-axis:

<img src="images/image5_line_all_gather.png" style="width:500px;"/>

*Figure 6: Line All-Gather execution on 2x4 MeshDevice *

```py
import ttnn

mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(2, 4), mesh_type=ttnn.MeshType.Ring)

# Construct test tensor of data; 8 chunks of 32x32
torch_tensor = torch.rand((1,1,32,128), dtype=torch.bfloat16)

# Convert to ttnn.Tensor, tilize and move onto devices across mesh DRAM
mesh_tensor = ttnn.from_torch(
    torch_input_tensor,
    layout=ttnn.TILE_LAYOUT,
    device=mesh_device,
    mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=3),
)

# Execute Line All-Gather on the tensor
output_tensor = ttnn.all_gather(mesh_tensor, dim=3, cluster_axis=0, mesh_device=mesh_device, topology=ttnn.Topology.Linear)
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
mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(2,4), mesh_type=ttnn.MeshType.Ring)

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
