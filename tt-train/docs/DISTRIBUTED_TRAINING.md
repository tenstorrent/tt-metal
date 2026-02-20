# Distributed Training Guide

This guide explains how to configure and run distributed training with TTML, covering Mesh Graph Descriptors (MGD files), device configuration, and parallelism strategies.

## Overview

TTML supports three types of distributed parallelism:
- **Data Parallel (DP/DDP)**: Replicate model across devices, shard data, synchronize gradients
- **Tensor Parallel (TP)**: Shard model parameters across devices, gather/reduce as needed
- **Pipeline Parallel (PP)**: Shard layers sequentially across devices

These can be combined (e.g., TP + DDP) to scale training across large device meshes.

## Mesh Graph Descriptors (MGD Files)

### What are MGD Files?

Mesh Graph Descriptors (MGD) are `.textproto` files that define the hardware topology and connectivity of your distributed system. They specify:
- Device topology (how devices are arranged in a mesh)
- Host topology (how hosts are connected)
- Inter-device connectivity and channels

### Setting MGD Files via Environment Variable

MGD files are specified using the `TT_MESH_GRAPH_DESC_PATH` environment variable:

```bash
export TT_MESH_GRAPH_DESC_PATH="/path/to/your/mesh_graph_descriptor.textproto"
```

### Where to Find MGD Files

Default MGD files are located in the tt-metal repository:

```
$TT_METAL_HOME/tests/tt_metal/tt_fabric/custom_mesh_descriptors/
```

Common examples include:
- `galaxy_1x32_mesh_graph_descriptor.textproto` - For 32-device Galaxy setups (1×32 mesh)
- `t3k_1x8_mesh_graph_descriptor.textproto` - For 8-device T3K setups (1×8 mesh)

You can also find additional MGD files in:
- `$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/` - Standard mesh descriptors
- `$TT_METAL_HOME/tt-train/sources/examples/python/multihost/*/configurations/` - Example configurations for multihost training

### Automatic MGD Selection

If `TT_MESH_GRAPH_DESC_PATH` is not set, the training script will automatically select an MGD file based on the number of devices:
- **8 devices**: Uses `t3k_1x8_mesh_graph_descriptor.textproto`
- **32 devices**: Uses `galaxy_1x32_mesh_graph_descriptor.textproto`

This automatic selection requires `TT_METAL_HOME` to be set.

### MGD File Format

MGD files use Protocol Buffer text format. Here's an example for a 1×32 Galaxy mesh:

```proto
mesh_descriptors {
  name: "M0"
  arch: WORMHOLE_B0
  device_topology { dims: [ 1, 32 ] }
  host_topology   { dims: [ 1, 1 ] }
  channels { count: 4 policy: RELAXED }
}

top_level_instance { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
```

**Important**: The `device_topology` dimensions in the MGD file define the physical device arrangement, which must be compatible with your logical mesh shape.

## Device Configuration

### Mesh Shape Matching

**Critical**: The mesh device shape specified in your MGD file must match the mesh shape you use when opening the device. The mesh shape in your training config's `device_config` determines how devices are logically organized.

For example, if your MGD defines a `device_topology { dims: [ 4, 8 ] }` (32 devices total), your `mesh_shape` should be compatible with this arrangement, such as:
- `[4, 8]` - 4 DP groups × 8 TP devices is fine
- `[1, 32]` - Single group with 32 TP devices will hang or crash with an error at open_device
- `[32, 1]` - 32 DP groups × 1 device each will also hang or crash with an error at open_device

### Device Config in Training YAML

In your training configuration YAML file, specify the device configuration:

```yaml
device_config:
  enable_tp: true      # Enable tensor parallelism
  enable_ddp: true     # Enable data parallelism
  mesh_shape: [4, 8]   # 4 DP groups × 8 TP devices = 32 devices
  device_ids: []       # Optional: specific device IDs to use
```

### Parallelism Strategy Selection

In the device config, you can specify which parallelism strategies to use:

- **`enable_tp`**: Enable tensor parallelism (shard model parameters)
- **`enable_ddp`**: Enable data parallelism (replicate model, shard data)
- **`enable_pp`**: Enable pipeline parallelism (shard layers sequentially)

### Mesh Shape Determination

In the NanoGPT training script (and other training scripts), the mesh shape is automatically determined from the `device_config` section of your YAML configuration:

```cpp
// From device_config in YAML
DeviceConfig config = parse_device_config(yaml_config);
MeshShape mesh_shape = config.mesh_shape;  // e.g., [4, 8]

// Open device with this mesh shape
initialize_device(mesh_shape, device_ids);
```

The mesh shape must match the total number of devices available and be compatible with your MGD file's device topology.

## Parallelism Axis Assignment

### Automatic Axis Determination

The parallelism axes are **automatically determined** based on the order you enable parallelism strategies in your device config:

1. **Data Parallel (DP) axis**: Always assigned to **axis 0** (first dimension) if `enable_ddp` is true
2. **Tensor Parallel (TP) axis**: Assigned to **axis 1** (second dimension) if `enable_tp` is true

This means:
- For `mesh_shape: [4, 8]` with both DDP and TP enabled:
  - DP uses axis 0 → 4 DP groups
  - TP uses axis 1 → 8 TP devices per group
- For `mesh_shape: [1, 32]` with only TP enabled:
  - TP uses axis 0 → 32 TP devices
- For `mesh_shape: [32, 1]` with only DDP enabled:
  - DP uses axis 0 → 32 DP groups

### Implementation Details

The axis assignment happens in `ParallelismContext`:

```cpp
uint32_t axis = 0;
if (config.enable_ddp) {
    m_ddp_axis = axis++;  // DP gets axis 0
    m_num_ddp_devices = mesh_device.shape()[m_ddp_axis.value()];
}
if (config.enable_tp) {
    m_tp_axis = axis++;    // TP gets axis 1 (if DP was enabled)
    m_num_tp_devices = mesh_device.shape()[m_tp_axis.value()];
}
```

**Key constraint**: The number of enabled parallelism strategies must equal the number of mesh shape dimensions:
- 1D mesh `[N]`: Can use either DP or TP (but not both)
- 2D mesh `[M, N]`: Can use both DP and TP together

## Complete Example

Here's a complete example for training Llama 8B with TP + DDP on a 32-device Galaxy:

### 1. Training Configuration (`training_llama8b_tp_ddp_galaxy.yaml`)

```yaml
training_config:
  project_name: "tt_train_llama8b_tp_dp"
  batch_size: 4  # Total batch size across all DP groups
  model_config: "configs/model_configs/llama8b_galaxy_tp_dp.yaml"

device_config:
  enable_tp: true
  enable_ddp: true
  mesh_shape: [4, 8]  # 4 DP groups × 8 TP devices = 32 devices
```

### 2. Set MGD File

```bash
export TT_METAL_HOME=/path/to/tt-metal
export TT_MESH_GRAPH_DESC_PATH=$TT_METAL_HOME/tests/tt_metal/tt_fabric/custom_mesh_descriptors/galaxy_1x32_mesh_graph_descriptor.textproto
```

Or let it auto-select (for 32 devices):
```bash
export TT_METAL_HOME=/path/to/tt-metal
# TT_MESH_GRAPH_DESC_PATH will be auto-set to galaxy_1x32_mesh_graph_descriptor.textproto
```

### 3. Run Training

```bash
./build/examples/nano_gpt/nano_gpt \
    --config configs/training_configs/training_llama8b_tp_ddp_galaxy.yaml
```

### 4. What Happens

1. The script reads `mesh_shape: [4, 8]` from `device_config`
2. It opens a device with mesh shape `[4, 8]` (32 devices total)
3. The MGD file must define a compatible device topology (e.g., `device_topology { dims: [ 4, 8 ] }`)
4. Parallelism context is initialized:
   - DP axis = 0 → 4 DP groups
   - TP axis = 1 → 8 TP devices per group
5. Model parameters are sharded across TP devices
6. Data batches are distributed across DP groups

## Troubleshooting

### MGD Shape Mismatch

**Error**: Device initialization fails or mesh shape doesn't match

**Solution**: Ensure the `device_topology` in your MGD file is compatible with your `mesh_shape`. For a `mesh_shape: [4, 8]`, your MGD should define a topology that can accommodate 32 devices arranged as 4×8.

### Parallelism Axis Mismatch

**Error**: "The number of parallelization axes must be equal to the number of mesh shape dimensions"

**Solution**:
- For 1D mesh `[N]`: Enable only one of `enable_ddp` or `enable_tp`
- For 2D mesh `[M, N]`: You can enable both `enable_ddp` and `enable_tp`

### MGD File Not Found

**Error**: MGD file path not found or `TT_METAL_HOME` not set

**Solution**:
- Set `TT_METAL_HOME` to your tt-metal repository root
- Explicitly set `TT_MESH_GRAPH_DESC_PATH` to your MGD file path
- Or ensure you're using 8 or 32 devices for automatic selection

### ParallelismContext Not Initialized

**Error**: `"ParallelismContext is not initialized."`

**Cause**: Code is trying to access the parallelism context (e.g., calling `get_parallelism_context()`, `get_tp_size()`, `get_ddp_size()`) before it has been initialized.

**Solution**:
1. Ensure you call `initialize_parallelism_context()` **after** opening the device but **before** using any distributed features:
   ```cpp
   // Correct order:
   initialize_device(mesh_shape, device_ids);
   ctx.initialize_parallelism_context({.enable_ddp = true, .enable_tp = true});
   // Now you can use get_parallelism_context()
   ```

2. In training scripts, the parallelism context should be initialized right after device initialization:
   ```cpp
   auto *device = &ttml::autograd::ctx().get_device();
   ttml::autograd::ctx().initialize_parallelism_context(
       {.enable_ddp = device_config.enable_ddp, .enable_tp = device_config.enable_tp});
   ```

3. If using distributed models (e.g., `DistributedLlama`), ensure the parallelism context is initialized before model construction, as the model may query parallelism settings during initialization.

### ParallelismContext Already Initialized

**Error**: `"ParallelismContext is already initialized."`

**Cause**: `initialize_parallelism_context()` is being called more than once. The parallelism context can only be initialized once per training session.

**Solution**:
1. **Check for duplicate initialization**: Ensure `initialize_parallelism_context()` is called exactly once in your code flow. Common causes:
   - Calling it in both initialization and model construction
   - Calling it in a loop or multiple code paths
   - Calling it in both C++ and Python if using mixed code

2. **Typical pattern**: Initialize once after device opening:
   ```cpp
   // Open device
   initialize_device(mesh_shape, device_ids);

   // Initialize parallelism context ONCE
   ttml::autograd::ctx().initialize_parallelism_context(config);

   // Use context throughout training
   // ... rest of training code ...
   ```

4. **In Python**: Same principle applies:
   ```python
   autograd_ctx = ttml.autograd.AutoContext.get_instance()
   autograd_ctx.open_device([mesh_rows, mesh_cols])

   # Initialize ONCE
   autograd_ctx.initialize_parallelism_context(
       DistributedConfig(enable_ddp=True, enable_tp=True)
   )
   ```

### Physical Chip ID Not Found / Topology Not Configured

**Error**:
```
TT_FATAL: Physical chip id X not found in control plane chip mapping.
You are calling for a chip outside of the fabric cluster.
Check that your mesh graph descriptor specifies the correct topology
```

**Cause**: The hardware topology has not been configured on the system. This error occurs when the fabric control plane cannot find the physical chips specified in your MGD file or mesh configuration.

**Solution**:
1. **Install and configure topology**:
   ```bash
   pip install tt-topology
   tt-topology -l mesh
   ```
   This configures the hardware topology to match your mesh configuration.

2. **Verify topology matches your mesh**: Ensure the topology configuration matches the number of chips you're trying to use. For example:
   - If using 8 chips with `mesh_shape: [1, 8]`, ensure all 8 chips are configured
   - If using 32 chips with `mesh_shape: [4, 8]`, ensure all 32 chips are configured

3. **Check device configuration**: Some device configurations (e.g., n300) may default to using fewer chips than available. If you want to use all chips:
   - Check your device config YAML file
   - Update `mesh_shape` to match the number of chips you want to use
   - Example: Change `mesh_shape: [1, 2]` to `mesh_shape: [1, 8]` to use all 8 chips on an n300 configuration

**Important Notes**:
- **Graceful shutdown**: When terminating a training process (e.g., with Ctrl+C), the fabric may not exit gracefully. If you encounter errors on subsequent runs, reset the cards:
  ```bash
  tt-smi -r
  ```
  or
  ```bash
  tt-smi -glx_reset
  ```
  on a galaxy

- **Topology persistence**: The topology configuration should persist across reboots, but may need to be reconfigured if hardware changes or after system updates.

## Additional Resources

- **MGD Format Documentation**: `$TT_METAL_HOME/tt_metal/fabric/MGD_README.md`
- **Multihost Training Examples**: `$TT_METAL_HOME/tt-train/sources/examples/python/multihost/`
- **Device Config Examples**: `$TT_METAL_HOME/tt-train/configs/training_configs/`
- **TP+DP Example (Python)**: `$TT_METAL_HOME/tt-train/sources/examples/linear_regression_tp_dp/linear_regression_tp_dp.py` - Complete example demonstrating combined Tensor Parallelism and Data Parallelism with proper mesh configuration, parallelism context initialization, and distributed tensor mapping
