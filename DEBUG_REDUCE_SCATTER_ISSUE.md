# Debugging "forward_coord or backward_coord is null" Error

## Root Cause Analysis

The error occurs in `reduce_scatter_program_factory.cpp` at lines 88-90 when both `forward_coordinate` and `backward_coordinate` are null.

### What's Happening

The `get_physical_neighbor_from_physical_coord` function:
1. Computes a `potential_neighbor` using `physical_coord.get_neighbor(tensor.device()->shape(), ...)`
2. Searches for that neighbor in `tensor.device_storage().coords`
3. Returns `nullopt` if the neighbor doesn't exist in `device_coords`

### Likely Causes

**Most Common Issue: Mismatch between device shape and tensor's device coordinates**

When you distribute a tensor with `mesh_shape_override = {1, 4}`, the tensor's `device_storage.coords` are created based on that override shape. However, `get_neighbor` uses `tensor.device()->shape()` which might be different.

## Diagnostic Steps

Add debug logging right before the fatal check:

```cpp
log_debug(tt::LogOp, "Mesh coordinate: {}", mesh_coordinate);
log_debug(tt::LogOp, "Device shape: {}", tensor_args.input_tensor.device()->shape());
log_debug(tt::LogOp, "Device coords count: {}", tensor_args.input_tensor.device_storage().coords.size());
for (const auto& coord : tensor_args.input_tensor.device_storage().coords) {
    log_debug(tt::LogOp, "  Device coord: {}", coord);
}
log_debug(tt::LogOp, "Cluster axis: {}", operation_attributes.cluster_axis.has_value() ?
    std::to_string(operation_attributes.cluster_axis.value()) : "nullopt");
log_debug(tt::LogOp, "Topology: {}", operation_attributes.topology);
log_debug(tt::LogOp, "Forward neighbor: {}", forward_coordinate.has_value() ?
    forward_coordinate.value().to_string() : "null");
log_debug(tt::LogOp, "Backward neighbor: {}", backward_coordinate.has_value() ?
    backward_coordinate.value().to_string() : "null");
```

## Common Fixes

### Fix 1: Explicitly Set cluster_axis

In your test, try explicitly setting `cluster_axis`:

```cpp
auto output_tensor = ttnn::reduce_scatter(input_mesh_tensor, dim, 1); // cluster_axis = 1
```

Or:

```cpp
auto output_tensor = ttnn::reduce_scatter(input_mesh_tensor, dim, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt);
```

### Fix 2: Check Tensor Distribution

The issue might be that when you replicate with `mesh_shape_override = {1, 4}`, the tensor's device coordinates don't match what `get_neighbor` expects.

Verify the tensor distribution matches the mesh device:

```cpp
// After distribute_tensor, check:
log_info(tt::LogTest, "Input tensor device shape: {}", input_mesh_tensor.device()->shape());
log_info(tt::LogTest, "Input tensor coords count: {}", input_mesh_tensor.device_storage().coords.size());
for (const auto& coord : input_mesh_tensor.device_storage().coords) {
    log_info(tt::LogTest, "  Coord: {}", coord);
}
```

### Fix 3: Ensure Mesh Shape Consistency

Make sure the `mesh_shape_override` in your test matches the actual mesh device shape or represents a valid submesh:

```cpp
// In your test, verify:
log_info(tt::LogTest, "Mesh device shape: {}", test_fixture.mesh_device_->shape());
// Should match or be compatible with mesh_shape_override = {1, 4}
```

### Fix 4: Check Topology Detection

The topology might be incorrectly detected. The code uses:
- `Topology::Linear` → `BoundaryMode::NONE` (neighbors only exist if within bounds)
- `Topology::Ring` → `BoundaryMode::WRAP` (wraps around)

If your topology is Linear but the code thinks it's Ring (or vice versa), neighbors might not be found.

## Code Flow to Check

1. **When `cluster_axis` is `nullopt`** (line 200-216 in `ccl_common.cpp`):
   - Uses linearized index approach
   - Should work if `device_coords` are ordered correctly

2. **When `cluster_axis` has a value** (line 169-199):
   - Uses `get_neighbor` with `tensor.device()->shape()`
   - Requires `device_coords` to match what `get_neighbor` computes

## Expected Values for Your Test

- **Mesh device shape**: Should be `{1, 4}` or compatible
- **Tensor device_storage.coords**: Should be `[(0,0), (0,1), (0,2), (0,3)]` for 4 devices
- **cluster_axis**: If nullopt, uses linearized approach. If set to 1, uses dimension 1 of mesh coordinates
- **Boundary mode**: For Linear topology with cluster_axis=1, should be `NONE`

## Quick Test Fix

Try this modification to your test:

```cpp
// After creating input_mesh_tensor, add:
auto mesh_shape = input_mesh_tensor.device()->shape();
log_info(tt::LogTest, "Tensor device shape: {}", mesh_shape);
log_info(tt::LogTest, "Tensor coords: {}", input_mesh_tensor.device_storage().coords.size());

// Explicitly set cluster_axis to match your mesh dimension
auto output_tensor = ttnn::reduce_scatter(
    input_mesh_tensor,
    dim,
    1,  // cluster_axis = 1 (the second dimension of {1, 4})
    std::nullopt,  // subdevice_id
    std::nullopt,  // memory_config
    std::nullopt,  // optional_output_tensor
    std::nullopt,  // num_links
    std::nullopt   // topology
);
```

## Most Likely Issue

Based on the code analysis, I suspect:

1. **cluster_axis is being set to 1** (or inferred)
2. **Boundary mode is NONE** (Linear topology)
3. **get_neighbor computes neighbors** using `tensor.device()->shape()`
4. **But device_storage.coords** might not include all coordinates from the device shape
5. **When replicating with mesh_shape_override**, the coords might only include the override shape coordinates, not the full device shape

**Check**: Does `tensor.device()->shape()` equal `{1, 4}`? And does `device_storage.coords` contain exactly `[(0,0), (0,1), (0,2), (0,3)]`?

If the device shape is different (e.g., `{2, 2}` or `{4, 1}`), but coords are `[(0,0), (0,1), (0,2), (0,3)]`, then `get_neighbor` will compute neighbors based on the wrong shape.
