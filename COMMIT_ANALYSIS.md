# Commit Analysis: Root Cause of "forward_coord or backward_coord is null" Error

## The Problematic Commit

**Commit**: `37aa0a796a7f3fc1ea3b7a59af22b603b907d246`
**Author**: Saad Jameel
**Date**: Thu Oct 30 18:33:28 2025
**PR**: #31370
**Title**: "Add support for cluster_axis=None with MxN meshes in all_gather, reduce_scatter, all_reduce and all_broadcast"

## What Changed

The commit added logic to handle `cluster_axis=None` for non-line topologies by recursively calling reduce-scatter along each mesh dimension.

### Key Code Addition in `reduce_scatter.cpp`:

```cpp
// If cluster_axis is None, but mesh shape is not 1xM or Mx1, then we call reduce-scatter on cluster_axis=1, then
// reduce-scatter on cluster_axis=0
if (cluster_axis == std::nullopt) {
    auto mesh_shape = input_tensor.device()->get_view().shape();
    if (!mesh_shape.is_line_topology()) {
        Tensor tensor = input_tensor;
        for (size_t i = 0; i < mesh_shape.dims(); ++i) {
            tensor = ttnn::reduce_scatter(
                tensor, dim, i, subdevice_id, memory_config, optional_output_tensor, num_links, topology);
        }
        return tensor;
    }
}
```

## The Bug

### Issue 1: Using `device()->get_view().shape()` Instead of Distribution Shape

The code uses `input_tensor.device()->get_view().shape()` to check if it's a line topology. However, **this returns the mesh device's physical shape**, not the tensor's distribution shape.

In your test:
- **Mesh device shape**: `{1, 4}` (from `GetDeterminedMeshShape()`)
- **Tensor distribution shape**: `{1, 4}` (from `mesh_shape_override`)
- **Tensor device_storage.coords**: `[(0,0), (0,1), (0,2), (0,3)]`

`MeshShape(1, 4).is_line_topology()` returns `true` (only one non-unit dimension), so the recursive path shouldn't be taken. **BUT** if there's any mismatch between what `get_view().shape()` returns and the actual distribution, or if the tensor's device coordinates don't match the device shape, neighbors won't be found.

### Issue 2: Recursive Calls Change Tensor Coordinates

When the recursive path IS taken (for non-line topologies), each recursive call does:
```cpp
tensor = ttnn::reduce_scatter(tensor, dim, i, ...);
```

After the first reduce-scatter call (e.g., with `cluster_axis=0`), the tensor's `device_storage.coords` change. Then when it calls reduce-scatter again with `cluster_axis=1`, it's operating on a tensor with different coordinates, and `get_physical_neighbor_from_physical_coord` can't find neighbors because:

1. It computes neighbors using `tensor.device()->shape()`
2. But searches for them in `tensor.device_storage().coords`
3. After the first reduce-scatter, the coords might not match what `get_neighbor` expects

### Issue 3: `get_view().shape()` May Not Match Distribution

The `get_view().shape()` might return the full mesh device shape (e.g., `{2, 2}` or `{4, 1}`), while the tensor was distributed with `mesh_shape_override = {1, 4}`. This causes:

- `is_line_topology()` check on wrong shape
- Neighbor computation using wrong shape
- Search in coords that don't match

## Root Cause Summary

**The core issue**: When `cluster_axis=None`, the code checks `input_tensor.device()->get_view().shape()` to decide if it's a line topology. However:

1. **If `get_view().shape()` returns a non-line topology** (even though distribution is line), it takes the recursive path
2. **The recursive path calls reduce-scatter with explicit `cluster_axis` values** (0, 1, etc.)
3. **But the tensor's `device_storage.coords` don't match what `get_neighbor` expects** based on `device()->shape()`
4. **Result**: Both forward and backward neighbors are `nullopt`

## Evidence

From the test:
- Test uses `mesh_shape_override = {1, 4}` (line topology)
- Mesh device shape should be `{1, 4}` (from `GetDeterminedMeshShape()`)
- But `get_view().shape()` might return something different
- Or the tensor's coords don't align with device shape after distribution

## Potential Fixes

### Fix 1: Use Tensor's Distribution Shape Instead of Device View Shape

Change:
```cpp
auto mesh_shape = input_tensor.device()->get_view().shape();
```

To:
```cpp
// Use the tensor's actual distribution shape from topology
auto mesh_shape = input_tensor.tensor_topology().distribution_shape();
// OR use device_storage coords to infer shape
```

### Fix 2: Check Actual Distribution, Not Device Shape

Instead of checking `device()->get_view().shape()`, check the actual distribution:
```cpp
if (cluster_axis == std::nullopt) {
    // Check if tensor is actually distributed as a line
    const auto& coords = input_tensor.device_storage().coords;
    if (coords.size() > 1) {
        // Check if coords form a line topology
        bool is_line = true;
        // ... check coords form a line ...
        if (!is_line) {
            // recursive path
        }
    }
}
```

### Fix 3: Ensure cluster_axis is Set Correctly

The test should explicitly set `cluster_axis`:
```cpp
auto output_tensor = ttnn::reduce_scatter(input_mesh_tensor, dim, 1); // explicit cluster_axis
```

## Related Commits to Investigate

1. `e6fe738ca3` - "mesh config reafctoring" (might have changed how shapes are determined)
2. `52c87777ec` - "Fixing big mesh case" (might have changed mesh shape handling)
3. `4afd292610` - "fix mesh_coord calculation for fill_cache" (might have changed coordinate handling)

## Recommendation

**Immediate workaround**: In your test, explicitly set `cluster_axis=1`:
```cpp
auto output_tensor = ttnn::reduce_scatter(input_mesh_tensor, dim, 1);
```

**Long-term fix**: The code should use the tensor's actual distribution shape/topology, not `device()->get_view().shape()`, when determining whether to take the recursive path.
