# Tensor Tests

This folder hosts tests for Runtime Tensors and their associated utilities.

These tests correspond to the tensor implementation in `tt_metal/impl/tensor/` and the experimental API in `tt_metal/api/tt-metalium/experimental/tensor/`.

## Test Files

- `test_tensor_sharding.cpp` - TensorSpec sharding validation
- `test_host_tensor.cpp` - Sanity tests for HostTensor (type traits, construction, copy/move semantics)
- `test_mesh_tensor.cpp` - Sanity tests for MeshTensor (type traits, construction, move semantics, device-based tests with MeshBuffer)
