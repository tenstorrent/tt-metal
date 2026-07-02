# Tensor Tests

This folder hosts tests for Runtime Tensors and their associated utilities.

These tests correspond to the tensor implementation in `tt_metal/impl/tensor/` and the experimental API in `tt_metal/api/tt-metalium/experimental/tensor/`.

## Test Files

- `test_tensor_sharding.cpp` - TensorSpec sharding validation
- `test_host_tensor.cpp` - Sanity tests for HostTensor (type traits, construction, copy/move semantics)
- `test_mesh_tensor.cpp` - Sanity tests for MeshTensor (type traits, construction, move semantics, device-based tests with MeshBuffer)
- `test_tensor_types.cpp` - Free-function tensor type helpers (e.g. `tile_size`)
- `test_tensor_layout.cpp` - TensorLayout alignment/physical-shape/strides/padding metadata, with optional H2D/D2H round-trip
- `test_create_tensor.cpp` - HostTensor->MeshTensor H2D/D2H round-trips and create-tensor combination sweeps
- `test_create_tensor_with_layout.cpp` - Layout-driven padding of an allocated MeshTensor (TILE vs ROW_MAJOR)
- `test_tensor_nd_sharding.cpp` - Legacy<->ND shard spec conversion, BufferDistributionSpec core/page-mapping, TensorSpec sharding helpers
- `test_vector_conversion.cpp` - Host<->tensor vector conversion on HostTensor (dtypes, layouts, block-float, borrowed storage)
- `common_tensor_test_utils.{hpp,cpp}` - Shared H2D/D2H round-trip helper (`test_utils::test_tensor_on_device`)
