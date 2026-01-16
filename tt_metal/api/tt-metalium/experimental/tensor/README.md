# Tensor in Runtime

This folder hosts the current effort on lowering Tensor concepts from TTNN to TT-Metal.

The goal is to make Tensor a first-class citizen in Metal Runtime while exposing a reasonable level of abstraction.

## Why this folder

Tensor migration represents work with significant transient states.
This folder allows Runtime to continue the effort with arbitrary steps and minimal distraction to other functions.
It also allows experimentation without the restrictions of our deprecation policy.

## Expected lifetime

TODO

## Maturity criteria

TODO

## Source of Initial Implementations

The initial placeholder structs in this directory were populated by copy-pasting existing implementations from the `ttnn` library. This was done to accelerate the refactoring process.

- **Source Commit:** `9f3856801448f589170defe41b23c8b9b43e33a2`

### Copied Files:

- `spec/memory_config/memory_config.hpp` from `ttnn/api/ttnn/tensor/memory_config/memory_config.hpp`
- `spec/memory_config/sharding_types.hpp` from `ttnn/api/ttnn/tensor/types.hpp`
- `spec/tensor_spec.hpp` from `ttnn/api/ttnn/tensor/tensor_spec.hpp`
- `spec/layout/alignment.hpp` from `ttnn/api/ttnn/tensor/layout/alignment.hpp`
- `spec/layout/page_config.hpp` from `ttnn/api/ttnn/tensor/layout/page_config.hpp`
- `spec/layout/tensor_layout.hpp` from `ttnn/api/ttnn/tensor/layout/tensor_layout.hpp`
- `topology/distributed_tensor_configs.hpp` from `ttnn/api/ttnn/distributed/distributed_configs.hpp`
- `topology/tensor_topology.hpp` from `ttnn/api/ttnn/distributed/tensor_topology.hpp`
- `tensor_types.hpp` from `ttnn/api/ttnn/tensor/types.hpp`
