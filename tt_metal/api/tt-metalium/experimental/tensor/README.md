# Tensor in Runtime

This folder hosts the current effort on lowering Tensor concepts from TTNN to TT-Metal.

The goal is to make Tensor a first-class citizen in Metal Runtime while exposing a reasonable level of abstraction.

## Why experimental

Tensor migration represents work with significant transient states.

This folder
- allows Runtime to continue the effort with arbitrary steps and minimal distraction to other functions.
- unblocks other efforts interested in Host/Device Tensor but not the TTNN migration
- allows experimentation without the restrictions of our deprecation policy.

## namespace

Headers in this folder are not in the experimental namespace as this staging area is meant to be short lived,
most of the concepts hosted are well tested, production code already.

## Header Mapping

The following headers were migrated from TTNN to this directory. Forward headers remain in the original TTNN locations for backwards compatibility.

| TTNN Source | TT-Metal Destination |
|-------------|----------------------|
| `ttnn/api/ttnn/tensor/types.hpp` | `tensor_types.hpp` |
| `ttnn/api/ttnn/tensor/tensor_spec.hpp` | `spec/tensor_spec.hpp` |
| `ttnn/api/ttnn/tensor/layout/alignment.hpp` | `spec/layout/alignment.hpp` |
| `ttnn/api/ttnn/tensor/layout/layout.hpp` | `spec/layout/layout.hpp` |
| `ttnn/api/ttnn/tensor/layout/page_config.hpp` | `spec/layout/page_config.hpp` |
| `ttnn/api/ttnn/tensor/layout/tensor_layout.hpp` | `spec/layout/tensor_layout.hpp` |
| `ttnn/api/ttnn/tensor/memory_config/memory_config.hpp` | `spec/memory_config/memory_config.hpp` |
| `ttnn/api/ttnn/distributed/distributed_configs.hpp` | `topology/distributed_tensor_configs.hpp` |
| `ttnn/api/ttnn/distributed/tensor_topology.hpp` | `topology/tensor_topology.hpp` |

## Life-time

This folder is expected to be short-lived, the effort is tracked by:
https://github.com/tenstorrent/tt-metal/issues/36373
