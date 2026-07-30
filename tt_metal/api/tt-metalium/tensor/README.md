# Tensor in Runtime

This folder hosts Tensor concepts lowered from TTNN to TT-Metal.

The goal is to make Tensor a first-class citizen in Metal Runtime while exposing a reasonable level of abstraction.

## Namespace

Headers live in `tt::tt_metal`. Most of the concepts hosted here are well tested, production code migrated from TTNN.

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

## Life-time

The migration effort is tracked by:
https://github.com/tenstorrent/tt-metal/issues/36373
