<!--
Verbatim copy. Source: Google Docs
URL:       https://docs.google.com/document/d/14gIEpKja0xPf9TZ7caCBMGSrbaB2aYi8iGHGrG-06R4
Owner:     Sapna Fun Bahadur Khatri
Created / last updated: 2026-08-12
Retrieved: 2026-08-13
Reproduced as authored, with these editorial changes:
  - repository links repointed from /tree/main/ to /tree/v0.76.0/ so this record
    keeps pointing at the tree that shipped (all 14 paths verified at the tag)
  - two stray "**" removed; they were bold-whitespace spans in the Google Doc
    that did not survive conversion
  - an "Editor's note" added under Level 2 recording the op folders that do not
    follow the documented shape
Level 3 rows marked <...> are placeholders in the source document, not omissions.
-->

# Kernel Ops Delivery - Folder Structure

**Feature(s) this release:** Quasar ResNet Kernel Ops
**Source:** [tenstorrent/tt-metal - ttnn](https://github.com/tenstorrent/tt-metal/tree/v0.76.0/ttnn)

## Level 1 - Common (used by Wormhole, Blackhole, Quasar, and any new designs)

 - [tt-metal/ttnn/cpp/ttnn/operations/](https://github.com/tenstorrent/tt-metal/tree/v0.76.0/ttnn/cpp/ttnn/operations) - the op library (33 op families; see breakdown below)
 - [tt-metal/ttnn/cpp/ttnn/kernel/](https://github.com/tenstorrent/tt-metal/tree/v0.76.0/ttnn/cpp/ttnn/kernel) - device-side kernel entry points shared across ops
 - [tt-metal/ttnn/cpp/ttnn/kernel_lib/](https://github.com/tenstorrent/tt-metal/tree/v0.76.0/ttnn/cpp/ttnn/kernel_lib) - shared device-kernel helper library
 - [tt-metal/ttnn/cpp/ttnn/graph/](https://github.com/tenstorrent/tt-metal/tree/v0.76.0/ttnn/cpp/ttnn/graph) - graph capture / trace support
 - [tt-metal/ttnn/cpp/ttnn-nanobind/](https://github.com/tenstorrent/tt-metal/tree/v0.76.0/ttnn/cpp/ttnn-nanobind) - Python binding layer
 - [tt-metal/ttnn/api/ttnn/](https://github.com/tenstorrent/tt-metal/tree/v0.76.0/ttnn/api/ttnn) - public C++ headers (tensor/, distributed/, graph/, common/, services/)
 - [tt-metal/ttnn/core/](https://github.com/tenstorrent/tt-metal/tree/v0.76.0/ttnn/core) - core implementation (tensor/, distributed/, graph/, services/)
 - [tt-metal/ttnn/ttnn/](https://github.com/tenstorrent/tt-metal/tree/v0.76.0/ttnn/ttnn) - Python API (operations/, distributed/, experimental_loader/)
 - [tt-metal/ttnn/tt_lib/](https://github.com/tenstorrent/tt-metal/tree/v0.76.0/ttnn/tt_lib) - legacy compatibility layer (fused_ops/, fallback_ops/)
 - [tt-metal/ttnn/examples/](https://github.com/tenstorrent/tt-metal/tree/v0.76.0/ttnn/examples) - reference ops (add/, lab_eltwise_binary/, lab_multicast/)
 - [tt-metal/ttnn/tutorials/](https://github.com/tenstorrent/tt-metal/tree/v0.76.0/ttnn/tutorials) - tutorials (basic_python/)
 - [tt-metal/ttnn/test/](https://github.com/tenstorrent/tt-metal/tree/v0.76.0/ttnn/test) - library-local tests

### Op families under ttnn/cpp/ttnn/operations/

| Family | Contents |
| --- | --- |
| matmul/ | Matrix multiply |
| conv/ | Convolutions |
| pool/ | Pooling |
| sliding_window/ | Sliding-window infrastructure for conv / pool |
| eltwise/ | Element-wise unary / binary / ternary |
| data_movement/ | "39 ops - transpose/, permute/, concat/, slice/, pad/, tilize*/, untilize*/, sharded/, reshape*/, fold/, sort/, gather/, scatter/" |
| reduction/ | Reductions incl. accumulation/ |
| normalization/ | "LayerNorm, RMSNorm, softmax" |
| transformer/ | Attention / SDPA |
| "ccl/, point_to_point/" | "Collective communication, point-to-point transfer" |
| moreh/ | Moreh op suite |
| "embedding/, embedding_backward/" | Embedding forward / backward |
| "kv_cache/, prefetcher/" | Inference serving support |
| "creation/, full/, full_like/, rand/, randn/, uniform/, bernoulli/" | Tensor creation |
| "copy/, core/, generic/, debug/, examples/" | Infrastructure |
| "index_fill/, loss/" | Misc |
| kernel_helper_functions/ | Shared kernel-side helpers |
| experimental/ | 34 experimental families incl. quasar/ (see Level 2) |

## Level 2 - Architecture (per chip family)

Ops are written once against the Metalium API, and the hardware abstraction layer resolves the chip underneath. Most of the library is therefore chip-independent and sits at Level 1. Architecture-specific ops are delivered in a dedicated tree.

| Level | Architecture | Folder | Contents |
| --- | --- | --- | --- |
| L2.1 | Wormhole (B0) | — | No architecture-specific op folder; runs the Level 1 op library unchanged |
| L2.2 | Blackhole | — | No architecture-specific op folder; runs the Level 1 op library unchanged |
| L2.3 | Quasar | [tt-metal/ttnn/cpp/ttnn/operations/experimental/quasar/](https://github.com/tenstorrent/tt-metal/tree/v0.76.0/ttnn/cpp/ttnn/operations/experimental/quasar) | "28 op folders - compute, convolution, layout, sharding, slicing, movement" |

### L2.3 Quasar - expanded

| Group | Ops |
| --- | --- |
| Compute | "matmul/ (incl. sparse/), binary/, binary_ng/, reduction/generic/, typecast/" |
| Convolution / vision | "conv2d/, pool_generic/, halo/, fold/" |
| Layout | "tilize/, tilize_with_val_padding/, untilize/, untilize_with_unpadding/, transpose/, reshape_view/" |
| Sharding | "interleaved_to_sharded/, sharded_to_interleaved/, reshard/" |
| Slicing / padding | "slice/, slice_write/, padded_slice/, pad/, op_slicing/" |
| Movement / placement | "move/, reallocate/, to_device/, to_layout/, to_memory_config/" |

Each op folder follows the standard shape: device/ (program factory), device/kernels/{compute, dataflow}/ (device kernels), and the host-side op definition.

> **Editor's note (not in the source document).** Verified against the `v0.76.0` tag:
> 22 of the 27 op folders match that shape. Four are host-side wrappers with no
> `device/` directory and no kernels — `reallocate/`, `to_device/`, `to_layout/` and
> `to_memory_config/` — and `reduction/` nests one level deeper, with the program
> factory at `reduction/generic/device/`. The source document's count of 28 op
> folders is 27 at the tag.

## Level 3 - Sub-architecture / program (the customer delivery)

The repository holds one Quasar op tree. There are no separate program directories - each program delivers the same experimental/quasar/ tree, distinguished by the release milestone and the feature list below.

| Level | Program | Delivered from | Feature this release |
| --- | --- | --- | --- |
| L3.1 | Quasar Horizon | ttnn/cpp/ttnn/operations/experimental/quasar/ | &lt;...&gt; |
| L3.2 | Quasar Trinity | ttnn/cpp/ttnn/operations/experimental/quasar/ | Quasar ResNet Kernel Ops |
| L3.3 | Quasar Saturn | ttnn/cpp/ttnn/operations/experimental/quasar/ | &lt;...&gt; |

**This delivery contains:** all of Level 1 plus the full Quasar op tree (experimental/quasar/).
