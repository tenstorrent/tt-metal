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

Table of Contents
=================

- [Kernel Ops Delivery](#kernel-ops-delivery---folder-structure)
- [LLK Delivery](#llk-delivery---folder-structure)
- [Runtime Delivery](#runtime-delivery---folder-structure)

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
| L3.1 | 2.0.1 | ttnn/cpp/ttnn/operations/experimental/quasar/ | Quasar ResNet Kernel Ops |
| L3.2 | 2.0.2 | ttnn/cpp/ttnn/operations/experimental/quasar/ | &lt;...&gt; |
| L3.3 | 2.0.3 | ttnn/cpp/ttnn/operations/experimental/quasar/ | &lt;...&gt; |

**This delivery contains:** all of Level 1 plus the full Quasar op tree (experimental/quasar/).


# LLK Delivery - Folder Structure

**Feature(s) this release:** LLK INT8 support; PDL-related LLK features; Quant / dequant kernels
**Source:** [tenstorrent/tt-metal - tt_metal/tt-llk](https://github.com/tenstorrent/tt-metal/tree/main/tt_metal/tt-llk)

## Level 1 - Common (used by Wormhole, Blackhole, Quasar, and any new designs)

- [tt-metal/tt_metal/tt-llk/common/](https://github.com/tenstorrent/tt-metal/tree/main/tt_metal/tt-llk/common)- cross-chip LLK helpers (ckernel_fence.h, llk_assert.h, tensor_shape.h, tensor_shape_coverage{,_math,_pack,_unpack}.h, sanitizer/)
- [tt-metal/tt_metal/tt-llk/tests/](https://github.com/tenstorrent/tt-metal/tree/main/tt_metal/tt-llk/tests)- shared LLK test harness (helpers/, python_tests/, sources/)
- [tt-metal/tt_metal/tt-llk/docs/](https://github.com/tenstorrent/tt-metal/tree/main/tt_metal/tt-llk/docs)- shared documentation (common/, llk/{l1,l2,l3}, performance_counters/, tests/)
- [tt-metal/tt_metal/tt-llk/infra/](https://github.com/tenstorrent/tt-metal/tree/main/tt_metal/tt-llk/infra)- shared build / formatting tooling (fix_cstdint.py, run_order_processing.py)

## Level 2 - Architecture (per chip family)

| **Level** | **Architecture** | **Folder** | **Contents** |
| --- | --- | --- | --- |
| L2.1 | Wormhole (B0) | [tt-metal/tt_metal/tt-llk/tt_llk_wormhole_b0/](https://github.com/tenstorrent/tt-metal/tree/main/tt_metal/tt-llk/tt_llk_wormhole_b0) | common/inc/sfpu/experimental, instructions/, llk_lib/{debug, experimental} |
| L2.2 | Wormhole (B0) | [tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/](https://github.com/tenstorrent/tt-metal/tree/main/tt_metal/hw/ckernels/wormhole_b0/metal/llk_api) | llk_sfpu |
| L2.3 | Blackhole | [tt-metal/tt_metal/tt-llk/tt_llk_blackhole/](https://github.com/tenstorrent/tt-metal/tree/main/tt_metal/tt-llk/tt_llk_blackhole) | common/inc/sfpu/experimental, instructions/, llk_lib/{debug, experimental} |
| L2.4 | Blackhole | [tt_metal/hw/ckernels/blackhole/metal/llk_api/](https://github.com/tenstorrent/tt-metal/tree/main/tt_metal/hw/ckernels/blackhole/metal/llk_api) | llk_sfpu |
| L2.5 | Quasar | [tt-metal/tt_metal/tt-llk/tt_llk_quasar/](https://github.com/tenstorrent/tt-metal/tree/main/tt_metal/tt-llk/tt_llk_quasar) | common/inc/{experimental, sfpu, internal}, instructions/ (assembly.yaml), llk_lib/ |
| L2.6 | Quasar | [tt_metal/hw/ckernels/quasar/metal/llk_api/](https://github.com/tenstorrent/tt-metal/tree/main/tt_metal/hw/ckernels/quasar/metal/llk_api) | llk_sfpu |

### L2.5-L2.6 Quasar - expanded

| **Folder** | **Contents** |
| --- | --- |
| [tt_llk_quasar/llk_lib/](https://github.com/tenstorrent/tt-metal/tree/main/tt_metal/tt-llk/tt_llk_quasar/llk_lib) | 29 headers — unpack (llk_unpack_common.h, llk_unpack_matmul.h, llk_unpack_tilize.h, llk_unpack_reduce.h, llk_unpack_unary_operand.h, llk_unpack_binary_operands.h, llk_unpack_*_broadcast_operands.h, llk_unpack_reduce_col_tilizeA_strided.h), math (llk_math_common.h, llk_math_matmul.h, llk_math_reduce.h, llk_math_eltwise_binary.h, llk_math_eltwise_binary_broadcast.h, llk_math_eltwise_binary_sfpu.h, llk_math_eltwise_unary_sfpu.h, llk_math_eltwise_ternary_sfpu.h, llk_math_eltwise_sfpu_common.h, llk_math_eltwise_unary_datacopy.h, llk_math_unary_broadcast.h, llk_math_transpose_dest.h), pack (llk_pack.h, llk_pack_common.h, llk_pack_matmul.h, llk_pack_untilize.h), plus llk_defs.h, llk_srcs.h, llk_sync.h, llk_memory_checks.h |
| [tt_llk_quasar/common/inc/](https://github.com/tenstorrent/tt-metal/tree/main/tt_metal/tt-llk/tt_llk_quasar/common/inc) | ckernel.h, ckernel_defs.h, ckernel_ops.h, ckernel_addrmod.h, ckernel_dest.h, ckernel_gpr_map.h, ckernel_instr_params.h, ckernel_pcbuf.h, ckernel_proj_params.h, ckernel_risc_atomics.h, ckernel_riscv_debug.h, ckernel_sfpu.h, ckernel_template.h, ckernel_trisc_common.h, ckernel_trisc_id.h, ckernel_vector.h, cmath_common.h, cpack_common.h, cunpack_common.h, llk_tdma_guard.h |
| [tt_llk_quasar/common/inc/sfpu/](https://github.com/tenstorrent/tt-metal/tree/main/tt_metal/tt-llk/tt_llk_quasar/common/inc/sfpu) | ckernel_sfpu_add.h, ckernel_sfpu_mul_int32.h, ckernel_sfpu_binary_comp.h, ckernel_sfpu_relu.h, ckernel_sfpu_sigmoid.h, ckernel_sfpu_silu.h, ckernel_sfpu_sqrt.h, ckernel_sfpu_typecast_fp16b_uint16.h, ckernel_sfpu_typecast_int32_fp32.h |
| [tt_llk_quasar/common/inc/experimental/](https://github.com/tenstorrent/tt-metal/tree/main/tt_metal/tt-llk/tt_llk_quasar/common/inc/experimental) | ckernel_sfpu_abs.h, ckernel_sfpu_fill.h, ckernel_sfpu_swiglu.h — include only if in scope |
| [tt_llk_quasar/instructions/](https://github.com/tenstorrent/tt-metal/tree/main/tt_metal/tt-llk/tt_llk_quasar/instructions) | assembly.yaml — instruction definitions |
| [tt_metal/hw/ckernels/quasar/metal/llk_api](https://github.com/tenstorrent/tt-metal/tree/main/tt_metal/hw/ckernels/quasar/metal/llk_api) | 57 headers<br>llk_math_binary_api.h<br>llk_math_common_api.h<br>llk_math_matmul_api.h<br>llk_math_reduce_api.h<br>llk_math_transpose_dest_api.h<br>llk_math_unary_datacopy_api.h<br>llk_pack_common_api.h<br>llk_pack_reduce_api.h<br>llk_pack_tile_api.h<br>llk_pack_untilize_api.h<br>llk_sfpu/ckernel_sfpu_binary.h<br>llk_sfpu/ckernel_sfpu_binary_max_min.h<br>llk_sfpu/ckernel_sfpu_binop_with_unary.h<br>llk_sfpu/ckernel_sfpu_clamp.h<br>llk_sfpu/ckernel_sfpu_comp.h<br>llk_sfpu/ckernel_sfpu_converter.h<br>llk_sfpu/ckernel_sfpu_exp.h<br>llk_sfpu/ckernel_sfpu_gelu.h<br>llk_sfpu/ckernel_sfpu_log.h<br>llk_sfpu/ckernel_sfpu_log1p.h<br>llk_sfpu/ckernel_sfpu_negative.h<br>llk_sfpu/ckernel_sfpu_piecewise_rational.h<br>llk_sfpu/ckernel_sfpu_polyval.h<br>llk_sfpu/ckernel_sfpu_quant.h<br>llk_sfpu/ckernel_sfpu_recip.h<br>llk_sfpu/ckernel_sfpu_relu.h<br>llk_sfpu/ckernel_sfpu_rsqrt.h<br>llk_sfpu/ckernel_sfpu_sigmoid.h<br>llk_sfpu/ckernel_sfpu_silu.h<br>llk_sfpu/ckernel_sfpu_softplus.h<br>llk_sfpu/ckernel_sfpu_sqrt.h<br>llk_sfpu/ckernel_sfpu_sqrt_custom.h<br>llk_sfpu/ckernel_sfpu_square.h<br>llk_sfpu/ckernel_sfpu_tanh.h<br>llk_sfpu/ckernel_sfpu_topk.h<br>llk_sfpu/ckernel_sfpu_trigonometry.h<br>llk_sfpu/ckernel_sfpu_typecast.h<br>llk_sfpu/ckernel_sfpu_where.h<br>llk_sfpu/llk_math_eltwise_binary_sfpu_add_int.h<br>llk_sfpu/llk_math_eltwise_binary_sfpu_binary_comp.h<br>llk_sfpu/llk_math_eltwise_binary_sfpu_init.h<br>llk_sfpu/llk_math_eltwise_binary_sfpu_macros.h<br>llk_sfpu/llk_math_eltwise_binary_sfpu_max_min.h<br>llk_sfpu/llk_math_eltwise_binary_sfpu_mul_int.h<br>llk_sfpu/llk_math_eltwise_ternary_sfpu_init.h<br>llk_sfpu/llk_math_eltwise_ternary_sfpu_macros.h<br>llk_sfpu/llk_math_eltwise_ternary_sfpu_where.h<br>llk_sfpu/llk_math_eltwise_unary_sfpu_binop_with_scalar.h<br>llk_sfpu/llk_math_eltwise_unary_sfpu_init.h<br>llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h<br>llk_sfpu/llk_math_eltwise_unary_sfpu_rsqrt.h<br>llk_unpack_AB_api.h<br>llk_unpack_AB_matmul_api.h<br>llk_unpack_AB_reduce_api.h<br>llk_unpack_A_api.h<br>llk_unpack_common_api.h<br>llk_unpack_tilize_api.h |

## Level 3 - Sub-architecture / program (the customer delivery)

The repository holds **one** Quasar architecture folder. There are no separate Horizon / Trinity / Saturn directories — each program delivers the **same tt_llk_quasar/ tree**, distinguished by the release milestone and the feature list below.

| **Level** | **Program** | **Delivered from** | **Feature this release** |
| --- | --- | --- | --- |
| L3.1 | 2.0.1 | tt_metal/tt-llk/tt_llk_quasar/ | LLK INT8 support; PDL-related LLK features; Quant / dequant kernels |
| L3.2 | 2.0.2 | tt_metal/tt-llk/tt_llk_quasar/ | <...> |
| L3.3 | 2.0.3 | tt_metal/tt-llk/tt_llk_quasar/ | <...> |

**This delivery contains:**all of Level 1 (common/, tests/, docs/, infra/) plus the full Quasar architecture folder (tt_llk_quasar/).


# Runtime Delivery - Folder Structure

**Feature(s) this release:**FD support for dispatch engine; Profiler debug tool support
**Source:** [tenstorrent/tt-metal - tt_metal](https://github.com/tenstorrent/tt-metal/tree/main/tt_metal)

## Level 1 - Common (used by Wormhole, Blackhole, Quasar, and any new designs)

- [tt-metal/tt_metal/api/](https://github.com/tenstorrent/tt-metal/tree/main/tt_metal/api)- public C++ API (tt-metalium/ headers, serialized descriptors)
- [tt-metal/tt_metal/impl/](https://github.com/tenstorrent/tt-metal/tree/main/tt_metal/impl)- core runtime (device/, dispatch/, program/, buffers/, allocator/, kernels/, trace/, sub_device/, event/, profiler/, debug/, context/, threading/)
- [tt-metal/tt_metal/llrt/](https://github.com/tenstorrent/tt-metal/tree/main/tt_metal/llrt)- low-level runtime (llrt_common/, hal/codegen/)
- [tt-metal/tt_metal/jit_build/](https://github.com/tenstorrent/tt-metal/tree/main/tt_metal/jit_build)- kernel JIT compilation and build orchestration
- [tt-metal/tt_metal/common/](https://github.com/tenstorrent/tt-metal/tree/main/tt_metal/common)- shared host utilities
- [tt-metal/tt_metal/detail/](https://github.com/tenstorrent/tt-metal/tree/main/tt_metal/detail)- internal host-side helpers
- [tt-metal/tt_metal/hostdevcommon/](https://github.com/tenstorrent/tt-metal/tree/main/tt_metal/hostdevcommon)- host / device shared definitions
- [tt-metal/tt_metal/logging/](https://github.com/tenstorrent/tt-metal/tree/main/tt_metal/logging)- runtime logging
- [tt-metal/tt_metal/distributed/](https://github.com/tenstorrent/tt-metal/tree/main/tt_metal/distributed)- multi-device / mesh (multihost/, flatbuffer/, layer_completion/)
- [tt-metal/tt_metal/fabric/](https://github.com/tenstorrent/tt-metal/tree/main/tt_metal/fabric)- fabric + CCL routing (builder/, ccl/, impl/, hw/, config/, mesh_graph_descriptors/, cabling_descriptors/)
- [tt-metal/tt_metal/hw/inc/api/](https://github.com/tenstorrent/tt-metal/tree/main/tt_metal/hw/inc/api)- device-side kernel API (dataflow/, compute/, debug/, tensor/, numeric/)
- [tt-metal/tt_metal/hw/inc/internal/](https://github.com/tenstorrent/tt-metal/tree/main/tt_metal/hw/inc/internal)- device-side internals (dataflow/, debug/, ethernet/, tensor/)
- [tt-metal/tt_metal/hw/inc/hostdev/](https://github.com/tenstorrent/tt-metal/tree/main/tt_metal/hw/inc/hostdev)- host / device message structs
- [tt-metal/tt_metal/hw/toolchain/](https://github.com/tenstorrent/tt-metal/tree/main/tt_metal/hw/toolchain)- linker scripts and crt0 (main.ld, script_tng.ld, tmu-crt0.S)
- [tt-metal/tt_metal/third_party/](https://github.com/tenstorrent/tt-metal/tree/main/tt_metal/third_party)- umd/, tracy/, tt-cluster-descriptors/
- [tt-metal/tt_metal/programming_examples/](https://github.com/tenstorrent/tt-metal/tree/main/tt_metal/programming_examples)- reference programs
- [tt-metal/tt_metal/python_env/](https://github.com/tenstorrent/tt-metal/tree/main/tt_metal/python_env)- Python environment definition
- [tt-metal/tt_metal/test/](https://github.com/tenstorrent/tt-metal/tree/main/tt_metal/test)- runtime-local tests

## Level 2 - Architecture (per chip family)

Runtime code is organized by RISC generation: tt-1xx covers Wormhole and Blackhole, tt-2xx covers Quasar. Each generation folder holds code shared by every chip in that generation, with the chip as a leaf beneath it. Both are delivered together.

| **Level** | **Architecture** | **Folder** | **Contents** |
| --- | --- | --- | --- |
| L2.1 | Wormhole (B0) | [llrt/hal/tt-1xx/](https://github.com/tenstorrent/tt-metal/tree/main/tt_metal/llrt/hal/tt-1xx),[hw/firmware/src/tt-1xx/](https://github.com/tenstorrent/tt-metal/tree/main/tt_metal/hw/firmware/src/tt-1xx),[hw/inc/internal/tt-1xx/](https://github.com/tenstorrent/tt-metal/tree/main/tt_metal/hw/inc/internal/tt-1xx),[hw/ckernels/wormhole_b0/](https://github.com/tenstorrent/tt-metal/tree/main/tt_metal/hw/ckernels/wormhole_b0) | wormhole/ leaf in each; soc_descriptors/wormhole_b0_80_arch.yaml, core_descriptors/wormhole_b0_*.yaml |
| L2.2 | Blackhole | [llrt/hal/tt-1xx/](https://github.com/tenstorrent/tt-metal/tree/main/tt_metal/llrt/hal/tt-1xx),[hw/firmware/src/tt-1xx/](https://github.com/tenstorrent/tt-metal/tree/main/tt_metal/hw/firmware/src/tt-1xx),[hw/inc/internal/tt-1xx/](https://github.com/tenstorrent/tt-metal/tree/main/tt_metal/hw/inc/internal/tt-1xx),[hw/ckernels/blackhole/](https://github.com/tenstorrent/tt-metal/tree/main/tt_metal/hw/ckernels/blackhole) | blackhole/ leaf in each; soc_descriptors/blackhole_140_arch.yaml, core_descriptors/blackhole_*.yaml |
| **L2.3** | **Quasar** | [llrt/hal/tt-2xx/](https://github.com/tenstorrent/tt-metal/tree/main/tt_metal/llrt/hal/tt-2xx),[hw/firmware/src/tt-2xx/](https://github.com/tenstorrent/tt-metal/tree/main/tt_metal/hw/firmware/src/tt-2xx),[hw/inc/internal/tt-2xx/](https://github.com/tenstorrent/tt-metal/tree/main/tt_metal/hw/inc/internal/tt-2xx),[hw/ckernels/quasar/](https://github.com/tenstorrent/tt-metal/tree/main/tt_metal/hw/ckernels/quasar) | quasar/ leaf in each; soc_descriptors/quasar_32_arch.yaml, core_descriptors/quasar_simulation_*.yaml |

### L2.3 Quasar - expanded

| **Folder** | **Contents** |
| --- | --- |
| [tt_metal/llrt/hal/tt-2xx/](https://github.com/tenstorrent/tt-metal/tree/main/tt_metal/llrt/hal/tt-2xx) | hal_2xx_common.cpp/.hpp, sources.cmake |
| [tt_metal/llrt/hal/tt-2xx/quasar/](https://github.com/tenstorrent/tt-metal/tree/main/tt_metal/llrt/hal/tt-2xx/quasar) | qa_hal.cpp/.hpp, qa_hal_tensix.cpp, qa_hal_dispatch.cpp, qa_hal_active_eth.cpp, qa_hal_idle_eth.cpp, *_asserts.hpp |
| [tt_metal/hw/firmware/src/tt-2xx/](https://github.com/tenstorrent/tt-metal/tree/main/tt_metal/hw/firmware/src/tt-2xx) | dm.cc, dmk.cc, dispatch_dm.cc, trisc.cc, trisck.cc |
| [tt_metal/hw/firmware/src/tt-2xx/quasar/](https://github.com/tenstorrent/tt-metal/tree/main/tt_metal/hw/firmware/src/tt-2xx/quasar) | noc.c, fds_functions.cpp, noc_address_translation_tables.cpp |
| [tt_metal/hw/inc/internal/tt-2xx/](https://github.com/tenstorrent/tt-metal/tree/main/tt_metal/hw/inc/internal/tt-2xx) | risc_common.h, dataflow_buffer.inl, dataflow_buffer/, noc_zero_l1.inl |
| [tt_metal/hw/inc/internal/tt-2xx/quasar/](https://github.com/tenstorrent/tt-metal/tree/main/tt_metal/hw/inc/internal/tt-2xx/quasar) | c_tensix_core.h, cfg_defines.h, core_config.h, dev_mem_map.h, dram_address_map.h, eth_l1_address_map.h, tensix*.h, stream_interface.h, noc/ (+ registers/), overlay/ (+ meta/) |
| [tt_metal/hw/ckernels/quasar/metal/](https://github.com/tenstorrent/tt-metal/tree/main/tt_metal/hw/ckernels/quasar/metal) | llk_api/ (+ llk_sfpu/), llk_io/, common/ |

## Level 3 - Sub-architecture / program (the customer delivery)

The repository holds one Quasar runtime tree. There are no separate program directories - each program delivers the same tt-2xx/quasar folders, distinguished by the release milestone and the feature list below.

| **Level** | **Program** | **Delivered from** | **Feature this release** |
| --- | --- | --- | --- |
| L3.1 | 2.0.1 | tt_metal/.../tt-2xx/quasar/ | FD support for dispatch engine; Profiler debug tool support |
| L3.2 | 2.0.2 | tt_metal/.../tt-2xx/quasar/ | <...> |
| L3.3 | 2.0.3 | tt_metal/.../tt-2xx/quasar/ | <...> |

**This delivery contains:** all of Level 1 plus the full Quasar generation and chip folders (llrt/hal/tt-2xx/, hw/firmware/src/tt-2xx/, hw/inc/internal/tt-2xx/, hw/ckernels/quasar/)
