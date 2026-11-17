<!--
Folder and file structure for the Nov-13 (v0.82.0) AI-IP release (Jira epic AIIPSW-18).

Source: the v0.80.0 document as merged to main
(releases/0.80.0/03-ai-ip-0.80.0-folder-and-file-structure.md). Structure,
headings, table layout and wording are kept.

Tree verified: commit bfc3f70fd49af2cbdc7d8263ad288daaf6eed049 (v0.82.0 release
commit), compared with 978f93b56e36b68318d11ecf7b101085a81d718e (v0.80.0).
Verified on: 2026-11-16, with tree-only git commands (ls-tree, name-only diff).
All repository links point to /tree/v0.82.0/.

Changes against the v0.80.0 document:
  Kernel Ops
  - ttnn/cpp/ttnn/operations/      32 -> 33 op families (index_fill_new/ added);
                                   moreh/ was in the count but not listed, now listed
  - experimental/ families         36 -> 39 (fused_msda/, gemma4/,
                                   nomic_embed_text_v2_moe/ added)
  - experimental/quasar/           29 -> 31 op folders (batch_norm/, softmax/ added,
                                   new "Normalization" group); shared common/
                                   (thread_plan.hpp) added
  - data_movement/                 39 ops, unchanged
  - expanded table: added rows for the model op tests
    (models/experimental/ops/quasar/tests/) and model code (qwen3_vl/,
    llama32_1b_quasar/)
  LLK
  - tt_llk_quasar/arch/            quasar_4row/ -> saturn/ and trinity/
  - tt_llk_quasar/common/inc/sfpu/ removed; its 11 headers are now in
                                   hw/ckernels/quasar/metal/llk_api/llk_sfpu/
  - tt_llk_quasar/common/inc/      cfg_defines.h, sfpu_impl.h, sfpu_reg.h added
  - tt_llk_quasar/llk_lib/         34 -> 39 headers (llk_unpack_tilize_to_dest.h,
                                   llk_unpack_unary_operand_to_dest.h,
                                   experimental/llk_{unpack,math,pack}_gather_compress.h)
  - hw/ckernels/quasar/.../llk_api 148 -> 174 headers (26 added in llk_sfpu/)
  - tt-llk/tests/                  hardware_free/ added
  - tt-llk/infra/                  check_*.py source checks listed
  - Blackhole llk_lib/hal/ was in the tree but not listed, now listed
  Runtime
  - tt_metal/fabric/               experimental/ added
  - tt_metal/hw/inc/internal/      local_mem/, profiler/ added
  - llrt/hal/tt-2xx/quasar/        qa_att_windows.hpp, qa_emulator_models.hpp added
  - hw/inc/internal/tt-2xx/quasar/ 70 -> 73 files (cfg_defines.h removed; tensor/,
                                   noc/att uncapped config and data, overlay/
                                   addrgen_state.h added)
  - soc_descriptors/               quasar_qsr3a2_arch.yaml added
  - core_descriptors/              quasar_qsr3a2_arch.yaml and
                                   quasar_qsr3a2_arch_tensix_dispatch.yaml added
  - hw/inc/internal/tt-2xx/        dataflow_buffer_test_helpers.h was in the tree
                                   but not listed, now listed
  - expanded table: added rows for noc/att/ and the ATT tests
    (tests/tt_metal/tt_metal/noc/)
  - hw/firmware/src/tt-2xx/quasar/ holds noc.c and noc_address_translation_tables.cpp
    only; fds_functions.cpp was not in the tree at either tag and is removed
    from the row (fds_functions.hpp is in hw/inc/internal/tt-2xx/quasar/overlay/)
-->

Table of Contents
=================

- [Kernel Ops Delivery](#kernel-ops-delivery---folder-structure)
- [LLK Delivery](#llk-delivery---folder-structure)
- [Runtime Delivery](#runtime-delivery---folder-structure)

# Kernel Ops Delivery - Folder Structure

**Feature(s) this release:** Quasar Qwen3-VL-2B related Kernel Ops (AIIPSW-85); Quasar YOLO related Kernel Ops (AIIPSW-86); Horizon port of Qwen3-VL-2B Kernel Ops from Quasar (AIIPSW-128); Trinity port of Llama Kernel Ops from Quasar (AIIPSW-27); Trinity port of YOLO Kernel Ops from Quasar (AIIPSW-35)
**Source:** [tenstorrent/tt-metal - ttnn](https://github.com/tenstorrent/tt-metal/tree/v0.82.0/ttnn)

## Level 1 - Common (used by Wormhole, Blackhole, Quasar, and any new designs)

 - [tt-metal/ttnn/cpp/ttnn/operations/](https://github.com/tenstorrent/tt-metal/tree/v0.82.0/ttnn/cpp/ttnn/operations) - the op library (33 op families; see breakdown below)
 - [tt-metal/ttnn/cpp/ttnn/kernel/](https://github.com/tenstorrent/tt-metal/tree/v0.82.0/ttnn/cpp/ttnn/kernel) - device-side kernel entry points shared across ops
 - [tt-metal/ttnn/cpp/ttnn/kernel_lib/](https://github.com/tenstorrent/tt-metal/tree/v0.82.0/ttnn/cpp/ttnn/kernel_lib) - shared device-kernel helper library
 - [tt-metal/ttnn/cpp/ttnn/graph/](https://github.com/tenstorrent/tt-metal/tree/v0.82.0/ttnn/cpp/ttnn/graph) - graph capture / trace support
 - [tt-metal/ttnn/cpp/ttnn-nanobind/](https://github.com/tenstorrent/tt-metal/tree/v0.82.0/ttnn/cpp/ttnn-nanobind) - Python binding layer
 - [tt-metal/ttnn/api/ttnn/](https://github.com/tenstorrent/tt-metal/tree/v0.82.0/ttnn/api/ttnn) - public C++ headers (tensor/, distributed/, graph/, common/, services/)
 - [tt-metal/ttnn/core/](https://github.com/tenstorrent/tt-metal/tree/v0.82.0/ttnn/core) - core implementation (tensor/, distributed/, graph/, services/)
 - [tt-metal/ttnn/ttnn/](https://github.com/tenstorrent/tt-metal/tree/v0.82.0/ttnn/ttnn) - Python API (operations/, distributed/, experimental_loader/)
 - [tt-metal/ttnn/tt_lib/](https://github.com/tenstorrent/tt-metal/tree/v0.82.0/ttnn/tt_lib) - legacy compatibility layer (fused_ops/, fallback_ops/)
 - [tt-metal/ttnn/examples/](https://github.com/tenstorrent/tt-metal/tree/v0.82.0/ttnn/examples) - reference ops (add/, lab_eltwise_binary/, lab_multicast/)
 - [tt-metal/ttnn/tutorials/](https://github.com/tenstorrent/tt-metal/tree/v0.82.0/ttnn/tutorials) - tutorials (basic_python/)
 - [tt-metal/ttnn/test/](https://github.com/tenstorrent/tt-metal/tree/v0.82.0/ttnn/test) - library-local tests

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
| "index_fill/, index_fill_new/, loss/, moreh/" | Misc |
| kernel_helper_functions/ | Shared kernel-side helpers |
| experimental/ | 39 experimental families incl. quasar/ (see Level 2) |

## Level 2 - Architecture (per chip family)

Ops are written once against the Metalium API, and the hardware abstraction layer resolves the chip underneath. Most of the library is therefore chip-independent and sits at Level 1. Architecture-specific ops are delivered in a dedicated tree.

| Level | Architecture | Folder | Contents |
| --- | --- | --- | --- |
| L2.1 | Wormhole (B0) | — | No architecture-specific op folder; runs the Level 1 op library unchanged |
| L2.2 | Blackhole | — | No architecture-specific op folder; runs the Level 1 op library unchanged |
| L2.3 | Quasar | [tt-metal/ttnn/cpp/ttnn/operations/experimental/quasar/](https://github.com/tenstorrent/tt-metal/tree/v0.82.0/ttnn/cpp/ttnn/operations/experimental/quasar) | 31 op folders and a shared common/ folder - compute, normalization, convolution, transformer, layout, sharding, slicing, movement |

### L2.3 Quasar - expanded

| Group | Ops |
| --- | --- |
| Compute | matmul/ (incl. sparse/), binary/, binary_ng/, reduction/generic/, typecast/ |
| Normalization | batch_norm/, softmax/ |
| Convolution / vision | conv2d/, pool_generic/, halo/, fold/ |
| Transformer | transformer/ (sdpa/, sdpa_decode/) |
| Layout | tilize/, tilize_with_val_padding/, untilize/, untilize_with_unpadding/, transpose/, reshape_view/ |
| Sharding | interleaved_to_sharded/, sharded_to_interleaved/, reshard/ |
| Slicing / padding | slice/, slice_write/, padded_slice/, pad/, op_slicing/ |
| Movement / placement | move/, reallocate/, to_device/, to_layout/, to_memory_config/ |
| Shared | common/ (thread_plan.hpp) |
| Model op tests | [models/experimental/ops/quasar/tests/](https://github.com/tenstorrent/tt-metal/tree/v0.82.0/models/experimental/ops/quasar/tests) - qwen3_vl_ops/ (Qwen3-VL-2B), yolo_ops/ (YOLO), oft_ops/ (OFT), gpt_oss_ops/, vad_variant_ops/ |
| Model code | [models/experimental/ops/quasar/qwen3_vl/](https://github.com/tenstorrent/tt-metal/tree/v0.82.0/models/experimental/ops/quasar/qwen3_vl) (Qwen3-VL-2B), [models/experimental/llama32_1b_quasar/](https://github.com/tenstorrent/tt-metal/tree/v0.82.0/models/experimental/llama32_1b_quasar) (Llama 3.2 1B) |

Each op folder follows the standard shape: device/ (program factory), device/kernels/{compute, dataflow}/ (device kernels), and the host-side op definition.


## Level 3 - Sub-architecture / program (the customer delivery)

| Level | Program | Delivered from | Feature this release |
| --- | --- | --- | --- |
| L3.1 | 2.0.0 (Quasar) | ttnn/cpp/ttnn/operations/experimental/quasar/ | Quasar Qwen3-VL-2B related Kernel Ops (AIIPSW-85); Quasar YOLO related Kernel Ops (AIIPSW-86) |
| L3.2 | 2.0.1 (Horizon) | ttnn/cpp/ttnn/operations/experimental/quasar/ | Horizon port of Qwen3-VL-2B Kernel Ops from Quasar (AIIPSW-128) |
| L3.3 | 2.0.2 (Trinity) | ttnn/cpp/ttnn/operations/experimental/quasar/ | Trinity port of Llama Kernel Ops from Quasar (AIIPSW-27); Trinity port of YOLO Kernel Ops from Quasar (AIIPSW-35) |

**This delivery contains:** all of Level 1 plus the full Quasar op tree (experimental/quasar/).


# LLK Delivery - Folder Structure

**Feature(s) this release:** Quasar OFT related LLK features (AIIPSW-100)
**Source:** [tenstorrent/tt-metal - tt_metal/tt-llk](https://github.com/tenstorrent/tt-metal/tree/v0.82.0/tt_metal/tt-llk)

## Level 1 - Common (used by Wormhole, Blackhole, Quasar, and any new designs)

- [tt-metal/tt_metal/tt-llk/common/](https://github.com/tenstorrent/tt-metal/tree/v0.82.0/tt_metal/tt-llk/common)- cross-chip LLK helpers (ckernel_fence.h, ckernel_sfpu_welfords_common.h, llk_assert.h, tensor_shape.h, tensor_shape_coverage{,_math,_pack,_unpack}.h)
- [tt-metal/tt_metal/tt-llk/tests/](https://github.com/tenstorrent/tt-metal/tree/v0.82.0/tt_metal/tt-llk/tests)- shared LLK test harness (helpers/, python_tests/, sources/, hardware_free/)
- [tt-metal/tt_metal/tt-llk/docs/](https://github.com/tenstorrent/tt-metal/tree/v0.82.0/tt_metal/tt-llk/docs)- shared documentation (common/, llk/{l1,l2,l3}, performance_counters/, tests/)
- [tt-metal/tt_metal/tt-llk/infra/](https://github.com/tenstorrent/tt-metal/tree/v0.82.0/tt_metal/tt-llk/infra)- shared build / formatting tooling (fix_cstdint.py, run_order_processing.py, check_*.py source checks)

## Level 2 - Architecture (per chip family)

| **Level** | **Architecture** | **Folder** | **Contents** |
| --- | --- | --- | --- |
| L2.1 | Wormhole (B0) | [tt-metal/tt_metal/tt-llk/tt_llk_wormhole_b0/](https://github.com/tenstorrent/tt-metal/tree/v0.82.0/tt_metal/tt-llk/tt_llk_wormhole_b0) | common/inc/sfpu/experimental, instructions/, llk_lib/{debug, experimental} |
| L2.2 | Wormhole (B0) | [tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/](https://github.com/tenstorrent/tt-metal/tree/v0.82.0/tt_metal/hw/ckernels/wormhole_b0/metal/llk_api) | llk_sfpu |
| L2.3 | Blackhole | [tt-metal/tt_metal/tt-llk/tt_llk_blackhole/](https://github.com/tenstorrent/tt-metal/tree/v0.82.0/tt_metal/tt-llk/tt_llk_blackhole) | common/inc/sfpu/experimental, instructions/, llk_lib/{debug, experimental, hal} |
| L2.4 | Blackhole | [tt_metal/hw/ckernels/blackhole/metal/llk_api/](https://github.com/tenstorrent/tt-metal/tree/v0.82.0/tt_metal/hw/ckernels/blackhole/metal/llk_api) | llk_sfpu |
| L2.5 | Quasar | [tt-metal/tt_metal/tt-llk/tt_llk_quasar/](https://github.com/tenstorrent/tt-metal/tree/v0.82.0/tt_metal/tt-llk/tt_llk_quasar) | arch/ (saturn, trinity variants), common/inc/{experimental, internal}, instructions/ (assembly.yaml), llk_lib/ |
| L2.6 | Quasar | [tt_metal/hw/ckernels/quasar/metal/llk_api/](https://github.com/tenstorrent/tt-metal/tree/v0.82.0/tt_metal/hw/ckernels/quasar/metal/llk_api) | llk_sfpu |

### L2.5-L2.6 Quasar - expanded

| **Folder** | **Contents** |
| --- | --- |
| [tt_llk_quasar/llk_lib/](https://github.com/tenstorrent/tt-metal/tree/v0.82.0/tt_metal/tt-llk/tt_llk_quasar/llk_lib) | 39 headers — unpack (llk_unpack_common.h, llk_unpack_matmul.h, llk_unpack_tilize.h, llk_unpack_tilize_to_dest.h, llk_unpack_reduce.h, llk_unpack_unary_operand.h, llk_unpack_unary_operand_to_dest.h, llk_unpack_binary_operands.h, llk_unpack_*_broadcast_operands.h, llk_unpack_reduce_col_tilizeA_strided.h), math (llk_math_common.h, llk_math_matmul.h, llk_math_reduce.h, llk_math_eltwise_binary.h, llk_math_eltwise_binary_broadcast.h, llk_math_eltwise_binary_sfpu.h, llk_math_eltwise_unary_sfpu.h, llk_math_eltwise_ternary_sfpu.h, llk_math_eltwise_sfpu_common.h, llk_math_eltwise_unary_datacopy.h, llk_math_unary_broadcast.h, llk_math_transpose_dest.h), pack (llk_pack.h, llk_pack_common.h, llk_pack_matmul.h, llk_pack_untilize.h), plus llk_defs.h, llk_srcs.h, llk_sync.h, llk_memory_checks.h, and 8 headers in experimental/ (incl. llk_unpack_gather_compress.h, llk_math_gather_compress.h, llk_pack_gather_compress.h) |
| [tt_llk_quasar/common/inc/](https://github.com/tenstorrent/tt-metal/tree/v0.82.0/tt_metal/tt-llk/tt_llk_quasar/common/inc) | cfg_defines.h, ckernel.h, ckernel_addrmod.h, ckernel_buf_desc.h, ckernel_debug.h, ckernel_defs.h, ckernel_dest.h, ckernel_gpr_map.h, ckernel_include.h, ckernel_instr_params.h, ckernel_ops.h, ckernel_pcbuf.h, ckernel_proj_params.h, ckernel_risc_atomics.h, ckernel_riscv_debug.h, ckernel_sfpu.h, ckernel_template.h, ckernel_trisc_common.h, ckernel_trisc_id.h, ckernel_vector.h, cmath_common.h, cpack_common.h, cunpack_common.h, llk_bfd_alloc.h, llk_reinit_guard.h, llk_tdma_guard.h, sfpu_impl.h, sfpu_reg.h |
| [tt_llk_quasar/common/inc/experimental/](https://github.com/tenstorrent/tt-metal/tree/v0.82.0/tt_metal/tt-llk/tt_llk_quasar/common/inc/experimental) | ckernel_sfpu_abs.h, ckernel_sfpu_fill.h, ckernel_sfpu_swiglu.h — include only if in scope |
| [tt_llk_quasar/instructions/](https://github.com/tenstorrent/tt-metal/tree/v0.82.0/tt_metal/tt-llk/tt_llk_quasar/instructions) | assembly.yaml — instruction definitions |
| [tt_metal/hw/ckernels/quasar/metal/llk_api](https://github.com/tenstorrent/tt-metal/tree/v0.82.0/tt_metal/hw/ckernels/quasar/metal/llk_api) | 174 headers (all Quasar SFPU kernels are in llk_sfpu/)<br>experimental/llk_math_eltwise_binary_custom_api.h<br>experimental/llk_math_matmul_custom_api.h<br>experimental/llk_math_reduce_custom_api.h<br>experimental/llk_math_reduce_custom_runtime_api.h<br>experimental/llk_unpack_AB_reduce_custom_api.h<br>experimental/llk_unpack_AB_reduce_custom_runtime_api.h<br>experimental/llk_unpack_AB_sub_bcast_col_custom_api.h<br>llk_math_binary_api.h<br>llk_math_common_api.h<br>llk_math_matmul_api.h<br>llk_math_reduce_api.h<br>llk_math_transpose_dest_api.h<br>llk_math_unary_datacopy_api.h<br>llk_pack_common_api.h<br>llk_pack_reduce_api.h<br>llk_pack_tile_api.h<br>llk_pack_untilize_api.h<br>llk_sfpu/ckernel_sfpu_abs.h<br>llk_sfpu/ckernel_sfpu_activations.h<br>llk_sfpu/ckernel_sfpu_add.h<br>llk_sfpu/ckernel_sfpu_add1.h<br>llk_sfpu/ckernel_sfpu_add_top_row.h<br>llk_sfpu/ckernel_sfpu_addcdiv.h<br>llk_sfpu/ckernel_sfpu_addcmul.h<br>llk_sfpu/ckernel_sfpu_alt_complex_rotate90.h<br>llk_sfpu/ckernel_sfpu_atan2.h<br>llk_sfpu/ckernel_sfpu_binary.h<br>llk_sfpu/ckernel_sfpu_binary_bcast.h<br>llk_sfpu/ckernel_sfpu_binary_bitwise.h<br>llk_sfpu/ckernel_sfpu_binary_comp.h<br>llk_sfpu/ckernel_sfpu_binary_fmod.h<br>llk_sfpu/ckernel_sfpu_binary_max_min.h<br>llk_sfpu/ckernel_sfpu_binary_pow.h<br>llk_sfpu/ckernel_sfpu_binary_remainder.h<br>llk_sfpu/ckernel_sfpu_binop_with_unary.h<br>llk_sfpu/ckernel_sfpu_bitwise.h<br>llk_sfpu/ckernel_sfpu_bitwise_not.h<br>llk_sfpu/ckernel_sfpu_cast_fp32_to_fp16a.h<br>llk_sfpu/ckernel_sfpu_cbrt.h<br>llk_sfpu/ckernel_sfpu_celu.h<br>llk_sfpu/ckernel_sfpu_clamp.h<br>llk_sfpu/ckernel_sfpu_clamped_silu_glu.h<br>llk_sfpu/ckernel_sfpu_comp.h<br>llk_sfpu/ckernel_sfpu_conversions.h<br>llk_sfpu/ckernel_sfpu_converter.h<br>llk_sfpu/ckernel_sfpu_copy_dest_values.h<br>llk_sfpu/ckernel_sfpu_cumprod.h<br>llk_sfpu/ckernel_sfpu_cumsum.h<br>llk_sfpu/ckernel_sfpu_digamma.h<br>llk_sfpu/ckernel_sfpu_div_int32.h<br>llk_sfpu/ckernel_sfpu_div_int32_floor.h<br>llk_sfpu/ckernel_sfpu_dropout.h<br>llk_sfpu/ckernel_sfpu_elu.h<br>llk_sfpu/ckernel_sfpu_ema.h<br>llk_sfpu/ckernel_sfpu_erf.h<br>llk_sfpu/ckernel_sfpu_erfc.h<br>llk_sfpu/ckernel_sfpu_erfinv.h<br>llk_sfpu/ckernel_sfpu_exp.h<br>llk_sfpu/ckernel_sfpu_exp2.h<br>llk_sfpu/ckernel_sfpu_expm1.h<br>llk_sfpu/ckernel_sfpu_expm1_cw.h<br>llk_sfpu/ckernel_sfpu_fill.h<br>llk_sfpu/ckernel_sfpu_fmod.h<br>llk_sfpu/ckernel_sfpu_gcd.h<br>llk_sfpu/ckernel_sfpu_gelu.h<br>llk_sfpu/ckernel_sfpu_hardmish.h<br>llk_sfpu/ckernel_sfpu_hardshrink.h<br>llk_sfpu/ckernel_sfpu_hardtanh.h<br>llk_sfpu/ckernel_sfpu_heaviside.h<br>llk_sfpu/ckernel_sfpu_i0.h<br>llk_sfpu/ckernel_sfpu_i1.h<br>llk_sfpu/ckernel_sfpu_identity.h<br>llk_sfpu/ckernel_sfpu_int_sum.h<br>llk_sfpu/ckernel_sfpu_is_fp16_zero.h<br>llk_sfpu/ckernel_sfpu_isclose.h<br>llk_sfpu/ckernel_sfpu_isinf_isnan.h<br>llk_sfpu/ckernel_sfpu_lcm.h<br>llk_sfpu/ckernel_sfpu_lerp.h<br>llk_sfpu/ckernel_sfpu_lgamma.h<br>llk_sfpu/ckernel_sfpu_log.h<br>llk_sfpu/ckernel_sfpu_log1p.h<br>llk_sfpu/ckernel_sfpu_logaddexp.h<br>llk_sfpu/ckernel_sfpu_logaddexp2.h<br>llk_sfpu/ckernel_sfpu_logical_not.h<br>llk_sfpu/ckernel_sfpu_logsigmoid.h<br>llk_sfpu/ckernel_sfpu_mac.h<br>llk_sfpu/ckernel_sfpu_mask.h<br>llk_sfpu/ckernel_sfpu_max_pool_indices.h<br>llk_sfpu/ckernel_sfpu_mish.h<br>llk_sfpu/ckernel_sfpu_mul_int32.h<br>llk_sfpu/ckernel_sfpu_negative.h<br>llk_sfpu/ckernel_sfpu_operand.h<br>llk_sfpu/ckernel_sfpu_piecewise_rational.h<br>llk_sfpu/ckernel_sfpu_polygamma.h<br>llk_sfpu/ckernel_sfpu_polyval.h<br>llk_sfpu/ckernel_sfpu_prelu.h<br>llk_sfpu/ckernel_sfpu_quant.h<br>llk_sfpu/ckernel_sfpu_rand.h<br>llk_sfpu/ckernel_sfpu_rdiv.h<br>llk_sfpu/ckernel_sfpu_recip.h<br>llk_sfpu/ckernel_sfpu_reduce.h<br>llk_sfpu/ckernel_sfpu_relu.h<br>llk_sfpu/ckernel_sfpu_remainder.h<br>llk_sfpu/ckernel_sfpu_replay_bank1.h<br>llk_sfpu/ckernel_sfpu_reshuffle_rows.h<br>llk_sfpu/ckernel_sfpu_rounding_ops.h<br>llk_sfpu/ckernel_sfpu_rpow.h<br>llk_sfpu/ckernel_sfpu_rsqrt.h<br>llk_sfpu/ckernel_sfpu_rsub_int32.h<br>llk_sfpu/ckernel_sfpu_selu.h<br>llk_sfpu/ckernel_sfpu_shift.h<br>llk_sfpu/ckernel_sfpu_sigmoid.h<br>llk_sfpu/ckernel_sfpu_sigmoid_appx.h<br>llk_sfpu/ckernel_sfpu_sign.h<br>llk_sfpu/ckernel_sfpu_signbit.h<br>llk_sfpu/ckernel_sfpu_silu.h<br>llk_sfpu/ckernel_sfpu_situ_glu.h<br>llk_sfpu/ckernel_sfpu_snake_beta.h<br>llk_sfpu/ckernel_sfpu_softcap.h<br>llk_sfpu/ckernel_sfpu_softplus.h<br>llk_sfpu/ckernel_sfpu_softshrink.h<br>llk_sfpu/ckernel_sfpu_softsign.h<br>llk_sfpu/ckernel_sfpu_sqrt.h<br>llk_sfpu/ckernel_sfpu_sqrt_custom.h<br>llk_sfpu/ckernel_sfpu_square.h<br>llk_sfpu/ckernel_sfpu_srcs.h<br>llk_sfpu/ckernel_sfpu_sub_int.h<br>llk_sfpu/ckernel_sfpu_tanh.h<br>llk_sfpu/ckernel_sfpu_tanh_derivative.h<br>llk_sfpu/ckernel_sfpu_tanhshrink.h<br>llk_sfpu/ckernel_sfpu_threshold.h<br>llk_sfpu/ckernel_sfpu_tiled_prod.h<br>llk_sfpu/ckernel_sfpu_topk.h<br>llk_sfpu/ckernel_sfpu_trigonometry.h<br>llk_sfpu/ckernel_sfpu_typecast.h<br>llk_sfpu/ckernel_sfpu_typecast_fp16b_uint16.h<br>llk_sfpu/ckernel_sfpu_typecast_fp32_to_uint16.h<br>llk_sfpu/ckernel_sfpu_typecast_int32_fp16b.h<br>llk_sfpu/ckernel_sfpu_typecast_int32_fp32.h<br>llk_sfpu/ckernel_sfpu_typecast_uint16_fp32.h<br>llk_sfpu/ckernel_sfpu_unary_comp.h<br>llk_sfpu/ckernel_sfpu_unary_max_min.h<br>llk_sfpu/ckernel_sfpu_unary_power.h<br>llk_sfpu/ckernel_sfpu_unary_shift.h<br>llk_sfpu/ckernel_sfpu_welfords.h<br>llk_sfpu/ckernel_sfpu_where.h<br>llk_sfpu/ckernel_sfpu_xielu.h<br>llk_sfpu/llk_math_eltwise_binary_sfpu_add_int.h<br>llk_sfpu/llk_math_eltwise_binary_sfpu_binary_comp.h<br>llk_sfpu/llk_math_eltwise_binary_sfpu_init.h<br>llk_sfpu/llk_math_eltwise_binary_sfpu_macros.h<br>llk_sfpu/llk_math_eltwise_binary_sfpu_max_min.h<br>llk_sfpu/llk_math_eltwise_binary_sfpu_mul_int.h<br>llk_sfpu/llk_math_eltwise_ternary_sfpu_init.h<br>llk_sfpu/llk_math_eltwise_ternary_sfpu_macros.h<br>llk_sfpu/llk_math_eltwise_ternary_sfpu_where.h<br>llk_sfpu/llk_math_eltwise_unary_sfpu_binop_with_scalar.h<br>llk_sfpu/llk_math_eltwise_unary_sfpu_init.h<br>llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h<br>llk_sfpu/llk_math_eltwise_unary_sfpu_rsqrt.h<br>llk_sfpu/llk_math_ema_sfpu_entry.h<br>llk_sfpu/llk_math_gcd_sfpu_entry.h<br>llk_sfpu/llk_math_lcm_sfpu_entry.h<br>llk_sfpu/llk_math_triangle_solve_sfpu_entry.h<br>llk_sfpu/llk_math_unary_max_min_sfpu_entry.h<br>llk_sfpu/llk_math_welfords_sfpu_entry.h<br>llk_sfpu/llk_triangle_solve_sync.h<br>llk_sfpu_srcs_api.h<br>llk_unpack_AB_api.h<br>llk_unpack_AB_matmul_api.h<br>llk_unpack_AB_reduce_api.h<br>llk_unpack_A_api.h<br>llk_unpack_common_api.h<br>llk_unpack_tilize_api.h |

## Level 3 - Sub-architecture / program (the customer delivery)

| **Level** | **Program** | **Delivered from** | **Feature this release** |
| --- | --- | --- | --- |
| L3.1 | 2.0.0 (Quasar) | tt_metal/tt-llk/tt_llk_quasar/ | Quasar OFT related LLK features (AIIPSW-100) |
| L3.2 | 2.0.1 (Horizon) | tt_metal/tt-llk/tt_llk_quasar/ |  |
| L3.3 | 2.0.2 (Trinity) | tt_metal/tt-llk/tt_llk_quasar/ (four-row headers from arch/trinity/, selected from the IP variant) |  |

**This delivery contains:**  All of Level 1 (common/, tests/, docs/, infra/) plus the full Quasar architecture folder (tt_llk_quasar/).


# Runtime Delivery - Folder Structure

**Feature(s) this release:** Horizon ATT / flat address UD, UE (AIIPSW-78); Horizon Runtime D3 verification of multiple programs across partitions (AIIPSW-80); Trinity debug tools: tt-npe (AIIPSW-39)

## Level 1 - Common (used by Wormhole, Blackhole, Quasar, and any new designs)

- [tt-metal/tt_metal/api/](https://github.com/tenstorrent/tt-metal/tree/v0.82.0/tt_metal/api)- public C++ API (tt-metalium/ headers, serialized descriptors)
- [tt-metal/tt_metal/impl/](https://github.com/tenstorrent/tt-metal/tree/v0.82.0/tt_metal/impl)- core runtime (device/, dispatch/, program/, buffers/, allocator/, kernels/, trace/, sub_device/, event/, profiler/, debug/, context/, threading/)
- [tt-metal/tt_metal/llrt/](https://github.com/tenstorrent/tt-metal/tree/v0.82.0/tt_metal/llrt)- low-level runtime (llrt_common/, hal/codegen/)
- [tt-metal/tt_metal/jit_build/](https://github.com/tenstorrent/tt-metal/tree/v0.82.0/tt_metal/jit_build)- kernel JIT compilation and build orchestration
- [tt-metal/tt_metal/common/](https://github.com/tenstorrent/tt-metal/tree/v0.82.0/tt_metal/common)- shared host utilities
- [tt-metal/tt_metal/detail/](https://github.com/tenstorrent/tt-metal/tree/v0.82.0/tt_metal/detail)- internal host-side helpers
- [tt-metal/tt_metal/hostdevcommon/](https://github.com/tenstorrent/tt-metal/tree/v0.82.0/tt_metal/hostdevcommon)- host / device shared definitions
- [tt-metal/tt_metal/logging/](https://github.com/tenstorrent/tt-metal/tree/v0.82.0/tt_metal/logging)- runtime logging
- [tt-metal/tt_metal/distributed/](https://github.com/tenstorrent/tt-metal/tree/v0.82.0/tt_metal/distributed)- multi-device / mesh (multihost/, flatbuffer/, layer_completion/)
- [tt-metal/tt_metal/fabric/](https://github.com/tenstorrent/tt-metal/tree/v0.82.0/tt_metal/fabric)- fabric + CCL routing (builder/, ccl/, impl/, hw/, config/, mesh_graph_descriptors/, cabling_descriptors/, experimental/)
- [tt-metal/tt_metal/hw/inc/api/](https://github.com/tenstorrent/tt-metal/tree/v0.82.0/tt_metal/hw/inc/api)- device-side kernel API (dataflow/, compute/, debug/, tensor/, numeric/)
- [tt-metal/tt_metal/hw/inc/internal/](https://github.com/tenstorrent/tt-metal/tree/v0.82.0/tt_metal/hw/inc/internal)- device-side internals (dataflow/, debug/, ethernet/, tensor/, local_mem/, profiler/)
- [tt-metal/tt_metal/hw/inc/hostdev/](https://github.com/tenstorrent/tt-metal/tree/v0.82.0/tt_metal/hw/inc/hostdev)- host / device message structs
- [tt-metal/tt_metal/hw/toolchain/](https://github.com/tenstorrent/tt-metal/tree/v0.82.0/tt_metal/hw/toolchain)- linker scripts and crt0 (main.ld, script_tng.ld, tmu-crt0.S)
- [tt-metal/tt_metal/third_party/](https://github.com/tenstorrent/tt-metal/tree/v0.82.0/tt_metal/third_party)- umd/, tracy/, tt-cluster-descriptors/
- [tt-metal/tt_metal/programming_examples/](https://github.com/tenstorrent/tt-metal/tree/v0.82.0/tt_metal/programming_examples)- reference programs
- [tt-metal/tt_metal/python_env/](https://github.com/tenstorrent/tt-metal/tree/v0.82.0/tt_metal/python_env)- Python environment definition
- [tt-metal/tt_metal/test/](https://github.com/tenstorrent/tt-metal/tree/v0.82.0/tt_metal/test)- runtime-local tests

## Level 2 - Architecture (per chip family)

Runtime code is organized by RISC generation: tt-1xx covers Wormhole and Blackhole, tt-2xx covers Quasar. Each generation folder holds code shared by every chip in that generation, with the chip as a leaf beneath it. Both are delivered together.

| **Level** | **Architecture** | **Folder** | **Contents** |
| --- | --- | --- | --- |
| L2.1 | Wormhole (B0) | [llrt/hal/tt-1xx/](https://github.com/tenstorrent/tt-metal/tree/v0.82.0/tt_metal/llrt/hal/tt-1xx),[hw/firmware/src/tt-1xx/](https://github.com/tenstorrent/tt-metal/tree/v0.82.0/tt_metal/hw/firmware/src/tt-1xx),[hw/inc/internal/tt-1xx/](https://github.com/tenstorrent/tt-metal/tree/v0.82.0/tt_metal/hw/inc/internal/tt-1xx),[hw/ckernels/wormhole_b0/](https://github.com/tenstorrent/tt-metal/tree/v0.82.0/tt_metal/hw/ckernels/wormhole_b0) | wormhole/ leaf in each; soc_descriptors/wormhole_b0_80_arch.yaml, core_descriptors/wormhole_b0_*.yaml |
| L2.2 | Blackhole | [llrt/hal/tt-1xx/](https://github.com/tenstorrent/tt-metal/tree/v0.82.0/tt_metal/llrt/hal/tt-1xx),[hw/firmware/src/tt-1xx/](https://github.com/tenstorrent/tt-metal/tree/v0.82.0/tt_metal/hw/firmware/src/tt-1xx),[hw/inc/internal/tt-1xx/](https://github.com/tenstorrent/tt-metal/tree/v0.82.0/tt_metal/hw/inc/internal/tt-1xx),[hw/ckernels/blackhole/](https://github.com/tenstorrent/tt-metal/tree/v0.82.0/tt_metal/hw/ckernels/blackhole) | blackhole/ leaf in each; soc_descriptors/blackhole_140_arch.yaml, core_descriptors/blackhole_*.yaml |
| **L2.3** | **Quasar** | [llrt/hal/tt-2xx/](https://github.com/tenstorrent/tt-metal/tree/v0.82.0/tt_metal/llrt/hal/tt-2xx),[hw/firmware/src/tt-2xx/](https://github.com/tenstorrent/tt-metal/tree/v0.82.0/tt_metal/hw/firmware/src/tt-2xx),[hw/inc/internal/tt-2xx/](https://github.com/tenstorrent/tt-metal/tree/v0.82.0/tt_metal/hw/inc/internal/tt-2xx),[hw/ckernels/quasar/](https://github.com/tenstorrent/tt-metal/tree/v0.82.0/tt_metal/hw/ckernels/quasar) | quasar/ leaf in each; soc_descriptors/quasar_32_arch.yaml, quasar_qsr3a2_arch.yaml, core_descriptors/quasar_simulation_*.yaml, quasar_qsr3a2_arch*.yaml |

### L2.3 Quasar - expanded

| **Folder** | **Contents** |
| --- | --- |
| [tt_metal/llrt/hal/tt-2xx/](https://github.com/tenstorrent/tt-metal/tree/v0.82.0/tt_metal/llrt/hal/tt-2xx) | hal_2xx_common.cpp/.hpp, sources.cmake |
| [tt_metal/llrt/hal/tt-2xx/quasar/](https://github.com/tenstorrent/tt-metal/tree/v0.82.0/tt_metal/llrt/hal/tt-2xx/quasar) | qa_hal.cpp/.hpp, qa_hal_tensix.cpp, qa_hal_dispatch.cpp, qa_hal_active_eth.cpp, qa_hal_idle_eth.cpp, qa_att_windows.hpp, qa_emulator_models.hpp, *_asserts.hpp |
| [tt_metal/hw/firmware/src/tt-2xx/](https://github.com/tenstorrent/tt-metal/tree/v0.82.0/tt_metal/hw/firmware/src/tt-2xx) | dm.cc, dmk.cc, dispatch_dm.cc, trisc.cc, trisck.cc |
| [tt_metal/hw/firmware/src/tt-2xx/quasar/](https://github.com/tenstorrent/tt-metal/tree/v0.82.0/tt_metal/hw/firmware/src/tt-2xx/quasar) | noc.c, noc_address_translation_tables.cpp |
| [tt_metal/hw/inc/internal/tt-2xx/](https://github.com/tenstorrent/tt-metal/tree/v0.82.0/tt_metal/hw/inc/internal/tt-2xx) | risc_common.h, dataflow_buffer.inl, dataflow_buffer/, dataflow_buffer_test_helpers.h, noc_zero_l1.inl, worker_go_signalling.h; tt-2.0.0/, tt-2.0.1/, tt-2.0.2/ (per-variant register definitions, meta/) |
| [tt_metal/hw/inc/internal/tt-2xx/quasar/](https://github.com/tenstorrent/tt-metal/tree/v0.82.0/tt_metal/hw/inc/internal/tt-2xx/quasar) | cache.h, core_config.h, dev_mem_map.h, dram_address_map.h, eth_l1_address_map.h, tensix*.h, stream_interface.h, plic.hpp, semaphore_cached_pool.h, noc/ (+ registers/, att/), overlay/, tensor/ |
| [tt_metal/hw/inc/internal/tt-2xx/quasar/noc/att/](https://github.com/tenstorrent/tt-metal/tree/v0.82.0/tt_metal/hw/inc/internal/tt-2xx/quasar/noc/att) | att.h, att_address.h, att_config.h, noc_address_backend_att.h, configs/ (per-chip ATT configs, incl. horizon_2x3_att_config.h), temporary_programming/ |
| [tests/tt_metal/tt_metal/noc/](https://github.com/tenstorrent/tt-metal/tree/v0.82.0/tests/tt_metal/tt_metal/noc) | ATT tests: test_quasar_att_address.cpp, test_quasar_att_config.cpp, test_quasar_att_image.cpp |
| [tt_metal/hw/ckernels/quasar/metal/](https://github.com/tenstorrent/tt-metal/tree/v0.82.0/tt_metal/hw/ckernels/quasar/metal) | llk_api/ (+ llk_sfpu/), llk_io/, common/ |

## Level 3 - Sub-architecture / program (the customer delivery)

| **Level** | **Program** | **Delivered from** | **Feature this release** |
| --- | --- | --- | --- |
| L3.1 | 2.0.0 (Quasar) | tt_metal/.../tt-2xx/quasar/ |  |
| L3.2 | 2.0.1 (Horizon) | tt_metal/.../tt-2xx/quasar/ | Horizon ATT / flat address UD, UE (AIIPSW-78); Horizon Runtime D3 verification of multiple programs across partitions (AIIPSW-80) |
| L3.3 | 2.0.2 (Trinity) | tt_metal/.../tt-2xx/quasar/ | Trinity debug tools: tt-npe (AIIPSW-39) |

**This delivery contains:** all of Level 1 plus the full Quasar generation and chip folders (llrt/hal/tt-2xx/, hw/firmware/src/tt-2xx/, hw/inc/internal/tt-2xx/, hw/ckernels/quasar/)
