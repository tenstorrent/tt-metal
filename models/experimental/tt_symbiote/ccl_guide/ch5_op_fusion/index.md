# Chapter 5: Op Fusion (CCL + Matmul)

Op fusion combines a collective communication operation and a matrix multiplication into a single Metal program that pipelines the two at the tile level — eliminating the DRAM round-trip between them.

All fused operations in this chapter are under `ttnn::experimental`. They are production-quality but carry no stability guarantee across releases.

---

## Table of Contents

| Section | File | Description |
|---------|------|-------------|
| 5.1 | [Why Fusion Matters](why_fusion.md) | DRAM round-trip cost, bandwidth math, FusedOpSignaler mechanism, when fusion helps |
| 5.2 | [Fused Ops](fused_ops.md) | `all_gather_matmul_async`, `matmul_reduce_scatter_async`, `strided_all_gather_minimal_matmul_async` |
| 5.3 | [Llama Fused Ops](llama_fused_ops.md) | `llama_all_gather_matmul_async`, `llama_rs_matmul`, `all_gather_concat`, `fused_rms_1_1_32_8192` |

---

Op fusion eliminates the DRAM round-trip between a collective and its following matmul by sharing an L1 circular buffer between them. See [§5.1](why_fusion.md) for the performance model, bandwidth math, and the FusedOpSignaler mechanism.

---

## New parameters in Ch5

| Parameter | Type | Appears in | Purpose |
|-----------|------|------------|---------|
| `all_gather_core_grid_offset` | `CoreCoord` | `all_gather_matmul_async` | Offsets the AllGather workers on the core grid to avoid overlap with matmul workers |
| `reduce_scatter_core_grid_offset` | `CoreCoord` | `matmul_reduce_scatter_async` | Same for the ReduceScatter side |
| `strided_all_gather_core_grid_offset` | `CoreCoord` | `strided_all_gather_minimal_matmul_async` | Core offset for the strided AllGather variant |
| `memory_config_ag` / `memory_config_mm` | `MemoryConfig` | all fused ops | Separate memory configs for the CCL and matmul outputs |
| `intermediate_memory_config_rs` | `MemoryConfig` | `matmul_reduce_scatter_async` | Scratch buffer config for the ReduceScatter accumulation stage |
| `global_cb` | `GlobalCircularBuffer` | Llama ops | Direct L1 circular buffer shared across devices (Llama only) |
| `intermediate_tensor` | `ttnn.Tensor` | Llama AG+matmul | Pre-allocated intermediate storage for the AllGather stage |
| `intermediate_packet_buffer` | `ttnn.Tensor` | `llama_rs_matmul` | Pre-allocated packet buffer for the ReduceScatter stage |

---

## Prerequisites

- Chapter 2 §2.1: AllGather async parameter conventions
- Chapter 4 §4.1: Why async matters (ERISC/Tensix separation, SubDevices, persistent buffers)
- Chapter 4 §4.2: Async primitives (`all_gather_async`, `reduce_scatter_minimal_async`)

---

*Next: [5.1 Why Fusion Matters](why_fusion.md)*
