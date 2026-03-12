# CCL User Guide

This guide covers the **Collective Communication Library (CCL)** in tt-metal — the subsystem that moves tensors between devices in a multi-chip Tenstorrent system. It is written for ML engineers using ttnn for multi-device inference and training, from first API call through advanced performance tuning.

The guide assumes familiarity with Python, basic ML concepts (tensors, matmul, MoE routing), and the ttnn tensor API. No prior knowledge of Ethernet hardware or Metal kernels is required.

---

## How to Use This Guide

| Goal | Recommended path |
|------|-----------------|
| New to CCL — start from scratch | Ch1 → Ch2 → Ch3 |
| Optimize inference throughput | Ch4 (async) → Ch5 (op fusion) |
| Building or debugging a MoE model | [Ch3 §3.2 — AllToAll](ch3_intermediate_operations/all_to_all.md) → Ch6 |
| Overlap CCL with compute | [Ch4 §4.1 — Why Async](ch4_async_overlap/why_async.md) → [Ch4 §4.3 — Overlap Patterns](ch4_async_overlap/overlap_patterns.md) |
| Tune topology, links, semaphores | Ch7 |
| Debug a hang or wrong output | [Ch7 §7.2 — Common Mistakes](ch7_advanced_tuning/subdevice_and_semaphores.md#common-mistakes) |

---

## Chapter Index

| # | Title | Description | Key operations |
|---|-------|-------------|----------------|
| 1 | [Introduction](ch1_introduction/index.md) | What CCL is, ERISC hardware, how CCL fits into ttnn | Concepts only |
| 2 | [Basic Operations](ch2_basic_operations/index.md) | The three fundamental collective ops | `all_gather`, `all_reduce`, `broadcast` |
| 3 | [Intermediate Operations](ch3_intermediate_operations/index.md) | Scatter, AllToAll, and root-targeted reduce | `reduce_scatter`, `all_to_all_dispatch/combine`, `reduce_to_root` |
| 4 | [Async Overlap](ch4_async_overlap/index.md) | Async variants, SubDevice partitioning, GlobalSemaphore, overlap patterns | `*_async` variants, SubDevice, GlobalSemaphore |
| 5 | [Op Fusion (CCL + Matmul)](ch5_op_fusion/index.md) | Fusing AllGather or ReduceScatter with a matmul into one program | `all_gather_matmul_async`, `matmul_reduce_scatter_async`, Llama fused ops |
| 6 | [MoE-Specific Patterns](ch6_moe_patterns/index.md) | Sparse token dispatch/combine for Mixture-of-Experts; DeepSeek patterns | `all_to_all_dispatch`, `all_to_all_combine`, `deepseek_minimal_broadcast` |
| 7 | [Advanced Tuning](ch7_advanced_tuning/index.md) | Topology, `num_links`, SubDevice, EDM kernel internals, program caching, trace mode | `topology`, `num_links`, `GlobalSemaphore`, EDM internals |

---

## Quick Reference

| Python call | What it does | Where to learn more |
|-------------|-------------|---------------------|
| `ttnn.all_gather(t, dim)` | Gather tensor slices from all devices along `dim` | [Ch2 — AllGather](ch2_basic_operations/all_gather.md) |
| `ttnn.all_reduce(t, math_op)` | All-gather + reduce across all devices | [Ch2 — AllReduce](ch2_basic_operations/all_reduce.md) |
| `ttnn.reduce_scatter(t, dim, math_op)` | Reduce across devices, scatter shards back | [Ch3 — ReduceScatter](ch3_intermediate_operations/reduce_scatter.md) |
| `ttnn.all_gather_async(t, ..., semaphore)` | Non-blocking AllGather; overlaps with compute | [Ch4 — Async Primitives](ch4_async_overlap/async_primitives.md) |
| `ttnn.all_gather_matmul_async(t, w, ...)` | Fused AllGather + Matmul sharing one L1 CB | [Ch5 — Fused Ops](ch5_op_fusion/fused_ops.md) |
| `ttnn.all_to_all_dispatch(t, indices, mapping)` | Sparse token dispatch to expert devices | [Ch6 — Dispatch & Combine](ch6_moe_patterns/dispatch_combine.md) |
| `ttnn.all_to_all_combine(t, metadata, mapping)` | Sparse token gather back from expert devices | [Ch6 — Dispatch & Combine](ch6_moe_patterns/dispatch_combine.md) |
| `ttnn.experimental.deepseek_minimal_broadcast(t, coord)` | Lightweight one-to-all broadcast for small tensors | [Ch6 — DeepSeek Patterns](ch6_moe_patterns/deepseek_patterns.md) |
| `ttnn.create_global_semaphore(dev, cores, val)` | Create a cross-core synchronization semaphore | [Ch4 §4.1](ch4_async_overlap/why_async.md#semaphores-in-async-operations) |
| `ttnn.reset_global_semaphore_value(sem, 0)` | Reset semaphore between iterations | [Ch7 §7.2](ch7_advanced_tuning/subdevice_and_semaphores.md#semaphore-reset-when-and-how) |

---

## Prerequisites

- **Python** — comfortable with function signatures, optional keyword arguments, and object lifetime.
- **Basic ML** — understand tensors, matrix multiply, and the concept of data-parallel / tensor-parallel training.
- **ttnn tensors** — know how to create a `ttnn.Tensor`, move it to a device with `ttnn.to_device`, and read it back. The [ttnn documentation](https://docs.tenstorrent.com/ttnn/) covers this.
- **Multi-device setup** — have a `MeshDevice` or a list of `IDevice*` objects ready. The guide assumes your devices are already initialized.

---

## Source Code Location

All CCL operations are implemented under:

```
ttnn/cpp/ttnn/operations/ccl/          # Core ops: all_gather, reduce_scatter, all_reduce,
                                        # broadcast, all_to_all_*, reduce_to_root, deepseek_*

ttnn/cpp/ttnn/operations/experimental/ccl/  # Experimental / Llama-specific fused ops:
                                             # all_gather_matmul_async, matmul_reduce_scatter_async,
                                             # llama_all_gather_matmul_async, llama_rs_matmul,
                                             # all_gather_concat, fused_rms_1_1_32_8192
```

EDM kernel source (ERISC firmware):

```
ttnn/cpp/ttnn/operations/ccl/kernels/edm/erisc_datamover.cpp
```

---

*This guide was generated for tt-metal. For the latest API signatures always check the C++ headers and nanobind binding files under the paths above.*
