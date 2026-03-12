# Chapter 4: Async/Overlapping Compute and Communication

## Overview

All operations covered in Chapters 2 and 3 are *synchronous*: the host dispatches the CCL op, and the runtime waits for every device to complete it before dispatching the next op. Synchronous collectives are correct, simple to reason about, and sufficient for most workloads — but they leave hardware idle. While ERISC cores are transferring data, Tensix compute cores sit idle. While Tensix cores are running the next matmul, ERISC cores sit idle.

Async operations break this serialism. By explicitly controlling when communication starts, when compute starts, and where the synchronization points are, you can overlap Ethernet transfers with Tensix compute and approach the theoretical roofline where neither resource is the bottleneck.

This chapter covers the experimental async CCL API, the hardware model that makes overlap possible, and the patterns used in production LLM inference kernels to achieve it.

> **Status note:** All operations in this chapter live under `ttnn::experimental`. Their APIs are stable enough for production use in specific model architectures (Llama, DeepSeek, ring-attention) but may change without deprecation notice. Check the experimental CCL directory (`ttnn/cpp/ttnn/operations/experimental/ccl/`) for the current state.

---

## Table of Contents

| Section | File | Description |
|---------|------|-------------|
| [4.1 Why Async Matters](why_async.md) | `why_async.md` | The blocking problem, the overlap model, SubDevice, sync vs async decision guide |
| [4.2 Async Primitives](async_primitives.md) | `async_primitives.md` | `all_gather_async`, `all_reduce_async`, `reduce_scatter_minimal_async`, `send_async`/`recv_async` APIs |
| [4.3 Overlap Patterns](overlap_patterns.md) | `overlap_patterns.md` | Pipeline patterns, ring-attention overlap, subdevice dispatch, common pitfalls |

The shared parameters (`topology`, `num_links`, `cluster_axis`, `memory_config`, `subdevice_id`) are documented in full in [Section 2.1 — Parameter notes](../ch2_basic_operations/all_gather.md#parameter-notes). This chapter focuses on the parameters that are new in async ops: `multi_device_global_semaphore`, `barrier_semaphore`, `persistent_output_buffer`, `chunks_per_sync`, `num_workers_per_link`, `num_buffers_per_channel`.

---

## Prerequisites

- Chapters 1–3 read in full
- Understanding of GlobalSemaphore creation: `ttnn.create_global_semaphore(mesh_device, core_range_set, 0)` (Section 1.3)
- Familiarity with SubDevice concepts; introduced in this chapter if not seen before

---

*Next: [4.1 Why Async Matters](why_async.md)*
