# Chapter 2: Basic Operations

## Overview

This chapter covers the three most commonly used CCL operations in working order: AllGather, AllReduce, and Broadcast/AllBroadcast. Each section walks through the conceptual data flow, the Python API with accurate parameter documentation drawn from the C++ headers, common usage patterns with illustrative code, hardware-level internals (how data actually moves across ERISC cores and the NOC), and the practical gotchas that only appear once you run on real hardware.

Chapter 1 established the vocabulary — ERISC cores, EDM, Ring vs. Linear topology, `MeshDevice`. This chapter assumes that foundation and goes directly into operational detail. Code snippets in this chapter use realistic tensor shapes and parameter values; they are illustrative rather than copy-pasteable test programs (Chapter 3 adds the mesh setup boilerplate needed for end-to-end examples).

The three files in this chapter cover:

- **AllGather** — the foundational "give everyone the full tensor" collective. Ubiquitous in tensor-parallel inference for reassembling activation shards.
- **AllReduce** — the "sum across all devices and give everyone the result" collective. The standard tool for gradient synchronization and partial-matmul reduction.
- **Broadcast / AllBroadcast** — one-to-all and all-to-all distribution primitives. Less symmetric than AllGather; crucial for MoE expert dispatching and position-embedding distribution.

---

## Table of Contents

| Section | File | Description |
|---------|------|-------------|
| [2.1 AllGather](all_gather.md) | `all_gather.md` | Concept, data flow, full API, ring/linear internals, gotchas |
| [2.2 AllReduce](all_reduce.md) | `all_reduce.md` | Concept, ReduceScatter+AllGather decomposition, API, gotchas |
| [2.3 Broadcast and AllBroadcast](broadcast.md) | `broadcast.md` | Directed broadcast, AllBroadcast, sender semantics, MoE use |

The shared parameters (`topology`, `num_links`, `cluster_axis`, `memory_config`, `subdevice_id`) are documented in full in [Section 2.1 — Parameter notes](all_gather.md#parameter-notes); subsequent sections note only operation-specific deviations.

---

## Prerequisites

- Chapter 1 read in full, especially Section 1.2 (Hardware Topology) and Section 1.3 (CCL in the TTNN Ecosystem)
- Familiarity with `ttnn.from_torch`, `ttnn.MeshDevice`, and `ttnn.ShardTensorToMesh`

---

*Next: [2.1 AllGather](all_gather.md)*
