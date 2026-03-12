# Chapter 3: Intermediate Operations

## Overview

This chapter covers four CCL operations that go beyond the symmetric collectives of Chapter 2. Where AllGather and AllReduce treat every device identically, the operations here introduce *asymmetry*: a scatter dimension that is reduced (ReduceScatter), sparse token routing based on expert assignment (AllToAllDispatch / AllToAllCombine), directed reduction to a single root device (ReduceToRoot), and local tensor slicing driven by mesh position (MeshPartition).

Chapter 2 established the behavioral contracts for topology, `num_links`, `cluster_axis`, `memory_config`, and `subdevice_id`. Those contracts apply unchanged here; this chapter notes only operation-specific deviations.

## Table of Contents

| Section | File | Description |
|---------|------|-------------|
| [3.1 ReduceScatter](reduce_scatter.md) | `reduce_scatter.md` | Row-parallel linear layers; first half of AllReduce decomposition; output shape, API, ring/linear data flow, kernel internals, gotchas |
| [3.2 AllToAllDispatch + AllToAllCombine](all_to_all.md) | `all_to_all.md` | MoE expert dispatch/combine; sparse token routing; tensor conventions, API, sparse kernel internals |
| [3.3 ReduceToRoot + MeshPartition](reduce_to_root.md) | `reduce_to_root.md` | SDPA tree reduction to one device; local tensor partitioning; multi-tensor API, MeshPartition slice, gotchas |

The shared parameters (`topology`, `num_links`, `cluster_axis`, `memory_config`, `subdevice_id`) are documented in full in [Section 2.1 — Parameter notes](../ch2_basic_operations/all_gather.md#parameter-notes); this chapter notes only operation-specific deviations.

---

## Prerequisites

- Chapter 1 read in full, especially Section 1.2 (Hardware Topology) and Section 1.3 (CCL in the TTNN Ecosystem)
- Chapter 2 read in full — particularly Section 2.1 (AllGather) and Section 2.2 (AllReduce), since ReduceScatter is the inverse of AllGather and the first phase of AllReduce
- Familiarity with Mixture-of-Experts (MoE) routing for Section 3.2

---

*Next: [3.1 ReduceScatter](reduce_scatter.md)*
