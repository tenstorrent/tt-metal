# Chapter 1: Introduction to CCL (Collective Communication Library)

## Overview

This chapter introduces the Collective Communication Library (CCL) as implemented in tt-metal and exposed through the TTNN Python API. CCL is the subsystem responsible for coordinating tensor data movement across multiple Tenstorrent devices — whether those devices are on the same board, connected via Ethernet links, or arranged in larger mesh configurations spanning multiple hosts.

Understanding CCL is foundational to writing efficient multi-device models in TTNN. Unlike single-device computation, multi-device workloads must carefully orchestrate *when* and *how* data crosses chip boundaries. A misconfigured AllGather that copies too much data, or a ReduceScatter that serializes when it could pipeline, can eliminate all the performance benefits of scaling to more chips. The chapters that follow this one go from basic usage through advanced kernel-level tuning — but everything rests on the conceptual foundation laid here.

This chapter covers three topics:

1. **What CCL is** — the problem it solves, the operations it provides, and analogies to familiar distributed-computing concepts (MPI, NCCL).
2. **Hardware topology** — how Tenstorrent chips are wired together, what ERISC cores and the NOC are, and why topology choice (Ring vs. Linear) matters at the API level.
3. **The TTNN ecosystem** — how CCL fits into the broader TTNN/tt-metal stack, the key C++ types you will encounter when reading source or writing custom ops, and where the relevant files live on disk.

---

## Table of Contents

| Section | File | Description |
|---------|------|-------------|
| [1.1 What Is CCL?](what_is_ccl.md) | `what_is_ccl.md` | Problem motivation, operation catalogue, analogies to MPI/NCCL |
| [1.2 Hardware Topology](hardware_topology.md) | `hardware_topology.md` | ERISC cores, EDM, NOC, Ring vs. Linear topology, mesh concepts |
| [1.3 CCL in the TTNN Ecosystem](ttnn_ecosystem.md) | `ttnn_ecosystem.md` | Stack placement, key types, directory map, Python API shape |

---

## Prerequisites

- Familiarity with basic TTNN tensor operations (`ttnn.matmul`, `ttnn.to_device`, etc.)
- A conceptual understanding of data-parallel or tensor-parallel training is helpful but not required
- No prior knowledge of Ethernet fabric or RISC-based kernel programming is assumed — those topics are introduced from scratch in [Section 1.2](hardware_topology.md)

---

## A Note on Terminology

Throughout this guide the following terms have specific meanings:

| Term | Meaning |
|------|---------|
| **device** | A single Tenstorrent chip (e.g., one Wormhole N150 or N300 die) |
| **chip** | Synonym for device in hardware contexts |
| **link** | A physical Ethernet connection between two ERISC cores on adjacent chips |
| **fabric** | The collection of all links connecting devices in a system |
| **collective** | An operation that requires coordination across all participating devices |
| **rank** | The index of a device within a communicating group (borrowed from MPI) |
| **mesh** | A logical 2-D grid of devices (rows = one axis, columns = another) |
| **sub-mesh** | A contiguous rectangular slice of a mesh used for one collective |

---

*Next: [1.1 What Is CCL?](what_is_ccl.md)*
