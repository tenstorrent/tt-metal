# Chapter 7: Advanced Tuning

This is the final chapter of the CCL user guide. The preceding chapters covered the API surface of each CCL operation family. This chapter goes one level deeper: it explains the mechanisms that determine performance — topology selection, link count, SubDevice partitioning, semaphore lifecycle, EDM kernel internals, program caching, and trace mode — so you can reason about bottlenecks and make informed tuning decisions.

## Contents

| Section | Topics |
|---------|--------|
| [§7.1 — Topology and Links](topology_and_links.md) | Ring vs Linear, `num_links`, `cluster_axis`, 2D mesh CCL |
| [§7.2 — SubDevice and Semaphores](subdevice_and_semaphores.md) | `SubDevice`, `subdevice_id`, `GlobalSemaphore` lifecycle, async overlap |
| [§7.3 — Kernel Internals](kernel_internals.md) | EDM kernel, NOC transfers, program caching, trace mode, L1 budget |

## Prerequisites

- Chapters 1–6: familiarity with all CCL op families and their Python APIs.
- This chapter is hardware-agnostic but assumes you are running on a Tenstorrent Wormhole or later chip with ERISC Ethernet cores.

## Key parameters introduced in this chapter

| Parameter | Type | Where used | Purpose |
|-----------|------|------------|---------|
| `topology` | `ttnn.Topology` | All ring ops | `Ring` or `Linear`; selects EDM routing mode |
| `num_links` | `int \| None` | All CCL ops | Number of ERISC channels used simultaneously |
| `cluster_axis` | `int \| None` | 2D-mesh ops | Which mesh dimension the CCL runs along |
| `subdevice_id` | `SubDeviceId \| None` | CCL and compute ops | Routes op to a specific Tensix partition |
| `cores` (GlobalSemaphore) | `CoreRangeSet` | `create_global_semaphore` | Which Tensix cores the semaphore lives on |
| `initial_value` | `int` | `create_global_semaphore` | Starting count; typically 0 |
| `reset_value` | `int` | `reset_global_semaphore_value` | Value to write between iterations |

---

*Back to [CCL User Guide Index](../index.md)*
