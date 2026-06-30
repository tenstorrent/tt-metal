# Quasar Dispatch-Engine Fast Dispatch — High-Level Summary

## What we're doing

On Quasar, we are moving the fast-dispatch (FD) prefetcher and dispatcher off the
Tensix worker grid and onto the dedicated **dispatch-engine cores** the silicon
provides. In UMD these are a distinct `CoreType::DISPATCH` (analogous to how DRAM
cores are their own type), with their own NOC tiles and 8 DM processors per core.

The intent is that this is **arch-driven and fully internal**. It is *not* exposed
through `DispatchCoreConfig`, `OpenDevice`, or `MeshDevice`. From a user's point of
view nothing changes; the prefetcher/dispatcher simply land on dispatch-engine cores
instead of consuming Tensix workers.

If you already understand FD, the mental model is: **same prefetcher/dispatcher
kernels, same command protocol, same `dispatch_core_manager` assignment model — but
a new core type underneath them, sourced from a different place, with its own HAL and
firmware.**

## Why

- Dispatch-engine cores are purpose-built for this role, so FD no longer has to steal
  Tensix compute cores.
- It matches how other non-user-targetable cores (DRAM, L2CPU) are already handled:
  defined by the SoC, resolved internally, never user-selectable.

## The key conceptual shifts

| Aspect | WH/BH (and Quasar today) | Quasar dispatch-engine (target) |
|--------|--------------------------|----------------------------------|
| Core pool source | Core-descriptor YAML (Tensix-relative) | **UMD SoC descriptor `dispatch:` list** |
| Effective core type | `CoreType::WORKER` (or ETH) from `DispatchCoreConfig` | **`CoreType::DISPATCH`**, resolved internally by arch |
| Processor | BRISC/NCRISC (or Quasar Tensix DM) | **Dispatch-engine DM** (DM0 = prefetch, DM1 = dispatch in v1) |
| Firmware | Tensix `dm.cc` | **`dispatch_dm.cc`** (DM-only base firmware) |
| Memory map | `DispatchMemMap(WORKER)` | `DispatchMemMap(CoreType::DISPATCH)` |
| User control | WORKER vs ETH, ROW vs COL | **None** — arch decides |

A few things deliberately **do not** change: the `dispatch_core_manager`
assignment API, the `DispatchTopology` node graph, the prefetcher/dispatcher kernels
themselves (`cq_prefetch.cpp` / `cq_dispatch.cpp`), the kernel-placement guard that
keeps user kernels off dispatch cores, and the L1-banking allocator's treatment of
dispatch cores. We reuse all of that; only the pool source, core type, HAL/firmware
path, and coordinate translation are new.

## How a dispatch-engine core gets "wired up"

Because `CoreType::DISPATCH` is a genuinely new core type in tt-metal (it only
existed in UMD before), it has to be taught to every layer that previously only knew
about WORKER / ETH / DRAM:

- **Core discovery / coordinates** — dispatch cores come from the SoC descriptor, and
  tt-metal addresses them via a synthetic logical `(index, 0)` → NOC0 mapping
  (identity TRANSLATED mapping, like L2CPU). The cluster coordinate-translation layer
  learns to handle `CoreType::DISPATCH`.
- **HAL** — a new `HalProgrammableCoreType::DISPATCH` with its own L1/memory map,
  launch/go-message addresses, an 8-DM processor count, and a JIT/firmware path that
  selects `dispatch_dm.cc` for firmware and the normal kernel build for the cq
  kernels.
- **Firmware** — `dispatch_dm.cc` runs on the dispatch-engine DMs; it is loaded at
  **every `CreateDevice`** (including slow dispatch), so the cores are alive and
  waiting for a GO before any program launches on them.
- **Memory map / settings** — `DispatchMemMap` and `DispatchSettings` gain a
  `CoreType::DISPATCH` case (reusing the worker L1 layout for now).
- **Semaphores / launch / runtime args** — the validation and addressing paths that
  previously assumed worker/eth/dram coordinates are extended to accept dispatch
  coordinates so `CreateSemaphore`, `SetRuntimeArgs`, and `LaunchProgram` work on
  these cores.

## Interim escape hatch

There is an opt-in env var, **`TT_METAL_TENSIX_DISPATCH_CORES=1`**, that forces the
old behavior: FD runs on Tensix workers from the core-descriptor YAML, exactly as
Quasar FD does today. This always wins when set, regardless of whether the SoC lists
dispatch-engine cores. It exists purely as a bring-up / comparison fallback and is the
same class of control as `TT_METAL_GTEST_ETH_DISPATCH`. With the env unset, the
default is the dispatch-engine path when the SoC defines dispatch cores, and a clear
FD-init failure if it does not.

## Bring-up strategy

The first runnable milestone is the **slow-dispatch (SD) microbenchmarks**
(`test_dispatcher` / `test_prefetcher`), not full FD. These tests bypass the FD host
stack: they build a program and launch `cq_prefetch.cpp` / `cq_dispatch.cpp` directly
on the dispatch-engine DMs (via internal-only helpers, never a public API). That lets
us validate the new core type, HAL, firmware, coordinates, semaphores, and kernel
load path end-to-end before tackling the full FD runtime integration. Full FD wiring
through `dispatch_core_manager` and the dispatch-kernel initializer is a later phase;
the only thing that changes there is the *caller* — the dispatch-engine core type,
DM-assignment model, and memory map are shared.

## DPRINT interaction (for those less familiar with it)

DPRINT is the on-device print mechanism: each core's firmware/kernels write formatted
messages into a small per-core L1 buffer, and a host-side "print server" polls those
buffers and emits the lines you see (e.g. `0:0-0:DM4: ...`). The host has to know
which cores to watch, where each core's print buffer lives in L1, and how to enable
it.

Because dispatch-engine cores are a brand-new core type, the print server had to be
taught about them, in three respects:

1. **Which cores to watch / how to select them.** The print server now includes
   `CoreType::DISPATCH` tiles in its scan, and the `TT_METAL_DPRINT_CORES=dispatch`
   selection (and a `TT_METAL_DPRINT_DISPATCH_CORES` variant) routes to these cores.
   On WH/BH and the interim Tensix path this is unchanged.
2. **Where the print buffer is.** Quasar's on-device print region reserves space for
   the compute (TRISC) processors first and the DM processors second. Dispatch-engine
   firmware is DM-only, so its print buffer sits in the **DM half** of that region —
   the host now reads/enables it at the correct offset, otherwise the firmware writes
   to one place and the host listens at another and you see nothing.
3. **How the lines are labeled.** Dispatch-engine cores are tagged with a `DE-`
   prefix in the print output (e.g. `0:0-0:DE-DM4: ...`) so they're visually
   distinct from Tensix DMs without baking that distinction into each message.

None of this changes how DPRINT works conceptually; it's the same buffer-per-core,
host-polls model. We simply extended "which cores exist and where their buffers are"
to cover the new core type. DPRINT is debug-only and orthogonal to correctness — the
microbenchmarks pass with it off — but it was essential for bring-up visibility.

## Net effect

When complete, on a Quasar SoC that defines dispatch engines, FD transparently runs
its prefetcher and dispatcher on dispatch-engine DM cores, Tensix workers stay free
for compute, and no user-facing API changes. The design keeps the door open for
multiple dispatch cores, multiple CQs, and other DM roles in the future without
further API changes.
