# Plan: Quasar Fast-Dispatch on Dispatch-Engine DM Cores

## Goal

Move Quasar fast-dispatch (FD) prefetcher and dispatcher off the Tensix worker grid and onto **DM processors on dedicated dispatch-engine cores** (`CoreType::DISPATCH` in UMD). This should be **arch-driven and internal** — not exposed through `DispatchCoreConfig`, `OpenDevice`, or `MeshDevice` APIs — while following the same patterns used for other non-user-targetable core types (DRAM, L2CPU, etc.).

---

## Current State (from codebase)

### What exists today

| Layer | Quasar FD today | WH/BH FD |
|--------|-----------------|----------|
| **Core pool source** | `core_descriptors/quasar_*.yaml` → `dispatch_cores` (often `[]`, or Tensix-relative coords in `quasar_simulation_2x3_arch_fast_dispatch.yaml`) | `core_descriptors/*_arch.yaml` → `dispatch_cores` relative to Tensix grid |
| **Effective core type** | `DispatchCoreConfig` → `CoreType::WORKER` via `get_core_type_from_config()` | WORKER or ETH |
| **Topology** | `quasar_single_chip_1cq`: PREFETCH_HD + DISPATCH_HD, no DISPATCH_S | 3+ nodes depending on CQ count |
| **Kernel creation** | `FDKernel::configure_kernel_variant()` → `experimental::quasar::CreateKernel` + `QuasarDataMovementConfig` on Tensix logical cores | `CreateKernel` + BRISC/NCRISC |
| **Memory layout** | `DispatchMemMap` keyed on `CoreType::WORKER`, with `are_fd_kernels_on_same_core=true` for 1CQ Quasar | Same map, split cores |
| **User API** | `DispatchCoreConfig` on `OpenDevice` / `MeshDevice` (ignored for Quasar-specific behavior today) | User-selectable WORKER vs ETH, ROW vs COL |

### UMD dispatch-engine support (partial)

UMD already has the **plumbing** for dispatch cores:

- `CoreType::DISPATCH` in coordinate types
- `SocArchDescriptor::get_dispatch_cores()` / `CoordinateManager::get_dispatch_cores()`
- Soc descriptor YAML schema field `dispatch:` (see `tt_metal/third_party/umd/docs/yaml_schemas/soc_descriptor.yaml`)
- Quasar coordinate manager passes `dispatch_cores` through to base `CoordinateManager`
- NOC control register base map includes `CoreType::DISPATCH` in `grendel_implementation.hpp`

**Gap in this tree:** hardcoded arch constants and several in-repo soc YAMLs are still empty, even though the **target emulator** already defines dispatch cores (see below):

```cpp
// tt_metal/third_party/umd/device/api/umd/device/arch/grendel_implementation.hpp
// Placeholder — runtime soc YAML is source of truth for v1; update when non-emulator variants ship.
static const std::vector<tt_xy_pair> DISPATCH_CORES_NOC0 = {};
```

- `tt_metal/soc_descriptors/quasar_32_arch.yaml` — no `dispatch:` field (no dispatch engines on this variant)
- `tt_metal/core_descriptors/quasar_*.yaml` — `dispatch_cores: []` (Tensix-relative; not the dispatch-engine path)
- `tt_metal/third_party/umd/tests/soc_descs/quasar_simulation_2x3.yaml` — **has** `dispatch: [0-2]` (matches Aether / emulator; only in-repo soc with dispatch engines)

### Target emulator soc descriptor

Initial development targets the Quasar 2×3 emulator build:

**Path:** `../tt-umd-simulators/build/emu-quasar-2x3_DISPATCH/soc_descriptor.yaml`

| Field | Value |
|-------|-------|
| Grid | 2 × 3 |
| `functional_workers` | `(0,1)`, `(1,1)` — Tensix compute cores |
| `dispatch` | `(0,2)` — **one** dispatch-engine core (NOC0) |
| `pcie` | `(1,2)` — host/NOC2AXI tile (Aether `NOC2AXI`) |
| `dram` | `(0,0)`, `(1,0)` |
| `worker_l1_size` | 4 MiB |
| **Dispatch-engine L1** | **4 MiB** (same as Tensix workers) |
| **DM processors per dispatch-engine core** | **8** (DM0–DM7); **all valid for FD** |

Core layout (NOC0 y/x):

```
y=2:  DISPATCH(0,2)   PCIE(1,2)
y=1:  TENSIX(0,1)     TENSIX(1,1)
y=0:  DRAM(0,0)       DRAM(1,0)
```

**Coordinate source of truth:** Aether `rtl/targets/2x3_DISPATCH/aether.yml` (NOC/testbench coordinates). UMD runtime soc YAML (`emu-quasar-2x3_DISPATCH`, in-repo `quasar_simulation_2x3.yaml`) matches Aether: `dispatch: [0-2]`, `pcie: [1-2]`. No change to UMD `router_only` for this variant.

**Variant availability (confirmed):**

- **Only this variant** has dispatch-engine cores today. All other Quasar soc variants (`1×3`, `8×4`, `quasar_32`, etc.) have **no** dispatch cores — their `dispatch:` lists are empty or absent.
- **Future variants** will add dispatch engines; some will expose **multiple** dispatch locations. The implementation should handle an ordered list of arbitrary length, but v1 only needs to support the single-core 2×3 emulator case.
- Quasar dispatch-engine FD is **conditional on the runtime soc descriptor** when the env is unset and dispatch engines are present. If `get_dispatch_cores()` is empty (env unset), the **default** is to fail FD init with a clear error. **`TT_METAL_TENSIX_DISPATCH_CORES=1`** always restores the legacy Tensix path (see below).

**Tensix fallback (interim, env-var only):**

- **`TT_METAL_TENSIX_DISPATCH_CORES=1`** — when set, Quasar FD **always** uses the interim **Tensix** path: `CoreType::WORKER`, dispatch cores from **core descriptor YAML** (`get_logical_dispatch_cores()`), existing Tensix `dm.cc` / `DispatchMemMap(WORKER)` — **even if** the soc descriptor lists dispatch-engine cores. Env overrides placement; default path uses dispatch-engine cores when the env is unset.
- Not exposed through `OpenDevice`, `MeshDevice`, or `DispatchCoreConfig` — same class of control as `TT_METAL_GTEST_ETH_DISPATCH`.
- Default (unset): use dispatch-engine cores from soc when present; FD init fails if the soc has no dispatch engines.

**Implications for implementation:**

- Dispatch cores are sourced from the **runtime soc descriptor** when the env is unset (emulator YAML → UMD `SocArchDescriptor::get_dispatch_cores()`), not from core descriptor YAML — matching the DRAM pattern. **`TT_METAL_TENSIX_DISPATCH_CORES=1`** always selects the interim Tensix pool instead.
- The 2×3 emulator exposes **a single dispatch-engine core** with **8 DM processors** and **4 MiB L1**. **v1:** prefetch on **DM0**, dispatcher on **DM1** on that core (`are_fd_kernels_on_same_core=true`). The design must support full flexibility for future use: multiple dispatch cores, multiple prefetcher/dispatcher pairs (e.g. per CQ), and specialized FD blocks on other DMs — without API changes.
- For v1, **runtime YAML loading is sufficient** — no need to populate `grendel::DISPATCH_CORES_NOC0` or other in-repo soc YAMLs until those variants gain dispatch engines.
- In-repo UMD test soc `quasar_simulation_2x3.yaml` mirrors the emulator dispatch entry and can be used for unit tests without the external build path.
- **No dispatch engines in soc (default, env unset):** Quasar FD **fails to init** with a clear error. With **`TT_METAL_TENSIX_DISPATCH_CORES=1`**, always use Tensix dispatch cores from core descriptor YAML instead (regardless of soc dispatch entries).

### Critical blockers in tt-metal

1. **`Cluster::get_virtual_coordinate_from_logical_coordinates`** only supports TENSIX/WORKER, DRAM, and ETH — it **throws** for other core types (`tt_metal/llrt/tt_cluster.cpp`).

2. **`DispatchMemMap`** only implements WORKER and ETH — no `CoreType::DISPATCH` path. Its constructor also builds a **`DispatchSettings`**, whose `switch` over core type likewise handles only WORKER/ETH (`default` throws) — both need a `CoreType::DISPATCH` case.

3. **`HalProgrammableCoreType`** has TENSIX, ETH, DRAM — **no DISPATCH yet**. Will add `HalProgrammableCoreType::DISPATCH` (confirmed). Device-side **`ProgrammableCoreType::DISPATCH`** in `core_config.h` is also required (confirmed). Quasar HAL currently registers DM firmware for Tensix only.

4. **`get_logical_dispatch_cores()`** always reads from core descriptor YAML (Tensix-relative), not from UMD soc descriptor.

5. **`dispatch_core_manager::get_dispatch_core_type()`** derives type from user-facing `DispatchCoreConfig`, which only maps to WORKER or ETH.

6. **`MetalContext::is_coord_in_range`** (used by `ProgramImpl::create_semaphore` / `CreateSemaphore`) only checks worker, ETH, and DRAM cores — it returns false for dispatch-engine coords, so `CreateSemaphore(..., CoreType::DISPATCH)` throws `Coordinates out of range`. Needs a `Cluster::is_dispatch_core` check.

7. **DISPATCH HAL `KERNEL_CONFIG` size** — `create_dispatch_mem_map` copies the Tensix mem-map sizes, where `KERNEL_CONFIG` size is 0 (Tensix runtime-arg validation uses the hardcoded `max_runtime_args` constant instead). `Kernel::validate_runtime_args_size` sizes DISPATCH runtime args from `get_dev_size(DISPATCH, KERNEL_CONFIG)`, so without an explicit size `SetRuntimeArgs` fails with "Max allowable is 0". Must populate it in the dispatch memmap.

8. **Static TLB not configured for dispatch-engine cores** — `ll_api::configure_static_tlbs` (`tlb_config.cpp`) only mapped **TENSIX** and **ETH** cores. SD `test_prefetcher` writes FetchQ entries via `Cluster::get_static_tlb_window` on the prefetch core; on the dispatch-engine path that is NOC0 `(0,2)`. Without a TLB mapping, UMD throws `TLB window for core (0, 2) not found` during `SDPrefetchDRAMToL1TestFixture.TestTerminate`. **Fix:** add a loop over `sdesc.get_cores(CoreType::DISPATCH, TRANSLATED)` in `configure_static_tlbs` (same `configure_tlb` call as workers). Empty on WH/BH socs with no `dispatch:` list. Also required for **Phase 4b** FD (`system_memory_manager.cpp` uses `get_static_tlb_window` on prefetcher cores). *(Fixed during Phase 4a bringup.)*

## Reference Patterns to Follow

### DRAM cores (best analog for “not user-targetable, soc-defined”)

- Defined in **UMD soc descriptor**, not core descriptor YAML
- `CoreType::DRAM` with logical `(channel, subchannel)` → NOC0 translation in `metal_soc_descriptor`
- `HalProgrammableCoreType::DRAM` with its own L1/memory map
- Excluded from compute grid via allocator (`AllocCoreType::Dispatch` vs compute)
- Not selectable via `DispatchCoreConfig`

### L2CPU / SECURITY / ROUTER (best analog for “UMD-only, no logical grid”)

- UMD maps NOC0 ↔ TRANSLATED with **identity** (no logical coordinates)
- Not programmable by users
- Comment in UMD: *“No logical coordinates available for DISPATCH cores”* — same as L2CPU today

### WH/BH dispatch (what to reuse vs replace)

**Reuse:**

- `dispatch_core_manager` assignment API (`prefetcher_core`, `dispatcher_core`, pool consumption from front)
- `DispatchTopology` node graph and kernel config classes
- `DispatchQueryManager`, inspector/debug helpers (`GetDispatchCores`)
- Kernel placement guard in `program.cpp` (no user kernels on dispatch cores)
- `AllocCoreType::Dispatch` in L1 banking allocator

**Replace for Quasar (default path):**

- Core pool source: soc descriptor `dispatch` list, not Tensix-relative YAML entries
- Core type resolution: internal `CoreType::DISPATCH`, not `DispatchCoreConfig`
- HAL / firmware / JIT path for dispatch-engine DM (not Tensix DM)
- Coordinate translation path in `Cluster`

**Interim Tensix path (opt-in via `TT_METAL_TENSIX_DISPATCH_CORES=1`):** when the env is set, **always** use interim behavior — pool from core descriptor YAML, `CoreType::WORKER`, Tensix `QuasarDataMovementConfig`, `DispatchMemMap(WORKER)` — regardless of soc dispatch-engine entries.

### What NOT to do (per requirements)

- Do **not** add `DispatchCoreType::DISPATCH` to public `DispatchCoreConfig`
- Do **not** add `dispatch_core_type` / `dispatch_core_axis` options for Quasar dispatch-engine selection
- Do **not** put dispatch-engine cores in `core_descriptors/quasar_*.yaml` `dispatch_cores` for the **default** dispatch-engine path (those entries are Tensix-grid-relative and used when **`TT_METAL_TENSIX_DISPATCH_CORES=1`** enables the interim fallback — including when soc dispatch-engine cores exist)

---

## Proposed Design

### 1. Internal dispatch core type resolution

Introduce an **arch-gated internal resolver** (e.g. `resolve_dispatch_core_type(ARCH, DispatchCoreConfig, soc_desc, rtoptions)`) used everywhere `dispatch_core_manager::get_dispatch_core_type()` is called:

| Arch | FD active | Soc has dispatch cores | `TT_METAL_TENSIX_DISPATCH_CORES` | Resolved `CoreType` |
|------|-----------|------------------------|----------------------------------|------------------------|
| QUASAR | yes | yes | unset | `CoreType::DISPATCH` |
| QUASAR | yes | no | unset | **FD init fails** with clear message |
| QUASAR | yes | any | `=1` | `CoreType::WORKER` (interim Tensix fallback; **env always wins**) |
| QUASAR | no | any | any | N/A (slow dispatch) |
| WH/BH | yes | n/a | n/a | From `DispatchCoreConfig` (WORKER/ETH) as today |

Register the env var in `rtoptions` (same pattern as `TT_METAL_GTEST_ETH_DISPATCH`). Log at info level when the Tensix fallback path is active so bringup logs show which mode is in use.

`DispatchCoreConfig` continues to exist on device-open APIs for WH/BH compatibility. On Quasar:

- **Core type / axis:** ignored for dispatch-engine selection (`dispatch_core_axis=COL` is silently ignored; no `TT_FATAL`)
- **`TT_METAL_GTEST_ETH_DISPATCH`:** likewise has no effect on Quasar
- Dispatch-engine FD is the default when the env is unset and the soc descriptor defines dispatch cores; **`TT_METAL_TENSIX_DISPATCH_CORES=1`** always forces the interim Tensix path

### 2. Dispatch core pool from UMD soc descriptor

Add something like `get_quasar_dispatch_cores(env, device_id)` that selects the core pool:

1. If **`TT_METAL_TENSIX_DISPATCH_CORES=1`** → return `get_logical_dispatch_cores()` from core descriptor YAML (interim Tensix path; requires non-empty YAML `dispatch_cores` for the variant, e.g. `quasar_simulation_2x3_arch_fast_dispatch.yaml`). **Env is checked first** — overrides soc dispatch-engine entries.
2. Else if soc `get_dispatch_cores()` is **non-empty** → return synthetic logical cores from the ordered dispatch list (dispatch-engine path).
3. Else → empty list → **FD init fails** with a clear message.

Wire this into `dispatch_core_manager::reset_dispatch_core_manager()` for Quasar.

Validate count and DM availability for the active topology:

- **v1 (single-chip 1CQ, 2×3 emulator):** 1 dispatch **tile**; prefetch on DM0, dispatcher on DM1 on that tile (`are_fd_kernels_on_same_core=true`). The **assignment pool** must still expose **≥ 2 logical entries** for that tile (duplicate `(0,0)`) because `prefetcher_core()` and `dispatcher_core()` each pop one slot — same pattern as interim Tensix YAML listing `[[1,-1], [1,-1]]` twice (see fix **C**)
- **Future (multi-CQ, multi-chip):** multiple dispatch cores and multiple prefetcher/dispatcher kernel instances; specialized blocks on other DMs. Pool and assignment logic must not hard-code core count, CQ count, or DM indices — only v1 wiring is fixed to DM0/DM1

For Quasar core descriptors, keep `dispatch_cores: []` on the **default** dispatch-engine path. Tensix-relative `dispatch_cores` entries in YAML (e.g. fast-dispatch sim variants) are used when **`TT_METAL_TENSIX_DISPATCH_CORES=1`** is set (including when soc dispatch-engine cores exist).

### 3. Logical coordinate convention (confirmed)

UMD will **not** grow a logical dispatch grid. tt-metal uses a **synthetic index → NOC0 mapping** only:

- `CoreCoord(index, 0)` where `index` is position in the ordered soc descriptor `dispatch:` list (v1: index 0 → NOC0 `(0,2)`)
- Implement translation in `metal_soc_descriptor` + extend `Cluster::get_virtual_coordinate_from_logical_coordinates` for `CoreType::DISPATCH`
- **`CoreType::DISPATCH` uses identity NOC0 ↔ TRANSLATED mapping** (same as L2CPU) — confirmed
- `dispatch_core_manager` continues to use `CoreCoord` as today; only the translation layer changes

### 4. HAL: `HalProgrammableCoreType::DISPATCH` (confirmed)

Extend **`HalProgrammableCoreType::DISPATCH`** (Quasar-only initially) and matching device-side **`ProgrammableCoreType::DISPATCH`** in `hw/inc/internal/tt-2xx/quasar/core_config.h`:

- Add `ProgrammableCoreType::DISPATCH` to device `core_config.h` (new index in enum, before `COUNT`); dispatch-engine kernels use the **`ProgrammableCoreType::DISPATCH` slot** in `kernel_config_base[]` / `sem_offset[]` in launch messages — not the Tensix slot
- Add `static_assert` mapping in `qa_hal.cpp` (alongside existing TENSIX/ETH asserts)
- Register L1/memmap/launch addresses in new **`qa_hal_dispatch.cpp`** (mirror `qa_hal_tensix.cpp` / DRAM registration pattern)
- **JIT / firmware build:** extend **`hal_2xx_common.cpp`** `srcs()` / `target_name()` so `HalProgrammableCoreType::DISPATCH` + `HalProcessorClassType::DM` + `is_fw` → **`dispatch_dm.cc`** (not `dm.cc`); execution kernels still use `dmk.cc`
- **L1 size: 4 MiB** — same as Tensix workers (`worker_l1_size` in soc descriptor)
- **`dev_mem_map`:** reuse the common Quasar Tensix `dev_mem_map.h` layout until a dispatch-specific difference is required
- **`KERNEL_CONFIG` size:** the DISPATCH memmap (`create_dispatch_mem_map`) copies the Tensix mem-map sizes, but Tensix leaves `mem_map_sizes[KERNEL_CONFIG] = 0` (its runtime-arg validation uses the hardcoded `max_runtime_args` constant). DISPATCH cores instead size runtime args from `get_dev_size(DISPATCH, KERNEL_CONFIG)` in `Kernel::validate_runtime_args_size`, so the dispatch memmap **must set** `KERNEL_CONFIG` size to the real region (gap between `KERNEL_CONFIG` base and `DEFAULT_UNRESERVED` base) — otherwise SetRuntimeArgs fails with "Max allowable is 0"
- **Processor count: 8 DMs** (DM0–DM7; all valid for FD)
- Extend `llrt::get_core_type()` to map virtual dispatch cores → `HalProgrammableCoreType::DISPATCH`
- Reuse existing Tensix dispatch config buffer features where layout matches; add a dispatch-specific `DispatchFeature` only if needed later

### 5. `DispatchMemMap` for dispatch-engine cores

Add `CoreType::DISPATCH` branch in `DispatchMemMap`:

- **L1 size: 4 MiB** — use `HalProgrammableCoreType::DISPATCH` addresses (same layout as Tensix worker L1 for now)
- **v1:** `are_fd_kernels_on_same_core=true` (single dispatch core; prefetch DM0 + dispatcher DM1 share L1 layout)
- **Future:** derive same-core vs split-core from topology + soc dispatch core count; support multiple CQ/kernel instances without redesign

Also add a `CoreType::DISPATCH` case in the **`DispatchSettings`** constructor (`dispatch_settings.cpp`): the `DispatchMemMap` constructor builds a `DispatchSettings(num_hw_cqs, core_type, ...)`, and its `switch` previously handled only `WORKER`/`ETH` (`default` throws `init_defaults not implemented for core type DISPATCH`). Route `CoreType::DISPATCH` through `init_worker_defaults` — the DISPATCH HAL memmap mirrors the Tensix (worker) L1 layout, so worker buffer sizes apply; `core_type_` is set to `DISPATCH` after the switch, overriding the `WORKER` value the builder sets internally.

Register in `MetalContext` and `DispatchTopology` for Phase 4b. **`DispatchKernelInitializer`** registers `DispatchMemMap(CoreType::DISPATCH)` when fast dispatch is enabled (replaces today’s Quasar-specific `DispatchMemMap(WORKER)` wiring — see Phase 4b):

```cpp
dispatch_mem_map_[CoreType::DISPATCH] = std::make_unique<DispatchMemMap>(CoreType::DISPATCH, ...);
```

SD tests (Phase 4a) construct or query `DispatchMemMap(CoreType::DISPATCH)` directly in the test harness — they do not go through `dispatch_kernel_initializer`.

### 6. DM assignment and extensibility (confirmed)

| | v1 (2×3 emulator, 1CQ) | Future |
|--|------------------------|--------|
| Dispatch cores | 1 (`index` 0) | Multiple per soc / multi-chip |
| Prefetcher | DM0 on core 0 | Per-CQ instances; any valid DM on any dispatch core |
| Dispatcher | DM1 on core 0 | Per-CQ instances; any valid DM on any dispatch core |
| Other DMs | Unused in v1 | Specialized FD blocks, DISPATCH_S, etc. |

Implementation requirements:

- FD kernel config carries **explicit core + DM index** via **`internal::CreateDispatchEngineKernel(...)`** (confirmed); no implicit BRISC mapping
- `dispatch_core_manager` and topology wiring must accept **N dispatch cores × 8 DMs** without hard-coding v1 layout
- v1 hard-codes only the **default assignment table** for `quasar_single_chip_1cq` (DM0/DM1 on core 0)

### 7. FDKernel / kernel placement changes

**`FDKernel::GetCoreType()`** — already delegates to `dispatch_core_manager`; will automatically return DISPATCH once resolver is updated.

**`configure_kernel_variant()`** — add Quasar + DISPATCH path:

- Today: Quasar + WORKER → `experimental::quasar::CreateKernel` with `num_threads_per_cluster=1`
- Target: Quasar + DISPATCH → **`internal::CreateDispatchEngineKernel(...)`** with explicit **core + DM** (wraps `CreateKernel` + `QuasarDataMovementConfig`):
  - **v1:** prefetch → core 0 / DM0; dispatcher → core 0 / DM1
  - **Future:** assignment driven by topology table (per CQ, per role); any DM0–DM7 on any dispatch core
- **`send_to_brisc_`** is irrelevant; processor selection is always explicit

**`Kernel` / `DispatchEngineKernel` configure path (confirmed):** add **`DispatchEngineKernel`** (mirror **`DramKernel`**) with a dedicated `configure(...)` path in `kernel.cpp` / `kernel.hpp` for `HalProgrammableCoreType::DISPATCH` — used by SD tests, `CreateDispatchEngineKernel`, and FDKernel.

**`get_programmable_core_type_index()`** — add DISPATCH → `HalProgrammableCoreType::DISPATCH`.

### 8. Firmware initialization

**`DISPATCH_MODE_DEV` vs dispatch-engine firmware (important):**

- `launch_msg.kernel_config.mode == DISPATCH_MODE_DEV` tells a **destination compute core** (Tensix worker, ETH, etc.) that it is receiving programs via fast dispatch — e.g. `dispatch.cpp` sets this on worker launch messages, and worker FW (`dm.cc`, `brisc.cc`, …) uses it to notify the dispatcher when a kernel completes.
- It does **not** describe how prefetch/dispatch kernels on dispatch-engine cores should run. Dispatch-engine cores are the **source** of FD traffic, not FD destinations.
- On dispatch-engine cores, initial launch messages use **`DISPATCH_MODE_HOST`** — same as Tensix dispatch cores today (`risc_firmware_initializer.cpp` overwrites dispatch cores to `DISPATCH_MODE_HOST` even when FD is enabled).

**Separate dispatch-engine DM firmware:**

Add **`hw/firmware/src/tt-2xx/dispatch_dm.cc`**, derived from `dm.cc` but with all Tensix-only logic removed:

| Keep (from `dm.cc`) | Remove / unused |
|---------------------|--------|
| DM0 GO/launch loop; DM0 orchestrates **DM1–DM7 subordinates** via the same **`subordinate_map_t`** struct (TRISC/Neo fields left unused) | TRISC deassert / `run_triscs` / `wait_subordinates` (Neo TRISC paths) |
| NOC init, bank tables, `firmware_config_init` with `ProgrammableCoreType::DISPATCH` | `set_deassert_addresses`, TRISC IC invalidate |
| Kernel load/run via `kernel_text_offset[hartid]` | TRISC/Neo subordinate sync (fields in struct remain but are not written) |
| DFB setup, overlay, profiler hooks as needed for cq kernels | Tensix-specific subordinate orchestration beyond DM-only sync |

**Subordinate model (confirmed):** reuse the **same `subordinate_map_t` struct** as Tensix `dm.cc`; TRISC/Neo fields are **unused** on dispatch-engine cores. **DM0** still orchestrates DM1–DM7 the same way Tensix `dm.cc` orchestrates subordinate DMs, minus all TRISC paths.

Dispatch-engine cores have **8 DMs, no TRISC** — the FW is DM-only. Do **not** reuse `dm.cc` directly for `HalProgrammableCoreType::DISPATCH`; wire **`dispatch_dm.cc`** in **`hal_2xx_common.cpp`** for DISPATCH + DM firmware builds.

**Phase 3 deliverable:** `dispatch_dm.cc` is loaded on dispatch-engine cores during **every `CreateDevice`** via **`risc_firmware_initializer`** — including **slow dispatch** — before any SD `LaunchProgram` on those cores. Required for Phase 4a regardless of FD runtime options.

**Host init split (confirmed):**

| Component | Phase 3 / 3b / 4a (SD) | Phase 4b (FD) |
|-----------|------------------------|---------------|
| **`risc_firmware_initializer`** | Load **`dispatch_dm.cc`** on each soc dispatch-engine core (all 8 DMs); initial launch/go mailboxes with **`DISPATCH_MODE_HOST`**; skip Tensix multicast / TRISC setup — **always at `CreateDevice`** on the default dispatch-engine path | Same |
| **DPrint server (Phase 3b)** | Attach to **`CoreType::DISPATCH`** tiles; init/enable L1 print buffers on dispatch-engine cores (FW + manually launched cq kernels in Phase 4a) | Same; **`dispatch_s` DRAM aggregation** remains Phase 4b when FD + dispatch_s enabled |
| **`dispatch_kernel_initializer`** | **Not used** — `init()` returns early when `!using_fast_dispatch()` (unchanged WH/BH gate) | Quasar path: register **`DispatchMemMap(CoreType::DISPATCH)`** (not `WORKER`); compile/configure prefetch/dispatch **execution** kernels (`cq_prefetch`, `cq_dispatch`, topology kernels) on dispatch-engine DMs — WH/BH pattern |

**Interim Tensix fallback (`TT_METAL_TENSIX_DISPATCH_CORES=1`):** unchanged — uses existing Tensix `dm.cc` on `CoreType::WORKER` dispatch cores from core descriptor YAML (no `dispatch_dm.cc` on dispatch-engine tiles when this path is active). Interim Tensix fallback coordinates in `quasar_simulation_2x3_arch_fast_dispatch.yaml` remain as-is for now.

### 9. Topology and assignment

**Scope:** implement and test **single-chip 1CQ** only for v1. Topology tables, pool sizing, and assignment APIs should be structured so multi-CQ and multi-chip extensions add nodes/assignments without redesign.

Keep `quasar_single_chip_1cq` graph (PREFETCH_HD → DISPATCH_HD) but change **where cores come from**:

| Node | Current (interim) | Target (v1) |
|------|-------------------|-------------|
| PREFETCH_HD | Tensix from YAML pool | Dispatch-engine core 0, **DM0** |
| DISPATCH_HD | Tensix from YAML pool | Dispatch-engine core 0, **DM1** |

`dispatch_core_manager` assignment logic (prefetcher/dispatcher pairing, MMIO device rules) largely stays; only the pool source, core type, and DM indices change.

Remove reliance on `quasar_simulation_2x3_arch_fast_dispatch.yaml` Tensix dispatch entries once dispatch-engine path works. **Compute grid benefit (confirmed):** both Tensix cores in the 2×3 grid become fully available for user workloads.

### 10. FD init when dispatch-engine path unavailable

When Quasar FD is requested and the **default** dispatch-engine path cannot be used:

| Condition | Behavior |
|-----------|----------|
| `TT_METAL_TENSIX_DISPATCH_CORES=1` | **Always** use interim Tensix dispatch cores from core descriptor YAML; `CoreType::WORKER` — even if soc lists dispatch-engine cores |
| Default (env unset), soc has dispatch engines | Use dispatch-engine cores (`CoreType::DISPATCH`) |
| Default (env unset), soc has no dispatch engines | **Fail during FD init** with a clear message |
| Either path | Do **not** silently downgrade to slow dispatch |

If the env var is set but core descriptor YAML has empty `dispatch_cores`, FD init should still fail with a message that YAML dispatch cores are required for the Tensix fallback path.

Slow dispatch remains available only when FD is not enabled via runtime options.

### 11. Allocator, validation, and exclusion

**L1 banking allocator** (`l1_banking_allocator.cpp`):

- Resolve dispatch cores via **`get_quasar_dispatch_cores()`** / **`resolve_dispatch_core_type()`** (respects `TT_METAL_TENSIX_DISPATCH_CORES=1` override — not raw soc lists alone)
- Use resolved `CoreType::DISPATCH` (or `WORKER` on Tensix fallback) for coordinate lookup instead of `dispatch_core_type` from `DispatchCoreConfig`
- Mark dispatch-engine NOC coords as `AllocCoreType::Dispatch`

**Kernel placement** (`program.cpp::validate_kernel_placement`):

- Compare against resolved dispatch core list from **`get_quasar_dispatch_cores()`** (not `DispatchCoreConfig` or soc-only lists)
- Allow kernels with `get_kernel_core_type() == CoreType::DISPATCH` on dispatch-engine cores (including `cq_prefetch.cpp` / `cq_dispatch.cpp` launched by SD tests)
- User kernels on dispatch-engine cores remain forbidden
- **Physical chip id:** when compile runs on a **`MeshDevice`**, use **`device->build_id()`** for `resolve_dispatch_core_type()` and service-core lookups — **`device->id()`** is the mesh id, not `Cluster::get_soc_desc` chip id (see fix **B**)

**Service cores:** `ServiceCoreManager` is BH/Galaxy Tensix dispatch-column only — **not applicable** to Quasar dispatch-engine cores. No extension needed; document exclusion.

### 12. Debug / tooling

Update consumers of dispatch core lists to handle `CoreType::DISPATCH`:

- `debug_helpers.hpp::GetDispatchCores`
- **DPRINT** — see **Phase 3b** (`dprint_server.cpp`, `debug_helpers.hpp::GetAllCores`, `device_print.h` callstack offsets)
- Watcher / profiler / inspector (`data.cpp`, `doAllDispatchCoresComeAfterNonDispatchCores`) — Phase 5
- NOC sanitize (`sanitize_noc_host.hpp`) — add dispatch-engine to valid core sets — Phase 5
- `jit_build` / `build_env_manager` — hash uses **resolved dispatch core type** (`CoreType::DISPATCH` vs `CoreType::WORKER` from `resolve_dispatch_core_type()`), not user `DispatchCoreConfig`, on Quasar — Phase 5

### 13. Soc descriptor handling

- **Only variant with dispatch engines today:** `../tt-umd-simulators/build/emu-quasar-2x3_DISPATCH/soc_descriptor.yaml` (`dispatch: [0-2]`, `pcie: [1-2]` — aligned with Aether)
- In-repo test mirror: `tt_metal/third_party/umd/tests/soc_descs/quasar_simulation_2x3.yaml` (same coordinates)
- **Do not** add `dispatch:` entries to other Quasar soc YAMLs until those variants actually gain dispatch engines
- **`router_only`:** no additional UMD changes for the 2×3 dispatch variant (remains `[]`; north-west tile is dispatch-engine, not router)
- **`grendel::DISPATCH_CORES_NOC0`:** leave empty for v1; runtime soc YAML is the source of truth. Update hardcoded constants only when a non-emulator variant ships with dispatch cores
- Add/extend UMD tests using the 2×3 soc with non-empty dispatch list (scaffolding exists in `test_soc_descriptor.cpp`)
- **Emulator soc path:** build output lives under `tt-umd-simulators` (outside tt-metal tree). Validate the built `emu-quasar-2x3_DISPATCH/soc_descriptor.yaml` matches in-repo `quasar_simulation_2x3.yaml` before bringup; tt-metal simulator runs must load the emulator-built YAML when using `emu-quasar-2x3_DISPATCH`
- **Future variants:** when new soc descriptors add one or more dispatch locations, no API changes — pool size and assignment strategy adapt to the ordered `dispatch:` list
- **Do not confuse soc fields:** UMD reads only the **`dispatch:`** list for `CoreType::DISPATCH` tiles. Lines like **`dispatch_cores: [[1,-1], …]`** in an emulator soc YAML are **tt-metal core-descriptor syntax** copied by mistake — UMD ignores them; they do **not** expand the FD assignment pool. Interim Tensix dispatch coords belong in **`tt_metal/core_descriptors/quasar_simulation_2x3_arch_fast_dispatch.yaml`**, not in the soc descriptor.

---

## Bringup Regressions and Fixes (Post Phase 2–3b)

Phase 2 (`resolve_dispatch_core_type()` + soc-based pool) and Phase 3b (DPRINT on `CoreType::DISPATCH`) landed before Phase 4b FD wiring was complete. Several integration gaps broke **Tensix-interim** and/or **default dispatch-engine** fast dispatch on the 2×3 emulator. Track these explicitly so Phase 4b does not reintroduce them.

### Summary

| Issue | Symptom | Affected path | Fix | Status |
|-------|---------|---------------|-----|--------|
| **A. `dispatch_s` on Quasar 1CQ** | `TT_ASSERT prefetch.cpp:295 downstream_kernels_.size() == 2` during FD init | Default dispatch-engine FD (and any path using `DispatchQueryManager` reset) | Restore pre–Phase 2 guard: `dispatch_s_enabled_` requires `arch != QUASAR` for 1CQ (topology has no `DISPATCH_S` node) | ✅ Fixed (`dispatch_query_manager.cpp`) |
| **B. Mesh id vs physical chip** | `Cannot access soc descriptor for 1 … Call initialize_device_driver(1)` during `ProgramImpl::compile` | **Both** paths when compiling via `MeshDevice` (e.g. `SingleDmL1Write`) | `validate_kernel_placement` must pass **`device->build_id()`** (physical chip), not **`device->id()`** (mesh id), to `resolve_dispatch_core_type()` / `is_service_core()` | ✅ Fixed (`program.cpp`) |
| **C. Dispatch pool size (1 tile, 2 FD roles)** | `No more available dispatch cores on device 0 to assign` during `populate_fd_kernels` / device open | Default dispatch-engine FD (`TT_METAL_TENSIX_DISPATCH_CORES` **unset**) | One soc dispatch tile is correct; **1CQ** topology still consumes **two** pool pops (`PREFETCH_HD` + `DISPATCH_HD`). Mirror interim Tensix YAML (**duplicate** the same logical core in the pool) or teach `dispatch_core_manager` to reuse one tile without double-pop | ⏳ **Required for Phase 4b** |
| **D. DPRINT “all dispatch” on engine path** | Misleading success: FW prints from engine tile, then FD fails with **C** | Dispatch-engine path with `TT_METAL_DPRINT_CORES=dispatch` | DPRINT working proves **`dispatch:`** and Phase 3 FW are fine; FD failure is pool assignment, not missing soc cores | Documented (not a separate code fix) |

**Regression test (2×3 emulator, fast dispatch):**

```bash
export TT_METAL_SIMULATOR=/path/to/emu-quasar-2x3_DISPATCH/
unset TT_METAL_SLOW_DISPATCH_MODE

# Interim Tensix path — must keep passing after Phase 4b changes:
export TT_METAL_TENSIX_DISPATCH_CORES=1
./build/test/tt_metal/unit_tests_legacy --gtest_filter='QuasarMeshDeviceSingleCardFixture.SingleDmL1Write'

# Default dispatch-engine path — blocked until fix **C**:
unset TT_METAL_TENSIX_DISPATCH_CORES
./build/test/tt_metal/unit_tests_legacy --gtest_filter='QuasarMeshDeviceSingleCardFixture.SingleDmL1Write'
```

### A. `dispatch_s_enabled` on Quasar 1CQ

**Cause:** Phase 2 `DispatchQueryManager::reset()` enabled `dispatch_s` when `resolved_dispatch_core_type == WORKER`, but Quasar **`quasar_single_chip_1cq`** has only **`PREFETCH_HD` + `DISPATCH_HD`** (no `DISPATCH_S`). WH/BH 1-CQ uses three nodes including `DISPATCH_S`.

**Fix:** Keep the historical Quasar guard:

```cpp
dispatch_s_enabled_ = (num_hw_cqs == 1 or resolved_dispatch_core_type == CoreType::WORKER) and
                        resolved_dispatch_core_type != CoreType::DISPATCH and arch != tt::ARCH::QUASAR;
```

**Note:** With **`TT_METAL_TENSIX_DISPATCH_CORES=1`**, `resolve_dispatch_core_type()` returns **`WORKER`**, so the `arch != QUASAR` term is what prevents spurious `dispatch_s` on Quasar 1CQ Tensix FD.

### B. `validate_kernel_placement` and `MeshDevice::id()`

**Cause:** Phase 2 `validate_kernel_placement()` calls `resolve_dispatch_core_type(env, device_id, …)`, which calls `get_soc_desc(device_id)`. When compile is invoked on a **`MeshDevice`**, `device->id()` returns **`mesh_id_`** (often **1** on a unit mesh), not the physical chip (**0**). The simulator only initializes soc descriptors for chip **0**.

**Fix:** Use **`device->build_id()`** at both `validate_kernel_placement` call sites in `ProgramImpl::compile()` — same as the remote JIT path already does. Rename the helper parameter to `physical_chip_id` to avoid repeat mistakes.

**Rule for future call sites:** any code that passes an id into **`Cluster::get_soc_desc`**, **`resolve_dispatch_core_type`**, **`get_quasar_dispatch_cores`**, or **`ServiceCoreManager`** chip-keyed APIs must use **physical chip id** (`build_id()` on `IDevice`, or explicit `ChipId` from device pool), **not** mesh id.

### C. Dispatch-core pool: one engine tile, two assignment slots (Phase 4b blocker)

**Cause:** `dispatch_core_manager` assigns FD roles by **popping** entries from the front of `available_dispatch_cores_by_device`. For **`quasar_single_chip_1cq`**:

| FD role | API | Pool pops |
|---------|-----|-----------|
| Prefetch HD | `prefetcher_core()` | 1 |
| Dispatch HD | `completion_queue_writer_core()` / `dispatcher_core()` | 1 |

**Interim Tensix path (`TT_METAL_TENSIX_DISPATCH_CORES=1`):** pool comes from core descriptor YAML — **`dispatch_cores: [[1,-1], [1,-1]]`** lists the **same** logical coord **twice**, so two pops succeed (same physical tensix).

**Default dispatch-engine path:** pool comes from soc — **`dispatch: [0-2]`** → **one** synthetic logical core **`(0,0)`** → first pop succeeds, second throws **`No more available dispatch cores`**.

DPRINT output (`DE-DM*: DISPATCH DM0-FW: initialized`) confirms the soc **`dispatch:`** list and Phase 3 firmware are working; the failure is **pool sizing**, not missing hardware.

**Required fix (pick one, prefer mirroring Tensix YAML):**

1. **Pool expansion (recommended, minimal):** In **`get_quasar_dispatch_cores_cached()`**, when the soc lists a **single** dispatch-engine core and FD needs multiple assignment slots, **duplicate** `(0,0)` in the returned vector — at minimum **`2 × num_hw_cqs`** entries for same-core 1CQ (prefetch + dispatch per CQ). Matches interim YAML behavior without changing assignment APIs.

2. **Same-core reuse (alternative):** In **`dispatch_core_manager`**, when Quasar + **`CoreType::DISPATCH`** + **`are_fd_kernels_on_same_core`**, **`dispatcher_core` / `completion_queue_writer_core`** reuse the core already assigned to **`prefetcher_core`** without a second pool pop (closer to hardware truth, more invasive).

**Validation:** After fix **C**, **`SingleDmL1Write`** and full FD init must pass with **`TT_METAL_TENSIX_DISPATCH_CORES` unset** on `emu-quasar-2x3_DISPATCH`.

### D. Preserving interim Tensix FD while landing Phase 4b

Requirements so **`TT_METAL_TENSIX_DISPATCH_CORES=1`** keeps working:

| Area | Must remain true |
|------|------------------|
| **`resolve_dispatch_core_type`** | Env **checked first** → **`WORKER`** when env set, regardless of soc `dispatch:` |
| **`get_quasar_dispatch_cores`** | Env set → pool from **core descriptor YAML** (duplicate coords), not soc list |
| **`DispatchQueryManager`** | **`arch != QUASAR`** guard for **`dispatch_s`** (fix **A**) |
| **`validate_kernel_placement`** | **`build_id()`** for soc/service lookups (fix **B**) |
| **`risc_firmware_initializer`** | Skip dispatch-engine **`dispatch_dm.cc`** init when env set (`assert_dispatch_cores` / engine init gated on `!get_use_quasar_tensix_dispatch_cores()`) |
| **DPRINT** | Tensix dispatch cores stay **`CoreType::WORKER`**; `TT_METAL_DPRINT_CORES=dispatch` uses WORKER loop, not DISPATCH reroute |
| **FD topology** | **`DispatchMemMap(WORKER)`** + Tensix **`QuasarDataMovementConfig`** on interim path until Phase 4b replaces initializer wiring |

Phase 4b must **not** remove or bypass the env override; default-path pool fix (**C**) must **not** break the YAML-backed pool when the env is set.

---

## Initial Bringup: SD Microbenchmark Tests

First validation milestone: run existing **slow-dispatch (SD)** microbenchmark tests with `cq_prefetch.cpp` and `cq_dispatch.cpp` on the dispatch engine, **without** full FD init and **without** exposing core locations through public device-open APIs.

**Note:** SD tests still call **`CreateDevice`** (slow dispatch). On the default dispatch-engine path, **`risc_firmware_initializer` loads `dispatch_dm.cc`** on soc dispatch cores at device open — same as fast dispatch. SD tests do **not** call `initialize_fast_dispatch()` or `dispatch_kernel_initializer`.

### Target tests

| Test binary | Relevant SD paths | Kernels placed today (Quasar interim) |
|-------------|-------------------|----------------------------------------|
| `test_prefetcher` | `SDPrefetchTestBase`, combined prefetch+dispatch SD path (`execute_generated_commands`) | `cq_prefetch.cpp` + `cq_dispatch.cpp` on Tensix `{0,0}` via `Common::sd_prefetch_core` / `Common::dispatch_core()` |
| `test_dispatcher` | SlowDispatch spoof-prefetch + dispatch path | `cq_dispatch.cpp` (+ spoof prefetch) on Tensix logical cores from `common.h` |
| `test_dispatch` (dispatch_program) | `MeshDispatchFixture` SD tests (eth/tensix bringup) | Does **not** launch `cq_*` kernels; must keep passing on Quasar with dispatch engine present (full Tensix compute grid available) |

Primary cq-kernel migration work is in **`test_prefetcher.cpp`** and **`test_dispatcher.cpp`**, which share helpers in `tests/tt_metal/tt_metal/perf_microbenchmark/dispatch/common.h`.

### Why SD tests fit this milestone

These tests **already bypass FD infrastructure**:

- Run with `TT_METAL_SLOW_DISPATCH_MODE=1` (or `get_fast_dispatch() == false` for SD fixtures)
- Build a `Program`, call `CreateKernel` for `cq_prefetch.cpp` / `cq_dispatch.cpp`, then `LaunchProgram`
- Do **not** call `initialize_fast_dispatch()`, `dispatch_core_manager` assignment, or `OpenDevice` dispatch options
- Manually stage L1/hugepage/FetchQ exactly like production (see `SDPrefetchTestBase::execute_generated_commands`)

Moving them to the dispatch engine is a **direct kernel placement change** plus HAL/coordinate support — not a user-facing API change.

### Test-only core resolution (not public API)

Replace hardcoded Tensix logical cores in `common.h`:

```cpp
// Today (Quasar interim — Tensix grid):
inline constexpr CoreCoord sd_prefetch_core = {0, 0};
inline CoreCoord dispatch_core(const IDevice* device) {
    return (device->arch() == ARCH::QUASAR) ? CoreCoord{0, 0} : CoreCoord{4, 0};
}
```

With **internal helpers** (test `common.h` and/or `tt::tt_metal::internal`, same stability rules as `api/internal/README.md`):

| Helper | v1 behavior |
|--------|----------------|
| `dispatch_engine_core(device, index)` | Synthetic logical `CoreCoord(index, 0)` → NOC0 via `CoreType::DISPATCH` |
| `sd_prefetch_core(device)` | `dispatch_engine_core(device, 0)` |
| `sd_dispatch_core(device)` | `dispatch_engine_core(device, 0)` (same core, different DM) |
| `prefetch_dm()` / `dispatch_dm()` | `DataMovementProcessor::RISCV_0` (DM0) / `RISCV_1` (DM1) |

Helpers call **`get_quasar_dispatch_cores()`** at runtime (soc `dispatch:` list when env unset; core descriptor YAML when `TT_METAL_TENSIX_DISPATCH_CORES=1`). On Quasar variants with no dispatch engines (env unset):

- **Default:** SD cq-kernel tests targeting the dispatch engine skip or fail setup (consistent with FD init failure).
- **`TT_METAL_TENSIX_DISPATCH_CORES=1`:** SD tests use the legacy Tensix logical cores from `common.h` / core descriptor YAML (interim path; env overrides dispatch-engine placement even when soc lists dispatch cores).

**Not exposed:** no `DispatchCoreConfig`, no `OpenDevice` / `MeshDevice` parameters, no Python/ttnn surface. Users cannot target dispatch-engine cores through normal APIs.

### Kernel placement mechanics

SD tests must switch from Tensix worker placement to dispatch-engine placement:

| Concern | Today (interim) | Target (dispatch engine) |
|---------|-----------------|---------------------------|
| Logical core | `{0,0}` on Tensix grid | `CoreCoord(0, 0)` synthetic index → soc dispatch NOC0 **`(0,2)`** |
| Physical core lookup | `device_->worker_core_from_logical_core(...)` | Internal `dispatch_engine_virtual_core(device, index)` using `CoreType::DISPATCH` |
| `CreateKernel` config | `QuasarDataMovementConfig{num_threads_per_cluster=1}` with DM auto-assign by creation order | **`internal::CreateDispatchEngineKernel(...)`** with explicit **DM0** (prefetch) / **DM1** (dispatch) on `HalProgrammableCoreType::DISPATCH` |
| Semaphores | `CreateSemaphore(..., CoreType::WORKER)` | `CreateSemaphore(..., CoreType::DISPATCH)` |
| L1 memmap | `dispatch_mem_map(CoreType::WORKER)` | `dispatch_mem_map(CoreType::DISPATCH)` |
| `FD_CORE_TYPE` / `DISPATCH_KERNEL` | Tensix programmable index; `DISPATCH_KERNEL=1` on FD/cq kernels | `HalProgrammableCoreType::DISPATCH` index; **keep `DISPATCH_KERNEL=1`** on `cq_prefetch` / `cq_dispatch` (same define as WH/BH FD kernels — enables dispatch-kernel code paths in profiler/sanitize/device-print) |

**Explicit DM pinning (confirmed):** use **`internal::CreateDispatchEngineKernel(device, core, dm_processor, ...)`** — do not rely on kernel creation order. SD tests and FD paths share this helper.

**Legacy kernels:** keep `is_legacy_kernel = true` for `cq_prefetch.cpp` / `cq_dispatch.cpp` during bringup (already used in SD path).

### Files to update for SD bringup (v1)

- `tests/tt_metal/tt_metal/perf_microbenchmark/dispatch/common.h` — core helpers, `make_sd_*_defines` phys coord args
- `tests/tt_metal/tt_metal/perf_microbenchmark/dispatch/test_prefetcher.cpp` — SD prefetch/dispatch launch path
- `tests/tt_metal/tt_metal/perf_microbenchmark/dispatch/test_dispatcher.cpp` — SD spoof-prefetch path
- `tt_metal/llrt/tt_cluster.cpp` — `CoreType::DISPATCH` coordinate translation (prerequisite)
- `tt_metal/impl/dispatch/dispatch_mem_map.cpp` — `CoreType::DISPATCH` branch (prerequisite)
- `tt_metal/impl/kernels/kernel.{hpp,cpp}` — **`DispatchEngineKernel`** configure path + `CoreType::DISPATCH` + explicit DM processors
- **`tt_metal/api/internal/dispatch/dispatch_engine_cores.hpp`** (required) — `dispatch_engine_core()`, **`CreateDispatchEngineKernel()`**, coordinate helpers; used by SD tests, FDKernel, allocator, validation, and tooling

### Relationship to full FD (later phase)

SD bringup and full FD share HAL/coordinate/memmap work but differ in orchestration:

```
SD tests (Phase 4a)          Full FD (Phase 4b)
─────────────────────        ─────────────────────────
CreateDevice (slow dispatch) initialize_fast_dispatch()
  → risc_firmware_initializer  → dispatch_kernel_initializer
    loads dispatch_dm.cc         (execution kernels, WH/BH pattern;
  (always on default path)        DispatchMemMap CoreType::DISPATCH)
  → DPrint server (Phase 3b)   → dispatch_s DRAM print agg (optional)
Manual Program + CreateKernel  FDKernel / DispatchTopology
Test common.h + internal       dispatch_core_manager pool
  dispatch_engine_cores.hpp
No user API exposure           No user API exposure (same)
```

Full FD later reuses the same dispatch-engine core type, DM assignment model, and memmap — only the **caller** changes from test harness to `dispatch_core_manager`.

### SD bringup acceptance criteria (2×3 emulator)

- `TT_METAL_SLOW_DISPATCH_MODE=1` + `emu-quasar-2x3_DISPATCH` simulator:
  - SD prefetcher tests in `test_prefetcher` pass with kernels on dispatch-engine core 0, DM0/DM1
  - SD dispatcher tests in `test_dispatcher` pass with `cq_dispatch.cpp` on dispatch engine
  - Both Tensix compute cores `(0,1)` and `(1,1)` remain available as worker targets (no interim dispatch Tensix reservation)
- **Status (partial):** ✅ `DispatchLinearWriteSDTestFixture` (dispatcher-only SD via `spoof_prefetch.cpp`); ✅ `SDPrefetchDRAMToL1TestFixture.TestTerminate` (prefetch+dispatch SD via `cq_prefetch.cpp` + FetchQ TLB path)
- No new public API for dispatch-engine core selection
- Running SD cq-kernel tests on the **default** dispatch-engine path requires soc dispatch cores and **`dispatch_dm.cc` loaded in Phase 3** (Phase **3b** for DPRINT); **`TT_METAL_TENSIX_DISPATCH_CORES=1`** runs SD cq-kernel tests on interim Tensix cores instead

---

## Suggested Implementation Phases

### Phase 0 — Prerequisites (UMD + spec)

- ~~Finalize dispatch-engine NOC coordinates for 2×3 emulator~~ — **done**; aligned with Aether (`dispatch: [0-2]`, `pcie: [1-2]`)
- ~~Confirm variant scope~~ — **only 2×3 emulator has dispatch engines**; other variants have none for now
- ~~DM count per dispatch-engine core~~ — **8 DMs (DM0–DM7), all valid for FD**
- ~~Dispatch-engine L1~~ — **4 MiB**; reuse common Quasar `dev_mem_map` until a difference is required
- ~~DM assignment~~ — **v1: DM0 prefetch, DM1 dispatch**; design for full N-core × 8-DM flexibility

### Phase 1 — Core discovery and coordinates (tt-metal + UMD)

- Read dispatch list from runtime soc descriptor; env unset + empty list → FD init failure
- Add `TT_METAL_TENSIX_DISPATCH_CORES` to `rtoptions` / `RunTimeOptions` (**checked first**; always forces Tensix pool when set)
- Synthetic logical coord mapping in `metal_soc_descriptor`
- Extend `Cluster::get_virtual_coordinate_from_logical_coordinates` for DISPATCH
- Extend **`MetalContext::is_coord_in_range`** to accept dispatch-engine coords (`Cluster::is_dispatch_core`) — required by `ProgramImpl::create_semaphore` / `CreateSemaphore(..., CoreType::DISPATCH)` in Phase 4a
- Internal `get_quasar_dispatch_cores()` / override in `get_logical_dispatch_cores()`

### Phase 2 — Internal type resolution

- `resolve_dispatch_core_type()` arch gate
- Update `dispatch_core_manager`, `DispatchQueryManager`, allocator, and program validation to use **`get_quasar_dispatch_cores()`** / **`resolve_dispatch_core_type()`** (not raw soc lists or `DispatchCoreConfig`)
- **Bringup fixes (see [Bringup Regressions](#bringup-regressions-and-fixes-post-phase-23b)):**
  - **`DispatchQueryManager::reset`:** restore **`arch != QUASAR`** guard on **`dispatch_s_enabled_`** for 1CQ (fix **A**)
  - **`program.cpp::validate_kernel_placement`:** pass **`device->build_id()`** (physical chip), not **`device->id()`** (mesh id), into `resolve_dispatch_core_type()` / service-core checks (fix **B**)

### Phase 3 — HAL and firmware

- `HalProgrammableCoreType::DISPATCH` + device **`ProgrammableCoreType::DISPATCH`** (new enum index; 4 MiB L1; shared Quasar `dev_mem_map`)
- **`qa_hal_dispatch.cpp`** — L1/memmap/launch registration for dispatch-engine cores; set **`KERNEL_CONFIG` mem-map size** explicitly (Tensix leaves it 0 since it uses the `max_runtime_args` constant, but DISPATCH runtime-arg validation reads `get_dev_size(DISPATCH, KERNEL_CONFIG)`)
- Extend **`hal_2xx_common.cpp`**: JIT routes DISPATCH DM firmware → **`dispatch_dm.cc`**
- New **`dispatch_dm.cc`** (DM-only FW derived from `dm.cc`; same `subordinate_map_t`, TRISC fields unused; DM0 orchestrates DM1–7)
- **`risc_firmware_initializer`:** load `dispatch_dm.cc` on soc dispatch-engine cores at **every `CreateDevice`**; `DISPATCH_MODE_HOST` — **must complete before Phase 4a SD `LaunchProgram`**
- **`DispatchMemMap` for `CoreType::DISPATCH`** (implementation; SD tests query directly in Phase 4a) — includes the **`DispatchSettings`** `CoreType::DISPATCH` case (reuse `init_worker_defaults`); `MetalContext` builds `DispatchMemMap(CoreType::DISPATCH)` during init on Quasar, so this must be in place in Phase 3
- **`DispatchEngineKernel`** + **`internal::CreateDispatchEngineKernel`** in required internal header
- **`dispatch_kernel_initializer`:** **not in Phase 3** — Phase 4b only (see below)
- **DPRINT (device-side only):** L1 `DPRINT_BUFFERS` in dispatch HAL memmap; `dprint.h` in **`dispatch_dm.cc`** — **host DPrint server attach is Phase 3b**

### Phase 3b — DPRINT enablement

Phase 3 provides device-side DPRINT infrastructure (L1 `DPRINT_BUFFERS` in dispatch HAL memmap, `dprint.h` in **`dispatch_dm.cc`**, JIT `-DDEBUG_PRINT_ENABLED` when DPRINT rtoptions are on). **Phase 3b** wires the **host DPrint server** so prints from dispatch-engine **firmware and kernels** are collected end-to-end — required before Phase 4a SD bringup when debugging with DPRINT.

**Why between Phase 3 and Phase 4a:** SD tests (Phase 4a) launch `cq_prefetch.cpp` / `cq_dispatch.cpp` on dispatch-engine DMs via `CreateDispatchEngineKernel`. Without Phase 3b, DPRINT calls in FW/kernels compile but the server never attaches to `CoreType::DISPATCH` tiles (buffers uninitialized, `TT_METAL_DPRINT_CORES=dispatch` does not match dispatch-engine core descriptors).

**Scope (host-side):**

- **`debug_helpers.hpp::GetAllCores`** — include soc dispatch-engine tiles as `{logical, CoreType::DISPATCH}` (synthetic index coords)
- **`DPrintServer::init_device` / `attach_device`** — iterate **`CoreType::DISPATCH`** (or equivalent) so dispatch-engine cores get init/enable magic on their L1 print buffers
- **`TT_METAL_DPRINT_CORES=dispatch`** — when `resolve_dispatch_core_type()` is `DISPATCH`, match entries from **`GetDispatchCores()`** with `CoreType::DISPATCH` (today the filter runs under the `CoreType::WORKER` loop and never matches). The selection is parsed as a **class name** stored under `CoreType::WORKER`, so the enable loop **reroutes** it to the `CoreType::DISPATCH` iteration and **`continue`s past the WORKER iteration**. ⚠️ Do **not** clear WORKER by setting its class to `RunTimeDebugClassNoneSpecified` — that falls through to the explicit-cores branch which calls `get_feature_cores(...).at(CoreType::WORKER)`; matching a class name makes `ParseFeatureCoreRange` return early without inserting a `WORKER` key, so `.at` throws `map::at` during `attach_device` (surfaces as a `map::at` exception in test `SetUp()`). Use `continue` instead. *(Fixed during Phase 4a bringup.)*
- **Explicit core targeting** — support synthetic dispatch logical coords (e.g. `(0,0)` → dispatch index 0) via env (extend `TT_METAL_DPRINT_CORES` parsing and/or add **`TT_METAL_DPRINT_DISPATCH_CORES`** with same syntax as worker/ETH/DRAM)
- **`get_enable_symbols_info()`** — `HalProgrammableCoreType::DISPATCH` legend (8-DM hex mask, same style as Quasar Tensix DM / ETH)
- **`device_print.h`** — kernel callstack PC/RA adjustment uses **`FD_CORE_TYPE` / `ProgrammableCoreType::DISPATCH`** slot in `kernel_config_base[]`, not hardcoded `ProgrammableCoreType::TENSIX`
- **`DPrintServer::Impl::get_core_buffers` DM sub-buffer split** — Quasar's on-device `DevicePrintMemoryLayout` always lays out the **TRISC sub-buffer first (3264 B)** then the **DM sub-buffer (1632 B)**. Dispatch-engine firmware compiles with **`COMPILE_FOR_DM`**, so `get_device_print_buffer()` returns `dprint_buf.buffer` = the **DM sub-buffer at offset +3264**. The host must therefore read/enable the DM sub-buffer at `structure_address + compute_size`, mirroring the DM half of the existing Quasar **TENSIX** split. Without this, `CoreType::DISPATCH` fell into the generic single-buffer path (offset 0 = TRISC region): firmware wrote at +3264 while host enabled/polled at +0 → **no prints** (and the buffer was never enabled). *(Fixed during Phase 4a bringup.)*
- **`GetRiscName` `DE-` prefix** — prepend **`DE-`** to the RISC name for `HalProgrammableCoreType::DISPATCH` cores so the DPRINT line prefix distinguishes dispatch-engine DMs from Tensix DMs (e.g. `0:0-0:DE-DM4: ...` vs `0:0-0:DM4: ...`) without baking the distinction into each DPRINT message. *(Added during Phase 4a bringup.)*
- **Kernel defines** — ensure **`DISPATCH_KERNEL=1`** on both **`cq_prefetch.cpp`** and **`cq_dispatch.cpp`** SD/FD define blocks (profiler/sanitize/device-print paths; plan requirement)

**Out of Phase 3b (later phases):**

- **`dispatch_s` DRAM print aggregation** (`DEVICE_PRINT_DISPATCH_ENABLED`) — Phase 4b when full FD + dispatch_s runs
- Watcher / profiler / inspector / NOC-sanitize / JIT build-hash updates for DISPATCH — Phase 5

**Acceptance criteria (2×3 emulator):**

- With DPRINT enabled (`TT_METAL_DPRINT_CORES=dispatch` or `TT_METAL_DPRINT_DISPATCH_CORES=all` / explicit dispatch logical coord), host collects prints from **`dispatch_dm.cc`** at `CreateDevice` (e.g. `"DM0-FW: initialized"`) on NOC0 `(0,2)` — line prefix marks the core as **`DE-DM<n>`** (e.g. `0:0-0:DE-DM4: ...`)
- After Phase 4a SD launch, DPRINT from **`cq_prefetch.cpp` / `cq_dispatch.cpp`** on DM0/DM1 appears on the host
- **`TT_METAL_TENSIX_DISPATCH_CORES=1`** — unchanged WH/BH-style DPRINT on Tensix dispatch cores from YAML (no regression)
- **Status: ✅ verified on 2×3 emulator** — `DispatchLinearWriteSDTestFixture.LinearWrite/49152B_3iter_4194304words_unicast` passes with end-to-end dispatch-engine DPRINT (`DE-DM*`) collected.

**Key files:** `impl/debug/dprint_server.cpp`, `impl/debug/debug_helpers.hpp`, `hw/inc/api/debug/device_print.h`, `llrt/rtoptions.cpp` (if new env var), SD `common.h` `make_sd_prefetch_defines` (`DISPATCH_KERNEL=1`)

### Phase 4a — SD microbenchmark bringup (**first runnable milestone**)

- Depends on **Phases 1–3 and 3b** (coordinates, type resolution, HAL, **`dispatch_dm.cc` at `CreateDevice`**, **DPRINT host attach for `CoreType::DISPATCH`**)
- **`dispatch_engine_cores.hpp`** + test `common.h` helpers + **`CreateDispatchEngineKernel`**
- Port `test_prefetcher` / `test_dispatcher` SD paths: `cq_prefetch.cpp` + `cq_dispatch.cpp` on dispatch engine NOC0 `(0,2)`, DM0/DM1, core index 0
- `CreateSemaphore` / memmap / coordinate lookup on `CoreType::DISPATCH`
- Kernel placement validation allows dispatch-engine cores for cq kernels (`DISPATCH_KERNEL=1` unchanged)
- **`DispatchEngineKernel::configure` must write binaries into the kernel-config ring buffer** via `llrt::write_binary_to_address(base_address + offsets[riscv_id])` — the same path as `DataMovementKernel` / `QuasarDataMovementKernel`. It must **not** use `test_load_write_read_risc_binary` (which loads to the JIT default address): SD `finalize_kernel_bins` places binaries in the kernel-config ring buffer and records `kernel_text_offset[]` accordingly, and `dispatch_dm.cc` jumps to `kernel_config_base + kernel_text_offset`. Loading to the wrong address makes the FW jump into empty/garbage memory after GO, so the kernels never complete and **`LaunchProgram` hangs** (no validation, no progress). *(Fixed during Phase 4a bringup.)*
- **`configure_static_tlbs` must map `CoreType::DISPATCH` cores** — SD `test_prefetcher` enqueues work by writing FetchQ slots through `get_static_tlb_window` on the prefetch physical core (dispatch engine `(0,2)` on 2×3). Extend `ll_api::configure_static_tlbs` in `tlb_config.cpp` to call `configure_tlb` for every `get_cores(DISPATCH, TRANSLATED)` entry, mirroring the existing TENSIX/ETH loops. Symptom without fix: `TLB window for core (0, 2) not found` (`tlb_manager.cpp`) in `SDPrefetchDRAMToL1TestFixture.TestTerminate`. *(Fixed during Phase 4a bringup.)*
- **Recommended SD prefetcher bringup order:** `SDPrefetchDRAMToL1TestFixture.TestTerminate` (`use_exec_buf_disabled`) first, then `DRAMToL1PagedRead` same param variant; defer `SmokeTest`, `RandomTest`, `HostTest`, and `use_exec_buf_enabled` until the issue-queue path is stable
- Confirm `test_dispatch` (dispatch_program) SD tests still pass with full Tensix compute grid

### Phase 4b — Full FD kernel integration

- **`dispatch_kernel_initializer`:** Quasar path registers **`DispatchMemMap(CoreType::DISPATCH)`** (replaces today’s `DispatchMemMap(WORKER)`); compile/configure prefetch/dispatch execution kernels — WH/BH pattern
- FDKernel processor assignment on dispatch-engine DMs (`configure_kernel_variant` → `CreateDispatchEngineKernel`)
- Topology pool wiring via `dispatch_core_manager`
- Remove Tensix-based Quasar dispatch YAML dependency for the **default** path only — keep YAML + **`TT_METAL_TENSIX_DISPATCH_CORES=1`** as interim fallback
- **Dispatch pool for 1CQ same-core (fix **C**, required):** expand soc-based pool (duplicate synthetic `(0,0)` entries) **or** reuse one tile across prefetch + dispatch HD without double-pop — see [Bringup Regressions](#bringup-regressions-and-fixes-post-phase-23b)
- **Regression gate:** `QuasarMeshDeviceSingleCardFixture.SingleDmL1Write` passes on 2×3 emulator **with and without** `TT_METAL_TENSIX_DISPATCH_CORES=1` after Phase 4b

### Phase 5 — Tooling and integration tests

- Watcher / profiler / inspector updates for `CoreType::DISPATCH`
- **`jit_build` / `build_env_manager`** — include resolved dispatch type (`DISPATCH` vs `WORKER`) in build hash on Quasar
- NOC sanitize — dispatch-engine in valid core sets
- UMD + tt-metal coordinate round-trip / soc descriptor tests
- **Out of scope for Phase 5:** re-doing Phase 4a SD cq-kernel migration (`test_prefetcher` / `test_dispatcher`) — that is the Phase 4a deliverable; **DPRINT host wiring is Phase 3b**

---

## Files Likely Touched (by area)

| Area | Key files |
|------|-----------|
| UMD | `grendel_implementation.hpp`, soc YAMLs, `coordinate_manager.cpp`, `quasar_coordinate_manager.cpp` |
| Core discovery | `llrt/core_descriptor.cpp`, new helper in `core_descriptor.hpp` or `metal_soc_descriptor.cpp` |
| Coordinates | `llrt/tt_cluster.cpp`, `llrt/metal_soc_descriptor.{hpp,cpp}`, **`llrt/tlb_config.cpp`** (`configure_static_tlbs` for `CoreType::DISPATCH`) |
| Device FW types | `hw/inc/internal/tt-2xx/quasar/core_config.h` (`ProgrammableCoreType::DISPATCH`) |
| Type resolution | `dispatch_core_common.cpp`, `dispatch_core_manager.{hpp,cpp}`, `llrt/rtoptions.{hpp,cpp}` |
| HAL | `hal_types.hpp`, **`qa_hal_dispatch.cpp`**, `qa_hal.cpp`, **`hal_2xx_common.cpp`**, `llrt/llrt.cpp` |
| Dispatch FW | `hw/firmware/src/tt-2xx/dispatch_dm.cc` (new; based on `dm.cc`, TRISC removed) |
| FD | `fd_kernel.cpp`, `prefetch.cpp`, `dispatch.cpp`, `dispatch_mem_map.cpp`, `dispatch_settings.cpp` (DISPATCH core-type case), `topology.cpp`, `kernel.{hpp,cpp}` |
| SD bringup tests | `tests/.../dispatch/common.h`, `test_prefetcher.cpp`, `test_dispatcher.cpp` |
| Internal (required) | `tt_metal/api/internal/dispatch/dispatch_engine_cores.hpp` — **`CreateDispatchEngineKernel`**, coordinate helpers |
| FW init | `risc_firmware_initializer.cpp`, `dispatch_kernel_initializer.cpp` |
| Allocator | `l1_banking_allocator.cpp` |
| Validation | `program.cpp`, `metal_context.cpp` (`is_coord_in_range` accepts DISPATCH coords) |
| Debug (DPRINT) | **Phase 3b:** `debug_helpers.hpp`, `dprint_server.cpp` (attach reroute, `get_core_buffers` DM split, `GetRiscName` `DE-` prefix), `device_print.h`, `rtoptions.cpp` |
| Debug (other) | **Phase 5:** `profiler.cpp`, inspector, watcher, NOC sanitize |

---

## Design Decisions (Confirmed)

| Topic | Decision |
|-------|----------|
| **DM assignment (v1)** | Prefetcher on **DM0**, dispatcher on **DM1** on dispatch-engine core 0 (NOC0 `(0,2)`) |
| **DM assignment (future)** | Full flexibility: multiple dispatch cores, multiple prefetcher/dispatcher kernels (e.g. per CQ), specialized blocks on other DMs; implementation must not hard-code v1 layout |
| **SD bringup** | SD tests place `cq_prefetch.cpp` / `cq_dispatch.cpp` via **test/internal helpers only** — no public API for dispatch-engine locations |
| **DMs per core** | 8 (DM0–DM7); all valid for FD |
| **L1 / memory map** | **4 MiB** L1; reuse common Quasar **`dev_mem_map`** until a dispatch-specific difference is required |
| **Logical coordinates** | **No UMD logical dispatch grid**; tt-metal synthetic **`CoreCoord(index, 0)` → NOC0** only |
| **Explicit DM pinning** | **`internal::CreateDispatchEngineKernel`** — DM0/DM1 explicit; no creation-order dependency |
| **NOC coordinates (2×3)** | **`dispatch (0,2)`**, **`pcie (1,2)`** — aligned with Aether |
| **Dispatch-engine FW** | New **`dispatch_dm.cc`** (DM-only; same `subordinate_map_t`, TRISC fields unused; DM0 orchestrates DM1–7). Not `dm.cc`. |
| **`dispatch_dm.cc` load** | **Every `CreateDevice`** on default dispatch-engine path via `risc_firmware_initializer` (slow + fast dispatch); required before Phase 4a SD `LaunchProgram` |
| **DPRINT on dispatch engine** | **Phase 3b** — host DPrint server attaches to `CoreType::DISPATCH`; device-side buffers from Phase 3 HAL; end-to-end before Phase 4a debug |
| **`dispatch_kernel_initializer`** | **Phase 4b only** — `DispatchMemMap(CoreType::DISPATCH)` + execution kernels; **skipped in slow dispatch / Phase 4a** (`!using_fast_dispatch()` early return) |
| **`DispatchEngineKernel`** | Dedicated configure path (like `DramKernel`); used by SD, FDKernel, and `CreateDispatchEngineKernel`. **`configure` writes binaries into the kernel-config ring buffer** via `write_binary_to_address(base + offsets[riscv_id])` (same as `DataMovementKernel`), **not** `test_load_write_read_risc_binary` — otherwise SD `LaunchProgram` hangs (FW jumps to wrong L1 address). |
| **Static TLB (dispatch engine)** | **`configure_static_tlbs`** maps `CoreType::DISPATCH` soc tiles (same as TENSIX/ETH) so `get_static_tlb_window` works for FetchQ writes on prefetch cores at NOC0 `(0,2)`; required for SD `test_prefetcher` and FD `system_memory_manager` |
| **`kernel_config_base[]` index** | Dispatch-engine kernels use **`ProgrammableCoreType::DISPATCH`** slot (new enum index), not Tensix |
| **`dispatch_engine_cores.hpp`** | **Required** internal header — coordinates, resolver helpers, `CreateDispatchEngineKernel` |
| **Execution kernel init** | **`dispatch_kernel_initializer`** (Phase 4b) — `DispatchMemMap(CoreType::DISPATCH)` + prefetch/dispatch execution kernels |
| **`DISPATCH_MODE_DEV`** | Set on **destination worker** launch messages only (`dispatch.cpp`); dispatch-engine cores use **`DISPATCH_MODE_HOST`** |
| **`DISPATCH_KERNEL` define** | **Keep `DISPATCH_KERNEL=1`** on cq/FD kernels on dispatch-engine DMs (profiler/sanitize paths) |
| **JIT build hash** | Uses **resolved dispatch type** (`DISPATCH` vs `WORKER` from `resolve_dispatch_core_type()`), not user `DispatchCoreConfig` |
| **`TT_METAL_TENSIX_DISPATCH_CORES=1`** | **Always** forces interim Tensix path when set — overrides soc dispatch-engine cores; interim YAML coords unchanged |
| **Interim Tensix FD regression tests** | **`SingleDmL1Write`** (and similar mesh FD tests) must pass with env set after Phase 4b; fixes **A** + **B** required |
| **Default dispatch-engine FD (1CQ pool)** | Soc **`dispatch: [0-2]`** is one tile; pool must allow **two** assignment pops for prefetch + dispatch HD (fix **C**) |
| **MeshDevice compile / validation** | Use **`build_id()`** for physical chip in soc/service APIs; **`id()`** is mesh id only |
| **No dispatch engines (default)** | Quasar **FD init fails** with clear message (env unset, empty soc dispatch list) |
| **`dispatch_core_axis` on Quasar** | **Silently ignored** (along with other `DispatchCoreConfig` dispatch-engine fields) |
| **Scope (v1)** | **Single-chip 1CQ** only; design extensible to multi-CQ and multi-chip |
| **Compute grid** | Moving FD to dispatch engine **frees Tensix cores** previously used as interim dispatch cores |
| **HAL / device types** | **`HalProgrammableCoreType::DISPATCH`** + **`ProgrammableCoreType::DISPATCH`** |
| **Identity translation** | **`CoreType::DISPATCH`:** NOC0 ↔ TRANSLATED identity (confirmed) |
| **`router_only` (UMD)** | **No change** for 2×3 dispatch variant |

---

## Recommended Sequencing

Implement **Phases 1 → 2 → 3 → 3b → 4a** as the **first runnable milestone**:

1. **Phase 1** — soc dispatch pool, synthetic coords, `TT_METAL_TENSIX_DISPATCH_CORES` (env checked first)
2. **Phase 2** — `resolve_dispatch_core_type()`, allocator/validation
3. **Phase 3** — HAL, **`ProgrammableCoreType::DISPATCH`** (new index), **`qa_hal_dispatch.cpp`**, **`dispatch_dm.cc`** at every `CreateDevice`, `DispatchMemMap`, **`DispatchEngineKernel`**, required **`dispatch_engine_cores.hpp`**
4. **Phase 3b** — DPRINT host attach for **`CoreType::DISPATCH`** (FW + cq-kernel prints collectible before SD bringup)
5. **Phase 4a** — SD `test_prefetcher` / `test_dispatcher` on dispatch engine `(0,2)`, DM0/DM1 (requires `dispatch_dm.cc` already loaded; no `dispatch_kernel_initializer`)

**Phase 4b** wires full FD through `dispatch_core_manager` and **`dispatch_kernel_initializer`** (`DispatchMemMap(CoreType::DISPATCH)`). Complete **bringup regression fixes A–C** (see [Bringup Regressions](#bringup-regressions-and-fixes-post-phase-23b)) before declaring Phase 4b done. **`TT_METAL_TENSIX_DISPATCH_CORES=1`** always selects interim Tensix fallback from core descriptor YAML when set. Multi-CQ / multi-chip extensions follow the same core pool and DM model without public API changes.
