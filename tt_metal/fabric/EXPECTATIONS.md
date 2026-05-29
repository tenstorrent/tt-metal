# tt_metal/fabric — Engineering Expectations

This document captures coding standards, architectural invariants, and recurring
pitfalls specific to the fabric layer. It is the first thing to read before
sending a PR that touches this directory.

---

## 1. C++ Best Practices

### Non-copyable, non-movable by default

Heavy state-bearing classes (`FabricContext`, `FabricBuilderContext`,
`FabricEriscDatamoverBuilder`, router builders) must be declared non-copyable
and non-movable:

```cpp
MyClass(const MyClass&) = delete;
MyClass& operator=(const MyClass&) = delete;
MyClass(MyClass&&) = delete;
MyClass& operator=(MyClass&&) = delete;
```

These objects own firmware configuration state. Accidental copies produce
silent divergence between what the host configures and what runs on device.

### Always use `TT_FATAL` / `TT_ASSERT` for preconditions

Every public method that has a precondition must enforce it at entry:

```cpp
void FooBuilder::set_channel(size_t idx) {
    TT_FATAL(idx < max_channels_, "Channel index {} out of range [0, {})", idx, max_channels_);
    ...
}
```

Do not use bare `assert()` — it compiles out in Release. `TT_FATAL` fires in
all build types and includes a message. `TT_ASSERT` is acceptable for
purely debug-time checks where the condition is expensive to evaluate.

### Use `std::optional` for absent values, never sentinel integers

Sentinel values (`-1`, `0xFFFFFFFF`, `UINT32_MAX`) for "not set" have caused
real OOB bugs in this codebase (see §4). Prefer `std::optional<T>` and use
`.value_or(sentinel)` only at the call site where firmware requires a raw
integer.

```cpp
// Bad
uint32_t teardown_sem_id = UINT32_MAX;  // means "not set"

// Good
std::optional<size_t> teardown_sem_id;
```

### Prefer named constants over magic numbers

Sizes, channel counts, and firmware offsets must come from `builder_config::*`
or named `constexpr` values — never appear as bare integers. A mismatch
between a loop bound and an array size is the single most common class of bug
in this layer.

### Headers shared between host and device must guard with `#if defined(KERNEL_BUILD) || defined(FW_BUILD)`

Headers under `hw/inc/` and host-facing headers like `fabric_edm_packet_header.hpp`
compile in both contexts. Use the standard guard:

```cpp
#if defined(KERNEL_BUILD) || defined(FW_BUILD)
#include "api/debug/assert.h"
#else
#include <tt_stl/assert.hpp>
#endif
```

Never include host-only headers (STL streams, `<filesystem>`, etc.) unconditionally
in a file that may be compiled for device.

### Keep the `FabricContext` / `FabricBuilderContext` split

`FabricContext` is immutable after construction — topology queries, packet
specs, mesh type. `FabricBuilderContext` holds mutable build-time state (EDM
configs, per-device state, tensix config).

Do not add mutable fields to `FabricContext`. If you need to cache something
that depends on build-time state, put it in `FabricBuilderContext` or compute
it on demand.

### Builder phases must be called in order

`FabricBuilder` has a strict lifecycle:

```
discover_channels() → create_routers() → connect_routers() →
compile_ancillary_kernels() → create_kernels()
```

Each phase asserts that the previous phase completed. Do not add logic that
bypasses or reorders phases. If a new phase is needed, add it explicitly with
its own guard flag and document it here.

---

## 2. Array and Buffer Sizing

### Array size and loop bounds must come from the same constant

This is the root cause of the OOB read fixed in #43690:
`receiver_channels_downstream_teardown_semaphore_id` was sized with
`max_downstream_edms` (8) but iterated with `num_max_sender_channels` (10).

Rule: if you declare `std::array<T, builder_config::FOO>`, every loop over
that array must be bounded by `builder_config::FOO`. If firmware requires a
different number of values, emit padding explicitly rather than reading past
the array end:

```cpp
for (uint32_t i = 0; i < builder_config::num_max_sender_channels; i++) {
    if (i < builder_config::max_downstream_edms) {
        args.push_back(my_array[i].value_or(-1));
    } else {
        args.push_back(-1);  // padding required by firmware ABI
    }
}
```

Document in a comment why the two constants differ, and open a follow-up
issue to reconcile them.

### Runtime args must match firmware ABI exactly

`get_runtime_args()` produces a vector whose layout is a firmware ABI.
Changing the order, count, or type of any element is a breaking firmware
change. When modifying `get_runtime_args()`:

1. Update the corresponding firmware reader in lock-step.
2. Run hardware validation on both WH and BH before merging.
3. Add a `static_assert` on the expected argument count where possible.

---

## 3. Routing and Topology

### Never hardcode topology assumptions

Code must not assume linear 1D, 2D mesh, or torus topology. Query
`FabricContext`:

```cpp
if (fabric_context.is_wrap_around_mesh(mesh_id)) { ... }
if (fabric_context.is_2D_routing_enabled()) { ... }
```

The same binary runs on N150, N300, T3K, Galaxy, and multi-chassis torus
configurations. Hardcoded topology assumptions have caused silent
misconfiguration on new platforms.

### `FabricConfig::FABRIC_1D_RING` on WH T3K uses `FabricType::MESH`

WH T3K does not have physical wrap-around connections. Despite the ring
config name, `get_fabric_type()` returns `MESH` for WH non-galaxy hardware
(issue #32146). Do not infer torus routing from the config enum alone —
query `is_wrap_around_mesh()`.

### VC assignment is topology-dependent

Virtual channel assignment (`IntermeshVCMode`) is set during `FabricContext`
initialization based on the number of meshes and their connectivity pattern.
Do not assign VCs manually. Use the `IntermeshVCConfig` from
`FabricBuilderContext` and the `IntermeshVCMode` / `IntermeshRouterType`
enums. Incorrect VC assignment causes deadlocks on multi-mesh topologies.

### `RoutingDirection::Z` requires explicit support checks

Z-direction routing (vertical device stacking) is not available on all
hardware. Before emitting Z-direction channels:

```cpp
if (fabric_context.has_z_router_on_device(control_plane, node_id)) { ... }
```

---

## 4. Known Recurring Pitfalls

### Pitfall: mismatched array size and loop bound (OOB read)

**Root cause:** Two `builder_config::` constants that are semantically related
but numerically different (`max_downstream_edms` vs `num_max_sender_channels`)
used in the same loop.

**Symptom:** Silent garbage data in Release builds; bounds-check assertion
crash in Debug builds with libstdc++ ≥ 15.

**Fix pattern:** See §2 above. Always use the array's own size as the loop
bound; emit padding for any firmware ABI overhang.

### Pitfall: FabricConfig → FabricType mismatch on T3K

**Root cause:** `FABRIC_1D_RING` implies torus semantics but WH T3K hardware
does not support wrap-around. `get_fabric_type()` special-cases this.

**Symptom:** Routing table generator produces unreachable routes on T3K.

**Fix:** Query `is_wrap_around_mesh()` before assuming ring topology.

### Pitfall: topology solver non-determinism from SAT solver

The topology solver uses a SAT backend (CaDiCaL). SAT solvers may return
different valid solutions across runs. Code that compares solution indices
or depends on a specific channel assignment order between runs is fragile.

**Fix:** Compare channel *properties* (direction, endpoint, VC) not
channel indices. Do not serialize channel indices to disk and reload them
in a different run without re-validating.

### Pitfall: channel trimming state not persisted across FabricBuilder reconstructions

Channel trimming state is loaded at `FabricContext` construction time and
does not reload automatically if the context is reconstructed. If you tear
down and re-initialize fabric mid-test, re-import channel trimming overrides
explicitly.

### Pitfall: `GRACEFULLY_TERMINATE` is non-functional

`TerminationSignal::GRACEFULLY_TERMINATE` is defined but not implemented in
firmware. It does not drain outstanding messages. Use
`IMMEDIATELY_TERMINATE` and design your teardown to tolerate in-flight
messages, or drain explicitly before signalling termination.

---

## 5. Testing

- CPU-only unit tests (topology solver, routing table generator, builder
  config) belong in `tests/tt_metal/tt_fabric/` and must not require hardware.
  Use the `cpu_only` label.

- Hardware tests that exercise EDM kernel behavior must run on both WH
  (N150/N300/T3K) and BH (N150/N300) configurations before merging.

- Any change to `get_runtime_args()` or EDM kernel ABI requires hardware
  validation — CI alone is insufficient because runtime arg mismatches
  often manifest as silent hangs, not crashes.

- New topology configurations (mesh descriptors in `mesh_graph_descriptors/`,
  cabling descriptors in `cabling_descriptors/`) must be accompanied by a
  CPU-only routing correctness test that exercises every (src, dst) pair.
