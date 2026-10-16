# `MetalEnv` public API review

Header under review: [`tt_metal/api/tt-metalium/experimental/context/metal_env.hpp`](../../tt_metal/api/tt-metalium/experimental/context/metal_env.hpp).
Related: [`Fabric-MetalEnv-Uplift-Plan.md`](Fabric-MetalEnv-Uplift-Plan.md) (fabric config fixed at construction; dispatch-only fabric).

Status: draft. `MetalEnv` has no Python bindings, and apart from `get_system_mesh()` nothing outside tests calls its getters (now `get_device_properties()` and the two device counts) or `create_*` methods. Reshaping it now is nearly free.

**Approach: conservative first step.** Fix the descriptor, and replace the positional `create_*` parameters with one options struct whose defaults match today's. Don't split parameters by scope, and don't solve query-before-open yet. Both stay possible later without changing the rest of the API shape (§4.2). The analysis for the deferred work is kept below (§2.4, §4.3-§4.5).

## Decisions so far

| Topic | Decision |
| --- | --- |
| Ownership model | Today's model is transitory. The end state has no global `MetalContext`; the env always owns its context, never the other way around. API decisions below assume the end state. |
| `MetalEnvDescriptor` | Option B: an aggregate with an explicit `MetalEnvTarget` and the fabric descriptor (§2.3). No dispatch settings in the descriptor for now. |
| Query before open | Deferred (§2.4). |
| Getters | Hardware queries move to a non-owning `MetalDeviceProperties` handle returned by `MetalEnv::get_device_properties()`; getters drop `get_` and the SFPU constants get a `sfpu_` prefix. `get_num_pcie_devices()` and `get_num_available_devices()` stay on `MetalEnv` (§3). |
| `create_*` parameters | One `CreateMeshDeviceOptions` struct holding all five settings, with today's defaults; `l1_bank_remap` is deleted. Splitting env-wide settings into the descriptor is deferred and has a non-breaking migration path (§4.2). |
| Dispatch default and `DispatchCoreConfig` | Deferred. Analysis in §4.3-§4.5: tt_metal should own ttnn's cluster-aware default, and `DispatchCoreConfig` could then become single-phase. |
| `get_system_mesh()` | Accept for now. The end state removes run-time resets of the control plane and system mesh, which removes the hazards (§5). |

---

## 1. Ownership model: current state is transitory

Today there are three ownership patterns. Which one applies depends on how the env was reached and in what order its methods were called.

1. **Legacy default env: the context owns the env.** A bare `MetalContext::instance()` creates a `MetalEnv` implicitly, builds a `MetalContext` on it, and sets `env_owned_ = true` so the context deletes the env (`metal_context.cpp:431-438`).
2. **Explicit env, system mesh queried first: the env owns the context.** `MetalEnv::get_system_mesh()` calls `ensure_context_registered`, which creates a context that `~MetalEnv` tears down (`metal_env.cpp:396-408, 625-631`).
3. **Explicit env, `create_*` called first: the mesh device owns the context.** Each `create_*` call makes a new context and marks the mesh device to destroy it on close (`metal_env.cpp:671-685`).

All contexts live in a process-global registry (`g_instances[MAX_CONTEXT_COUNT]`, silicon always at slot 0), and bare `instance()` falls back to "any live context" as a bridge for legacy callers (`metal_context.cpp:402-419`).

**End state.** There is no global `MetalContext` and no registry. A `MetalEnv` always owns its context. Everything that exists only to bridge the global registry is scaffolding and should not shape the public API:

- the `env_owns_context` branches in the three `create_*` methods;
- `ensure_context_registered` / `has_registered_context`;
- `SubDevice`'s fallback to the default context (`get_env_or_default`, `sub_device.cpp:24-29`);
- the bare `instance()` fallback.

A visible symptom of the transitory model: whether a mesh device's close destroys the context depends on whether the user called `get_system_mesh()` first. That goes away in the end state and is not tracked as a separate issue here.

---

## 2. `MetalEnvDescriptor`

### 2.1 Problems

- **String literals don't compile.** `MetalEnvDescriptor("blackhole_P150.yaml")` is ambiguous between `explicit MetalEnvDescriptor(const std::string&)` and `explicit MetalEnvDescriptor(std::optional<std::string>)`. Confirmed with gcc and clang. This has been true since the class was introduced, and it's why every caller wraps the argument.
- **A failed mock lookup silently opens silicon.** `experimental::get_mock_cluster_desc_name(arch, n)` returns `nullopt` for unsupported pairs. Most call sites pass the result straight in (e.g. `test_mock_allocator.cpp:38`, `test_query_op_constraints_mock_device.cpp:52`, many in `test_integration.cpp`), and `nullopt` means silicon.
- **`""` means different things per constructor.** Through the `string` constructor it means silicon. Through `std::optional<std::string>{""}` it makes `is_mock_device()` true, but `RunTimeOptions::set_mock_cluster_desc("")` is a no-op, so the cluster opened is silicon.
- **The name says "path", but a bare filename is accepted** and searched for (`rtoptions.hpp:904-915`).
- **`mock_cluster_desc_path()` dereferences `nullopt`** (undefined behavior) on a silicon descriptor.
- **Configuring fabric on silicon reads badly:** `MetalEnvDescriptor(std::nullopt, fabric)`.
- **`protected` members exist only because the internal `ContextDescriptor` inherits from it.** Through that base, `ContextDescriptor::is_mock_device()` means "Mock but not Emule" when built by `init_context_descriptor` (`metal_context.cpp:728-729`), but "any mock path" when built by `init_risc_fw_context_descriptor` (`:757`).

### 2.2 History: how it got this shape

No one designed the constructor set. It accreted as an internal convention was promoted to public API, and each later feature appended a constructor.

| Commit | PR | Date | Change |
| --- | --- | --- | --- |
| `ea7b2e14572` | #37453 | 2026-02-13 | Splits initialization into classes and introduces the internal `ContextDescriptor`: one positional bag of fabric, dispatch, allocator and mock settings. The mock cluster is `std::string_view mock_cluster_desc_path = ""`, and `is_mock_device()` is `!mock_cluster_desc_path_.empty()`. This is where "empty string means silicon" comes from. |
| `1e9bbdabff5` | #38815 | 2026-03-09 | "Add a query layer": introduces `MetalEnv` and carves `MetalEnvDescriptor` out of `ContextDescriptor` **as its base class**, hence the `protected` members. The mock path was the only env-level field, so "no argument means silicon, a string means mock" was the whole descriptor. Both the `const std::string&` constructor (keeping the internal `""` convention) and the `std::optional<std::string>` constructor were added in this PR, so the literal-string ambiguity dates from day one. `ContextDescriptor` converts `""` to `nullopt` itself before calling the base. |
| `76612dd48f9` | #39563 | 2026-03-11 | Lowers fabric, control plane and system mesh to `MetalEnv`. Adds `FabricConfigDescriptor` as an aggregate, but attaches it through a third positional constructor `(optional<string>, FabricConfigDescriptor)` instead of turning the descriptor into an aggregate too. |
| `d02dc4a32e0` | #39655 | 2026-03-17 | Adds `create_mesh_device` / `create_unit_mesh_device` / `create_unit_meshes` / `create_sub_device`, copying `MeshDevice::create`'s positional parameter list verbatim, including `l1_bank_remap`. `ContextDescriptor` stops carrying fabric settings and reads them from the env. |
| `158cd09f9c1` | #43754 | 2026-05-26 | Mock and silicon coexistence. Exports `get_mock_cluster_desc_name(arch, n)` returning `std::optional<std::string>`, which pairs with the `optional` constructor. This is what makes the silent silicon fallback easy to hit. The recommended usage in `mock_device.hpp:47-51` calls `.value()`, but most tests don't. |
| `3a77992ea44` | #55418 | 2026-09-04 | Hides `ControlPlane` from the public `MetalEnv` API. No descriptor change. |

### 2.3 Options

**A. Named factories on the existing class.** `MetalEnvDescriptor::silicon(FabricConfigDescriptor = {})`, `::mock(std::string cluster_desc, FabricConfigDescriptor = {})`, `::mock(tt::ARCH, uint32_t num_chips, ...)`, the last failing loudly on an unsupported pair. Fixes all bugs in §2.1, but every new field adds a parameter to every factory.

**B. Aggregate with an explicit target (decision).** Matches `FabricConfigDescriptor`, supports designated initializers (and keyword arguments in Python), and extends naturally to Simulator and Emule targets, which today are chosen only by environment variables.

```cpp
class MetalEnvTarget {
public:
    static MetalEnvTarget silicon();
    static MetalEnvTarget mock(std::string cluster_desc);           // filename or path
    static MetalEnvTarget mock(tt::ARCH arch, uint32_t num_chips);  // TT_FATAL if unsupported
    bool is_mock() const;
    const std::string& mock_cluster_desc() const;                   // TT_FATAL if !is_mock()
};

struct MetalEnvDescriptor {
    MetalEnvTarget target = MetalEnvTarget::silicon();
    FabricConfigDescriptor fabric = {};
    // A `dispatch` member is added here only if the §4.2 split happens.
};

MetalEnv env({.target = MetalEnvTarget::mock(tt::ARCH::BLACKHOLE, 2),
              .fabric = {.fabric_config = tt_fabric::FabricConfig::FABRIC_2D}});
```

**C. Minimal fix.** Keep the class, drop the `optional` constructors, reject `""`.

With A or B, `ContextDescriptor` stops inheriting from the descriptor. It already holds `env_`, so it can ask the env whether it is a mock. The descriptor would also be the natural home for settings that today come only from environment variables: the custom mesh graph descriptor path (`TT_MESH_GRAPH_DESC_PATH`) and fast versus slow dispatch.

### 2.4 Deferred: querying the cluster before creating the env

Not part of the first step. Existing callers keep using the legacy global queries (`ttnn.cluster.get_cluster_type()` etc.) until this is revisited. The analysis below is kept for then.

Users sometimes pick fabric or dispatch config from cluster properties. Real examples:

- `tests/scripts/common.py:281-296` forces a COL dispatch axis on Blackhole unless both fabric and tensix config are set.
- `models/common/tests/conftest.py:185-203` picks a fabric config from the mesh shape.
- Many tests branch on `ttnn.cluster.get_cluster_type()` (T3K, Galaxy, P150, ...).

Today these queries go through the implicit default context, which opens the silicon cluster. With fabric and dispatch config fixed at env construction, that becomes chicken-and-egg: the answer lives on the env, but the env needs the answer.

**Option 1: discovery before the env.** A cheap, non-owning query over a `MetalEnvTarget` that doesn't open devices:

```cpp
struct ClusterInfo {
    tt::ARCH arch;
    ClusterType cluster_type;
    uint32_t num_devices;
    uint32_t num_pcie_devices;
    bool has_remote_devices;
};

ClusterInfo info = MetalEnvTarget::silicon().discover();
MetalEnv env({.target = MetalEnvTarget::silicon(), .fabric = choose_fabric(info)});
```

This is the physical-device pattern from other frameworks (enumerate and query, then create with a CreateInfo).

- Pros:
  - Descriptor fields stay concrete, so creating and querying use the same types.
  - It supports any user policy, not just the ones the library encodes.
  - It is also the natural home for the query the uplift plan wants under dispatch-fabric option B ("does opening these device ids under fast dispatch need fabric?"). The env could expose the same `ClusterInfo` later.
- Precedent:
  - Silicon arch is already discovered without opening devices, via `PCIDevice::enumerate_devices_info()` (`get_platform_architecture.hpp:19-41`).
  - `Cluster::get_cluster_type_from_cluster_desc` already runs a throwaway UMD topology discovery when not given a descriptor (`tt_cluster.cpp:88-111`).
- Cons and open questions:
  - Silicon topology discovery might run twice: once for `discover()`, once inside the env. To avoid that, the target could cache the discovered UMD `ClusterDescriptor` and hand it to `Cluster`, which today only has `Cluster(RunTimeOptions&)`.
  - Need to confirm UMD topology discovery doesn't hold the `CHIP_IN_USE` lock past the call.
  - `get_physical_architecture()` caches in a function-local static, so it can't reflect hardware changes within one process. That is probably fine.

**Option 2: "auto" values resolved at construction (your proposal).** Descriptor fields accept "unset", and the env resolves them for the discovered cluster.

- The asymmetry you raised is real, but it only hurts when one type plays both roles. Precedent and its cost: `DispatchCoreConfig` has an optional axis that `resolve_dispatch_core_axis` fills from arch and tensix config (`dispatch_core_common.cpp:19-29`). Because one type is both request and result, `MetalContext::initialize` has to resolve before comparing, or an unset request compares unequal to the stored resolved value and tears down a context that actually matches (`metal_context.cpp:172-178`).
- Mitigation: two types per config, a request type with optional fields in the descriptor and a resolved type with concrete fields returned by the env (e.g. `env.get_dispatch_config()`). This also gives "requested vs effective" a clean answer (§6).
- Pros: no boilerplate for the common case; discovery runs once.
- Cons: covers only policies the library encodes, so users with custom rules still need option 1; doubles the number of config types. An "auto" fabric default would reintroduce the implicit dispatch-only fabric switch that the uplift plan's option B removes, so fabric should not get an "auto" for that purpose.

**Option 3: resolver callback in the descriptor.** E.g. `std::function<DispatchDescriptor(const ClusterInfo&)> dispatch_policy`, called by the env after discovery.

- Pros: discovery runs once; custom policies work; the resolved types are concrete.
- Cons: the descriptor stops being a plain value (can't be compared, printed or hashed, and configs are compared today); user code runs inside env construction, so exception and re-entrancy rules must be defined; less natural in Python.

**Option 4: two-phase env (construct, then configure).** Config would be fixed at first use, not at construction. Same shape as the uplift plan's dispatch-fabric option C, with the same objection.

**Assessment.** Option 1 is the foundation: it's the only option that handles arbitrary user policy while keeping one concrete type per config. Option 2 can be layered on later for policies the library already owns, provided request and resolved types are distinct. Dispatch config is the obvious candidate, and the history shows the pre-env query for it exists only because tt_metal never owned a cluster-aware default (§4.5). With that default in tt_metal, dispatch needs no discovery at all, and option 1's remaining users are custom policies and fabric choices (such as picking fabric from the mesh shape). For those, a throwaway `MetalEnv` is an acceptable stopgap, so option 1 is an optimization, not a prerequisite. The main thing to verify before committing to option 1 is the cost and locking behavior of silicon topology discovery.

---

## 3. Getters

Original surface (one method per property on `MetalEnv`): `get_arch`, `get_arch_name`, `get_num_pcie_devices`, `get_num_available_devices`, `get_l1_size`, `get_dram_alignment`, `get_l1_alignment`, `get_num_dataflow_buffers`, `get_max_worker_l1_unreserved_size`, `get_eps`, `get_nan`, `get_inf`, plus `get_descriptor` and `get_system_mesh`.

Facts:

- Most of them mirror the free functions in `tt::tt_metal::hal::` (`tt_metal/hal.cpp`) one-to-one. The free functions route through the default context, so they're wrong once several envs exist, and they have no future once the global context is gone. The env getters are their replacement.
- Most are architecture constants (L1 size, alignments, dataflow-buffer count, eps/NaN/inf), not per-env state.
- Only `get_num_pcie_devices()` and `get_num_available_devices()` come from `Cluster`; the rest come from `Hal`.
- `get_arch_name()` duplicates UMD's `tt::arch_to_str(get_arch())`.
- The uplift plan's PR9 adds `get_fabric_config()`, `get_fabric_topology()` and the fabric size getters, so the surface would grow by about five.

**Decision: a non-owning properties handle.** The ten hardware queries move to `MetalDeviceProperties`, a small copyable class that holds a `const MetalEnvImpl*` and is returned by value from `MetalEnv::get_device_properties() const`.

- Getters drop the `get_` prefix: `arch`, `arch_name`, `l1_size`, `dram_alignment`, `l1_alignment`, `num_dataflow_buffers`, `max_worker_l1_unreserved_size`. The SFPU constants become `sfpu_eps`, `sfpu_nan`, `sfpu_inf`, matching the Python `MeshDevice.sfpu_eps/nan/inf`.
- `get_num_pcie_devices()` and `get_num_available_devices()` stay on `MetalEnv`. They describe the env's cluster, not the device architecture, so they would be misleading on a handle that a `MeshDevice` could also return.
- The handle reads from the env on each call, so it never goes stale. It dangles only if the `MetalEnv` is destroyed first, the same contract as `get_system_mesh()`. It stores a pointer rather than a reference so it stays copy-assignable.
- Callers write `env.get_device_properties().l1_alignment()`. `MetalEnvImpl::get_hal()` and `get_cluster()` gained const overloads so the handle can hold a const pointer.
- The class is declared in `metal_env.hpp` for now; the final location is a review decision.
- Follow-up: `MeshDeviceImpl::metal_env()` already exists, so `MeshDevice::get_device_properties()` can return the same handle. That would give ttnn ops a public, remote-only-safe route to alignments, which is what blocks most of the tier 3 sites in [`Hal-Query-Callers.md`](Hal-Query-Callers.md).

**Alternatives considered:**

- **Value struct filled at env construction** (e.g. `const ArchProperties& get_arch_properties() const`, a coworker's preference). A copy can't dangle, and the struct can be printed, compared and bound to Python as read-only fields. It also fits the idea of passing an "arch facts" struct down the conv and pool helper chains. Parameterized queries such as `alignment(BufferType)` don't fit it. It can be layered on the handle later, e.g. `MetalDeviceProperties::snapshot()`. It also composes with §2.4 option 1: if `ClusterInfo` exists, the env can expose `get_cluster_info()` for the device counts.
- **Overload the `hal::` free functions with `const MetalEnv&`.** Most mechanical migration, but it keeps the `get_` names in a namespace that has no future once the global context is gone.
- **One method per property on `MetalEnv`** (the previous state). Every new property is a new method, and the class keeps growing with PR9.

Also considered and not chosen: trimming to only the getters with real callers, and making the `hal::` free functions take a `tt::ARCH`. The latter needs a check that none of the `Hal` constructor flags (2-erisc mode, profiler DRAM bank size, DRAM-backed CQ, simulator) move the addresses these values depend on, in particular `KERNEL_CONFIG` for `max_worker_l1_unreserved_size`.

---

## 4. `create_*` parameters

### 4.1 Where each parameter ends up

| Parameter | Where it ends up | Scope |
| --- | --- | --- |
| `l1_bank_remap` | Unused on every `MetalEnv` path: `ScopedDevices` hardcodes `{}` (`mesh_device.cpp:165-174`), `initialize_impl` ignores its copy (`:1650`), `create_unit_meshes` marks it unused (`:577`). | Dead |
| `num_command_queues`, `dispatch_core_config`, `worker_l1_size` | Stored once on `MetalContext`. A different value on a later call logs "Closing and re-initializing MetalContext with new parameters" and calls `teardown()` (`metal_context.cpp:180-186`). I found no guard against devices opened by an earlier call. | Per env |
| `l1_small_size`, `trace_region_size` | Passed through `ContextDescriptor` and `DeviceManager` into each device's allocator. | Per device |

Removing `l1_bank_remap` loses nothing: the sub-device manager builds its own remap separately (`sub_device_manager.cpp:288`).

Other problems with the current signatures:

- Four positional `size_t` parameters can be swapped without a compile error.
- `num_command_queues` is `size_t` in the API and `uint8_t` internally; `device_id` is `int` rather than `ChipId`; `create_unit_meshes` takes `const std::vector<int>&`.
- Naming is inconsistent: `create_unit_mesh_device` versus `create_unit_meshes` (and `MeshDevice::create_unit_mesh`).
- The header includes the heavy `mesh_device.hpp` only for these signatures and their default constants.

### 4.2 Decision: one options struct now, split later if needed

**First step.** All five settings go into one struct whose defaults are exactly today's. Behavior is unchanged, and `l1_bank_remap` is deleted.

```cpp
struct CreateMeshDeviceOptions {
    size_t l1_small_size = DEFAULT_L1_SMALL_SIZE;
    size_t trace_region_size = DEFAULT_TRACE_REGION_SIZE;
    // The fields below are env-wide today (§4.1): all create_* calls on one env must agree on them.
    uint8_t num_command_queues = 1;
    DispatchCoreConfig dispatch_core_config = {};
    size_t worker_l1_size = DEFAULT_WORKER_L1_SIZE;
};

std::shared_ptr<distributed::MeshDevice> create_mesh_device(
    const distributed::MeshDeviceConfig& config, const CreateMeshDeviceOptions& options = {});
std::shared_ptr<distributed::MeshDevice> create_unit_mesh(ChipId device_id, const CreateMeshDeviceOptions& options = {});
std::map<ChipId, std::shared_ptr<distributed::MeshDevice>> create_unit_meshes(
    ttsl::Span<const ChipId> device_ids, const CreateMeshDeviceOptions& options = {});

auto mesh = env.create_mesh_device(config, {.trace_region_size = 64 << 20, .num_command_queues = 2});
```

Notes:

- **"Sensible defaults" means today's defaults.** In particular, `dispatch_core_config` stays `DispatchCoreConfig{}`: WORKER, with the axis resolved by `MetalContext::initialize`. The C++ versus ttnn divergence on N300/T3K (§4.3) is unchanged by this step.
- **The signature fixes land in the same PR**, since every `create_*` signature changes anyway: `create_unit_mesh_device` becomes `create_unit_mesh`, `int` becomes `ChipId`, and `std::vector<int>` becomes `Span`.
- **Keep `CreateMeshDeviceOptions` a plain aggregate**: no constructors, no methods. That keeps designated initializers working and leaves fields free to be moved or retyped later.
- **Recommended small guard.** A later `create_*` on the same env whose env-wide fields differ from the ones in use while devices are open should `TT_FATAL`, instead of silently tearing the context down (`metal_context.cpp:180-186`). Scope it to the `MetalEnv` path so the legacy `MeshDevice::create` behavior is unchanged. It turns today's silent hazard into a clear error, and it's the same rule the later split would enforce.
- With the struct, the header can forward-declare `MeshDevice` and `MeshDeviceConfig` instead of including `mesh_device.hpp`. It still needs the small `DispatchCoreConfig` and default-constant headers.

**Why a later split stays viable.** The `create_*` signatures never change again. Only fields move:

1. Add the env-wide settings to `MetalEnvDescriptor` (the end state sketched below).
2. Change the three env-wide fields in `CreateMeshDeviceOptions` to `std::optional<...> = std::nullopt`. `nullopt` means "use the env's value"; a value that is set must equal the env's (`TT_FATAL` otherwise, the same guard as above). Designated initializers and assignments of plain values keep compiling; only code that reads these fields back needs changing.
3. Deprecate those fields, then remove them.

**Deferred end state (split by scope)**, for reference:

```cpp
struct DispatchDescriptor {             // in MetalEnvDescriptor
    uint8_t num_command_queues = 1;
    std::optional<DispatchCoreType> dispatch_core_type = std::nullopt;  // nullopt: resolved for the cluster (§4.5)
    std::optional<DispatchCoreAxis> dispatch_core_axis = std::nullopt;  // nullopt: resolved for the cluster (§4.5)
    std::optional<size_t> worker_l1_size = std::nullopt;                // nullopt: arch default (§4.4)
};

struct CreateMeshDeviceOptions {
    size_t l1_small_size = DEFAULT_L1_SMALL_SIZE;
    size_t trace_region_size = DEFAULT_TRACE_REGION_SIZE;
};
```

### 4.3 Deferred: choosing dispatch config from the cluster

Same problem as §2.4, and the same options apply. Notes specific to dispatch:

- Today there are two cluster-aware default rules, and they disagree:
  - tt_metal (`resolve_dispatch_core_axis`, `dispatch_core_common.cpp:19-29`) fills only the axis: COL on Blackhole when tensix config is DISABLED, otherwise ROW. The type stays whatever `DispatchCoreConfig`'s default constructor set: WORKER.
  - ttnn (`create_cluster_aware_dispatch_config`, `ttnn/core/device.cpp:17-65`) fills both. The type is ETH on N300, N300_2x2 and T3K, and WORKER elsewhere. The axis is COL on Blackhole unless tensix config is MUX, so for UDM it picks COL where tt_metal picks ROW. It also rejects ETH+COL, and ROW on Blackhole without MUX.
  - So a C++ caller that passes `DispatchCoreConfig{}` and a Python caller that passes `None` get different dispatch config on N300/T3K, and on Blackhole with UDM.
- The ttnn rule reads cluster type and arch through `GetClusterType()` and `hal::get_arch()`, which go through the default context. That is the chicken-and-egg in concrete form.
- `tests/scripts/common.py:281-296` adds a third layer on top of ttnn's.
- `num_command_queues` moving to the env is not a behavior loss: opening meshes with different CQ counts on one env already tears the context down today.

### 4.4 Deferred: if `DispatchCoreConfig` becomes single-phase

Assumption: `DispatchCoreConfig` loses its optional axis, so every value is concrete and the type can't express "default". Today the axis is its only two-phase field (`dispatch_core_common.hpp:20-21`, with `get_dispatch_core_axis()` fataling when unset); the type is always set, WORKER by default.

Implications:

1. **"Default" moves from the type to the descriptor.** `DispatchDescriptor` holds `std::optional<DispatchCoreType>` and `std::optional<DispatchCoreAxis>` as separate fields, where `nullopt` means "the library picks for this cluster". The env resolves them during construction, when it knows the cluster and the fabric config, and exposes the result as a concrete `get_dispatch_core_config()`. This is §2.4 option 2 with the request/resolved split done by the descriptor rather than by a second value type. That largely answers the asymmetry concern: the value type returned by the env is the same type users pass to lower-level APIs, and only the descriptor fields are optional.
2. **The default constructor has to go.** No single concrete default is valid everywhere: the plain Blackhole core descriptor has no `row` section at all (§4.5). So `DispatchCoreConfig{}` and `ttnn.DispatchCoreConfig()` stop compiling. Roughly 30 files use them, including the default arguments of `MeshDevice::create*`, `CreateDevice` and the current `MetalEnv::create_*`. Those default arguments become `std::optional<DispatchCoreConfig> = std::nullopt`, which keeps source compatibility for callers that pass a real config. The ttnn and Python entry points already take `optional` (`ttnn-nanobind/device.cpp:254, 287`).
3. **Partial specification survives in the descriptor.** Because the descriptor's fields are optional one by one, "ETH, default axis" (about 20 files set only the type today) is `.dispatch = {.dispatch_core_type = DispatchCoreType::ETH}`, resolved inside the env. Neither a public resolver nor a second, two-phase request type is needed for this. Legacy entry points that take a `DispatchCoreConfig` (for example `MeshDevice::create`) can only accept a complete one or `nullopt`; callers that set only the type there move to the env API or pass both fields.
4. **Discovery before the env is not needed for dispatch**, provided tt_metal owns a sensible default (§4.5). The env resolves with the cluster it already has. What remains is genuinely custom policy, for example "use ETH on T3K only for this model". For that, the fallback is a throwaway `MetalEnv` used for the query and destroyed before creating the real one. It's acceptable for rare cases, with two costs: a full silicon cluster bring-up (open, topology discovery) runs twice, and the throwaway env must be destroyed first, since only one silicon env may exist. Pre-env discovery (§2.4 option 1) becomes an optimization for that path rather than a prerequisite.
5. **The default rules must be merged into one, owned by tt_metal.** See §4.5 for the history and a proposed rule. ttnn's `create_cluster_aware_dispatch_config` and the Python validation then go away. ttnn passes the user's optionals through unchanged.
6. **Resolution order is fabric first, then dispatch.** The axis default depends on `FabricConfigDescriptor::fabric_tensix_config`. Env construction handles that naturally.
7. **`worker_l1_size` has the same pattern hidden in a `size_t`.** `DEFAULT_WORKER_L1_SIZE` is `0`, which `validate_worker_l1_size` replaces with the arch's default unreserved size (`common_values.hpp:17`, `metal_context.cpp:93-96`). To be consistent it should become `std::optional<size_t>` in `DispatchDescriptor`, with the resolved value available from the env. `l1_small_size` and `trace_region_size` default to a real `0`, not a sentinel, so they're unaffected.
8. **A concrete config still isn't the final placement.** On Quasar, the core type actually used comes from the per-device SoC descriptor and `TT_METAL_TENSIX_DISPATCH_CORES`, and the configured type is ignored (`dispatch_engine_cores.cpp:185-200`). The result can be `CoreType::DISPATCH`, which `DispatchCoreType` can't express. So the per-device step `resolve_dispatch_core_type(env, device_id, config)` stays. It derives placement from config and isn't a "default", but the plan shouldn't promise that the resolved config describes placement on Quasar. Separately, it's worth deciding whether a Quasar config that asks for ETH should be rejected rather than ignored.
9. **Things that get simpler.** `MetalContext::initialize` no longer has to resolve before comparing (`metal_context.cpp:172-178`). Equality and hashing of `DispatchCoreConfig` become meaningful, including as part of `CoreDescriptorCacheKey`. The fatal in `get_dispatch_core_axis()` goes away.

Effect on the plan (when revisited): the §4.2 split carries items 1, 2 (for the env API) and 7, and needs the rule merge (item 5, §4.5) first. Nothing in the dispatch path depends on public discovery. The first step's `CreateMeshDeviceOptions::dispatch_core_config` keeps the two-phase `DispatchCoreConfig` and becomes `optional` during the split (§4.2, step 2).

### 4.5 Deferred: history of the "cluster-aware" resolver, and a proposed tt_metal default

**Hypothesis:** ttnn resolves dispatch config itself because tt_metal's resolver was broken. **Finding:** partly true. The axis resolver worked, but it was implicit and global, and every copy of it drifted. The *type* resolver was never broken in tt_metal, because tt_metal never had one: C++ has always defaulted to WORKER. ttnn added a policy that tt_metal lacked, then kept it on the ttnn side.

| Commit | PR | Date | Change |
| --- | --- | --- | --- |
| `509b2159ac3` | (issue #9514) | 2024-07 | `WH_ARCH_YAML` removed from the runtime. ETH dispatch on N300 becomes an explicit `dispatch_core_type` argument. Test conftests keep reading `WH_ARCH_YAML=..._eth_dispatch.yaml` as a flag (#11059). |
| `9b09061033c` | #15316 | 2024-11 | Adds the dispatch axis (row/col) for Llama decode on Galaxy. The axis is `optional` in `DispatchCoreConfig`, and `get_dispatch_core_axis()` fills it lazily from `MetalContext::instance()`'s arch (Blackhole: COL). This is the origin of the two-phase type. |
| `fc1fafe5308` | #26061 | 2025-08 | Removes `WH_ARCH_YAML` from models. Adds a Python `ttnn.DispatchCoreConfig` subclass that chooses the **type** by cluster type (ETH on N300, T3K, N300_2x2) and re-implements the axis default (Blackhole: COL) and validation in Python. The C++ default stays WORKER, so the C++ path and the Python path now disagree on N300/T3K. |
| `eec01a46fac` | #28138 | 2025-09 | Makes tensix MUX the default on Blackhole in Python and moves dispatch to ROW when MUX is on. The condition is written twice in the same PR: C++ "COL iff tensix == DISABLED", Python "ROW iff tensix == MUX". They are equivalent because the enum only has these two values. |
| `70f52f3dc51` | #32008 | 2025-11 | Reverts the MUX default on Blackhole: ROW dispatch lost 2-4 compute cores and some CCLs no longer fit. Adds a third layer in `tests/scripts/common.py` that forces COL unless both fabric and tensix config are set, overriding an explicit user ROW. |
| `6890f1d0e9f` | #32179 | 2025-11 | Adds `FabricTensixConfig::UDM`. The two copies from #28138 silently diverge: tt_metal picks ROW for UDM, ttnn picks COL and rejects ROW. |
| `73e676e1acc` | #44670 | 2026-07 | Fixes #42615: on Wormhole `ttnn.open_device` (Python default: ETH) and `ttnn.CreateDevice` (C++ default: WORKER) returned different grids (8x7 vs 8x8). The fix moves ttnn's Python resolver into ttnn C++ (`create_cluster_aware_dispatch_config`). It still reads global state (`GetClusterType()`, `hal::get_arch()`) and stays in ttnn. The C++ `MeshDevice::create` default is still WORKER. |
| `ab3eb1c2e21` | #55003 | 2026-09 | Removes the lazy global lookup from `get_dispatch_core_axis()`. It had been guarded by `instance_exists()` since the MetalEnv work, which traded a hang for a silently wrong grid on Blackhole and mock contexts. The axis is now resolved by `MetalContext::initialize` or explicitly; reading an unset axis is an error. |

So the ttnn resolver exists for two reasons:

- **The type:** ttnn wants the largest compute grid by default ("preferably the full grid", #42615), and tt_metal never provided a cluster-aware type default.
- **The axis:** Python needed a concrete value to validate before tt_metal saw it, and the tt_metal default lived behind a global lookup.

Neither reason needs a query before the env once tt_metal resolves both fields at env construction.

**Why tt_metal defaults to WORKER.** Mostly legacy, but with an undocumented justification that still holds.

- *Legacy.* WORKER dispatch came first. ETH dispatch was added in `bc4508f72a4` (#2371, 2024-03, "run fast dispatch on unused ETH cores on N300") as an alternate core descriptor, selected by pointing the `WH_ARCH_YAML` env var at it. #9514 turned the env var into a `DispatchCoreType dispatch_core_type = DispatchCoreType::WORKER` parameter, preserving "unset means the plain descriptor", plus a `TT_METAL_GTEST_ETH_DISPATCH` override for tests. #15316 wrapped it into `DispatchCoreConfig` unchanged. No commit or issue argues for cluster-unaware WORKER; it was carried forward each time.
- *The justification that holds: WORKER is the only type valid everywhere.* ETH dispatch runs only on **idle** ethernet cores. The listed ETH dispatch cores that are active links are dropped (`core_descriptor.cpp:291-295`); multi-CQ tops up from the control plane's inactive-eth list (`dispatch_core_manager.cpp:269-276`); and an empty result is fatal (`core_descriptor.cpp:303`). So whether ETH works depends on the product (P100 has no ethernet), on cabling (a P150 in a LoudBox/QuietBox, an all-links-connected Galaxy), and on topology. Quasar ignores the type entirely. WORKER has a descriptor section for every product.
- *Features that need WORKER.* The real-time profiler needs a spare tensix core that only WORKER dispatch leaves free, and disables itself under ETH (`realtime_profiler_manager.cpp:194-199`). `FabricSwitchManager` hard-codes WORKER for switch chips (`fabric_switch_manager.cpp:47`).

Implication for the proposed default: an ETH default is safe only where idle ethernet cores are guaranteed, which is why it has to be a per-product table (N300/T3K boards have idle links by construction) and not a rule derived from the descriptor. WORKER stays the fallback. The table still needs a check against idle-core counts for cabled configurations, and should document the RT-profiler trade-off.

**The core descriptor YAMLs are the ground truth, and every hand-written rule approximates them.** Sections present, with the compute grid each gives:

| Descriptor | Product | `row` | `col` |
| --- | --- | --- | --- |
| `wormhole_b0_80_arch.yaml` (WORKER) | N150 (`nebula_x1`) | 8x8 | 7x9 |
| | N300 / T3K (`nebula_x2`) | 8x7 | 7x8 |
| | WH Galaxy | 8x9 | 7x10 |
| `wormhole_b0_80_arch_eth_dispatch.yaml` (ETH) | N150, N300 / T3K | 8x8 | absent |
| | WH Galaxy | 8x10 | absent |
| `blackhole_140_arch.yaml` (WORKER) | unharvested | absent | 13x10 |
| `blackhole_140_arch_eth_dispatch.yaml` (ETH) | unharvested | absent | 14x10 |
| `blackhole_140_arch_fabric_mux.yaml` (tensix MUX or UDM) | unharvested | 14x9 | 13x10 |

The file is chosen in `core_descriptor.cpp:94-109`: ETH first, then any non-DISABLED tensix config selects the `fabric_mux` file, otherwise plain. The section by axis is picked at `:188`.

What this shows:

- **Blackhole COL without tensix is not a preference; it's the only option** (no `row` section). tt_metal's rule matches the data.
- **For UDM, both axes exist**, so both resolvers produce a valid default. ttnn's validation ("ROW is not supported on Blackhole unless MUX") is wrong for UDM.
- **ETH on Blackhole exists only as `col`.** ttnn's validation "COL is not supported for ETH" is wrong on Blackhole, where ETH+COL is the only valid ETH combination. Both resolvers pick ROW for ETH on Blackhole with tensix enabled, which has no section (inferred; not run).
- **ETH gives the full grid on N300/T3K.** That is ttnn's allow-list. N150 ties at 8x8, so it stays WORKER. But ETH would also give a larger grid on WH Galaxy (8x10 vs 8x9) and on Blackhole (14x10 vs 13x10), and ttnn does not choose it there. So the allow-list is not simply "largest grid". It encodes a judgement, most likely about ethernet-core availability (fabric uses the active links; P100 has no ethernet) or ETH-dispatch maturity on Blackhole. That needs confirming with the dispatch owners before it's turned into a rule.

**Proposed tt_metal default**, resolved at env construction per product, after fabric:

1. If the type is unset: ETH on `nebula_x2` products (N300, N300_2x2, T3K), WORKER elsewhere. This is ttnn's allow-list moved into tt_metal as an explicit per-product table. A derived "largest grid" rule would silently switch WH Galaxy and Blackhole to ETH.
2. If the axis is unset: the preferred axis among the sections present in the chosen descriptor. Only one exists for Blackhole plain/ETH and for Wormhole ETH. Where both exist, keep today's tt_metal choice (ROW) unless the UDM owners prefer COL.
3. Validate an explicit (type, axis) by checking that the descriptor has the section, instead of hand-written rules. This removes the Blackhole ETH+COL false rejection and the UDM ROW false rejection by construction.

Behavior changes: C++ callers on N300/T3K get ETH by default, which matches Python today and resolves #42615 for the C++ path too. Python users on Blackhole with UDM get ROW instead of COL, unless item 2 picks COL. `tests/scripts/common.py`'s override then becomes redundant; with tensix DISABLED it forces what the default already gives.

---

## 5. `get_system_mesh()`: accepted for now

Recorded for completeness. Both issues disappear in the end state, where fabric config is fixed at construction and nothing resets the control plane or system mesh at run time (see the uplift plan's "Runtime fabric reconfiguration: Remove").

- **The returned reference can dangle.** On N300/T3K under fast dispatch with fabric DISABLED, `DeviceManager::open_devices` switches fabric to `FABRIC_1D` (`device_manager.cpp:276-291`). `set_fabric_config` then runs `system_mesh_.reset()` (`metal_env.cpp:339`) because the control plane already exists. A `SystemMesh&` taken before `create_mesh_device` points at freed memory. The `ReconfigureFabricForDispatch` test reaches this path but re-fetches the reference.
- **It changes context ownership** (§1). Goes away with env-owns-context.

One small change is still worth making in the end state: return `const SystemMesh&`. `SystemMesh` only has const methods.

---

## 6. Other issues

Only issues that stay relevant in the end state are listed.

- **`get_descriptor()` returns the requested settings, not the effective ones.** Today `DeviceManager` can promote fabric from DISABLED to `FABRIC_1D` and adjust the routing-plane count, so `get_descriptor().fabric_config_descriptor()` can disagree with what's running. In the end state, runtime promotion is gone (uplift plan option B), but if "auto" values are adopted (§2.4 option 2) the two diverge by design. Either way: document `get_descriptor()` as "what was requested" and add resolved getters (the uplift plan's PR9 `get_fabric_config()` is one).
- **`create_sub_device` is a workaround, and deleting it needs the underlying fix.**
  - *Why it exists.* #39655 (`d02dc4a32e0`) found that `SubDeviceImpl::validate()` called `MetalContext::instance().hal()`, which implicitly opens the silicon context in mock-only flows. It threaded a `MetalEnvImpl*` into `SubDeviceImpl` and added `MetalEnv::create_sub_device` as the env-bound way to build one. The public `SubDevice(cores)` constructor passes `nullptr` and falls back to `MetalContext::instance(DEFAULT_CONTEXT_ID)` (`sub_device.cpp:24-29, 80-81`).
  - *The public constructor still has the bug.* It's what everyone uses: about 100 files, mostly `ttnn.SubDevice([...])`, while `create_sub_device` has zero callers. With no context yet it implicitly creates the silicon default, the #50041 hazard for an explicit silicon env. With mock and silicon envs coexisting, it validates against whichever context the fallback returns, possibly a different arch.
  - *The env is the wrong owner for this check.* `env_` is read only by `validate()` (`sub_device.cpp:50-62`), and only for `get_programmable_core_type_count()`. The other check, rejecting `IDLE_ETH`, needs nothing. Every `SubDevice` is eventually applied to a device through `SubDeviceManager`, whose `validate_sub_devices()` already checks grid bounds, active ethernet cores and overlap against the real device (`sub_device_manager.cpp:202-240`). The core-type-count check belongs there too, since the device knows its HAL. Anything richer a `SubDevice` might validate in future (grid, harvesting, coordinates) is also a device property, not an env property.
  - *Fix.* Keep the env-free `IDLE_ETH` check in the constructor. Move the core-type-count check into `SubDeviceManager::validate_sub_devices()`. Drop `SubDeviceImpl::env_`, `get_env_or_default` and the env parameters (the internal full-grid sub-device in `MeshDeviceImpl::initialize_impl`, `mesh_device.cpp:1672`, just stops passing it). `MetalEnv::create_sub_device` then has no reason to exist, now or later. The only behavior change: an out-of-range core type fails at `create_sub_device_manager` instead of at `SubDevice` construction. No test expects a throw at construction.
- **`MetalEnv::descriptor_` is dead and was never live** (`metal_env.hpp:159`). It has been in the header since `MetalEnv` was introduced (#38815), but no `MetalEnv` member has ever read or assigned it: `get_descriptor()` has forwarded to `impl_->get_descriptor()` since that first commit, and `MetalEnvAccessor` only exposes `impl()`. It looks like a leftover from before the pimpl split. There's no future need either: the descriptor lives in `MetalEnvImpl` for the object's whole life, and if the impl constructor throws, no `MetalEnv` exists to query. Keeping it is actively misleading, because it's default-constructed, i.e. it describes a silicon env even for a mock one.
- **Header hygiene.** The header uses `std::map`, `std::vector` and `std::string` without including them, relying on transitive includes. Typo: double space at line 34 ("at a time  due").
- **Internal: `ContextDescriptor` inherits from the public descriptor** (§2.1). The `is_mock_device()` meaning depends on which constructor built it. Resolved by §2.3 A or B; until then, worth making Emule handling consistent between `init_context_descriptor` and `init_risc_fw_context_descriptor`.

---

## Suggested PR order

### First step

| PR | Scope | API change |
| --- | --- | --- |
| 1 | Delete the dead `MetalEnv::descriptor_`. Move `SubDevice`'s core-type-count check into `SubDeviceManager::validate_sub_devices()`, drop `SubDeviceImpl::env_` and the default-context fallback, then delete `MetalEnv::create_sub_device` (§6). Header include fixes. | Removal (no callers); fixes the implicit silicon context from `SubDevice(cores)` |
| 2 | `MetalEnvDescriptor` as an aggregate with `MetalEnvTarget` and `fabric` (§2.3 B). `ContextDescriptor` stops inheriting. Migrate tests. | Breaking, experimental only |
| 3 | `CreateMeshDeviceOptions` with all five settings and today's defaults; `l1_bank_remap` deleted; `create_unit_mesh` rename; `ChipId` and `Span` parameter types; the mismatch guard (§4.2). | Breaking, experimental only; no behavior change except the guard |

The three are logically independent. All of them edit `metal_env.hpp` and the same tests, so land them in sequence to avoid conflicts.

### Deferred

| Item | Scope | Depends on |
| --- | --- | --- |
| Split by scope | Env-wide settings move into `MetalEnvDescriptor`; `CreateMeshDeviceOptions` fields become `optional` overrides, then are removed (§4.2). | PR2, PR3, and the dispatch default below |
| tt_metal owns the dispatch default | Per-product type table (ETH on `nebula_x2`), axis from the core descriptor's sections, validation by section lookup; then delete ttnn's resolver and the `tests/scripts/common.py` override (§4.5). Confirm the WH Galaxy and Blackhole ETH exclusions with the dispatch owners. It fixes the C++ versus Python divergence independently of `MetalEnv`, so it can go ahead whenever it's wanted. | — |
| `DispatchCoreConfig` single-phase | Drop the optional axis and the default constructor (§4.4). | The dispatch default above |
| Query before open | Cluster discovery before env, or a throwaway `MetalEnv` (§2.4). | — |
| `MeshDevice::get_device_properties()` | Return the same `MetalDeviceProperties` handle from `MeshDevice`, then migrate ttnn ops off the `hal::` free functions (§3). | The `MetalDeviceProperties` change |
| Fabric getters shape | Decide with the uplift plan's PR9: extend `MetalDeviceProperties`, add a separate fabric properties handle, or keep them on `MetalEnv` (§3). | — |
