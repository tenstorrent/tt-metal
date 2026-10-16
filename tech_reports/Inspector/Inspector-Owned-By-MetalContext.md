# Inspector owned by MetalContext: re-evaluation of the owner's proposal

Context: PR #59756 (Stage 1 of the inspector MetalEnv uplift) introduced a `SessionRegistry` in
`inspector.cpp`. The inspector owner prefers no registry: hooks should get the inspector from the
`MetalContext` in use, and only a hardware `MetalContext` exposes an RPC endpoint. The owner
offered two ways to reach it: pass the context to the static `Inspector::` functions, or get an
inspector instance from the context. This note re-checks that idea against the code and sketches
a plan for implementing it from scratch on a new branch.

Assumptions:

- `CreateProgram()` will migrate to an env-aware version, so a `Program` always carries the
  correct context (`ProgramImpl::get_context_id()`).
- Context 0 (`DEFAULT_CONTEXT_ID`) is not necessarily the hardware context. Nothing here may treat
  it as "the hardware session".

## Verdict

The proposal is feasible, and it fits the code better than the registry did.

- `MetalContext` already owns the session (`inspector_data_`, created in `initialize()`, reset in
  `teardown()`, only for non-mock clusters). What is global today is only the **lookup** from a
  hook to the session. The proposal replaces that lookup and changes nothing about ownership.
- Mock contexts never create a session. Once a hook asks "the context of this object" for its
  inspector, mock events are dropped for free. That was the whole point of Stage 2 of the
  original plan.
- No process-wide session state is left, so a second hardware context is structurally fine. Each
  context has its own session; only the RPC port and log directory would collide (see D5).

The cost is that every hook needs a context at once: there is no global fallback to migrate hooks
gradually. Stage 2 and most of Stage 3 of the original plan collapse into one change. The
remaining tricky parts are `MeshWorkload` (no context of its own) and destroy events from late
destructors. The `mesh_workload_*` lifecycle, `add_program` and `set_program_binary_status` events and the unused
`getMeshWorkloads` RPC are removed first, which takes `MeshWorkload` out of the destructor problem
and leaves `emit_debug_entry` as its only Inspector hook.

## Comparing the owner's two variants

**Variant 1: static functions take the context.**

```cpp
Inspector::program_compile_started(ctx, this, device, build_key);
Inspector::mesh_device_created(this, parent_id);  // context derived inside from the object
```

**Variant 2: get the inspector from the context.**

```cpp
if (auto* insp = ctx.inspector()) {
    insp->program_compile_started(this, device, build_key);
}
if (auto* insp = metal_context().inspector()) {
    insp->mesh_device_created(this, parent_id);
}
```

There are two forms of Variant 2. In 2a, `ctx.inspector()` returns `inspector::Data*` directly.
In 2b, `Inspector` becomes a per-context instance class in a light header, owned by
`MetalContext`, with `Data` as its implementation.

| Aspect | Variant 1 (static + context) | Variant 2a (`Data*`) | Variant 2b (instance `Inspector`) |
| --- | --- | --- | --- |
| Call-site churn | Only hooks without an object-derived context change (about 15 sites) | All of about 40 sites change | All of about 40 sites change |
| "No session" handling (mock, disabled, failed init) | Once, inside the facade | `if (auto* i = ...)` at every site | Same, or return a no-op instance |
| `noexcept` and `try`/`catch` around logging | Centralized, as today | Moves into `Data` methods | Inside the instance methods, as today |
| Include weight at call sites | `inspector.hpp` only (forward declarations) | `data.hpp`, which pulls in `capnp/ez-rpc.h` and the generated RPC header | Light header if `Data` stays out of it |
| Destroy hooks in late destructors | Lookup that never creates a context lives once, in the facade | Every destructor site does the lookup | Same as 2a |
| Context derived from the object (mesh device, socket, buffer) | Derived inside, so call sites don't change | Call site derives it | Call site derives it |
| Several hardware sessions | Fine: the context decides the session | Fine | Fine |
| How explicit the dependency looks | The context appears as an argument where it isn't derivable | Most explicit: `ctx.inspector()` at every site | Same as 2a |
| Testability | Through the context | Through the context | Easiest to substitute an instance |

Discussion:

- Both variants put the session behind the context, so they are architecturally equivalent. They
  differ in where the "find the session, bail if absent, catch everything" boilerplate lives.
  Variant 1 keeps it once per hook. Variant 2 repeats the lookup and null check at every call site.
- Variant 2a is not practical: about 40 call sites in core runtime files would include the
  Cap'n Proto headers.
- Variant 2b is the clean "instance" form. `MetalContext` owns an `Inspector` object (light
  header, `Data` as its implementation), and call sites write `ctx.inspector().hook(...)`. A no-op
  instance for mock or disabled contexts would remove the null checks. It costs touching every
  call site plus a class reshape. It also needs a story for destroy hooks that run when the
  context may already be gone.
- Recommendation: **Variant 1** for this change. It's the smallest diff that removes all global
  lookup, keeps error handling in one place, and leaves object-derived hooks untouched. Variant 2b
  can follow mechanically if the owner prefers instance syntax: the facade's per-hook bodies
  become member functions, and only the lookup moves. Worth asking the owner which syntax they
  meant.

## How each hook gets its context (Variant 1)

| Hook | Where the context comes from | Call-site change |
| --- | --- | --- |
| `mesh_device_created/initialized/destroyed` (4) | `MeshDeviceImpl::metal_context()` | None |
| `mesh_socket_created` (3) | `socket->get_mesh_device()` | None |
| `global_semaphore_created/reset` (2) | `buffer->device()` | None |
| `mesh_buffer_allocated` (4) | `mesh_buffer->device()` | None |
| `mesh_buffer_deallocated` (3) | `mesh_device_.lock()`; can be expired (D2) | Pass the locked device or null |
| `program_created/destroyed` (2) | `program->get_context_id()`, with a lookup that never creates a context (D2) | None |
| `program_compile_*`, `program_kernel_compile_finished` (6) | `program.cpp::compile()` already resolves the device's `MetalContext` | Add a `MetalContext&` argument (or derive from the program) |
| `set_dispatch/dispatch_s/prefetcher_core_info` (3) | `descriptor_.metal_context()` | Add a `MetalContext&` argument |
| `enable_kernel_path_collection`, `get_kernel_elf_path` (DPRINT, 2) | `DPrintServer::context_` | Add a `MetalContext&` argument |
| `emit_debug_entry`, `release_trace` (via experimental API) | Derived from the workload's programs; `MeshDeviceImpl` for release | See D3 |
| `initialize`, `serialize_rpc` (2) | Take `MetalContext&` | None |

## Difficulties

### D1. MeshWorkload has no context

With `Program` carrying the correct context, only `MeshWorkload` is left without one.
`MeshWorkloadImpl()` takes no arguments, and changing that is a non-experimental API change.

- The `mesh_workload_created/destroyed/add_program/set_program_binary_status` hooks are removed, so
  `MeshWorkloadImpl` only calls Inspector through `emit_debug_entry`.
- `emit_debug_entry` takes the context from any program in the workload
  (`program.impl().get_context_id()`). Nothing is stored in `MeshWorkloadImpl`.
- A workload with no programs has no context to derive. Decide explicitly how its entries are handled
  instead of dropping them silently.
- If programs from different contexts are mixed in one workload, it's a usage error. `TT_ASSERT` it.
- `EmitMeshWorkloadDebugEntry(workload, ...)` keeps its signature.
- Binding `MeshWorkload` at construction stays with #38445.

### D2. Destroy events must neither be dropped nor create contexts

- `rpc_get_mesh_devices` dereferences the stored `MeshDeviceImpl*`, so a dropped
  `mesh_device_destroyed` would leave a dangling pointer for the RPC thread. That hook is safe,
  because `~MeshDevice` runs it before `close_impl()` while the context is alive.
- Destructors of `Program` can run after their context has been torn down or
  destroyed (for example, Python GC at exit). Looking the context up by id there must not create
  one, but `MetalContext::instance(id)` may implicitly create the default context or fall back to
  another one. This needs a small lookup that never creates a context, such as
  `MetalContext* MetalContext::find_instance(ContextId)` (no creation, no fallback). An erase that
  finds nothing is harmless, because program ids are process-unique.
- `mesh_buffer_deallocated` can fire after the buffer's `MeshDevice` is gone, where
  `MeshBuffer::device()` `TT_FATAL`s. Entries are keyed by pointer and never dereferenced, so a
  dropped event leaves a stale entry, not a crash. Fix: keep the mesh id with each buffer, socket
  and semaphore entry, and purge them on `mesh_device_destroyed`. This change is internal to the
  inspector.
- A session is recreated on every `MetalContext::initialize()`. Objects from an earlier cycle
  arrive at a fresh session, so destroy and erase hooks must look up and never insert. Today
  several use `operator[]`.

### D3. Experimental API and ttnn

- `IsEnabled()` and `ShouldCaptureTensorSpecs()` have no context. ttnn's `enqueue_mesh_workload`
  (in `ttnn/api/ttnn/device_operation.hpp`) is the only caller in the repo, and it has the
  `mesh_device` at hand. Add `IsEnabled(const MeshDevice&)` and
  `ShouldCaptureTensorSpecs(const MeshDevice&)`, and switch ttnn. That is a three-line change in a
  widely included header, so it costs rebuild time, not risk.
- Then deprecate the no-arg forms, or remove them (they are experimental). If they must stay for a
  while, `IsEnabled()` can be a process-wide count of live sessions, which is not a lookup.
  `ShouldCaptureTensorSpecs()` can't be answered without an env.
- `ReleaseTraceDebugEntries(trace_id)` is called only from `MeshDeviceImpl::release_mesh_trace`.
  Call the internal `Inspector::release_trace(ctx, trace_id)` there, and keep the experimental
  function as a thin wrapper or deprecate it.
- `Inspector::get_rpc_server()` has no callers outside the inspector; it appears only in
  `docs/source/tt-metalium/tools/inspector.rst`. Remove the static, and update the docs to use the
  session's `get_rpc_server()`.

### D4. Hook vs. teardown race

Hooks on other threads read `inspector_data_` while `teardown()` resets it. This race exists today
(Copilot flagged it on PR #59756), and the watcher and DPRINT servers have the same pattern. If
wanted, the private accessor can read through a `std::atomic<inspector::Data*>` mirror in
`MetalContext` that `teardown()` clears first. A full fix (a `shared_ptr` session handle) stays a
separate follow-up.

### D5. A second hardware session

There is no guard. With per-context sessions nothing is shared in code, so a second hardware
context gets its own correct session. The only collisions are the configured endpoints. Both
sessions read the same `TT_METAL_INSPECTOR_LOG_PATH` and RPC port, because those come from
environment variables. Consequences:

- The second `Logger` wipes the shared log directory, and both then write to it.
- The second RPC server fails to bind. `TT_INSPECTOR_THROW` then either fails context init or
  warns and continues without RPC, depending on `initialization_is_important`.

When multiple hardware contexts become real, the fix is local: a per-session suffix for the port
and the log directory, plus tt-triage discovering several endpoints. The process-wide error-policy
atomics would just be stored twice with the same values.

### D6. Unaffected

- The `Data`/`Logger` explicit-context work from PR #59756 (`context_`/`env_` members, settings
  read through `env_.get_rtoptions()`, `Logger` taking rtoptions) carries over as it is.
- The error-policy atomics behind `TT_INSPECTOR_LOG`/`TT_INSPECTOR_THROW` are process-wide policy,
  not a session lookup. They can stay, or move into `Data` later, since every hook has `data` in
  hand.

## Target shape (Variant 1 sketch)

```cpp
// metal_context.hpp (internal)
class MetalContext {
public:
    static MetalContext* find_instance(ContextId id);  // no implicit creation, no fallback
private:
    friend class Inspector;
    inspector::Data* inspector_data() const { return inspector_data_.get(); }  // null: mock, disabled, failed
    std::unique_ptr<inspector::Data> inspector_data_;
};

// inspector.hpp (internal)
class Inspector {
public:
    static std::unique_ptr<inspector::Data> initialize(MetalContext&, std::optional<int> rank, uint64_t fw_hash);
    static bool is_enabled(const MetalContext&);
    // Context derived from the object:
    static void mesh_device_created(const distributed::MeshDeviceImpl*, std::optional<int> parent) noexcept;
    static void program_created(const detail::ProgramImpl*) noexcept;  // get_context_id() + find_instance
    // Context passed explicitly:
    static void program_compile_started(const MetalContext&, const detail::ProgramImpl*, const IDevice*, uint64_t) noexcept;
    static void set_dispatch_core_info(const MetalContext&, ...) noexcept;
    // Context derived from the workload's programs:
    static void emit_debug_entry(const distributed::MeshWorkloadImpl*, ...) noexcept;
};
```

## High-level plan (new branch from `main`, implemented from scratch)

PR #59756 is not reworked. The new branch starts from `main` and reuses the `Data`/`Logger` ideas
from it where they apply. Proposed commits:

1. `Data`/`Logger` take an explicit `MetalContext&` and read settings from its env. There is no
   registry: `Inspector::initialize(MetalContext&, ...)` just builds the session that
   `MetalContext` stores.
2. Add `MetalContext::find_instance()` and the private `inspector_data()` accessor. Route the
   object-derived hooks (mesh device, socket, semaphore, buffer allocate, program create and
   destroy) through `obj -> ctx -> inspector_data()`. Make destroy and erase hooks look up and
   never insert.
3. Add `MetalContext&` to the compile, dispatch-core-info and DPRINT hooks, and update their call
   sites.
4. Derive the `MeshWorkload` context from its programs (D1). Purge per-mesh entries on
   `mesh_device_destroyed` (D2).
5. Experimental API: add the `MeshDevice` overloads, switch ttnn, and route `release_trace`
   internally (D3).
6. Remove `Inspector::get_rpc_server()` and the dead `program_set_binary_status`, and update
   `inspector.rst`.

Steps 1 to 5 must land together. Without a global fallback, any hook left out (compile, workload,
the experimental no-arg queries) would silently stop reporting, and tt-triage would regress.
Only step 6 is separable cleanup.

Validation:

- Build `tt_metal` and `ttnn`.
- Run `unit_tests_inspector` (`test_rpc_startup`), `unit_tests_debug_tools`, and the tt-triage
  end-to-end tests on silicon.
- Run a DPRINT smoke test.
- Extend `MetalContextIntegrationTest.Coexisting*` to assert that the hardware session sees no mock
  mesh devices, buffers or programs (over RPC, as `test_rpc_startup` does).

## Questions for the inspector owner

1. Which variant did you mean: static functions taking the context (Variant 1), or
   `ctx.inspector().hook(...)` (Variant 2b)?
2. Experimental `IsEnabled()` / `ShouldCaptureTensorSpecs()` / `ReleaseTraceDebugEntries()`:
   replace them with `MeshDevice` overloads outright, or deprecate them first?
3. (Resolved) `mesh_workload_created/destroyed` and the `getMeshWorkloads` registry are removed,
   so there is no "created" record to move.
