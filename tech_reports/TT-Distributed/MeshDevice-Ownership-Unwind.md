# `MeshDevice` ownership: why the factories return `shared_ptr`, and what it takes to unwind

Header under review: [`tt_metal/api/tt-metalium/mesh_device.hpp`](../../tt_metal/api/tt-metalium/mesh_device.hpp) (`MeshDevice::create`, `create_unit_mesh`, `create_unit_meshes`, `create_submesh(es)`).
Related: [`MetalEnv-API-Review.md`](../TT-Fabric/MetalEnv-API-Review.md) (end state: the env always owns its context).

Status: prerequisites 0–4 are implemented on branch `ruizhang/meshdevice-ownership`; steps 5–7 are pending. D2 (`UnitMeshGroup`, §4), D3 (non-owning `tensor.device()`) and D5 (raw-pointer `get_parent_mesh()` / `get_submeshes()`) are decided. Line numbers are as of 2026-10-15, before those commits.

## Summary

- **Accretion, not necessity.** `shared_ptr` and `enable_shared_from_this` arrived in Sep 2024 (`7738225beda`, #12719) so that `initialize()` could store `shared_from_this()` in the `SystemMesh` singleton's map of strong references. That registry caused use-after-free bugs and was downgraded to `weak_ptr` within two months (`919d46aeb7c`, `0e5fb148837`). The return type stayed, and later features were built on top of it.
- **Only one step was motivated by Python.** Submesh ownership was inverted so that a submesh keeps its parent alive (`dbe14d445ad`, #18694), "because the parent mesh device must be kept alive longer than the children devices, however the current python code wasn't designed to handle that."
- **Most uses are superficial, and all of these can be references or raw pointers:**
  - the `shared_ptr` parameters on sockets, services and `MeshWorkload`;
  - `IDevice::get_mesh_device()`, which is a downcast in disguise;
  - the `Device` → `MeshDevice` back-pointer (`TODO #20966`);
  - the owning fields in streaming services.
- **Three structural dependencies remain. None of them needs `shared_ptr<MeshDevice>` as the public return type.**
  1. **Tensors observe the device weakly.** `MeshBuffer` holds a `weak_ptr<MeshDevice>` so that deallocating after the device is destroyed becomes a no-op. That needs an expiry signal, not ownership. A liveness token, already used by `PersistentL1Arena`, does the same job.
  2. **Unit meshes jointly own a hidden parent.** This covers every result of `create_unit_mesh(es)`, and therefore `ttnn.open_device`, `ttnn.CreateDevices` and `MetalEnv::create_unit_meshes`. Each result is a 1x1 submesh of an uninitialized parent that only the submeshes own, and on the `MetalEnv` path that parent destroys the `MetalContext`. This is the one case that genuinely needs shared lifetime, but it belongs on the group, not on each device handle. **Decided (D2):** make the group explicit as a `UnitMeshGroup` type that owns its members. The existing `create_unit_meshes` becomes a deprecated wrapper over it, and `MetalEnv` owning the context is the later cleanup (§4).
  3. **A submesh keeps its parent alive.** In C++ this is an ordinary "parent outlives child" precondition; in Python it is a `keep_alive`.
- **The Python `keep_alive` on `create_submesh` is backwards.** `keep_alive<1, 0>` makes the parent pin the child, which contradicts its own comment. It is left over from the Oct 2024 design, and the C++ `parent_mesh_` hides the mistake. Fix it regardless of the rest.
- **Every Python device handle owns the device.** That includes `tensor.device()` and `ttnn.open_device`, even though both are bound with `rv_policy::reference`, because nanobind upgrades raw pointers to `enable_shared_from_this` types into shared ownership. As a result, when a device gets destroyed depends on Python reference counts.
- **Two end states, same prerequisites.** Once nothing inside tt_metal needs shared ownership, there are two ways to finish. They complement each other: the recommendation is B first, then A's `unique_ptr` signatures as the replacement for the `shared_ptr` factories (see "Comparing Path A and Path B").
  - **Path A, `unique_ptr`:** the factories return `unique_ptr`. It is a small change that keeps today's construction code, but it makes heap allocation part of the API and changes existing signatures.
  - **Path B, plain object:** `MeshDevice` gets public constructors and becomes a plain, still non-movable object. Callers can keep it on the stack, as a member, in a `std::optional`, or behind any smart pointer. It changes no existing signature, but initialization has to move into the constructors.
- **Non-movability does not block Path B.** C++17 guaranteed copy elision lets an object be built directly in its final storage, and the repo returned a non-movable `DeviceMesh` by value in March 2024. Non-movability is required anyway, because command queues, buffers, programs and Python all capture the wrapper's address.

## Unwind plan

There are two possible end states. They share the same prerequisites:

- **Path A, `unique_ptr`.** The factories keep their names and return `std::unique_ptr<MeshDevice>`. The device always lives on the heap.
- **Path B, plain object.** `MeshDevice` gets public constructors that leave it fully initialized. Callers choose where it lives: stack, member, `std::optional`, `unique_ptr` or `shared_ptr`.

Both paths keep `MeshDevice` non-movable (§7.1), and both require prerequisites 0–7 first.

### Prerequisites (both paths)

Each step can land on its own. They are listed in dependency order.

| # | Change | Size | Breaks / risk |
| --- | --- | --- | --- |
| 0 | `create_submesh(es)` bindings: `keep_alive<1, 0>` → `keep_alive<0, 1>` (§5) | XS | `del submesh` in Python now releases the submesh while the parent is still alive. That is the intended behaviour. |
| 1 | Replace `IDevice::get_mesh_device()` (returns `shared_ptr`) with a non-owning lookup, and make `Device::mesh_device` a raw `MeshDevice*` (#20966, §2.1) | S | About 20 call sites. The profiler callers already only call `.get()`. |
| 2 | Add a liveness token to `MeshDeviceImpl`. `MeshBuffer` stores `MeshDevice*` plus a weak token. FFT plan caches key on `id()` instead of `weak_ptr` (§3) | M | The token must be reset first thing in `~MeshDevice`. You lose "`lock()` pins the device during deallocation", which only matters for a race that is already a bug. |
| 3 | Change `shared_ptr<MeshDevice>` parameters to `MeshDevice&`. **Public:** sockets, stream services, `MeshWorkload`, socket nanobind bindings. **Internal, used during initialization:** `init_realtime_profiler_socket`, `init_streaming_profiler`, `RealtimeProfilerManager`, `streaming_profiler::Receiver::create`. The internal ones already store only raw pointers (§2.1). | M, mechanical | The public part is an API change: add `MeshDevice&` overloads first and deprecate the `shared_ptr` ones. The internal part must land before either end state, because neither a `unique_ptr`-owned object nor an object still under construction can produce a `shared_ptr` to itself. |
| 4 | Owning `shared_ptr<MeshDevice>` fields (H2D/D2H/D2D stream services, tt-train `core::MeshDevice` and `core::Device`) become borrowed, with a documented precondition (§2.1) | S–M | Use-after-close becomes the caller's responsibility. The services already check `is_initialized()` in their destructors. |
| 5 | Unit meshes: add `UnitMeshGroup`, which owns its members and the shared state (today, the hidden parent). `MetalEnv::create_unit_meshes` returns it directly. `MeshDevice::create_unit_meshes` / `create_unit_mesh` keep their signatures and are reimplemented on top of it with `shared_ptr`s whose control block owns the group, then deprecated (§4.1) | M | Additive: existing callers keep compiling and keep today's behavior. Members of a group cannot outlive it. Coupled with step 6 (§4.1, "Sequencing"). Touches `close_impl`, the parent-chain checks for a command queue still in use, and the root walk in `dispatch_context`. |
| 6 | A submesh holds a non-owning pointer to its parent. The parent's list of children becomes raw pointers, and children unregister in their destructor. **`get_parent_mesh()` and `get_submeshes()` change to raw pointers** (`MeshDevice*`, `std::vector<MeshDevice*>`): an accepted API break (D5), since once the parent is not shared-owned there is no `shared_ptr` left to return (§5) | M | C++ gains the precondition "the parent outlives its submeshes". Python is covered by step 0. For unit meshes, the group (step 5) is what owns the parent. In-repo users of the two accessors: `dispatch_context.cpp`, `unit_mesh_utils.cpp`, `metal_env.cpp`, `test_mesh_device.cpp`, `test_dispatch_context.cpp`. `get_submeshes()` is also bound in Python (the conftests iterate it); it would return non-owning references to the live children. |
| 7 | Remove `enable_shared_from_this` from `MeshDevice` | XS once 1–6 land | The factories can keep returning `shared_ptr`. In Python, `tensor.device()` stops creating new owning handles and returns the existing wrapper instead (D3, confirmed). Needs the full Python CI, not only single-chip. |

Steps 1–4 are worth doing even if neither end state is pursued. They remove fragile code (§8) and are mostly mechanical.

### Path A: factories return `unique_ptr`

| # | Change | Size | Breaks / risk |
| --- | --- | --- | --- |
| A1 | New return types:<br>• `create`, `create_submesh` → `std::unique_ptr<MeshDevice>`<br>• `create_submeshes` → `std::vector<std::unique_ptr<MeshDevice>>`<br>• the same for `MetalEnv::create_mesh_device`<br>Unit meshes are not part of this step: they move to `UnitMeshGroup` in step 5, and `create_unit_mesh(es)` stay as deprecated wrappers until removed.<br>In the implementation, `shared_ptr<MeshDevice>(new ...)` becomes `unique_ptr`. The two-phase initialization stays. | S | Callers that copy the result, or that spell out the container types (§7.2). |
| A2 | `get_parent_mesh()` → `MeshDevice*`, `get_submeshes()` → `std::vector<MeshDevice*>` | S | Callers that stored the result. |
| A3 | nanobind: include `nanobind/stl/unique_ptr.h`. Python takes ownership of returned devices. | S | None for Python user code. |

### Path B: plain object

| # | Change | Size | Breaks / risk |
| --- | --- | --- | --- |
| B1 | Public root-mesh constructors (config plus options, and a `MetalEnv&` variant) that do everything the factory does today, with a guard that runs `close_impl` if initialization throws (§7.3) | M | Nothing; it is additive. The work is moving the post-wrap half of `MeshDeviceImpl::create` into the constructor. |
| B2 | Submesh constructor `MeshDevice(MeshDevice& parent, shape, offset)`. It registers with the parent and unregisters in its destructor. A parent destroyed with live submeshes aborts with a diagnostic. | M | Nothing; additive. |
| B3 | Unit meshes: nothing new for users beyond step 5's `UnitMeshGroup`. Internally, the group can hold its members by value (node-based storage with a pass-key constructor) instead of through `unique_ptr` (§7.3) | S | Depends on prerequisite 5. |
| B4 | nanobind: bind the constructors with `nb::init<...>`, the submesh one with `keep_alive<1, 2>`. `ttnn.open_mesh_device` and the other Python entry points construct `ttnn.MeshDevice(...)`. | S | None for Python user code. |
| B5 | Deprecate the `shared_ptr`-returning factories. Either remove them or replace them with Path A's `unique_ptr` signatures as a heap convenience. | — | The normal deprecation schedule (D4). |

The next section compares the two paths and gives the recommended order.

## Comparing Path A and Path B

Path A changes how the existing factories hand out a device. Path B changes what a device is. Both end with deterministic destruction, Python user code unchanged, and the same lifetime rules: a parent outlives its submeshes, a device outlives anything that borrows it, and tensors only observe. They differ in what callers must change, how much device bring-up code moves, where a device can live, and how ownership can be handed around.

### At a glance

| | Path A: `unique_ptr` | Path B: plain object |
| --- | --- | --- |
| **API and migration** | | |
| Existing signatures | Five factories and two accessors change their return types | None until B5. The constructors are additions. |
| Changes forced on C++ callers | Sites that copy the result, or that spell out `std::map`/`std::vector` of `shared_ptr<MeshDevice>` | None until B5. At B5, the same as A if B5 adopts A's signatures. |
| A caller that still wants shared ownership | Declares the type: `std::shared_ptr<MeshDevice> d = MeshDevice::create(...);`, after which copies work | `std::make_shared<MeshDevice>(cfg)` |
| Python user code | Unchanged | Unchanged |
| **Implementation** | | |
| Construction code | Unchanged: wrap the impl, then initialize | The post-wrap half of `MeshDeviceImpl::create` moves into the constructors |
| Cleanup when initialization throws | Automatic. The `unique_ptr` owns the object before initialization starts, so `~MeshDevice` runs `close_impl`, as it does today. | Has to be written. A constructor that throws never runs `~MeshDevice`, so a guard must call `close_impl`. |
| Submesh | `create_submesh` returns `unique_ptr` | Constructed from `MeshDevice& parent` |
| Unit meshes | `UnitMeshGroup` (step 5), the same in both paths | `UnitMeshGroup`; internally it may hold members by value |
| Python binding | Factories return `unique_ptr`; nanobind hands ownership to the Python object | `nb::init` constructs the device inside the Python object |
| **Using a device** | | |
| Where a device can live | Heap only | Stack, member, `std::optional`, `std::deque`/`std::map` node, `unique_ptr`, `shared_ptr` |
| Handing ownership to another owner, or returning it from a helper | Move the `unique_ptr` | A device held by value stays where it was built. To hand it around, hold it in a `unique_ptr` (the A form). |
| Created later, or reopened (gtest `SetUp()`, tt-train `AutoContext`) | `dev_ = MeshDevice::create(...);` and `dev_.reset();` | `dev_.emplace(cfg);` and `dev_.reset();` on a `std::optional<MeshDevice>` |
| When the device is destroyed | When its `unique_ptr` is reset or destroyed | At the end of its scope, or with its owner. This is the most visible in the code. |
| Heap allocation | One per device | None needed. This does not matter in practice: opening a device takes seconds. |
| **Fit with the codebase** | | |
| Closest existing pattern | `MeshBuffer::create`, a static factory returning a smart pointer | `MetalEnv`: `explicit MetalEnv(MetalEnvDescriptor = {})`, non-copyable and non-movable ([`metal_env.hpp:75-83`](../../tt_metal/api/tt-metalium/experimental/context/metal_env.hpp)). `Program` and `MeshWorkload` also have public constructors, though they are movable. |
| `MetalEnv` end state (the env owns the context) | `MetalEnv::create_mesh_device` returns `unique_ptr` | A `MeshDevice(MetalEnv&, ...)` constructor takes the env explicitly |
| **Cost and risk** | | |
| Size after prerequisites | S, plus caller fixes | M for B1 and B2, then S |
| Main risk | Churn at a single breaking change | Getting cleanup right when device bring-up fails inside a constructor. These failure paths are hard to exercise in tests. |
| Reversibility | Adding B later is purely additive | A's signatures are B5. Neither path rules out the other. |

### When each path fits

**Path A alone fits if:**
- the goal is the smallest change, and heap-only devices are acceptable;
- no caller needs a device as a local or a member without a pointer;
- device bring-up code should not be touched.

**Path B (then A's signatures) fits if:**
- `MeshDevice` should look like `MetalEnv`, the other non-movable object that owns hardware;
- there should be no single breaking change: existing callers keep compiling until the deprecated factories are removed;
- tests and tt-train would benefit from devices with scoped or `std::optional` lifetimes instead of pointers.

### Recommendation

The paths complement each other. B is what makes every storage choice possible; A is the convenient heap form.
1. Do prerequisites 0–7.
2. Do B1–B4. They are additive, so nothing breaks.
3. Do B5, replacing the `shared_ptr` factories with Path A's `unique_ptr` signatures.

If only one path is funded, A is cheaper. But it builds heap allocation into the API, and it is a breaking change on its own.

### Rough sizing of the breaking step (A1, or B5 in Path B)

- About 43 factory call sites in C++ outside `tests/`, and about 111 in `tests/`.
- About 143 C++ call sites of `create_submesh(es)`.
- About 290 lines outside `tests/` that name `shared_ptr<MeshDevice>` (`tt_metal` 220 including 52 in `programming_examples`, `ttnn` 60, `tt-train` 9, `tools` 5), and about 680 in `tests/`.
- Python has about 1,000 call sites of `open_mesh_device`, `open_device`, `CreateDevices` and `create_submesh`. Neither path renames those APIs, so none of these need edits.

## Decisions needed

| # | Question | Options | Recommendation |
| --- | --- | --- | --- |
| D1 | End-state handle type | (a) Path A only; (b) Path B only; (c) B, then A's signatures as the heap convenience; (d) keep the `shared_ptr` factories and only do the prerequisites | (c). A constructor supports stack, member, `optional::emplace`, `make_unique`, `make_shared` and `nb::init`, whereas a factory forces one ownership model on everyone. See "Comparing Path A and Path B". |
| D2 | How unit meshes share their lifetime once `MeshDevice` stops being shared-owned | (b) an explicit `UnitMeshGroup` type that owns its members, with the old `create_unit_meshes` kept as a deprecated wrapper; alternative (a): an internal shared owner behind the hidden parent, with no API change. Separately, (c) `MetalEnv` owns the context. | **Decided: (b)**, with (c) as the end-state cleanup. (a) is the alternative. (c) is not a competing option: it removes the context-teardown duty from whichever of (a)/(b) is chosen. See §4. |
| D3 | Should a Python handle obtained from `tensor.device()` keep the device alive? | Yes (today, by accident) / No | **Confirmed: No.** Explicit `close()` is the documented model, and tensors already observe the device weakly. Takes effect in step 7. Code that kept a device alive only through such a handle (for example `d = tensor.device()` after dropping the original) will see it destroyed. |
| D5 | Accept the `get_parent_mesh()` / `get_submeshes()` change to raw pointers in step 6? | Accept / keep `shared_ptr` accessors | **Accepted.** There is no compatibility path once the parent is not shared-owned. |
| D4 | Deprecation window for public APIs that take `shared_ptr<MeshDevice>` | — | Add the `MeshDevice&` overloads first, then follow the usual deprecation schedule. |

---

# Details

## 1. How it got here

| Date | Commit | Change | Stated reason |
| --- | --- | --- | --- |
| 2024-03-07 | `6202d06587c` (#5733) | First multi-device type, `ttnn::multi_device::DeviceMesh`. Copy and move are deleted. `open_device_mesh()` returns `DeviceMesh&` into a global map. pybind uses the default holder with `return_value_policy::reference`. | — |
| 2024-03-14 | `e7dade43c26` | `open_device_mesh()` returns `DeviceMesh` **by value** (non-movable, relying on C++17 elision). Python switches to constructing it through `py::init`. | "cleanup memory management" |
| 2024-08-30 | `db3a60ff6e3` (#12118) | Renamed `DeviceMesh` → `MeshDevice` | — |
| 2024-09-19 | `7738225beda` (#12719) | **Adds `enable_shared_from_this` and `static shared_ptr<MeshDevice> create()`.** The pybind holder becomes `std::shared_ptr<MeshDevice>`. `initialize()` calls `SystemMesh::map_mesh_device(shared_from_this(), ...)`, which stores the result in `unordered_map<MeshDeviceID, shared_ptr<MeshDevice>>`. | Mesh virtualization. The commit says nothing about lifetimes. |
| 2024-10-03 | `a4a9a4ccde2` (#12962) | Submeshes. The parent owns its children (`vector<shared_ptr<MeshDevice>>`, commented "Parent owns submeshes and responsible for their destruction"). Each child holds a `weak_ptr` to its parent. Python gets `keep_alive<1, 0>`. | — |
| 2024-11-20 | `919d46aeb7c` (#15250) | `SystemMesh` switched to holding weak references. | "Fix UAF error when `close_devices()` not invoked … changing the held ownership of MeshDevice handles in SystemMesh to be a weak reference to avoid double delete." |
| 2024-12-06 | `0e5fb148837` | `SystemMesh` destruction decoupled from `MeshDevice` destruction. | Use-after-free caused by singleton destruction order. |
| 2025-01-14 | `2c4a641fde4` (#16570) | `MeshDevice : public IDevice` | API unification. |
| 2025-02-24 | `4a0562cb607` (#18181) | `MeshBuffer` changes from a raw `MeshDevice*` to `weak_ptr<MeshDevice>`, obtained via `shared_from_this()`. | "crash if MeshDevice is deallocated or closed before MeshBuffer". Reproduced with C++ gtests. |
| 2025-03-07 | `dbe14d445ad` (#18694) | Adds `create_unit_mesh(es)` with a hidden parent. **Ownership is inverted**: children hold a `shared_ptr` to the parent, and the parent holds `weak_ptr`s to the children. | "…the current python code wasn't designed to handle that. So this PR inverts ownership relationship." |
| 2025-04-22 | `3623ec57394` (#18067) | `IDevice::get_mesh_device()` returns `shared_ptr`. `Device` gains a `weak_ptr<MeshDevice>` back-pointer marked `TODO #20966: Remove`. | "This is currently used exclusively by profiler." |
| 2025-12 | `eca462c1226`, `95476914c4e` | Moved to nanobind as `nb::class_<MeshDevice>` with `shared_ptr` interop. Holder semantics are unchanged. | — |
| 2026-09-23 | `edfa0407898` (#57279) | The impl is built before it is adopted, which establishes the "`pimpl_` is never null" invariant. | Fix for a null `pimpl_`. |

Before `7738225beda`, the Python-facing class was a plain non-movable object. The one reason for the original `shared_ptr` (a strong registry) was reverted, but the return type was never revisited.

## 2. Inventory of dependencies

### 2.1 Superficial (mechanical to change)

| Site | What it does | Replacement |
| --- | --- | --- |
| `const shared_ptr<MeshDevice>&` parameters on `H2DSocket`, `D2HSocket`, `MeshSocket`, stream services, `MeshWorkload::get_sem_base_addr` and similar | Only dereferenced. The sockets already store a raw pointer: "Helpers below still take a shared_ptr; the socket itself stores only a raw `MeshDevice*`" ([`h2d_socket.cpp:542-544`](../../tt_metal/distributed/h2d_socket.cpp)). Internal callers convert with `shared_from_this()` only to satisfy those signatures ([`d2h_socket.cpp:463`](../../tt_metal/distributed/d2h_socket.cpp), [`tensor_prefetcher_manager.cpp:734`](../../tt_metal/impl/buffers/tensor_prefetcher_manager.cpp)). | `MeshDevice&` |
| `IDevice::get_mesh_device()` → `shared_from_this()` ([`mesh_device.cpp:2250`](../../tt_metal/distributed/mesh_device.cpp)) | Mostly used to ask "is this `IDevice` a mesh?" in `extract_context_id` ([`metal_context.cpp:106-116`](../../tt_metal/impl/context/metal_context.cpp)). The profiler callers only call `.get()`. | `dynamic_cast`, or a virtual returning `MeshDevice*` |
| `Device::mesh_device` (`weak_ptr`, [`device_impl.hpp:177-179, 245`](../../tt_metal/impl/device/device_impl.hpp)) | Back-pointer from a physical chip to the mesh currently using it. It is set when the mesh is created and re-pointed explicitly in `close_impl` ([`mesh_device.cpp:1047-1050`](../../tt_metal/distributed/mesh_device.cpp)). | Raw pointer, set and cleared by the mesh |
| Owning fields: `H2DStreamService::mesh_device_` ([`h2d_socket_service.hpp:258`](../../ttnn/api/ttnn/services/h2d_socket_service.hpp)), `D2HStreamService` ([`d2h_socket_service.hpp:149`](../../ttnn/api/ttnn/services/d2h_socket_service.hpp)), D2D stream service ([`d2d_stream_service.cpp:377, 432`](../../ttnn/core/tensor/d2d_stream_service.cpp)), tt-train [`core/mesh_device.hpp:26`](../../tt-train/sources/ttml/core/mesh_device.hpp) and [`core/device.hpp:27`](../../tt-train/sources/ttml/core/device.hpp) | Defensive ownership that makes use-after-close impossible. tt-train's `AutoContext` is already the single owner, through a `unique_ptr`. | Borrow, with the precondition "the device outlives the service" |
| FFT plan caches: `device_weak` ([`stockham_host.hpp:64, 69`](../../ttnn/cpp/ttnn/operations/experimental/fft/device/stockham_host.hpp), [`apply_twiddles_host.hpp:53`](../../ttnn/cpp/ttnn/operations/experimental/fft/device/apply_twiddles_host.hpp)) | Detects that a cached `MeshDevice*` address has been reused by a new device | Key on `MeshDevice::id()`, which is unique per process |

### 2.2 Structural

These are covered in §3 (tensors observe the device weakly), §4 (unit meshes and their hidden parent) and §5 (submesh → parent).

### 2.3 Already non-owning

The prevailing pattern is already a raw `MeshDevice*`. It is used by `MeshCommandQueue`, `MeshEvent`, `GlobalSemaphore`, `GlobalCircularBuffer`, `ProgramImpl`, `TensorPrefetcherManager`, `CrossNodeDFB`, `PrefetcherPipe`, ttnn's default device ([`ttnn/core/device.cpp:97-102`](../../ttnn/core/device.cpp)), `Tensor::device()`, and the attribute structs of most ttnn device operations.

Rough counts in public headers (counting matching lines):

| Spelling | `tt_metal/api` | `ttnn/api` |
| --- | --- | --- |
| `shared_ptr<MeshDevice>` | 37 | 18 |
| `MeshDevice&` | 60 | 11 |
| `MeshDevice*` | 44 | 49 |

## 3. Structural problem 1: tensors observe the device weakly

**The chain.** `Tensor` → `DeviceStorage` → `MeshTensor` → `shared_ptr<MeshBuffer>` → `weak_ptr<MeshDevice>` ([`mesh_buffer.hpp:150-182`](../../tt_metal/api/tt-metalium/mesh_buffer.hpp)). A tensor never keeps its device alive.

**Why this case is real.** Tensors routinely outlive the device object:
- Python GC destroys stray tensors after the device at shutdown. A comment in [`mesh_buffer.cpp:406-410`](../../tt_metal/distributed/mesh_buffer.cpp) says so explicitly.
- Tensors created inside ops have no Python-visible relationship to the device.
- During `~MeshDevice`, the program cache destroys cached programs, which free their buffers. `weak_ptr::lock()` returns null at that point, so those frees take the "device is gone" path.

**What the `weak_ptr` protects.** The backing `Buffer` stores the `MeshDevice*` as a raw `IDevice* device_` ([`mesh_buffer.cpp:273`](../../tt_metal/distributed/mesh_buffer.cpp)). When the `weak_ptr` has expired, `MeshBuffer::deallocate()` marks that buffer as deallocated so it never touches the dangling pointer ([`mesh_buffer.cpp:400-441`](../../tt_metal/distributed/mesh_buffer.cpp)). `MeshBuffer::device()` `TT_FATAL`s when the pointer has expired ([`:444-448`](../../tt_metal/distributed/mesh_buffer.cpp)).

**What it does not protect.** A device that has been closed but not destroyed. `ttnn.close_mesh_device` only calls `mesh_device->close()` ([`ttnn/core/distributed/api.cpp:66`](../../ttnn/core/distributed/api.cpp)), so the common Python teardown path is handled by `is_initialized()` checks ([`mesh_buffer.cpp:416`](../../tt_metal/distributed/mesh_buffer.cpp), [`buffer.cpp:736`](../../tt_metal/impl/buffers/buffer.cpp)), not by the `weak_ptr`.

**Why this forces shared ownership today.** A `weak_ptr` needs a control block, so the `MeshBuffer` constructor calls `mesh_device->shared_from_this()`. That only works if the device is owned by a `shared_ptr`.

**Replacement: a liveness token.** The codebase already uses this pattern ([`persistent_l1_arena.hpp:108-110`](../../tt_metal/impl/allocator/persistent_l1_arena.hpp)):

```cpp
    // Liveness token handed to every Seal. Nulled and released in the destructor so a
    // Seal destroyed after the arena unseals into nothing instead of freed memory.
    std::shared_ptr<PersistentL1Arena*> liveness_;
```

Applied to `MeshDevice`, a sketch:

```cpp
// MeshDeviceImpl: owned uniquely; the token is the only shared thing.
std::shared_ptr<const void> liveness_ = std::make_shared<char>();
std::weak_ptr<const void> liveness_token() const { return liveness_; }
// ~MeshDevice resets liveness_ before close_impl(), so buffers freed during teardown see "gone".

// MeshBuffer
MeshDevice* mesh_device_;
std::weak_ptr<const void> device_alive_;
```

How it differs from `weak_ptr<MeshDevice>`: `lock()` on a `weak_ptr` also keeps the device alive for the duration of the call. Checking a token does not. That only matters if one thread destroys the device while another frees buffers on it, and that is already a bug.

The same family of patterns already exists in `DriscL1Arena` (a `shared_ptr` arena with weak handles, [`mesh_device_impl.hpp:205-211`](../../tt_metal/distributed/mesh_device_impl.hpp)) and `HostRegion`.

## 4. Structural problem 2: unit meshes jointly own a hidden parent

Every unit mesh is a submesh of a parent that the caller never sees ([`mesh_device.cpp:620-633`](../../tt_metal/distributed/mesh_device.cpp)):

```cpp
    auto mesh_device_impl = std::make_unique<MeshDeviceImpl>(
        std::move(scoped_devices), std::move(mesh_device_view), std::shared_ptr<MeshDevice>(), ctx);
    auto mesh_device = std::shared_ptr<MeshDevice>(new MeshDevice(std::move(mesh_device_impl)));

    auto submeshes = mesh_device->create_submeshes(MeshShape(1, 1));
    // ...
    std::map<int, std::shared_ptr<MeshDevice>> result;
    for (size_t i = 0; i < device_ids.size(); i++) {
        result[device_ids[i]] = submeshes[i];
    }
```

`create_unit_mesh` (single device) calls `create_unit_meshes({id})`, so `ttnn.open_device`, `ttnn.CreateDevice(s)` ([`ttnn-nanobind/device.cpp:70-81, 281-297`](../../ttnn/cpp/ttnn-nanobind/device.cpp)) and `MetalEnv::create_unit_meshes` all return children of a hidden parent.

**What the hidden parent is responsible for:**

- **Destroying the `MetalContext`** on the `MetalEnv` path: "Unit meshes are submeshes, so they always have a parent to own the context" ([`metal_env.cpp:859-863`](../../tt_metal/impl/context/metal_env.cpp), [`mesh_device.cpp:1158-1164`](../../tt_metal/distributed/mesh_device.cpp)).
- **Taking back its chips.** When a child closes, the physical devices are re-pointed to the parent ([`mesh_device.cpp:1047-1050`](../../tt_metal/distributed/mesh_device.cpp)).
- **Command-queue checks.** It is consulted by the parent-chain checks for a command queue still in use ([`:1054-1080, 1169-1177`](../../tt_metal/distributed/mesh_device.cpp)) and by the root walk in `dispatch_context` ([`dispatch_context.cpp:192-195`](../../tt_metal/distributed/dispatch_context.cpp)).
- **It is never initialized.** The physical devices themselves are already shared separately, through `shared_ptr<ScopedDevices>`.

**Why this one genuinely needs shared lifetime.** Today N handles are owned independently. The shared teardown (the context, and the final device close) has to run when the last of them goes away, in any order. Once steps 6–7 stop children from sharing ownership of the parent, something else has to provide that.

**Decision (D2): make the group explicit.**
- **Chosen, (b):** a `UnitMeshGroup` type owns its members and the shared state. The existing `create_unit_meshes` stays, reimplemented on top of the group, and goes through the formal deprecation process (§4.1).
- **End-state cleanup, (c):** `MetalEnv` owns the context ([#21500](https://github.com/tenstorrent/tt-metal/issues/21500), the MetalEnv review's end state). This is not a competing option. It removes one duty, the context teardown, from whichever owner is chosen, and it can't happen until the global-registry scaffolding is gone.
- **Alternative, (a):** an internal shared owner behind the hidden parent, with no API change (§4.2).

### 4.1 `UnitMeshGroup` (chosen)

**Why a type and not just an owner.** The group is a real concept that today is only implicit:
- **Unit meshes are opened and closed together.** That is why the hidden parent exists at all: "Creating MeshDevices 1 by 1 for each device id is not equivalent to creating/closing them at the same time" (`dbe14d445ad`). A group-level `close()` says this directly. Today it is a side effect of whichever handle happens to die last.
- **The group outlives (c).** After `MetalEnv` owns the context, the group stops driving context teardown, but it is still the unit that is opened and closed together and the home for membership and metadata. So it is not an API commitment that (c) would undo.
- **`std::map<int, std::shared_ptr<MeshDevice>>` is rigid.** It hard-codes an `int` key, chip-id ordering, and shared ownership of each element, and it has no room for metadata or group-level operations.
- **Code already rediscovers the group by hand.** ttnn's `unit_mesh::aggregate` / `disaggregate` find it through `get_parent_mesh()` and `get_submeshes()` ([`unit_mesh_utils.cpp:56-67, 115`](../../ttnn/core/tensor/unit_mesh/unit_mesh_utils.cpp)).

**Invariant and name.** Exactly one 1x1 member per chip, hence `UnitMeshGroup`. Members are borrowed from the group and **must not outlive it**.

**Interface sketch:**

```cpp
// Owns its members. Movable: members live on the heap or in node-based storage, so their addresses (which the
// runtime captures, §7.1) never change. Not copyable.
class UnitMeshGroup {
public:
    static UnitMeshGroup open(const std::vector<ChipId>& chip_ids, const MeshDeviceOptions& options = {});

    bool contains(ChipId chip) const;
    MeshDevice& at(ChipId chip) const;
    std::size_t size() const;
    std::vector<ChipId> chip_ids() const;
    /* begin()/end() over (ChipId, MeshDevice&) */

    void close();  // closes all members together
};
```

`MetalEnv::create_unit_meshes` returns a `UnitMeshGroup` directly. It is experimental and has only test callers, so no deprecation is needed there.

**What the type deliberately does not promise.** These keep later changes (including (c)) non-breaking:
- **No parent accessor.** How the shared state is held (today a hidden, never-initialized parent `MeshDevice`) is an implementation detail.
- **The chip-id key is not the core contract.** `at(ChipId)` matches today's map. A lookup by coordinate may be needed later (see "Scope"), because a member of a carved multi-host mesh can sit on a remote chip.
- **No promise that destroying the group destroys the `MetalContext`.** That stops being true after (c).

**Transition: the old API becomes a thin wrapper.** `MeshDevice::create_unit_meshes` keeps its signature (`std::map<int, std::shared_ptr<MeshDevice>>`) and builds a group. Dropping *shared ownership of `MeshDevice`* does not prevent returning `shared_ptr<MeshDevice>`: a `shared_ptr` may point at one object while its control block owns another. Here each returned pointer points at a member, and its control block keeps the whole group alive. Nothing inside tt_metal depends on these handles (they exist only for old callers), and they go away when the old API is removed.

```cpp
auto group = std::make_shared<UnitMeshGroup>(UnitMeshGroup::open(chip_ids, options));
std::map<int, std::shared_ptr<MeshDevice>> result;
for (ChipId id : group->chip_ids()) {
    MeshDevice* member = &group->at(id);
    // The captured `group` keeps the group (and so the chips) alive until the last member's handle is gone.
    // The deleter closes the member when ITS last handle goes away, like destroying a unit mesh does today.
    result[id] = std::shared_ptr<MeshDevice>(member, [group](MeshDevice* m) { m->close(); });
}
```

- **Why a custom deleter and not the plain aliasing constructor.** `std::shared_ptr<MeshDevice>(group, &group->at(id))` would be simpler, but it changes behavior. Today, when the last handle to one unit mesh goes away, that child is destroyed and its `close_impl` runs (command queues and allocators released), even while its siblings live; the chips close only when the hidden parent dies. With plain aliasing a member would live until *all* handles are gone. The deleter keeps today's per-member teardown, and gives each member its own `use_count()`. Explicit `close()` on a device works either way. The group's own destruction still closes everything together.
- This is exactly today's "last one out tears everything down", with no hidden-parent ownership trick and no other internal machinery. It is why (b) makes (a) unnecessary.
- **Keep these handles out of Python.** While `MeshDevice` still derives from `enable_shared_from_this`, nanobind may try to attach its own control block when such a pointer is passed back into C++. The Python API binds `UnitMeshGroup` instead, as below.
- All in-repo callers keep compiling and behaving the same. That is about 48 C++ call sites, almost all test fixtures, benchmarks and tools; ttnn's binding and `cluster_validation_utils.cpp` are the only non-test product code. It includes fixtures that store the map in explicitly typed members and copy the `shared_ptr`s into `devices_` vectors (about 1,200 uses in tests).
- `create_unit_mesh` (single device) becomes one of these handles onto a one-member group.
- Deprecate both with the repo's convention: a `deprecated` attribute whose message names the removal date, as the existing ones in `mesh_device.hpp` do ("... will be removed after September 9th, 2026"). The repo builds with `-Wno-deprecated-declarations` ([`CMakeLists.txt:211`](../../CMakeLists.txt)), so in-repo callers won't be warned and have to be migrated by hand. External users see the warning unless they also silence it.

**Python.**
- Bind `UnitMeshGroup` with the mapping protocol: `__getitem__`, `__iter__`, `__len__`, `keys` / `values` / `items`, plus `close()`. Dict-style code keeps working.
- `__getitem__` returns members with `rv_policy::reference_internal`, so a member's Python handle keeps its group alive. This is the "N handles → hidden parent" relationship that `keep_alive` could not express before (§6).
- `ttnn.CreateDevices` has no in-repo callers. It can return the group directly, or follow the same deprecation as the C++ API.

**Sequencing with step 6.** Steps 5 and 6 are coupled:
- The group has to own the hidden parent, so unit-mesh members must stop owning it through `parent_mesh_`. That is step 6's change, applied to unit meshes.
- These handles carry no `enable_shared_from_this` state, so `shared_from_this()` on such a member throws. After steps 1–3a, its only remaining caller is `create_submesh(es)`, which uses it to share ownership of the parent.

So either land steps 5 and 6 together, or land step 5 first while the group still creates its members through `std::shared_ptr` internally, so `shared_from_this()` keeps working until step 7.

**Step 5 as implemented (first PR).** Step 5 landed in its "group first" form:
- `UnitMeshGroup` (`tt-metalium/unit_mesh_group.hpp`) holds its members as `shared_ptr` internally, so behavior is unchanged and `shared_from_this()` keeps working.
- The group offers `open(chip_ids, options)`, `size` / `empty` / `contains`, `at(ChipId)` returning `MeshDevice&`, `chip_ids()`, iteration yielding `pair<ChipId, MeshDevice&>`, `close()`, and move-only semantics (plus a default-constructed empty group to move-assign into). Destroying a group closes what is left.
- Options are a new stable `MeshDeviceOptions` (`tt-metalium/mesh_device_options.hpp`): the five settings `MeshDevice::create` already takes positionally. It had to be stable because the API-tier check forbids a stable header from including an experimental one. `CreateMeshDeviceOptions` stays in `metal_env.hpp` as an alias, so `MetalEnv` callers are unaffected.
- `MetalEnv::create_unit_meshes` returns the group. This changes an experimental API, and the only in-repo callers were two tests.
- `MeshDevice::create_unit_meshes` / `create_unit_mesh` are untouched for now. They become wrappers (with the aliasing and closing deleter described above) in the change that removes shared ownership of the members, so the wrapper is written once.
- The Python binding is deferred to step 7. Before that, a handle returned for a member would co-own it through `enable_shared_from_this`, which contradicts the documented `reference_internal` contract. Binding it later avoids shipping transitional semantics.

**Scope: only `create_unit_meshes` for now.** There is a second, incompatible kind of "unit mesh group" today: an opened, initialized mesh carved with `create_submeshes(MeshShape(1, 1))`. It is the only kind `unit_mesh::aggregate` works with, because `aggregate` needs the parent's allocator. On `create_unit_meshes` results it throws, with a misleading "remote-only MeshDevice" message, because the hidden parent was never initialized.
- **Possible later extension:** something like `mesh.create_unit_mesh_group()` could return the same type. All its members would still be 1x1, so the name still fits. The group would *borrow* the parent instead of owning it, which is why the type doesn't expose its parent.
- **Not this type:** general `create_submeshes(shape)` / `create_submesh(shape, offset)` results (1x4, 2x2, possibly overlapping) are not unit meshes. Arbitrary submesh sets are already modeled by the parent/submesh tree. A uniform tiling into equal tiles would be a separate type, of which `UnitMeshGroup` could become the 1x1 special case. It would not be a rename.

**Open points.**
- The deprecation date for `create_unit_mesh(es)`.
- The end-state form of the single-device `create_unit_mesh`: a one-member `UnitMeshGroup`, or a plain 1x1 mesh.
- Whether `get_parent_mesh()` on unit meshes stays public. It becomes a raw `MeshDevice*` accessor in step 6 (D5, accepted). If the group replaces it, the hidden parent can become a pure implementation detail, or disappear after (c).
- How exposed external consumers (tt-mlir, tt-xla, ...) are. The in-repo footprint is small, which suggests they are lightly affected, but the formal deprecation process applies regardless.

### 4.2 Alternative: an internal shared owner (a)

Each unit-mesh child's impl holds an internal `shared_ptr` to a private group object that owns the hidden parent and its duties. The public API is unchanged.

| | (a) internal shared owner | (b) `UnitMeshGroup` (chosen) |
| --- | --- | --- |
| Public API change | None | Additive: a new type; the old API becomes a deprecated wrapper |
| Internal machinery | A private group object behind the hidden parent | None beyond the group itself; aliasing provides the shared semantics for the old API |
| Group-level operations (close together, membership) | Stay implicit | Explicit |
| Hidden parent and `get_parent_mesh()` on unit meshes | Stay | Can be retired along with the old API |
| After (c) | Shrinks, but still has to own the parent object while `get_parent_mesh()` is public | Stays useful as the group |
| Transition burden | None | Migrate about 48 call sites by hand (deprecation warnings are silenced in-repo); the old API lingers until removed |

(a) is the right choice only if no public API change is acceptable at all. It keeps the map's rigidity and the implicit group, and its machinery does not fully go away after (c).

## 5. Structural problem 3: a submesh keeps its parent alive

**Ownership has flipped once already.** In Oct 2024 the parent owned its children (`a4a9a4ccde2`). In Mar 2025 this was inverted (`dbe14d445ad`): a submesh keeps its parent alive through `parent_mesh_`, and the parent tracks its children through `weak_ptr` ([`mesh_device_impl.hpp:183-185`](../../tt_metal/distributed/mesh_device_impl.hpp)).

**For user-created submeshes, this is an ordinary precondition.** "The parent outlives its submeshes" is a standard rule in C++, the same one views and iterators follow, and in Python it is exactly what `keep_alive` expresses. The Python annotation, however, is backwards ([`distributed_nanobind.cpp:302-312`](../../ttnn/core/distributed/distributed_nanobind.cpp)):

```cpp
        .def(
            "create_submesh",
            &MeshDevice::create_submesh,
            nb::arg("submesh_shape"),
            nb::arg("offset") = nb::none(),
            nb::keep_alive<1, 0>())  // Keep MeshDevice alive as long as SubmeshDevice is alive
        .def(
            "create_submeshes",
            &MeshDevice::create_submeshes,
            nb::arg("submesh_shape"),
            nb::keep_alive<1, 0>())  // Keep MeshDevice alive as long as SubmeshDevices are alive
```

**What `<1, 0>` actually does.** nanobind's `keep_alive<Nurse, Patient>` keeps the Patient alive for as long as the Nurse exists, so `<1, 0>` keeps the *returned submesh* alive while `self` (the parent) exists. That matched the Oct 2024 design and contradicts the comment. Today the parent stays alive only because of the C++ `parent_mesh_`. The side effect is that a submesh's Python wrapper cannot be freed while the parent's wrapper exists. The correct annotation is `keep_alive<0, 1>`.

## 6. The Python side

**Binding.** `nb::class_<MeshDevice>(mod, "MeshDevice", nb::dynamic_attr())` ([`distributed_nanobind.cpp:159`](../../ttnn/core/distributed/distributed_nanobind.cpp)). There is no `__init__`; instances come only from factories, which return `shared_ptr`.

**Every handle owns the device.** From nanobind's docs (`docs/ownership_adv.rst`, section `enable_shared_from_this`): if a raw pointer is returned and a `shared_ptr` already exists, "you will get shared ownership on the Python side regardless of whether you used `rv_policy::take_ownership` or `rv_policy::reference`." So:
- `Tensor.device()` and `Tensor.devices()` ([`pytensor.cpp:1282-1304`](../../ttnn/cpp/ttnn-nanobind/pytensor.cpp), `rv_policy::reference`) return handles that keep the device alive.
- `rv_policy::reference, // cleanup has to happen in c++ land` on `ttnn.open_device` ([`ttnn-nanobind/device.cpp:81`](../../ttnn/cpp/ttnn-nanobind/device.cpp)) has no effect.

**Closing.** `ttnn.close_mesh_device` and `ttnn.close_device` call `close()` explicitly. The destructor is a backstop: `~MeshDevice` calls `close_impl`, which is idempotent ([`mesh_device.cpp:2055-2058`](../../tt_metal/distributed/mesh_device.cpp)). The fixtures in `conftest.py` close submeshes, then the parent, then reset the default device.

**What `keep_alive` can and cannot express:**

| Relationship | `keep_alive` works? | Notes |
| --- | --- | --- |
| Submesh → parent | Yes | `keep_alive<0, 1>` (§5) |
| Service or socket → device | Yes | `keep_alive<1, 2>` on the constructor, or a documented precondition |
| Tensor → device | Not needed | Tensors must *observe*, not own (§3). This is solved in C++ and does not depend on the bindings. Per-op `keep_alive` would be impractical anyway, because tensors are created inside ops. |
| N returned unit meshes → hidden parent | No | `keep_alive` only relates arguments and return values. With `UnitMeshGroup` (§4.1) the group is one Python object, and members returned by `__getitem__` with `rv_policy::reference_internal` keep it alive. |
| C++ globals (default device, tt-train `AutoContext`) | No | The default device is already a raw pointer that callers reset. |

**After prerequisite 7** (no more `enable_shared_from_this`):
- `tensor.device()` with `rv_policy::reference` finds the existing wrapper in nanobind's instance map instead of creating a new owning handle.
- If the device has already been destroyed, `MeshBuffer::device()` fails its token check before a dangling pointer can escape.

**At the end state:**
- **Path A:** the factories return `unique_ptr`, and nanobind hands ownership to the Python object.
- **Path B:** `nb::init` constructs `ttnn.MeshDevice(...)` inside the Python object, and Python owns it.
- **Both:** `close_mesh_device` keeps working, and the destructor remains the backstop. Python user code does not change.

## 7. End states in detail

### 7.1 Why `MeshDevice` stays non-movable (both paths)

**Origin.** Copy and move were deleted in the very first `DeviceMesh` commit (`6202d06587c`) as boilerplate. At that point the class held only a `vector<Device*>`.

**Why it matters now: the wrapper's address is the device's identity.** The factory wraps the impl and then passes the wrapper's address into the subsystems it creates ([`mesh_device.cpp:508-517`](../../tt_metal/distributed/mesh_device.cpp)):

```cpp
    auto mesh_device = std::shared_ptr<MeshDevice>(new MeshDevice(std::move(mesh_device_impl)));

    mesh_device->pimpl_->initialize_impl(
        mesh_device.get(),
        num_command_queues,
        l1_small_size,
        trace_region_size,
        worker_l1_size,
        l1_bank_remap,
        /*minimal=*/false);
```

Who keeps that address:

- Every `MeshCommandQueue` and the `SubDeviceManagerTracker` store it.
- Every `MeshBuffer`'s backing `Buffer` holds it as `IDevice* device_`.
- Programs, global semaphores and global circular buffers, the `Device` back-pointer, ttnn's default device, the nanobind instance map, and every `MeshDevice*` returned by `Tensor::device()` also depend on it.

A move would leave all of those dangling, and would also reintroduce the null `pimpl_` state that #57279 eliminated. Making moves safe via `pimpl` would mean re-pointing every internal reference to the impl and fixing up the public `MeshDevice*` pointers on each move. That is not worth it.

**Returning a non-movable object by value already works.** `e7dade43c26` did:

```cpp
inline DeviceMesh open_device_mesh(const DeviceGrid& device_grid, const DeviceIds& device_ids) {
    return DeviceMesh(device_grid, device_ids);  // DeviceMesh(DeviceMesh&&) = delete
}
```

C++17 guaranteed elision constructs the object directly in the caller's storage, so it is never relocated. That is exactly the address stability the back-pointers need. The same commit also showed the limit: pybind could not return it by value, so Python switched to `py::init`.

### 7.2 Path A: factories return `unique_ptr`

**Signatures.**

```cpp
static std::unique_ptr<MeshDevice> create(const MeshDeviceConfig& config, /* ... */);
std::unique_ptr<MeshDevice> create_submesh(const MeshShape& shape, const std::optional<MeshCoordinate>& offset = std::nullopt);
std::vector<std::unique_ptr<MeshDevice>> create_submeshes(const MeshShape& shape);
MeshDevice* get_parent_mesh() const;
std::vector<MeshDevice*> get_submeshes() const;
```

Unit meshes are covered by `UnitMeshGroup` (§4.1). `create_unit_mesh(es)` keep their `shared_ptr` signatures as deprecated wrappers until they are removed.

**Implementation.** In the construction sites ([`mesh_device.cpp:508, 763`](../../tt_metal/distributed/mesh_device.cpp); the hidden parent at `:622` moves into `UnitMeshGroup` with prerequisite 5), `std::shared_ptr<MeshDevice>(new MeshDevice(...))` becomes `std::unique_ptr<MeshDevice>(new MeshDevice(...))`. The rest of each factory stays as it is.

That includes cleanup when initialization fails. The `unique_ptr` owns the object before `initialize_impl` runs, so a throw destroys it, and `~MeshDevice` runs `close_impl`, exactly as today.

**Effect on callers:**

| Caller code | After A1 |
| --- | --- |
| `std::shared_ptr<MeshDevice> d = MeshDevice::create(...);` | Compiles. `unique_ptr` converts implicitly to `shared_ptr`. |
| `auto d = MeshDevice::create(...); d->foo(); f(d.get()); g(*d);` | Compiles. |
| `auto d = ...; other = d;`, or passing `d` by value | Breaks. Use `std::move`, or borrow with `*d` / `d.get()`. |
| `std::map<int, std::shared_ptr<MeshDevice>> m = MeshDevice::create_unit_meshes(...);` | Unaffected: `create_unit_meshes` keeps its signature (a deprecated wrapper over `UnitMeshGroup`) until it is removed. |
| `std::vector<std::shared_ptr<MeshDevice>> s = mesh->create_submeshes(...);` | Breaks for the same reason. |

**Python.**
- With `nanobind/stl/unique_ptr.h` included, a returned `unique_ptr` hands ownership to the Python object.
- `ttnn.CreateDevices` returns a `UnitMeshGroup` bound with the mapping protocol (§4.1), or keeps its current form through the deprecation period.
- `create_submesh` keeps `keep_alive<0, 1>`.

**Limit.** The device is always on the heap. Keeping one by value needs Path B.

### 7.3 Path B: plain object

**Shape.**

```cpp
class MeshDevice : public IDevice {
public:
    explicit MeshDevice(const MeshDeviceConfig& config, const MeshDeviceOptions& options = {});
    MeshDevice(MetalEnv& env, const MeshDeviceConfig& config, const MeshDeviceOptions& options = {});
    MeshDevice(MeshDevice& parent, const MeshShape& submesh_shape, std::optional<MeshCoordinate> offset = {});
    ~MeshDevice() override;

    MeshDevice(const MeshDevice&) = delete;
    MeshDevice(MeshDevice&&) = delete;  // address is identity (§7.1)
    // ...
};
```

`MeshDeviceOptions` stands for today's positional parameters: `l1_small_size`, `trace_region_size`, `num_command_queues`, `dispatch_core_config` and `worker_l1_size`. It would be the same struct as `CreateMeshDeviceOptions` in the MetalEnv review.

**Prefer constructors over a static factory that returns by value.** `static MeshDevice create(...) { return MeshDevice(...); }` does compile, thanks to guaranteed elision. But:
- it cannot be used with `optional::emplace` or with containers;
- it changes what `auto d = MeshDevice::create(...)` deduces, which breaks every `d->` in existing callers.

Keep the name `create` for the heap form from Path A, and give value semantics through constructors.

**Initialization moves into the constructor.** Today the factory builds an empty shell and then initializes it through a pointer. A plain object has no pointer to hand back until construction finishes, so everything that follows the wrap in `MeshDeviceImpl::create` ([`mesh_device.cpp:508-532`](../../tt_metal/distributed/mesh_device.cpp)) moves into the constructor:

```cpp
MeshDevice::MeshDevice(const MeshDeviceConfig& config, const MeshDeviceOptions& options) :
    pimpl_(MeshDeviceImpl::build(config, options)) {  // today's pre-wrap half: ScopedDevices, view, impl
    try {
        pimpl_->initialize_impl(this, options);        // command queues, sub-device tracker
        pimpl_->attach_local_devices(this);            // was set_mesh_device(shared_ptr)
        pimpl_->distributed_context_->barrier();
        // device_manager: initialize_profiler(), initialize_fabric_and_dispatch_fw()
        pimpl_->init_realtime_profiler_socket(*this);  // MeshDevice& after prerequisite 3
        pimpl_->init_streaming_profiler(*this);
    } catch (...) {
        pimpl_->close_impl(this);  // ~MeshDevice does not run when a constructor throws
        throw;
    }
}
```

Notes on the sketch:
- `this` is already the final address, whether the object is on the stack, in an `optional`, in a container node, on the heap, or inside a Python object. So the subsystems capture the right address.
- No `shared_ptr` to the object exists during construction. That is why prerequisite 3 must convert the helpers used during initialization.
- `close_impl` already tolerates a partially initialized device. Today's factory reaches it through `~MeshDevice` whenever initialization throws.

**Submeshes.**
- The submesh constructor takes the parent by reference. It shares the parent's `ScopedDevices`, initializes with the parent's allocator settings, and registers itself with the parent under the parent's API lock. Its destructor unregisters it.
- The parent's child list becomes `std::vector<MeshDevice*>`.
- "The parent outlives its submeshes" is a precondition. The parent's destructor should abort with a diagnostic if children are still registered, which turns a use-after-free into a clear error. This applies to Path A too.
- For a by-value `create_submeshes`, return a `std::deque<MeshDevice>` filled with `emplace_back(parent, shape, offset)`. A `deque` never relocates its elements and can itself be moved. Otherwise keep the heap convenience, `std::vector<std::unique_ptr<MeshDevice>>`.

**Unit meshes.** Users see `UnitMeshGroup` (§4.1) in both paths. What Path B changes is how the group may hold its members internally: by value instead of through `unique_ptr`, for example in a `std::map<ChipId, MeshDevice>` filled with `try_emplace(id, PassKey{}, *this, id)`. Map nodes never relocate, so members keep stable addresses, and the group stays movable because moving a map does not move its elements.

The pass key makes a constructor callable only from inside tt_metal, while still letting `std::map` and `std::deque` construct elements in place:

```cpp
class MeshDevice : public IDevice {
public:
    class PassKey {
        PassKey() {}  // user-provided, so PassKey{} cannot bypass it via aggregate initialization
        friend class MeshDevice;
        friend class MeshDeviceImpl;
        friend class UnitMeshGroup;
    };
    // Public, but only tt_metal can produce a PassKey.
    MeshDevice(PassKey, UnitMeshGroup& group, ChipId chip);
};
```

**Python.** `nb::init<...>` constructs the C++ object directly inside the Python object. A non-movable type is fine, and its address stays stable for as long as the Python object exists:

```cpp
nb::class_<MeshDevice>(mod, "MeshDevice", nb::dynamic_attr())
    .def(nb::init<const MeshDeviceConfig&, const MeshDeviceOptions&>())
    .def(nb::init<MeshDevice&, const MeshShape&, std::optional<MeshCoordinate>>(),
         nb::keep_alive<1, 2>());  // the submesh (self) keeps its parent alive
```

- `ttnn.open_mesh_device(...)` constructs `ttnn.MeshDevice(...)`. That is what `e7dade43c26` did with `py::init` in March 2024.
- `mesh.create_submesh(...)` can stay as a Python method that calls the submesh constructor.
- `ttnn.CreateDevices` returns a `UnitMeshGroup` (§4.1).
- No binding ever has to return a `MeshDevice` by value, which nanobind cannot do for a non-movable type.

**What callers write if they opt in:**

| Need | Code |
| --- | --- |
| A device for one scope | `MeshDevice dev(cfg); run(dev);` |
| A class member | `MeshDevice dev_;`, initialized in the constructor's member-initializer list |
| Created later or reopened (gtest `SetUp()`, tt-train `AutoContext`) | `std::optional<MeshDevice> dev_;` with `dev_.emplace(cfg);` to open and `dev_.reset();` to close |
| On the heap, or shared | `std::make_unique<MeshDevice>(cfg)`, `std::make_shared<MeshDevice>(cfg)` |

**Pitfalls for callers:**
- **Declaration order.** Declare objects that borrow the device (submeshes, sockets, services) after the device, so they are destroyed first. Tensors don't matter, because they only observe the device.
- **No statics or globals.** A static device is destroyed during static destruction, possibly after the `MetalContext` it depends on. A global `shared_ptr` has the same hazard today, but a plain object makes it easier to fall into.
- **Things that need a move.** There is no reassignment or swap, no `std::vector<MeshDevice>`, and no returning a named `MeshDevice` from a helper. Use `std::optional`, `std::deque` or `std::map`, or return a temporary.

## 8. Costs of the status quo

- **`shared_from_this()` throws during destruction.** `IDevice::get_mesh_device()` throws `bad_weak_ptr` while the device is being destroyed. `buffer.cpp` has to catch that on a path called from a `noexcept` destructor ([`buffer.cpp:143-163`](../../tt_metal/impl/buffers/buffer.cpp)).
- **Unchecked null dereference in the profiler.** It calls `device->get_mesh_device().get()` and uses the result without a null check, so it breaks once a chip's mesh has gone away ([`profiler.cpp:2934`](../../tt_metal/impl/profiler/profiler.cpp), [`tt_metal_profiler.cpp:456, 457, 698, 716, 1105`](../../tt_metal/impl/profiler/tt_metal_profiler.cpp)).
- **Destruction timing follows reference counts.** Python handles, including the ones `tensor.device()` hands out, extend the device's life without anyone intending it. If `close()` was not called explicitly, `~MeshDevice` runs `close_impl`, including a distributed barrier ([`mesh_device.cpp:1042-1045`](../../tt_metal/distributed/mesh_device.cpp)), at whatever point the last reference happens to drop.
- **Ownership runs upward.** A child owns its parent, and the parent owns the context, so teardown order falls out of reference counts instead of being visible in the code.
- **The Python annotation is wrong** (§5), and the C++ ownership hides the mistake.
