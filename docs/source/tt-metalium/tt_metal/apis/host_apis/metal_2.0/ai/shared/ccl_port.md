# Porting CCL Ops to Metal 2.0 — Fabric Connections and GlobalSemaphores

> What a Metal 2.0 port adds for an op whose kernels send packets to other chips over the fabric, or synchronize through a `GlobalSemaphore`. Read it alongside the [port recipe](../port/metal2_port.md) and [TTNN integration](ttnn_factory.md). It covers only what differs for these ops. Everything else in the recipe applies unchanged, including the syntax-swap invariant: the ported kernels send the same packets to the same places as the legacy ones.

## Read this first

**Audience:** the auditor and the porter of a CCL op: anything under `ccl/` or `experimental/ccl/`, and any other op whose factory calls a `tt::tt_fabric::append_*` helper or passes a `GlobalSemaphore`'s `address()` to a kernel.

**What's covered, in one table.** Each unsupported row is an [Appendix A](../audit/metal2_audit.md#appendix-a-metal-20-feature-compatibility) entry, so the audit gates on it before a port starts.

| Legacy construct | Metal 2.0 | Where |
|---|---|---|
| Caller-supplied `GlobalSemaphore` (in `operation_attributes_t`) | `GlobalSemaphoreParameter` + binding; on a cache hit, the translated override supplies it again | [GlobalSemaphores](#globalsemaphores) |
| Op-allocated `GlobalSemaphore` (`WorkloadDescriptor::semaphores`) | `MeshWorkloadArtifacts::global_semaphores` | [Op-allocated](#op-allocated-globalsemaphores) |
| Kernel reads the semaphore's address from an argument (for `noc_semaphore_*`, fabric packet headers, helpers) | `GlobalSemaphore(global_sem::<name>).address()`; every use of the address unchanged | [Kernel side](#kernel-side-the-device-object) |
| `append_fabric_connection_rt_args` (one sender, or `FabricConnectionManager` with present-flags) | `FabricConnectionsSpec` + `FabricConnectionBinding` | [Fabric connections](#fabric-connections) |
| `append_routing_plane_connection_manager_rt_args` with one link per worker | the same, with several targets | [RoutingPlaneConnectionManager](#routingplaneconnectionmanager) |
| RPCM with auto-selected or per-route links | **Unsupported** | [Appendix A](../audit/metal2_audit.md#routingplaneconnectionmanager-with-per-route-or-auto-selected-links--unsupported) |
| Fabric mux (`FabricMuxConfig`, mux V2) | **Unsupported** | [Appendix A](../audit/metal2_audit.md#fabric-mux--unsupported) |
| `MeshSocket`, `*FusedOpSignaler`, `GlobalCircularBuffer` | **Unsupported** | [Appendix A](../audit/metal2_audit.md#appendix-a-metal-20-feature-compatibility) |

**Program shape.** CCL factories build one program per mesh coordinate (a per-coordinate `create_descriptor`, or a `WorkloadDescriptor`), so they target [`MeshWorkloadSpecFactoryConcept`](ttnn_factory.md#per-coordinate-programs-meshworkloadspecfactoryconcept). Keep one `PerCoordProgram` per coordinate, with `range = ttnn::MeshCoordinateRange(coord)`. Fabric targets resolve per device, and a device at the end of a line simply has fewer connections in its own spec. Don't merge coordinates into wider ranges: that is an optimization, not port work. **A ported-from factory that already merges** (`merge_per_device_programs = true`, e.g. `dit_fused_distributed_rmsnorm`) ports only if the merge survives whole, per [TTNN integration](ttnn_factory.md#per-coordinate-programs-meshworkloadspecfactoryconcept): splitting it adds host dispatch work on every call, and a port must be performance-neutral. One `ProgramSpec` must resolve its fabric targets on every device of its range, and a kernel's bound sides are fixed per compiled program. So the legacy range ports as one entry only on a 1D fabric (2D targets are absolute peers) and only when every device in it has the same set of sides, as on a ring. `pad_per_device_runtime_args = true` is the cue that it doesn't: the legacy merge pads the devices with fewer connections. Otherwise stop. Fabric connection args that differ per device become framework-owned per-device run args on their own.

**Gen1 only.** Fabric connection bindings are rejected on 2nd-gen architectures, and the fabric calls of the device `GlobalSemaphore` exist only in kernels built with fabric enabled on Wormhole and Blackhole.

**The Device 2.0 gate and CCL kernels.** Raw `GlobalSemaphore` primitives, the fabric's device API (including `safe_get_noc_addr` from `tt_metal/fabric/hw/inc/noc_addr.h`), and calls the Device 2.0 migration marked `legacy primitive retained` do not fail the [Device 2.0 prerequisite](../audit/metal2_audit.md#device-20-prerequisite). Other Device 1.0 idioms (raw `noc_async_*` on hand-built addresses, address generators, program semaphores by raw address) still do, including as arguments to a sanctioned call.

**Not a precedent:** `experimental/kda/qkv_causal_conv1d_silu` is already on `MeshWorkloadSpecFactoryConcept`, but it reaches the fabric through `tt::tt_metal::internal::compute_fabric_connection_rt_args`, an internal API that predates fabric connection specs. Don't copy it; use the specs below.

---

## GlobalSemaphores

A `GlobalSemaphore` is a user-managed resource, like a tensor: the program declares a parameter, a data-movement kernel binds it, and each execution supplies the object. The host headers are `workload/specification/global_semaphore_parameter.hpp`, `KernelSpec::global_semaphore_bindings` in `kernel_spec.hpp`, and `ProgramRunArgs::global_semaphore_args` in `program_run_args.hpp`.

```cpp
// Legacy (per coordinate):
writer_rt_args.push_back(static_cast<uint32_t>(operation_attributes.semaphore.address()));

// Metal 2.0:
spec.global_semaphore_parameters = {
    GlobalSemaphoreParameter{.unique_id = GlobalSemaphoreParamName{"out_ready"}}};
writer.global_semaphore_bindings = {GlobalSemaphoreBinding{
    .global_semaphore_parameter_name = GlobalSemaphoreParamName{"out_ready"}, .accessor_name = "out_ready"}};
run_params.global_semaphore_args = {
    {GlobalSemaphoreParamName{"out_ready"}, std::cref(operation_attributes.semaphore)}};
```

- **One parameter per distinct semaphore**, named for its role (`out_ready`, `barrier`). The address RTA, its index constant (`kSenderWriterSemaphoreAddr`), and any `// smuggled-rta-ok` comment on it drop with it ([Dropped Plumbing](../port/metal2_port.md#dropped-plumbing)).
- **A semaphore indexed out of a vector** (`semaphore.at(i)`) is still one parameter per element the kernel reads, each bound to `std::cref(semaphore.at(i))`. When the count isn't fixed per program, use one parameter per element of the count you were given; the count is already in the program hash, or the legacy op was wrong (record it, don't fix it).
- **A semaphore the kernel selects at run time** (`get_common_arg_val<uint32_t>(semaphore_0 + direction)` over a contiguous block of address args) becomes one parameter per slot and an array of their addresses indexed the same way: `const uint32_t sem_addrs[] = {GlobalSemaphore(global_sem::sem_0).address(), GlobalSemaphore(global_sem::sem_1).address()};` then `sem_addrs[direction]`. The selection expression is unchanged. Selection that isn't a plain index over the block has no mapping: stop.
- **An optional caller semaphore** (passed as address `0` plus a flag when absent) is a conditionally bound resource ([rule 6](../port/metal2_port.md#kernel-side-whitelist)): bind it only when present, and gate the kernel's uses on a host-emitted define (there is no `global_sem::get_token_if_present`). That makes presence a compile-time property, so it ports only if presence is already in the program hash (the default hash covers an `std::optional` attribute; check a custom hash). Otherwise stop.
- **Only data-movement kernels bind a `GlobalSemaphore`.** A compute kernel that reads one is [Appendix A](../audit/metal2_audit.md#globalsemaphore-in-a-compute-kernel--unsupported).
- **The argument is a non-owning reference,** resolved to the address when the run args are set. The object must outlive the program's executions, which a caller-supplied or parked semaphore does.

### Caller-supplied: rebind it in the override

Most CCL ops take their semaphores from the caller (`operation_attributes_t::semaphore`, a vector, or nested in a fused op's params). The legacy factory keeps them out of the program hash and patches the address in `override_runtime_arguments`, because a cache hit may bring a different semaphore. **Translate that patch, don't drop it:** the ported override returns `global_semaphore_args` alongside its other run args, on every hit.

```cpp
// Legacy override_runtime_arguments (per coordinate): patch the address in place.
args[kSenderWriterSemaphoreAddr] = operation_attributes.semaphore.address();

// Metal 2.0 override_runtime_arguments(attrs, tensor_args, ret, range):
tt::tt_metal::ProgramRunArgs run_args;
// ... the tensor args, as for any override (the override owns them) ...
run_args.global_semaphore_args = {
    {GlobalSemaphoreParamName{"out_ready"}, std::cref(attrs.semaphore)}};
return run_args;
```

The adapter accepts a binding to a caller semaphore only from a factory that has an override; without one it throws at the cache miss, because the hit would keep the miss's address. **It checks only that the override exists, not what it returns:** an override that forgets a caller semaphore keeps the miss's address silently. Put every caller-semaphore parameter in `global_semaphore_args` on every hit. So:

- **The legacy factory patches the semaphore in its override** → the case above. No device-op edit.
- **The legacy factory doesn't re-apply the semaphore on a hit** → the audit gates on it ([Appendix A](../audit/metal2_audit.md#caller-globalsemaphore-not-re-applied-on-a-cache-hit--unsupported)). If the legacy hit takes the fast path, the op keeps a stale address today and the fix is the op owner's; if it re-runs the factory (a descriptor factory with no `Buffer*` bindings), the op is correct and the port would have to add an override, which the recipe doesn't cover yet.
- **Don't add `operation_attributes_t::global_semaphores`.** The adapter rebinds semaphores held there by itself, without an override, but adding the field is a device-op edit the port doesn't make. If the device op already has it, bind to its elements and the adapter does the rest.
- **The WRAP shape.** Some factories exist only to make that patch: a `legacy (MeshWorkload)` struct whose `create_mesh_workload` delegates to `DescriptorMeshWorkloadAdapter<Inner>` and whose `override_runtime_arguments` re-applies semaphore addresses (e.g. `SemaphoreArgPatch`, `record_semaphore_arg_slot`, `apply_semaphore_arg_patch`). Port the inner factory's `create_workload_descriptor` (or `create_descriptor`) as the `create_mesh_workload_artifacts`, move the wrapper's patch into the Metal 2.0 override as above, and give the ported factory the wrapper's struct name so `program_factory_t` is unchanged. The wrapper body, the inner legacy entry point, and the patch helpers are dropped plumbing.

### Op-allocated GlobalSemaphores

A factory that allocates a semaphore in its body (`ttnn::global_semaphore::create_global_semaphore`) parks the result where the program cache keeps it. Legacy uses `WorkloadDescriptor::semaphores`, so these factories are already per-coordinate and target `MeshWorkloadSpecFactoryConcept`:

```cpp
// Legacy:
auto sem = ttnn::global_semaphore::create_global_semaphore(mesh_device, cores, 0, BufferType::L1);
tt::tt_metal::distributed::Synchronize(mesh_device, std::nullopt, sub_device_ids);
workload_descriptor.semaphores.push_back(sem);
// ... sem.address() into the kernels' runtime args ...

// Metal 2.0:
MeshWorkloadArtifacts artifacts;
artifacts.global_semaphores.push_back(
    ttnn::global_semaphore::create_global_semaphore(mesh_device, cores, 0, BufferType::L1));
tt::tt_metal::distributed::Synchronize(mesh_device, std::nullopt, sub_device_ids);
// Fill global_semaphores completely before taking references: growing the vector moves its elements.
const auto& sem = artifacts.global_semaphores.back();
// ... per coordinate: a GlobalSemaphoreParameter, bindings, and std::cref(sem) in run_params ...
```

- **Keep the allocation call and the `Synchronize` exactly as legacy has them,** every argument included. Don't swap `ttnn::global_semaphore::create_global_semaphore` for `tt::tt_metal::CreateGlobalSemaphore`: they write the initial value through different command queues.
- **Parked semaphores keep their address,** so a cache hit needs nothing for them. A factory that mixes both kinds rebinds only the caller's in its override.
- **Conditional allocation** (allocate only when the caller passed none) ports as written: the same condition picks which object the `std::cref` refers to.

## Kernel side: the device object

The kernel constructs a `GlobalSemaphore` from its binding token and takes the address from it. Every use of the address stays exactly as it is. Include `api/dataflow/global_semaphore.h`.

```cpp
// Legacy:
uint32_t out_ready_addr = get_arg_val<uint32_t>(arg_idx++);

// Metal 2.0:
uint32_t out_ready_addr = GlobalSemaphore(global_sem::out_ready).address();
```

- **Change nothing downstream.** Packet headers, `fabric_*` calls, helpers in or outside the op directory, `noc_semaphore_*` on the address: all stay as legacy has them. `address()` returns the value the host's `GlobalSemaphore::address()` put in the legacy argument, so the kernel computes exactly what it did.
- **Don't convert uses to the object's methods in the port.** The methods (`wait`, `up`, `set_multicast`, the fabric `up` overloads, …) are the form for new code, but converting is not a syntax swap: some add a fence or a barrier the legacy code didn't have, and the fabric ones fill packet headers differently. That's a separate, measured change, not port work.
- **One construction per semaphore,** where the legacy kernel read the argument, keeping the legacy variable's name.
- **Don't call `GlobalSemaphore(uint32_t)`** in a ported kernel. That constructor is for non-Metal-2.0 kernels, and reaching it means the address is still an argument.
- **Program semaphores are separate.** A program `Semaphore` (`get_semaphore(id)`, `SemaphoreSpec`) ports per the recipe; only the `GlobalSemaphore` rows above change.

## Fabric connections

A fabric connection is one kernel instance's use of one fabric router. The program declares a matrix of them, a data-movement kernel binds it, and the runtime allocates the connection's semaphores and writes its args. These are framework-owned args, so the legacy connection RTAs disappear from the kernel's argument list. The declarations are experimental and live in advanced options: `ProgramSpec::advanced_options.fabric_connections` and `KernelSpec::advanced_options.fabric_connection_bindings` (`experimental/metal2_host_api/fabric_connections_spec.hpp`, `advanced_options.hpp`). Read the header's PLACEMENT and KERNEL ACCESS notes before you start.

**Key facts the mapping rests on:**
- **One spec, one link per node.** A `FabricConnectionsSpec` has `targets` and `links` (`Table<NodeCoord, uint32_t>`: each worker core and the link it uses for *every* target). Each node of a binding kernel must be in `links`, and each node in `links` must run a binding kernel. A kernel whose nodes don't all connect is split into a binding `KernelSpec` and a non-binding one.
- **Targets by fabric mode.** `Fabric1DTargets` (routing directions) on a 1D fabric; `Fabric2DTargets` (peer `FabricNodeId`s) on a 2D fabric. The wrong kind is rejected, so the factory picks by `tt::tt_fabric::is_2d_fabric_config(tt::tt_fabric::GetFabricConfig(*mesh_device))`.
- **The binding token** (`fabric::<accessor_name>`) has one slot per target, in `targets` order. A kernel source bound on some devices and not others looks it up with `fabric::get_token_if_present<"<accessor_name>">()`, which is `nullptr` where it isn't bound.
- **`FABRIC_2D` is defined for a binding kernel exactly when the fabric is 2D.** A legacy define the op sets by hand is harmless; leave it. Legacy `append_fabric_connection_rt_args` never defined it, so a kernel with `#if defined(FABRIC_2D)` code that the legacy factory compiled *without* the define on a 2D fabric would switch branches under the port: stop and report it.

### Resolving a legacy peer to a target

Legacy helpers take a destination `FabricNodeId`. On a 2D fabric that is the target as is. On a 1D fabric, legacy `append_fabric_connection_rt_args` picks **the first direction, in the order N, S, E, W, Z, in which the peer is an immediate neighbour**: a workaround for routing tables without wraparound links. Reproduce exactly that. Don't use `pipeline_get_forwarding_direction`, which follows the routing tables and can choose the long way round on a ring's wrap.

```cpp
// Same resolution as legacy append_fabric_connection_rt_args (tt_metal/fabric/fabric.cpp).
tt::tt_metal::experimental::FabricTargets fabric_target(
    const tt::tt_metal::distributed::MeshDevice& mesh_device,
    tt::tt_fabric::FabricNodeId self,
    tt::tt_fabric::FabricNodeId peer) {
    using tt::tt_fabric::RoutingDirection;
    if (tt::tt_fabric::is_2d_fabric_config(tt::tt_fabric::GetFabricConfig(mesh_device))) {
        return tt::tt_metal::experimental::Fabric2DTargets{peer};
    }
    for (const auto direction :
         {RoutingDirection::N, RoutingDirection::S, RoutingDirection::E, RoutingDirection::W, RoutingDirection::Z}) {
        const auto neighbours = tt::tt_fabric::pipeline_get_chip_neighbors(mesh_device, self, direction);
        const auto it = neighbours.find(*peer.mesh_id);
        if (it != neighbours.end() && std::ranges::find(it->second, peer.chip_id) != it->second.end()) {
            return tt::tt_metal::experimental::Fabric1DTargets{direction};
        }
    }
    TT_THROW("Could not find any forwarding direction from {} to {}", self, peer);
}
```

Put it in the factory file's anonymous namespace. Headers: `<tt-metalium/experimental/fabric/fabric.hpp>` and `<tt-metalium/experimental/fabric/pipeline_builder.hpp>`.

### One sender, or a FabricConnectionManager

The dominant legacy pattern: per worker core, a present-flag and the connection args for each side, read on device by `FabricConnectionManager::build_from_args`.

```cpp
// Legacy (per coordinate, per worker core `core` on link `link`):
writer_rt_args.push_back(forward_node.has_value());
if (forward_node) {
    tt::tt_fabric::append_fabric_connection_rt_args(
        *mesh_device, self_node, *forward_node, link, desc, core, writer_rt_args);
}
writer_rt_args.push_back(backward_node.has_value());
if (backward_node) { /* the same, with *backward_node */ }
```

Metal 2.0 gives **each side its own one-target spec**, declared and bound only on the devices that have that side. Each side's `links` lists exactly the worker cores whose legacy present-flag for that side was 1, with the link the legacy loop gave them. When every writer core opens both sides, both specs share one table, as below:

```cpp
using namespace tt::tt_metal::experimental;
tt::tt_metal::Table<tt::tt_metal::NodeCoord, uint32_t> links;   // writer cores that open this side, and their links
for (uint32_t link = 0; link < num_links; ++link) { links.emplace(worker_cores[link], link); }

auto add_side = [&](const char* name, tt::tt_fabric::FabricNodeId peer) {
    spec.advanced_options.fabric_connections.push_back(FabricConnectionsSpec{
        .unique_id = FabricConnectionsSpecName{name},
        .targets = fabric_target(*mesh_device, self_node, peer),
        .links = links});
    writer.advanced_options.fabric_connection_bindings.push_back(FabricConnectionBinding{
        .fabric_connections_spec_name = FabricConnectionsSpecName{name}, .accessor_name = name});
};
if (forward_node) { add_side("forward", *forward_node); }
if (backward_node) { add_side("backward", *backward_node); }
```

```cpp
// Legacy kernel:
auto fabric_connection = FabricConnectionManager::build_from_args<
    FabricConnectionManager::BuildFromArgsMode::BUILD_AND_OPEN_CONNECTION_START_ONLY>(arg_idx);

// Metal 2.0 kernel:
auto fabric_connection = FabricConnectionManager::build_from_optional_bindings<
    FabricConnectionManager::BuildFromArgsMode::BUILD_AND_OPEN_CONNECTION_START_ONLY>(
    fabric::get_token_if_present<"forward">(), fabric::get_token_if_present<"backward">());
```

- **Cores that open different sides** (one worker per direction, e.g. `all_reduce_async`): each side's spec lists only its own cores, and the writer splits into one `KernelSpec` per distinct set of sides (same source; each binds the sides its cores open). The kernel's `build_from_optional_bindings` line works unchanged in every split. Sharing one `links` table here would put two connections on one router, which the exclusive-link check rejects.
- **Use two one-target specs even on a ring,** where both sides always exist. It's the same kernel code on every device, and it also covers a side that uses a different link from the other.
- **A single sender:** `WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(arg_idx)` → `WorkerToFabricEdmSender::build_from_binding(fabric::<name>)`, with a one-target spec. A kernel that reads its own present-flag first (`if (get_arg_val<uint32_t>(i++)) { …build_from_args… }`) tests the token instead: `if (const auto* t = fabric::get_token_if_present<"<name>">()) { …build_from_binding(*t)… }`.
- **Only the connection args go.** The flags and connection words leave the kernel's runtime args; every other argument becomes named per [rule 4](../port/metal2_port.md#kernel-side-whitelist), so the positional cursor that used to skip past the connection args goes too.
- **Several worker cores share one spec** (one entry each in `links`). A kernel that opens connections on some of its cores and not others is split per the header's rule.

### RoutingPlaneConnectionManager

```cpp
// Legacy:
tt::tt_fabric::append_routing_plane_connection_manager_rt_args(
    *mesh_device, self_node, dst_nodes, {link}, desc, kernel_id, core, rt_args, api_type);
// kernel: open_connections(mgr, num_connections, arg_idx);

// Metal 2.0: one spec, one target per route in the legacy order.
// kernel: open_connections<NOC>(mgr, fabric::<name>);
```

- **Kernel calls keep their template arguments:** `open_connections<NOC>(mgr, n, idx)` → `open_connections<NOC>(mgr, fabric::<name>)`; `RoutingPlaneConnectionManager::build_from_args<mode, NOC>(idx, n)` → `build_from_binding<mode, NOC>(fabric::<name>)` (its default mode only builds, while `open_connections` opens).
- **Targets** are the legacy `dst_nodes` in the same order, resolved per [above](#resolving-a-legacy-peer-to-a-target); slot `i` of the token is route `i`. The direction overload (a list of `eth_chan_directions`, skipping absent neighbours, returning the count it built) becomes the neighbours present on that device, in the same order: directions on a 1D fabric, the neighbour nodes on a 2D fabric. The legacy count is the token's `num_targets()`. A device with no routes doesn't declare the spec (a spec with no targets is rejected).
- **Links:** the port carries only a single explicit link (`{link}`, or the same value for every route). An empty list (auto-select per route) or different links per route is [Appendix A](../audit/metal2_audit.md#routingplaneconnectionmanager-with-per-route-or-auto-selected-links--unsupported).
- **`sem_args_are_l1_addresses = true`** (the caller-supplied-semaphore variant) has no Metal 2.0 form: stop.
- The legacy `api_type` sets an `API_TYPE_*` define that no ttnn kernel reads; it drops.

### What the runtime checks that legacy didn't

The program now fails at build time where legacy may have silently misrouted:
- **No neighbour in a direction** (an edge device given that target). The fix is the per-coordinate spec listing only the sides it has, as above.
- **Two connections of one program on the same (direction, link) of a device.** Legacy configurations reach this on a 2-device ring (forward and backward are the same chip), with a duplicated cluster axis, or with more links than a direction forwards. **Don't set `fabric_connection_options.disable_exclusive_fabric_link_check`** to make it pass. The header allows it only when the op's kernels serialize their use of the shared router; whether they do is the op owner's call, not a port decision. Report the configuration as a finding.
- **Semaphores per node:** each target uses two program semaphores on its node. Legacy allocated the same ones; a failure here means the spec's `links` covers more nodes than legacy connected.

## Unsupported today

Summarized here; the recognition signals are in [Appendix A](../audit/metal2_audit.md#appendix-a-metal-20-feature-compatibility).

- **Fabric mux**, in any form. A factory that chooses the mux or a direct connection at run time is gated whole: the port is one factory, and splitting it is a device-op change.
- **Different links for the targets of one RPCM**, including auto-selected links.
- **A `GlobalSemaphore` in a compute kernel.**
- **A caller `GlobalSemaphore` that no override re-applies on a cache hit.**
- **`MeshSocket`**: the socket's config buffer address is a runtime arg, and Metal 2.0 has no parameter for it.
- **`*FusedOpSignaler`**: fused matmul+CCL ops read runtime args across ops by position. The Metal 2.0 matmul forks reject a fused op on purpose.
- **`GlobalCircularBuffer`**, already gated.
