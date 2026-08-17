# CCL Helpers PR (#46413) — response to team-meeting review

Addresses the three tt-metal-side concerns from the review. **These are proposals, not yet
implemented+built:** the machine this was drafted on (bh-34) has no `clang-20` toolchain and its
device is down (firmware mismatch), so the C++ below has not been compiled or run here. The plan is
grounded in a full read of the current helper (`ccl_helpers_dataflow.hpp`/`.inl`, the host header,
and the `point_to_point` factory); file:line cites are the current PR head (`d53c7013`).

---

## #1 — reduce the number of function TYPES (not the steps)

Two consolidations remove distinct types without removing any sequential step:

1. **Collapse the unicast + multicast atomic-inc into one inc type (highest value, low risk).**
   `arm_inc` → `AtomicIncChannel::inc(addr)` and `arm_multicast_inc` → `MulticastIncChannel::multicast_inc(addr)`
   are near-duplicates: both `set_state<Val|Flush>` on a fresh header (`.inl:242` vs `:269`) and both
   issue `<DstAddr>` with an identical `uint64_t remote_sem_noc_addr` (`.inl:252` vs `:280`). The only
   differences are cast mode (`fabric_unicast_*` vs `fabric_multicast_*`) and that the multicast arm
   takes its own route. Since cast mode is fixed at arm time, ONE `IncChannel` with a single
   `inc(addr)` can serve both — the arm variant bakes multicast-ness into the header. Removes **1
   channel type + 1 arm type** and unifies two issue-method names that already share a signature. The
   lost compile-time "this is multicast" tag carries no safety (both just take a NOC addr; route is
   already bound). Callers (`all_gather` writer 192-213, `point_to_point` writer_send:60) keep working.

2. **Collapse the one-shot `signal` / `signal_once` family 4 → 2.** `signal(num_hops,…)` (`hpp:409`) and
   `signal_once(num_hops,…)` (`hpp:424`) are thin wrappers that only call `unicast_route(n)` before the
   route-info forms (`:406`/`:416`). `unicast_route()` is already public (`hpp:226`), so drop the two
   `num_hops` overloads; the one caller (`reader_receive.cpp:36`) passes `unicast_route(n)` — one token.

Leave separate (genuinely different shapes, merging would burden the common path): `arm_unicast_write`
vs `arm_scatter_write`; `write` vs `write_page`. `DirectConn`/`MuxConn` are already the consolidated
`ConnT`-template model the review wants — the atomic-inc merge should follow that same pattern.

**Net: −1 channel type, −1 arm type, −2 one-shot overloads, 0 steps removed.**

---

## #2 — host helpers unclear; the migrated factory should bridge op-author ↔ fabric

The kernel helper is a clean typestate; the host helper is today a set of **à-la-carte packers**, so the
factory author is still forced to know fabric internals. In `point_to_point/device/host/send_program_factory.cpp`:

- **Clear:** `ccl_packet_dims(...)` (send:34-36) and `ccl_dm_route(...)` (send:93-94) — good bridges.
- **Unclear #1:** the author hand-builds a `std::vector<uint32_t>` of **9 scalars in a hardcoded order**
  (send:145-155) that must byte-match the kernel's `get_arg_val` indices (writer_send.cpp:19-27) — two
  files kept in sync by hand, no shared struct.
- **Unclear #2:** `append_ccl_fabric_rt_args(...)` appends the connection block *starting at index 9*
  (send:160-161); the kernel re-hardcodes `size_t conn_arg_idx = 9` (writer_send.cpp:33). A magic
  number duplicated host↔kernel.
- **Unclear #3:** the author then hand-promotes the plain vector into a `KernelDescriptor::RTArgList`,
  knowing which slots are `Buffer*`/semaphore bindings (idx 0 buffer, idx 8 semaphore) for cache-hit
  address patching (send:163-169).

**The gap:** the author naturally knows tensors + mesh coords + topology + page accounting; they should
NOT have to know the RT-arg *wire layout*, the has_fwd/has_bwd flag encoding, which slots are bindings,
or the magic `conn_arg_idx = 9`.

**Bridge (proposed):** a single per-direction args-builder that owns the whole layout —

```cpp
// host — the SOLE definition of the host↔kernel arg layout
KernelDescriptor::RTArgList build_ccl_send_writer_args(
    const Buffer* in, const Buffer* out, const PacketDims&, const DmRoute&,
    const GlobalSemaphore&, const CoreCoord&, /* link_idx, etc. */);
```

returning a ready `RTArgList` with `Buffer*`/semaphore bindings already placed and the connection block
already appended at the right offset — paired with a kernel-side `FabricStreamSender` ctor that knows
its own offset instead of hardcoding `= 9`. This subsumes send steps 3-5, deletes the magic index and
the two-file sync, and leaves the author passing only op-author-level inputs. (Also fix the host
header's stale line 22-24 "intentionally not Python-bound" — contradicts #3.)

---

## #3 — pybind the host helpers (generated ops are Python host code)

Binding site: `ttnn/cpp/ttnn-nanobind/fabric.cpp`, `ttnn::fabric::bind_fabric_api(nb::module_&)` (the
`ttnn._ttnn.fabric` submodule, `__init__.cpp:258/272`); add `mod.def(...)` at the end + `#include
".../ccl_helpers_dataflow_host.hpp"`. POD returns follow the `CclPacketDims` `nb::class_` precedent.

| Host helper | Status | Binding note |
|---|---|---|
| `ccl_packet_dims`, `ccl_dm_route` | bound on `ccl_help_eval` (commit `5b8156d`) but **NOT on the current `ccl_help` PR head** — carry them over | pure-compute, already have the `nb::class_` POD pattern |
| `make_ccl_semaphore` | C++-only → bind | one-liner: `mod.def("make_ccl_semaphore", &...::make_ccl_semaphore, nb::arg("mesh_device"), nb::arg("initial_value")=0)`; `GlobalSemaphore` already bound |
| `append_ccl_fabric_rt_args` | C++-only → bind | highest value (owns the has_fwd/has_bwd footgun). Don't expose the out-param vector — follow `setup_fabric_connection`'s lambda that builds + *returns* a `std::vector<uint32_t>`; note it *appends*, so take the in-progress list and return it extended. All arg types already bindable (`FabricNodeId`, `CoreCoord`, `ProgramDescriptor&`). |
| `append_ccl_line_route_ct_args` | C++-only, **templated** → bind via wrapper | wrap a lambda over 4 `std::vector<uint32_t>` returning the packed `ct_args` (templates can't bind directly); pure concatenation, value is owning the fwd/bwd × uni/mcast order in one place |

If #2's `build_ccl_*_args` bridge lands, it becomes the primary thing to pybind (a generated Python op
calls it directly), and `append_ccl_fabric_rt_args` binding is subsumed.

---

## Blocker / next step

The C++ above needs a `clang-20` build to compile+verify — unavailable on bh-34 (only clang-14), and the
device is down (firmware 19.9 vs 19.12). On a machine with the toolchain I'll implement all three, build,
and (for #1/#2) re-run the migrated `point_to_point`/`all_gather` kernels on the sim to confirm no
regression, then push the real diff to this PR.
