# CCL Helpers PR (#46413) — team-meeting review: IMPLEMENTED

The three tt-metal-side concerns from the review are implemented on this branch
(`wransom/ccl_help_review`, based on the PR head `d53c7013`), built with clang-20, and verified on
the multichip craq-sim (see Verification below). Commits:

- `119b39b` — pybind `ccl_packet_dims` + `ccl_dm_route` (carried over from the eval branch)
- `783aae1` — #1: consolidate helper function types
- `dde0032` — #2: host helper owns the fabric arg layout end-to-end; #3: pybind `make_ccl_semaphore`

---

## #1 — fewer function TYPES, same steps (`783aae1`)

- **One armed atomic-inc channel.** `AtomicIncChannel<ConnT, IncCast>` (Unicast default / Multicast)
  replaces the `AtomicIncChannel` + `MulticastIncChannel` pair. `arm_inc(val)` and
  `arm_inc(mcast_route, val)` are overloads (`arm_multicast_inc` is gone) and the issue verb is
  `inc(addr)` for both casts (`multicast_inc` is gone). Cast mode is baked into the header at arm
  time; issue dispatch is `if constexpr` — zero cost. The two paths were near-duplicates (same
  `set_state<Val|Flush>` + `with_state<DstAddr>` shape, differing only in the unicast vs multicast
  fabric entry point).
- **One spelling per one-shot.** `signal(route, …)` / `signal_once(cursor, …, route, …)` drop their
  `num_hops` twins; `unicast_route(n)` is public, so the caller writes `signal(unicast_route(n), …)`.

Net: −1 channel type, −1 arm name, −1 issue name, −2 one-shot overloads; zero sequential steps
removed. Left separate on purpose: `arm_unicast_write` vs `arm_scatter_write` (genuinely different
issue shapes) and `write` vs `write_page` (a step-level convenience, not a distinct type).

## #2 — host helper bridges op-author ↔ fabric (`dde0032`)

The factory previously hand-laid a 9-scalar arg vector that had to byte-match the kernel's
`get_arg_val` indices, duplicated a magic `conn_arg_idx = 9` in host AND kernel, and hand-promoted
`Buffer*`/semaphore bindings after the fact.

- `append_ccl_fabric_rt_args` → **`build_ccl_fabric_rt_args`**: RETURNS the connection block, and
  the block goes **FIRST** in the kernel's runtime args. The kernel consumes it with a cursor from 0
  (the `FabricStreamSender` ctor advances it) and reads the op's args after — **the magic offset is
  gone from both sides** and the helper is the sole owner of the wire layout.
- The `point_to_point` factories now push op args as their natural types straight into the
  `RTArgList` (a `Buffer*` records the cache-hit binding) — the placeholder-then-promote pass is
  deleted in both factories.

So the op author supplies only what they naturally know — buffers, mesh coords, topology, page
accounting (`ccl_packet_dims`), the route (`ccl_dm_route`) — and never the wire layout.

## #3 — pybind the host helpers (`119b39b` + `dde0032`)

Generated ops assemble their `MeshProgramDescriptor` from Python host code, so the host surface is
now bound under `ttnn._ttnn.fabric`:

| Python | What it owns |
|---|---|
| `ccl_packet_dims(dtype, page_size, num_pages, alignment)` → `CclPacketDims` | packet framing; the bf16 `bit_floor` + both packing regimes |
| `ccl_dm_route(mesh_device, src, dst, topology)` → `CclDmRoute` | 1-D route; the fwd/bwd sign reversal + ring short-way |
| `make_ccl_semaphore(mesh_device, initial_value=0)` → `GlobalSemaphore` | allocation + the cache-miss cross-device `Synchronize`; park on `MeshProgramDescriptor.semaphores` |
| `setup_fabric_connection(...)` (pre-existing) | the fabric-connection runtime args from Python |

The host header's stale "intentionally not Python-bound" note is corrected, and its authoring guide
now describes the conn-block-first contract.

---

## Verification

- Host build: clean (clang-20; the refactored factories + pybind compile in-tree).
- Python: `import ttnn` from this build; all three bindings present and callable.
- Multichip craq-sim (results):
  - `point_to_point` nightly suite: **27 passed, 0 failed** (BH 8xP150 sim, mesh (2,4), FABRIC_1D,
    12 min) — exercises the unicast `arm_inc`/`inc`, `signal`, and the conn-block-first layout in
    BOTH factories/kernels. Includes `cache_hit_with_output_tensor`, which validates the
    conn-first `Buffer*` binding positions across program-cache hits. The 2 deselected cases are
    the `with_device_delay` variants — they busy-wait on device clock cycles, which never
    terminates on a ~kHz simulator (sim limitation, not a helper issue).
  - `all_gather_async` `gather_dim_0 × fabric_linear` barrier case: **passed** (WH T3K all-MMIO
    sim, (1,8), 68 s) — exercises the multicast `arm_inc` overload + unified `inc`, scatter, and
    MuxConn through the rewritten `minimal_default_writer`.

## Note on the pybind carry-over

`ccl_packet_dims`/`ccl_dm_route` were bound on the eval branch (`5b8156d`) but that commit was NOT
on the PR head — carried over here so the binding ships with the helper PR rather than the eval
scaffolding.
