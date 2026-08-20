# Metal 2.0 ProgramSpec (pybound) — cross-core findings

Box: Blackhole P150b (Gen1), branch `mstaletovic/agent_eval`, `ttnn.CONFIG.validate_program_args = True`.

Probes: `tests/ttnn/unit_tests/operations/metal2_play/cross_core/`
(`test_cross_core.py`, `xc_specs.py`, `kernels/`). **21/21 green**, all on device.

What actually works, end to end and verified numerically:

| probe | topology | result |
|---|---|---|
| `test_ring_unicast_rotates` | N-core ring, each core NoC-writes into its successor's DFB entry | PASS at (cols,tiles/core,entries) = (4,1,1) (4,4,2) (8,2,2) (8,8,4) |
| `test_raw_mcast_broadcasts` | 1 sender → row, hand-rolled `sem::` + rect | PASS at cols=4, 8 |
| `test_family_mcast_broadcasts` | same, via `ttnn.mcast_spec.McastFamily` | PASS at cols=4, 8 |
| `test_dfb_address_*` | measured L1 addresses per node | see BLOCKED-2 |

Cross-core data movement is fully expressible from Python. Every friction below is about *how*.

---

## BLOCKED

### BLOCKED-1 — There is no cross-node DFB, and no way to fake one from the host

The wiring you actually want for "core 0 reads DRAM and feeds cores 1..N-1" is: producer bound
where the data is read, consumer bound where it is used. That is exactly what the validator
rejects.

```python
# producer = sender kernel on node (0,0) only; consumer = writer on the whole row
sender.dfb_bindings  = [ttnn.producer_of("recv", "recv")]   # WorkUnitSpec: node (0,0)
writer.dfb_bindings  = [ttnn.consumer_of("recv", "recv")]   # WorkUnitSpec: nodes (0..3,0)
```
```
TT_FATAL: Local DFB 'recv' is malformed at node 1-0: 0 producer instance(s) (none) and
1 consumer instance(s) ('writer'). A local DFB lives in shared SRAM on each node, so every node it
is instantiated on must run exactly one producer and one consumer kernel instance. This node has a
consumer but no producer — ensure a producer kernel covers it (via its WorkUnitSpec membership).
```
The pure two-node FIFO (producer on node 0, consumer on node 1) fails the mirror-image check
(`... malformed at node 0-0: 1 producer instance(s) ('sender') and 0 consumer instance(s) (none)`).

`CrossNodeDataflowBufferSpec` exists in `dataflow_buffer_spec.hpp` as a sketch, is explicitly
"not yet supported", and is **not pybound at all** — `dir(ttnn._ttnn.program_spec)` has no
`CrossNode*` entry, and `ProgramSpec`'s nanobind ctor hardcodes `.cross_node_dataflow_buffers = {}`
(`ttnn/cpp/ttnn-nanobind/program_specs.cpp:~470`). So it is not reachable even by accident.

**Verdict: the one abstraction that would make cross-core movement first-class does not exist; every cross-core op is a hand-built protocol over a node-local FIFO.**

### BLOCKED-2 — You cannot ask the host for a buffer's address, so a peer's buffer has no name

To NoC-write into a peer's DFB you need `(virtual_x, virtual_y, addr)`. The coords are fine —
`device.worker_core_from_logical_core()` on the host, `UnicastEndpoint` in the kernel. `addr` is the
problem: there is no host query for a DFB's L1 address in the spec path (the descriptor path's
`ttnn.get_cb_address` has no ProgramSpec analog), and the kernel has no `dfb::recv.address_on(x,y)`.

The only address you can obtain is **your own**: `dfb_recv.get_write_ptr()`. So every cross-core
kernel here writes its own local address into a remote NoC address and hopes:

```cpp
dfb_recv.reserve_back(1);
const uint32_t entry = dfb_recv.get_write_ptr();          // MY address...
noc.async_write(stage, peer, tile_bytes, {.offset_bytes = 0},
                {.noc_x = next_x, .noc_y = next_y, .addr = entry});   // ...used as THEIRS
```

I measured whether that is even true (`test_dfb_address_*`, node 0 holding `{recv, stage}` and
nodes 1..3 holding `{pad, recv, stage}` with `pad` declared first):

```
[uniform node sets]   recv=0x1b380 ×4      stage=0x1b400 ×4
[differing node sets] recv=0x1b580 ×4      stage=[0x1b680, 0x1b600, 0x1b600, 0x1b600]
```

- **DFB base addresses are program-GLOBAL.** `recv` sits at the same address on the lean node as on
  the fat nodes, even though the lean node has no `pad` beneath it. Reusing your own DFB address as
  a peer's is *correct* — but by an unstated, unchecked global-allocator property, not by contract.
- **Scratchpad base addresses are PER-NODE.** `stage` is at `0x1b680` on node 0 and `0x1b600` on
  nodes 1..3. The identical trick applied to a `Scratchpad` silently writes into the wrong place on
  an asymmetric grid.

Two node-local resources, opposite placement rules, nothing in the API or the headers saying so.

**Verdict: cross-core addressing rests on a measured, undocumented invariant that holds for DFBs and does NOT hold for scratchpads.**

### BLOCKED-3 — A non-zero semaphore initial value is not reachable from Python

`SemaphoreAdvancedOptions::initial_value` exists in C++ (deprecated, but there). The nanobind
`SemaphoreSpec` exposes only `unique_id` and `target_nodes` — no `advanced_options` at all
(`program_specs.cpp:142-152`), confirmed by `hasattr(ttnn.SemaphoreSpec, "advanced_options") == False`.

That removes the natural cross-core credit primitive: a semaphore pre-loaded with `num_entries`
which a remote producer decrements. Every credit protocol has to be re-expressed as
"receiver signals first, sender waits" — which is what my ring does, and what mcast_pipe's
`pre_handshake` does. It works, but it is the workaround, not the design.

**Verdict: the flow-control primitive is present in C++ and absent in Python.**

### BLOCKED-4 — Work units may not overlap, so a grid-wide kernel must be re-listed per role

The obvious factoring — one work unit per role, plus one grid-wide unit for the writer everyone
shares — is illegal:

```python
work_units = [
    ttnn.WorkUnitSpec(name="send",  kernels=[K_SENDER],   target_nodes=node0),
    ttnn.WorkUnitSpec(name="recv",  kernels=[K_RECEIVER], target_nodes=rest),
    ttnn.WorkUnitSpec(name="write", kernels=[K_WRITER],   target_nodes=whole_row),   # <-- overlaps
]
```
```
TT_FATAL: WorkUnitSpecs 'send' and 'write' overlap in target nodes
```
(`program_spec.cpp:1720`; the check is pairwise `nodes_intersect` over all work units.)

The legal form is to list `writer` in **every** role's work unit:
```python
ttnn.WorkUnitSpec(name="send", kernels=[K_SENDER,   K_WRITER], target_nodes=node0),
ttnn.WorkUnitSpec(name="recv", kernels=[K_RECEIVER, K_WRITER], target_nodes=rest),
```
This works (`test_raw_mcast_broadcasts`), so **a kernel in two work units is fine and
`target_nodes` composes to the union** — the DFB, semaphore and scratchpad node sets all came out
as the union, as documented. But the reading is inverted from what "work unit" suggests: a work
unit is a *node partition* that lists its kernels, not a group of kernels with a node set. Add an
Nth role and every shared kernel's membership list grows.

**Verdict: `target_nodes` composition works exactly as documented; the disjointness rule just makes the natural factoring inexpressible.**

---

## UGLY

### UGLY-1 — You cannot NoC-write into your own DFB

`noc.async_write(scratchpad, dfb, ...)` — a plain local L1→L1 copy — does not compile:

```
tt_metal/hw/inc/api/dataflow/dataflow_buffer.h:392:26: error: static assertion failed:
  DataflowBuffer without mcast range can only be used as L1 destination
  note: '(Noc::AddressType::NOC == Noc::AddressType::LOCAL_L1)' evaluates to false
```

`Noc::async_write` always resolves its **destination** as `AddressType::NOC`, and
`noc_traits_t<DataflowBuffer>::dst_addr` `static_assert`s on `LOCAL_L1`. So a DFB can be an
`async_read` destination and an `async_write_multicast` destination, but never an `async_write`
destination — not even to itself. The error message names the mcast case, which is misleading: I
was not multicasting, I was doing a local copy.

I worked around it with `MCAST_INCL_SRC` loopback so the sender's own entry is filled by the same
multicast. The other route is `CoreLocalMem<uint32_t> slot(dfb.get_write_ptr())` and a store loop
(what `addr_report_*.cpp` does), which bypasses the DFB abstraction entirely.

**Verdict: an asymmetric hole in the Noc/DFB trait table; the diagnostic points at the wrong feature.**

### UGLY-2 — Raw mcast: two hand-counted fan-outs for one rectangle

One rectangle, two different destination counts, because the data mcast is loopback and the
semaphore mcast is not:

```cpp
const uint32_t dests_incl = get_arg(args::dests_incl);  // data mcast (loopback: counts me)
const uint32_t dests_excl = get_arg(args::dests_excl);  // semaphore mcast (excludes me)
...
noc.async_write_multicast<NocOptions::MCAST_INCL_SRC>(stage, dfb_recv, tile_bytes, dests_incl, ...);
ready.inc_multicast(noc, x_start, y_start, x_end, y_end, 1, dests_excl);
```
```python
"dests_incl": {c0: cols},
"dests_excl": {c0: cols - 1},
```

Miscount either and the sender hangs on the write ack or the semaphore never reaches quorum. The
rectangle itself is four bare `uint32` runtime args in **virtual** coordinates, with the per-NoC
corner ordering (NOC_0 = low corner first, NOC_1 = high corner first) as the caller's problem, and
nothing checking that the rectangle covers the receiver work unit's nodes. Blackhole's harvested
NoC columns 8/9 are also the caller's problem for the fan-out arithmetic.

I wanted to write something like
`noc.async_write_multicast(stage, dfb::recv, tile_bytes, nodes::receivers)` and have the rectangle,
the corner order and both counts derived from the work unit I already declared.

Same op via `McastFamily.attach()`: `mcast_family_mover.cpp` is 31 code lines vs 38 for
`mcast_sender.cpp` + 14 for `mcast_receiver.cpp`, and the kernel spells **no** coordinate, no count
and no semaphore id — `MCAST_ARGS(bcast)` reads names the host wrote. Host side is one
`mcast.attach(spec, run_args, kernels=[K_MOVER], cores=cores)` versus 7 runtime args + a
`SemaphoreSpec` + virtual-coord math. The helper is a clear win where it applies.

### UGLY-3 — The helper only covers 1D mcast families

`ttnn/ttnn/mcast_spec.py::McastFamily.__init__` hardcodes `ttnn.Mcast1D(...)` and takes a
`Mcast1DShape` (`PerRow` / `PerColumn`). `ttnn.Mcast2D` is pybound with the same
`compile_time_args()` / `runtime_args()` / `is_sender()` surface (single sender → one rectangle,
the matmul-1d topology), but has no spec-path wrapper. A 2D broadcast drops you back to UGLY-2.

### UGLY-4 — Semaphore placement is unchecked

`test_semaphore_placement_disjoint_from_binding_kernels` declares the semaphore on node `(7,7)`
while the only kernels that bind it run on `(0..3,0)`:

```
[sem disjoint placement ACCEPTED SILENTLY] program ran and produced correct data
```

No host error. `SemaphoreSpec.target_nodes` is the one placement in the model that must be stated
explicitly rather than derived from bindings (per `semaphore_spec.hpp`), and it is also the one
placement nothing validates against the kernels that bind it. Getting it wrong yields a kernel
happily reading and writing an L1 word the runtime never zeroed on that node — a hang or a silent
pass, not an error. Compare the neighbouring checks, which are sharp:
`Kernel 'mover' references unknown semaphore 'ready'`, `Kernel 'mover' references unknown DFB 'recv'`.

### UGLY-5 — `ttnn.generic_op` rejects a write-only program

```
TT_FATAL: io_tensors must contain at least one input tensor and one output tensor, got 1 tensors.
```
The address-report probe writes a report and reads nothing, so I had to allocate a dummy 32×32
tensor, put it in the io list, and never bind it. This is a `generic_op` constraint, not a
ProgramSpec one — the spec was happy with a single `TensorParameter`.

---

## BROKE THE MODEL

### BTM-1 — The receiver is declared PRODUCER of a buffer it never writes

This is the shape of every cross-core receiver here. `mcast_receiver.cpp`, in full:

```cpp
void kernel_main() {
    Noc noc;
    DataflowBuffer dfb_recv(dfb::recv);
    Semaphore ready(sem::ready);

    dfb_recv.reserve_back(1);
    ready.down(1);
    dfb_recv.push_back(1);
}
```

Host side it is `ttnn.producer_of("recv", "recv")`. It binds no tensor, no scratchpad, and touches
no data. Its entire purpose is to satisfy "every node hosting the DFB runs exactly one producer
instance" on behalf of a remote core that did the actual writing. The named-binding model says
`producer_of` means "this kernel fills this buffer"; here it means "this kernel will tell the local
consumer when someone else has".

There is no honest alternative: BLOCKED-1 shows the truthful wiring is rejected, and dropping the
binding gives `DFB 'recv' has no producer`. So the fake endpoint is mandatory, not a shortcut.

### BTM-2 — Back-pressure has to be smuggled back over the NoC by hand

`reserve_back`/`push_back` flow control is node-local: my `push_back` informs my local consumer,
never my remote producer. So the remote producer has no credit and is free to overrun a ring slot.
`ring_mover.cpp` rebuilds the missing half as a second semaphore going the other way:

```cpp
dfb_recv.reserve_back(1);
const uint32_t entry = dfb_recv.get_write_ptr();
if constexpr (use_credit) {
    space.up(noc, prev_x, prev_y, 1);   // tell my PREDECESSOR my slot is claimed and free
    space.down(1);                      // wait for my SUCCESSOR to say the same
}
noc.async_write(stage, peer, tile_bytes, {.offset_bytes = 0},
                {.noc_x = next_x, .noc_y = next_y, .addr = entry});
noc.async_write_barrier();
arrived.up(noc, next_x, next_y, 1);
arrived.down(1);
dfb_recv.push_back(1);
```

Two semaphores and six runtime args to reproduce, across the NoC, what `reserve_back` gives you for
free inside one node. Note the credit protocol is also what keeps the *addresses* legal: it forces
both sides to the same loop iteration, hence the same ring slot, hence the same `get_write_ptr()`
(BLOCKED-2). Correctness of the address and correctness of the flow control are the same
hand-written invariant.

**Honest null:** with the credit compiled out (`use_credit=0`, cols=8, 16 tiles/core, 4 entries)
the ring still produced correct data — `[no reverse credit] corrupted cores: []`. The cores stay
close enough to lockstep here (every core does the same DRAM read per iteration) that the skew
never materialised. The hazard is structural, not reproduced.

### BTM-3 — `sem::name` is a bare integer, and arithmetic on it is legal

`genfiles.cpp:~200` emits semaphores as `constexpr std::uint32_t <name> = <id>u;` in namespace
`sem` — unlike DFBs, which get an opaque `DFBBindingToken`, and tensors, which get a
`TensorBindingToken<...>` type alias. `Semaphore`'s constructor takes a plain `uint32_t`. So
`Semaphore(sem::arrived + 1)` compiles and silently targets whatever semaphore holds the next id —
and `Semaphore(some_runtime_arg)` compiles too. The type safety the DFB tokens buy does not exist
for semaphores. (Observed in the generator, not exercised on device.)

---

## WIN

### WIN-1 — The two-kernel index mismatch is a compile error, not silent corruption

In the descriptor path, two kernels agree on a CB by both being handed the same integer, usually
through separate compile-time-arg lists. Disagreeing is silent, and shows up as a hang or garbage.
Here the writer is promised the accessor name `out_dfb` while `tile_writer.cpp` says `dfb::recv`:

```
tests/.../kernels/tile_writer.cpp:17:34: error: 'recv' is not a member of 'dfb'
```

The generated `kernel_bindings_generated.h` is per kernel and contains exactly the names that
kernel was bound, so *there is no integer to get wrong*. Same for `sem::` — the C++ suite's
`SemaphoreAccessorNameLoopback` test exists precisely because the two kernels can name the same
semaphore differently (`sem::signal` / `sem::waiter`) and cannot disagree about its id.

This is the single biggest ergonomic gain in cross-core work, where the same buffer is named by
three or four kernels.

### WIN-2 — Role specialization by disjoint node sets, with one DFB

Multiple KernelSpecs may bind the same DFB role as long as their node sets are disjoint. That kills
the `if (is_sender) { ... } else { ... }` branch that the single-kernel mcast pattern needs:

```python
sender   = KernelSpec("sender",   "mcast_sender.cpp",   dfb_bindings=[producer_of("recv","recv")])
receiver = KernelSpec("receiver", "mcast_receiver.cpp", dfb_bindings=[producer_of("recv","recv")])
# WorkUnitSpec "send" -> node (0,0);  "recv" -> nodes (1..N-1, 0)
```

Two small honest kernels (38 + 14 lines) instead of one branchy one, no `is_sender` runtime arg, no
dead code compiled onto the receivers, and the per-node census proves the coverage is exactly right
— including the degenerate `cols == 1` case, which the runtime-branch version has to test for.
Verified on device at cols=4 and cols=8.

### WIN-3 — The per-node census error messages are genuinely good

Every cross-core misconfiguration I could construct produced a message naming the offending node,
the kernels present at it, and the fix:

```
Local DFB 'recv' is malformed at node 1-0: 0 producer instance(s) (none) and 1 consumer
instance(s) ('writer'). ... This node has a consumer but no producer — ensure a producer kernel
covers it (via its WorkUnitSpec membership).
```
That is a debugging session you no longer have. The descriptor-path equivalent is a hang.

### WIN-4 — Semaphores are usable end to end from Python, with no id bookkeeping

`SemaphoreSpec` + `SemaphoreBinding` + `sem::name` works exactly as advertised: the ring's two
semaphores and the raw mcast's one needed no id allocation, no RTA plumbing, and no agreement
between kernels beyond the spec name. Compared with legacy `CreateSemaphore` + hand-passed ids, the
only thing lost is the initial value (BLOCKED-3) — and `McastFamily` has to *fake* an id assignment
(`config.sem_ids = [0, 1]`, "adopting placeholders keeps `owned_semaphores()` empty") to bridge to
the `Mcast1D` host helper, which still thinks it owns id allocation.

---

## What I'd want from the API

1. **A cross-node DFB**, even a restricted one: `producer_consumer_map` for unicast pipelines and a
   one-producer/many-consumer form for broadcast. It is the sketch in `dataflow_buffer_spec.hpp`;
   everything above is a workaround for its absence.
2. **A name for a peer's buffer.** Either kernel-side (`dfb::recv.at(node)` yielding a NoC address)
   or host-side (a `get_dfb_address(spec, "recv")` analogous to `ttnn.get_cb_address`). Today the
   only address available is your own, and reusing it is correct for DFBs and wrong for
   scratchpads — with nothing stating either.
3. **Document the placement rules I had to measure**: DFB base addresses are program-global,
   scratchpad base addresses are per-node. This is the invariant every cross-core kernel rests on.
4. **A receiver-side endpoint role.** `remote_producer_of("recv")` would let the census pass without
   the fake `producer_of`, and would tell the reader that a remote core fills this buffer.
5. **Cross-node credit.** Either automatic (as part of the cross-node DFB) or, minimally, expose
   `SemaphoreSpec` initial value so a credit counter can start at `num_entries` instead of forcing
   receiver-signals-first.
6. **Multicast that consumes a node set, not a rectangle.** `async_write_multicast(src, dfb::recv,
   size, nodes::receivers)` — deriving corner order per NoC, both fan-out counts, and the harvested-
   column correction. Today all four are the caller's arithmetic, in virtual coordinates, in
   runtime args.
7. **Validate `SemaphoreSpec.target_nodes` against the nodes of the kernels that bind it** (or drop
   the field and derive it, like DFBs and scratchpads).
8. **Let a kernel be declared once with its node set**, rather than requiring shared kernels to be
   re-listed in every disjoint work unit.
9. **Give semaphores an opaque binding token** like DFBs and tensors, so `sem::a + 1` stops
   compiling.
10. **Generalize `mcast_spec.McastFamily` to `Mcast2D`**, and let `async_write` accept a local L1
    destination so a scratchpad→DFB copy does not need a loopback multicast.
