# Regime-A fused all-gather matmul: design specification

## Goal and contract

Build a new multi-device TTNN op that consumes `in0[M, K/TP]`, K-sharded over an **even** tensor-parallel
group (`TP=4` or `8` normally), and a device-local `in1[K, N]`, and produces the replicated result
`all_gather(in0, dim=-1) @ in1`. It must preserve the accepted single-chip Regime-A choices—`Ns`, `Pk`,
`Sm`, core placement, local in0 ring, in1 DRAM-sharded readers, reduction, tails, precision, and fusions.
The objective is to hide fabric all-gather behind in1 DRAM reads and matmul, not to perform a complete
all-gather followed by matmul.

The op must support both ring and linear device topologies through the established CCL/fabric APIs and
**mux v2**. Do not add multi-device branches to `regime_a_matmul`; implement a separate op, initially named
`ttnn.experimental.all_gather_regime_a_matmul_async`.

## Device-level schedule

Use a balanced bidirectional all-gather. For the ring, send each transport chunk in both directions. With
even `TP`, a device receives work in waves:

```text
wave 0 (local):       K/TP
waves 1..TP/2-1:    2K/TP  (one full shard from each direction)
final antipode wave: K/TP   (half of the antipode payload on each direction)
```

The linear topology uses the repository's existing balanced bidirectional line schedule, with the same
local-first and progressive-consumption rules. Never process a duplicate shard, and never wait for the
whole gathered tensor. A received block is made compute-visible as soon as its payload is safe; it is then
forwarded if another device still needs it.

Only the **first N traversal** forwards A over fabric. If a worker owns several N sub-blocks, later
traversals reuse resident A and must not resend it. Assign global K blocks to the existing `Pk` groups so
each availability wave gives useful work to as many groups as possible; a purely contiguous assignment can
leave most `Pk` groups idle during fabric startup. The in1 reader must use the identical global-K order.

## On-chip dataflow and ownership

Keep the current local pipeline:

```text
fabric/local A arrival -> fixed L1 slot -> local in0 writer publishes CB0
                       -> eight-core in0 ring -> progressive matmul
in1 DRAM banks         -> bank-adjacent in1 readers -> matmul
matmul partials        -> existing Pk reduction -> fusion once -> output
```

CB0 already has capacity for the worker's complete `K/Pk` slice and remains resident across N sub-blocks.
The in0 writer must remain CB0's sole producer: fabric workers write fixed L1 staging slots and signal
readiness; the writer performs `cb_push_back`. Use per-slot epochs/credits rather than one ambiguous
cumulative counter, because bidirectional arrivals can be out of order and the antipode is split.

An A transport chunk crosses fabric **once per destination device**. `Ns` groups need identical A; they may
share forwarding work, but must not emit duplicate fabric copies. After ingress, distribute the payload to
all local `Ns` consumers over NoC. `Sm` groups own distinct M rows and may receive distinct slices. Do not
put fabric work on the critical in1-reader RISC. Compare one designated injector per direction against
striping chunks across eligible in0-ring cores/`Ns` groups; choose by measurement, not symmetry.

## Two implementation phases

1. **DRAM-staged reference.** Read local A once for both local compute and fabric egress. Remote ingress
   writes transport chunks to a deterministic DRAM staging region, then signals readiness; local readers
   wait per chunk and read it into the normal in0 path. This is the simplest correctness and overlap
   baseline, but its extra fabric-to-DRAM write plus DRAM-to-L1 read may compete with in1 and cap performance.
2. **Direct-L1 streaming.** Fabric ingress writes bounded L1 slots, signals payload readiness, and receives a
   credit only after all local consumers release a slot. Partition/seed the eight local ring stripes without
   rereading A from DRAM. Size the window from authoritative L1 accounting; do not assume the whole gathered
   A fits. Retain DRAM staging as an A/B diagnostic until direct L1 is proven correct and faster.

Transport granularity is independent of compute `kb`: transfer `C*kb` blocks to amortize fabric overhead,
but publish contained `kb` blocks progressively. Optimize for the default **4 KiB fabric packet**. If fabric
is measured critical, repeat the same configuration at **8 KiB** and report the difference. Signal once per
transport chunk/slot, not once per packet; payload must precede readiness, and source reuse must wait for the
appropriate flush/credit. Drain writes and non-posted atomics before kernel exit.

## Correctness, fusion, and performance gates

- Exact tile ownership: every global K tile is consumed once on every device; ring/line, both directions,
  tails, split antipode, cache replay, and fresh semaphore sets must be tested.
- Preserve BF16/compute fidelity and all supported epilogues: bias, activation, addcmul, and output chunking.
  Apply epilogues only after the complete global-K result has reached the existing local reduction endpoint.
- Test `TP=4` and `8`, narrow and wide N, `Pk>1`, `Ns>1`, `Sm>1`, multiple N sub-blocks, and non-divisible
  logical tails. Use watcher and per-RISC profiling; host dispatch time is not evidence of overlap.
- Record four baselines with identical shapes/configs: single-chip full-K matmul, standalone all-gather,
  unfused all-gather+matmul, and fused AGMM. Report all relaunches and
  `overlap_efficiency = max(T_mm, T_ag) / T_fused`; also report fabric and DRAM bytes.
- Galaxy RevB DRAM peak is **448 GB/s**, not the p150 value of 512 GB/s. All bandwidth percentages on this
  machine must use 448 GB/s and identify the board revision. Phase A is successful when correct and visibly
  overlapped; the production target is direct-L1 fused time within 10% of `max(T_mm, T_ag)` on DRAM-bound
  goldens, with no material regression versus unfused execution.

Start by reading the production `regime_a_matmul`,
`experimental/ccl/all_gather_minimal_matmul_async`, and current mux-v2 unit tests. Reuse fabric connection,
packet, teardown, and flow-control primitives; do not invent a private mux protocol.

## Appendix A: balanced bidirectional delivery for direct-L1 (per-core stream plan)

This is §"Device-level schedule" expressed for direct-L1, where the transport unit is one consumer core's
cb0 slot 0 rather than a device-level shard. It is a **required correction**: the first direct-L1
implementation assigned each ring position a single direction by `ring_pos` parity, so a stripe travelled
`tp-1` hops from its origin instead of `tp/2`. Same bytes, worse latency, and it inflates `T_ready_max` —
the leading term of the ring bound.

**Glossary.** Terms coined for this appendix, plus the two indices that are easy to confuse.

| term | meaning |
|---|---|
| **stripe** | the transport unit: the contents of ONE core's cb0 **slot 0**, `W*M_block*K_block` tiles. Identified by `(sl, p)`. The only part of a core's K slice that is not produced by the on-chip ring. |
| **`p`** — ring position | the core's index `0..7` in its 8-core on-chip bank ring (`ring_pos` in the plan/kernels). NOT a device index. |
| **`sl`** — slice | the within-bank slice index `0..preaders-1` (`= kk*mfac + nn*Sm + mm`, the plan's `CorePlan::slice`). Distinct from `kk`, which is only the `Pk` group; a rank owns `ppr` positions in EACH of the `preaders` slices. |
| **`ppr`** | ring positions per source rank, `= 8/tp` (requires `tp \| 8`). Rank of a position is `r = p / ppr`. |
| **rank** — `d`, `r` | device index `0..tp-1` within the TP group along `cluster_axis`. `d` = this device, `r` = a stripe's origin rank. |
| **origin** | the device with `d == r`: the one that reads the stripe from its LOCAL in0 shard rather than receiving it. |
| **reach** — `f`, `b` | how many hops a given stripe is relayed forward / backward before stopping. `f + b = tp-1` always. |
| **terminal** | a device at reach (`fd == f` or `bd == b`): it consumes the stripe and relays it no further. |
| **wave / depth** — `w` | hops travelled, so arrivals group into waves by arrival time. `w = 0` is local. |
| **antipode** | at even `tp`, the device `tp/2` away from the origin — equidistant in both directions, hence the one that has to be assigned a direction. |

`fd = (d-r) mod tp` and `bd = (r-d) mod tp`, so `fd + bd = tp`.

**The antipode is split, per stripe.** At even `tp` the device at distance `tp/2` is equidistant both ways,
so one direction must carry it — and if that is always the same direction, that link carries
`tp/2 * K/tp` against the other's `(tp/2-1) * K/tp` (at tp=4: 2x). Split it at STRIPE granularity, never at
byte granularity: give each stripe a flag deciding which direction owns its extra hop,

```text
via_fwd(sl, p) = ((p mod ppr) + sl) mod 2 == 0
reach          = (f, b) = (tp/2, tp/2 - 1) if via_fwd else (tp/2 - 1, tp/2)      [f + b = tp-1 always]
```

alternating over the `ppr * preaders` stripes each rank owns, so half of every rank's payload reaches its
antipode each way. The index is the SLICE `sl`, not the `Pk` group `kk`: at tp=8 `ppr == 1`, so `p mod ppr`
is always 0 and the alternation has to come from the second term, and `sl` gives `preaders` distinct values
where `kk` gives only `Pk`. Because the split is per stripe, **a core still receives exactly one stripe and
one credit** — no partial slot, no second arrival to reconcile.

**Per-core rule.** For core `(sl, p)`, exactly one case applies (`fd+bd = tp` with `f+b = tp-1` makes the two
arrival cases mutually exclusive and jointly exhaustive):

| case | role | sends |
|---|---|---|
| `fd == 0` | origin: read the stripe from the LOCAL in0 shard | forward **and** backward |
| `1 <= fd <= f` | arrives on the forward stream at depth `fd` | forward, iff `fd < f` |
| `1 <= bd <= b` | arrives on the backward stream at depth `bd` | backward, iff `bd < b` |

Arrival depth `w = 0 if fd == 0 else (fd if fd <= f else bd)`; max depth `tp/2`. Relay source remains the
consume destination, so no relay buffer exists.

**Waves.** Depth `w > 0` delivers rank `d-w`'s stripes via forward and rank `d+w`'s via backward; at
`w = tp/2` those are the same (antipode) rank, arriving half each way. Per device:

```text
wave 0                 ppr stripes = K/tp    (local, no fabric)
waves 1..tp/2-1       2ppr stripes = 2K/tp   (one full shard from each direction)
wave tp/2 (antipode)   ppr stripes = K/tp    (half on each direction)
```

which sums to `K` and reproduces the device-level wave table above. tp=4: `K/4, K/2, K/4`. tp=8:
`K/8, 2K/8 x3, K/8`.

**Invariants preserved.** Every stripe crosses `tp-1` hops, exactly as today, so total fabric bytes are
unchanged and no tile crosses fabric twice per destination. Sends per device stay at `ppr*(tp-1)` per ring
slice: origins now send twice, but the two terminals send nothing. And with the antipode split the two
directions carry **exactly half each**, so the mux client counts stay even — the same split the current
implementation has, reached without its `tp-1` hop depth.

**Scope.** Host-only: the direction/depth assignment in the direct-L1 stream plan. The kernel already
supports a core driving two muxes (the LINE-origin case) and gates on a single arrival semaphore, and the
stripe-granular antipode split is what keeps that true — a byte-level split would require two credits per
slot and per-slot epochs.

**STATUS-A: implemented, correct, and NOT the default.** Behind `TT_AGMM_DIRECT_L1_BALANCED=1`; the default
keeps one direction per ring position (depth `tp-1`). It passes the same 32/40 as the default, but measured
on `medium`/ring/2 links it is neutral at tp=4 (120.16 vs 120.19) and **+5.2 us / +3.8% at tp=8** (141.6 vs
136.4, repeated interleaved samples). It does reproducibly improve the fabric-only term
(`TT_AGMM_ABLATE=nowait`: -1.4 us at tp=4, -2.5 us at tp=8); what it loses is on the *dependent* path.

Why the depth win does not convert: **total link bytes are identical** — every stripe crosses `tp-1` hops
under either schedule (the invariant above) — so on a shape that is NoC-occupancy-bound there is no
throughput to gain, only latency, which is not the binding constraint. Meanwhile the arrival pattern gets
burstier: `2K/tp` per wave from both neighbours at once instead of `K/tp`, concentrating ingress against the
on-chip ring. That is the same mechanism that made a deferred fabric drain worse (see
`AGMM_DIRECT_L1_DESIGN.md`).

Expected to win once the dependency stall is removed by per-wave rings, at which point the burstiness stops
gating — re-measure it then. Keep the spec's schedule as the target; the flag is the switch.

## Appendix B: the wavefront — arrival-ordered consumption over the unicast in0 ring

Delivers the spec's "never wait for the whole gathered tensor / each availability wave gives useful work"
on the direct-L1 path. The on-chip distribution stays a UNICAST ring; multicast is explicitly out of scope
until further notice.

### THE FIX, in one line

**Each core consumes its chunks in the order they become available, instead of in ring order.** Same chunks,
same fabric traffic, same on-chip traffic, same unidirectional unicast ring — only the order changes.
Forwarding moves to the same order, so a core waiting on its own late chunk stops holding up the cores
behind it.

### Worked example: why the current order stalls (the picked config, device 0)

`(Pk,Ns,Sm,kb,nsb) = (10,1,1,2,1)`, 80 cores = 8 banks x 10 K-splits. Each K-split owns 16 of the 160 global
K-tiles; the 8 cores of a split need the SAME 16 tiles and differ only in which N columns they compute. Those
16 tiles sit in cb0 as **8 chunks of 2 K-tiles** (x 8 M-tiles = 16 tiles = 32 KB each). This device holds only
40 of the 160 global K-tiles, so each split takes 4 tiles from each device — **2 chunks per device**:

```text
chunk        :   0    1    2    3    4    5    6    7
lives on dev :   0    0    1    1    2    2    3    3
fabric hops  :   0    0    3    1    2    2    1    3     <- device 0; 0 == already local
```

Core at ring position `p` fetches chunk `p` (locally or over fabric); the ring then circulates chunks until
every core has all 8. Consumption order today is `p, p-1, p-2, ...`, giving (40 time units per fabric hop,
1 per on-chip hop):

```text
core | consumption order (chunk @ when available)                | stalls at
  0  | 0@0    7@121  6@42   5@83   4@84   3@45   2@126  1@7      | step 1
  1  | 1@0    0@1    7@122  6@43   5@84   4@85   3@46   2@127    | step 2
  2  | 2@120  1@1    0@2    7@123  6@44   5@85   4@86   3@47     | step 0  <--
  7  | 7@120  6@41   5@82   4@83   3@44   2@125  1@6    0@7      | step 0  <--
```

**Core 2 is the whole problem.** It must consume chunk 2 first, chunk 2 is one of the two last-arriving
chunks, so it idles from t=0 to t=120 and only then starts its matmul — while chunks 1 and 0 have been in its
L1 since t~1. Sorted by availability it could finish **6 of its 8 chunks before t=120**. Two cores per ring
are always in this position (two chunks always arrive last), and they set the makespan: `T_gather + T_matmul`.

### The two clocks — do not re-derive the wrong conclusion here

It is easy to "prove" this is unfixable by arguing that after `r` steps a core can only possess `r+1` chunks,
so it has nothing else to consume. That conflates two clocks: `r+1` counts **receive hops**, but a chunk
reaches core 2 as soon as its upstream neighbours pass it along, which does not depend on core 2's own
progress. Core 2 can be blocked at consume-step 0 and already hold two received chunks — the table above
shows exactly that (`1@1`, `0@2` in hand while step 0 waits until t=120). The freedom is real.

### A cb0 slot index IS the consumption-order index

Worth stating before the phases, because an earlier draft of this appendix got it wrong ("slot `p` holds
position `p`"). Compute does not pop cb0 per block — it waits cumulatively and addresses each block by an
explicit ascending offset:

```cpp
cb_wait_front(in0_cb, (k_block + 1) * in0_block_num_tiles);   // cumulative; popped once, after N reuse
matmul_blocks(..., k_block * in0_block_num_tiles);            // explicit offset into the resident slice
```

`k_block` walks ascending, so "the stripe consumed s-th" and "the stripe in slot s" are the same statement.
A fixed position->slot layout would therefore fix the ORDER too, not decouple from it. The only lever on
consumption order is **which stripe lands in which slot** — which is what phase 1 makes host-controlled, and
what phase 3 then chooses differently.

("chunk" here is the same 32 KB object appendix A's glossary calls a **stripe** — one cb0 slot's worth.)

### Two structural facts this rests on

1. Each chunk travels 7 hops (`o -> o+1 -> ... -> o+7`), so **each core forwards exactly 7 chunks over 7
   steps, one per step, and WHICH chunk it forwards at each step is free** — constrained only by
   receive-before-forward. Today that choice is implicitly "the one I just consumed", and the kernel WAITS
   before it forwards, so **a core blocked on its own consume wait forwards nothing and the stall propagates
   down the ring.** That is why forwarding has to move to availability order too, not just consumption.
2. A chunk's availability at core `c` is `T_arr(wave(o)) + hops(o->c)*delta`, which depends on `c`. So the
   consumption order is **per-core**, not device-global (device-global is true of *fabric* arrival, not of
   on-chip availability). It is still fully static and host-computable: 8 cores x 8 chunks per ring.

   Convenient consequence: `hops(o->c+1) = hops(o->c) + 1` for every chunk except `c+1`'s own, so adding one
   hop preserves the relative order. A core's consume order is therefore its predecessor's FORWARD order with
   its own chunk inserted — which means received chunks still land in ascending slot order and the kernel's
   existing receive COUNT remains a valid gate (see phase 4).

### Bound

```text
today       T_ready_max + T_mm          the core owning a last-arriving chunk owes its WHOLE matmul
this        T_ready_max + T_mm/8        after its last chunk a core owes only 1/8 (one chunk) of its work
```

Per-wave rings would give the intermediate `T_ready_max + T_mm/tp`. It is NOT the plan: availability ordering
is both finer-grained and a smaller change, and needs no new distribution structure. Same for a bidirectional
on-chip ring or NoC multicast — both were considered while the mechanism was mis-diagnosed, and neither is
needed once consumption is reordered.

### Phases, each with its own gate

| # | change | gate |
|---|---|---|
| 0 | Allow `nopayload`+`nowait` together (ablation composition) | **DONE** — closed the 2x2; see Target below. Fixed plumbing is 1.3 us, not the 13.4 us previously attributed to it |
| 1 | Make the slot assignment schedule-driven: `own_slot`, `peer_slot` and a per-step `{src_slot, dst_slot}` become host-emitted writer args (`RingSlotArg`, 17..32) instead of being derived from the step counter | **DONE** — bit-identical (`468b790b823e2959` pre/post, and direct-L1 == staged), 12/12 both paths, perf neutral (118.3 vs 120.2) |
| 2 | `RingSchedule.consume_pos[position][step]` becomes the single source of truth: the in1 reader takes the consume order as args 7..7+G-1, and the writer's `own_slot`/`dst_slot` are DERIVED from it by inverting the successor's map. Still today's rotation | **DONE** — bit-identical (`468b790b823e2959`, both paths), 12/12 both paths, perf neutral (118.95). Guards: schedule must be a permutation per position, and derived writer args must equal phase 1's `(s, s+1)` |
| 3 | **Consume in availability order, not ring order.** Host computes `avail(o, c) = fabric_hops(o)*WAVE + on_chip_hops(o->c)*DELTA`, sorts each core's 8 chunks by it, and emits that as the schedule. Forwarding follows the same order and is issued BEFORE the consume wait; the own-chunk wait AND the fabric relay both move out of the prologue into a once-only `ensure_own()` at the step that first needs that chunk | **DONE** — 117.80 us vs 121.60 rotation (-3.1%) and far more stable (0.16 vs 3.11 us spread), interleaved samples. 12/12 phase gate on all three modes; 32/40 full on direct-L1 (the 8 are `Ns>1`). Stall 17.5 -> 14.2 us, so ~3 of ~17 recovered — the order is right but something else still serialises |
| 4 | Per-slot readiness — only if needed | the spec's per-slot-epoch clause targets out-of-order BIDIRECTIONAL FABRIC arrivals; under direct-L1 each core takes exactly one fabric stripe and the on-chip receive order is static from phase 2, so verify the existing monotone count still suffices rather than adding machinery |
| 5 | Re-measure `TT_AGMM_DIRECT_L1_BALANCED=1` | its burstiness penalty should shrink once the ordering constraint is gone |

Phase 1 and 2 being bit-identical is what separates "plumbing is wrong" from "numerics moved" — the
writer/in1-reader desync is the failure mode this branch has hit twice.

### Phase-gate test suite (not the full 40)

Twelve tests, ~1 minute, run after every phase instead of the full suite:

```bash
-k "(medium and ring) or (large and ring) or sm2 or pk4 or (cache_replay and ring) or (single_chip and medium)"
```

Covers the perf shape (`medium`/ring at tp4 and tp8), a second shape (`large`), `Sm>1` (which caught the
semaphore-budget regression), a second `Pk`, program-cache replay with fresh semaphores, and op-vs-op parity.
Deliberately excludes `small` (picker chooses Ns=2 -> refused), line topology, and the fused epilogues
(orthogonal to the gather). Run the full 40 only before committing a phase.

### Target — from the closed 2x2 (phase 0, done)

|            | payload | no payload |
|------------|--------:|-----------:|
| **wait**   |  120.38 |      88.39 |
| **no wait**|  100.79 |      76.51 |

with `floor` (no fabric at all) = 75.24. Which decomposes the whole thing:

```text
floor                            75.24
+  1.3   fabric FIXED cost       60 clients' open + credit + close, no payload, no waiting
+ 24.3   payload OCCUPANCY       bytes actually moving
+ 19.6   WAITING                 the dependency stall
= 120.38
```

**The no-stall bound is `nowait` = 100.8 us**, not `nopayload`. `nowait` runs every read, every FP32 add,
every fabric send, and stalls on nothing, so it is exactly the makespan a perfect schedule reaches.
`nopayload` (88.4) is not a target — it is an artifact of deleting bytes that real work has to move.

So the wavefront is worth **~19.6 us (16%)**: 120.4 -> ~101. That finally puts the fused path ~14 us BELOW
the unfused Phase-0 composition (115.1), but still ~14 us above the 86.9 us gate. Closing the rest needs the
24.3 us of payload occupancy to shrink, and that is bytes over links at `num_links=2`, which is the hardware
maximum on this axis — so the gate is probably not reachable for this shape by scheduling alone.

Two earlier claims are retracted by this table: that the fabric's fixed cost was ~13.4 us and "a client-count
problem" (it is 1.3 us; the rest was credit-chain latency, which is waiting and therefore hideable), and that
per-wave rings could reach the gate.
