# AGMM Phase 2: direct-L1 streaming design

Status: IMPLEMENTED and measured on `bh-glx-120-c02u02`. Gated behind `TT_AGMM_DIRECT_L1=1`; DRAM staging
(Phase 1) stays the default and the A/B oracle, per the design spec's "retain DRAM staging as an A/B
diagnostic until direct L1 is proven correct and faster".

**Result: 143.4 -> 120.7 us (-15.8%) on medium/tp4/ring/2-link.** It does not reach the gate. See
"Measured" below for the four baselines and where the remaining time goes; the short version is that the
predicted ~93 us was optimistic in assuming L1-to-L1 gather traffic is free, and the ring's zero-overlap
bound is intact and now the single largest recoverable term.

## Why Phase 1 cannot meet the gate

`medium` = 256x5120x2560, bf16, tp=4, ring, 2 links. The single-chip matmul's 83.0 us corresponds to
30.14 MB / 83 us = 363 GB/s = 81% of the 448 GB/s Galaxy RevB peak, so this shape is DRAM-bandwidth-bound
and surplus bytes convert directly into time.

| traffic (per device)    | Phase 1 (DRAM-staged) | Phase 2 (direct-L1) |
|-------------------------|----------------------:|--------------------:|
| in0 shard read          |               0.66 MB |             0.66 MB |
| local -> staging write  |                  0.66 |                   - |
| fabric egress reads     |                  1.97 |     - (from L1)     |
| fabric ingress writes   |                  1.97 |     - (into L1)     |
| matmul in0 read         |                  2.62 |     - (resident)    |
| in1                     |                 26.21 |               26.21 |
| out                     |                  1.31 |                1.31 |
| **total**               |           **35.4 MB** |         **28.2 MB** |
| **roofline @ 363 GB/s** |            **97.5 us** |         **77.6 us** |

The gate is `1.1 * max(T_mm, T_ag) = 1.1 * max(83.0, 41.3) = 91.3 us`. Phase 1's own roofline is 97.5 us,
so it misses by 7% even with a perfect scheduler. This is byte count, not scheduling: no amount of stripe
reordering, core placement, or transfer batching can fix it.

Phase 2's roofline is *below* the single-chip matmul, because the gathered 3/4 of the activation never
round-trips DRAM. The floor for this shape is in1 streaming alone: 26.21 MB / 363 GB/s = 72 us (in1 is 87%
of all traffic).

This also reinterprets the `TT_AGMM_ABLATE=nowait` result. The "~30 us of contention that overlap cannot
recover" is a Phase-1 staging artifact, not a physical limit: 5.25 MB of the surplus (~14.5 us) is pure
staging round-trip that Phase 2 deletes.

## What this does NOT fix

Phase 2 attacks the byte count (the ~30 us half of the ablation split), not the ~37 us dependency stall.
The stall has a separate, structural cause -- see "the ring bound" below. Expected landing zone with the
ring retained is therefore ~93 us, not 78 us.

### The ring bound (why the arrival-order permutation patch was abandoned)

In `in0_ring_reduce_writer.cpp` the ring has core `c` at step `s` consuming shard `sigma(c-s mod G)`;
step 0 is its own shard, and step `s>0` blocks on the predecessor's forward. Unrolling:

```
t(c,s)   >= T_ready(sigma(c-s)) + (s+1)*delta        delta = per-step cost
makespan >= max_c max_s [T_ready(sigma(c-s)) + (s+1)*delta]
          = T_ready_max + G*delta
```

Every ring position appears at every step across the `G` cores (a Latin square), so **the bound does not
depend on sigma** -- the permutation of stripes onto ring positions is irrelevant. Concretely: the core
that owns the last-arriving shard cannot begin *any* of its G steps until that shard lands, and then still
owes the entire matmul. So `fused >= T_gather + T_matmul`: zero overlap, by construction.

That bound retro-explains every measurement on this branch: 150 ~= 74 + 83; dedicated gather cores bought
0.5 us; batching helped only because it lowered `T_ready_max`; `nowait` drops by exactly the `T_ready` term.
`scratchpad/arrival_order_AND_ablations_WIP.patch` is therefore worth at most one hop `delta`, not the 37 us
it was aimed at, and was dropped rather than debugged.

### What this bound does NOT rule out (read this before concluding the stall is unfixable)

The derivation above holds `sigma` variable but keeps ONE thing fixed: that core `c` consumes shard `c-s` at
step `s`. There are two independent freedoms here and the Latin-square argument only kills the first:

1. **`sigma`** -- which shard sits at which ring position. Irrelevant, as proved above.
2. **The CONSUMPTION ORDER** -- which of the shards a core already holds it works on next. Untouched by the
   argument, and it is where the whole stall lives.

Because those got conflated, this section was read for a while as "the stall is structural, only per-wave
rings can fix it". It is not. Concretely, at the picked config (Pk=10, tp=4) on device 0, chunks 2 and 7
arrive last (3 fabric hops); core 2 is forced to consume chunk 2 FIRST, so it idles through the entire
gather and then owes all 8 of its steps -- while chunks 1 and 0 have been sitting in its L1 since t~1.
Sorted by availability it could finish 6 of its 8 chunks before the last wave lands.

The fix is therefore **consume in availability order rather than ring order**: same shards, same fabric
traffic, same on-chip traffic, same unidirectional unicast ring, only the order changes. Forwarding moves to
the same order so a core waiting on its own late shard stops holding up the cores behind it. See
`REGIME_A_AGMM_DESIGN_SPEC.md` appendix B for the schedule and phasing.

Note also that the `t(c,s) >= T_ready(sigma(c-s)) + (s+1)*delta` inequality counts RECEIVE hops, not the
consuming core's own steps: a shard reaches core `c` when its upstream neighbours pass it along, which does
not depend on `c` having consumed anything. Reading `s` as both "hops travelled" and "steps this core has
completed" is what makes the stall look unfixable.

## Preconditions verified

**CB0 already holds the whole gathered slice, so no credits are needed.** `regime_a_matmul_plan.hpp:208`
sizes `cb0_tiles = M_block_capacity * K_slice_capacity`, and `K_slice_capacity = rup(cdiv(Kt, Pk), kb*8)`
where on the fused path `Kt` is *global* K -- the planner is deliberately built against the staging buffer
(`program_factory.cpp:643`). The fused path passing 40/40 at tp=4 and tp=8 proves those configs are
L1-feasible today.

Consequence: if fabric ingress writes a remote stripe into its **final** CB0 slot, nothing in the program
ever overwrites it. Slot 0 is written only by fabric (remote positions) or DRAM (local position); slot `s>0`
only by the ring forward. Zero reuse anywhere, so there is no bounded window, no flush/credit handshake --
only the per-shard arrival semaphores that already exist (`wave_fwd_sem` / `wave_bwd_sem`).

## Dataflow

Fabric clients are **all consumer cores**, not the 8 masters.

    core (kk,p) on device d  --fabric-->  core (kk,p) on device d+/-1
                                          (same core index, same L1 address, same size)

Rationale: under store-and-forward a device relays what it received. With direct-L1 a received stripe lives
in some consumer's CB0 slot 0, so an 8-master relay would need (a) cross-core L1 reads and (b) its own relay
buffer, because a master's relay partition (`bank_id % m_groups` M-slice of a whole rank) differs from what
any single core consumes. That buffer does not fit: ~160 KB/arrival x 3 live arrivals at tp=4, on top of
CB0's 640 KB, against ~1.5 MB L1. Reusing slots to make it fit reintroduces credits.

With relay source == consume destination there is no relay buffer and no extra L1. Each core receives
exactly one stripe in its lifetime and relays it once; the existing on-chip ring does all remaining
distribution, unchanged.

Falls out for free:
- **Antipode split becomes position-level.** Rank `r` is owned by `8/tp` ring positions, so at tp=4 assign
  one of its two positions forward and the other backward -- no byte-level split of a single core's payload.
  At tp=8 (1 position/rank) balance by alternating direction across Pk groups.
- **`my_dir` is a pure function of `p`**, since the rank a position owns is fixed and the device rank is.

Cost: mux clients scale with `preaders` (`4*preaders` per direction: 16 at Pk=4, 32 at Pk=8) instead of a
fixed 8. `channels_per_mux` is already derived dynamically (`program_factory.cpp:515`), but mux cores come
from the top row only (`mux_core_for`, `:536`), so this needs a bounds `TT_FATAL`.

## Address mapping

Destinations need no runtime table -- the consumer's mapping inverts in closed form. Consumers evaluate
`gk(l) = (l/run_len)*shard_stride + stripe_base + (l%run_len)`. For a global K tile `g`:

```
r    = g / k_shard_tiles                 // source rank
off  = g % k_shard_tiles
kk   = off / run_len                     // owning Pk group
l    = r*run_len + (off % run_len)       // capacity-local index
p    = l / (W*kb)                        // ring position == fabric entry point
slot0_offset = ((wb*M_block + m)*K_block + k) * tile_bytes
```

Slot-0 layout is `[wb][m][k]`, matching the step-0 read loop's `p += tile_bytes` walk.

Destination *core coords* are free: every device runs the same program with the same core assignment, so
ring group `q` / position `b` is core `cores[b*preaders + q]` on the peer exactly as locally.

## Gating risk: CB0 base-address symmetry across devices

The whole approach assumes the same program places CB0 at the same L1 offset on every device. The existing
code leans on the DRAM analogue and says so (`writer:582-586`: "Mesh tensors share an address across the
mesh, so the local address of a page is also its address on the peer").

**A host-side check is NOT available, contrary to the first version of this plan.** The idea was to
`TT_FATAL` in `create_mesh_workload`, which does hold every per-device program at once. But CB L1 addresses
are not assigned at `CreateCircularBuffer` time -- they are assigned when the program is compiled/finalized
(`Program::allocate_circular_buffers`), which happens at enqueue, after `create_at` has returned. There is
no address to compare at that point, and `mkcb` discards the `CBHandle` anyway.

What we do instead: **the sender uses its OWN `get_write_ptr(in0_cb)` as the destination address.** The peer
core has the same core index, runs the same kernel, with the same CB config, so its cb0 base is the sender's
cb0 base. No host support needed, and no new assumption beyond the one the DRAM path already makes
explicitly (`writer:582-586`) -- both reduce to "identical programs place identical objects at identical
addresses". If DRAM buffer symmetry holds across the mesh, CB symmetry holds by the same construction.

Residual risk: divergent L1 allocator state across devices (e.g. unequal pre-existing L1 buffers) would
break it silently. Not currently reachable -- the mesh allocates the same tensors everywhere -- but it is
the thing to suspect first if direct-L1 produces per-device-varying corruption. A one-time handshake
(each core fabric-writes its cb0 base to its peer and compares) would close it if that ever happens.

## Scope

- `Ns > 1` is refused on the direct-L1 path initially. With `Ns == 1` the shard's K range partitions
  cleanly across `(Pk group, position)`, and `Sm > 1` is also a clean partition (distinct M rows), so total
  fabric bytes are unchanged from today. `Ns > 1` ring groups need *identical* in0, so a naive scatter
  would emit duplicate fabric copies -- forbidden by the spec ("must not emit duplicate fabric copies");
  doing it right needs a NoC replication step after ingress. DRAM staging keeps covering `Ns > 1`.
- Measurement to report: the four spec baselines plus `overlap_efficiency`, and DRAM/fabric bytes, to
  confirm the 97.5 -> 77.6 us roofline shift rather than assuming it.

## Measured

`medium` = 256x5120x2560 bf16, tp=4, ring, 2 links, on `bh-glx-120-c02u02`. Device FW duration from the
tracy device profiler via `tools/mm_sweep/picker_gen/agmm_bench_worker.py`; each number is the median over
24-48 timed iterations of the MAKESPAN (max over the tp devices of that device's slowest core), because a
fused multi-device op is only finished when its slowest device is. Host wall is not used anywhere.

The four spec baselines, plus both fused paths:

| baseline                                        |     us |
|-------------------------------------------------|-------:|
| single-chip full-K matmul (1 device, fabric off) |   77.8 |
| full-K matmul on the mesh (Phase-0's 2nd op)     |   79.0 |
| standalone all-gather (Phase-0's 1st op)         |   36.6 |
| unfused all-gather + matmul (Phase 0)            |  115.1 |
| **fused Phase 1 (DRAM-staged)**                  | **143.4** |
| **fused Phase 2 (direct-L1)**                    | **120.7** |
| Phase 2, `TT_AGMM_ABLATE=nowait`                 |  101.7 |

    gate = 1.1 * max(T_mm, T_ag) = 1.1 * 79.0 = 86.9 us
    overlap_efficiency = max(T_mm, T_ag) / T_fused:   Phase 1 = 0.551,  Phase 2 = 0.654

T_ag and T_mm are read out of the SAME Phase-0 process (it is literally `all_gather_async` then the full-K
matmul, two device ops per call), so they are at identical shapes and config by construction rather than by
being configured to match.

Phase 2 decomposes cleanly, and with the ablations composable (`nopayload,nowait`) the 2x2 closes, so every
term below is measured rather than inferred:

|             | payload | no payload |
|-------------|--------:|-----------:|
| **wait**    |  120.38 |      88.39 |
| **no wait** |  100.79 |      76.51 |

    75.24  floor -- TT_AGMM_ABLATE=nogather (no fabric at all)
  +  1.3   fabric FIXED cost   -- 60 clients' open + credit + close, no payload, no waiting
  + 24.3   payload OCCUPANCY   -- bytes actually moving        (nowait - nopayload,nowait)
  + 19.6   WAITING             -- the dependency stall         (full - nowait)
    -----
   120.38

The fixed cost of having 60 fabric clients instead of 8 masters is **1.3 us** -- negligible. An earlier
version of this section split the same wall into "25.7 us of fabric traffic" and attributed ~13 us of it to
connect/credit overhead; that was wrong, and the `nopayload,nowait` cell is what corrects it. Of the fabric's
apparent cost, only 24.3 us is occupancy; the rest is latency, and latency is hideable.

The floor is 3 us BELOW the 79.0 us mesh matmul, which is the byte-count win showing up as a measurement
rather than an argument: the direct-L1 program reads roughly a quarter of the in0 bytes a full-K matmul does
(only the origin cores read, and only their own stripe).

Note also that the fused path's 25.7 us of gather cost is well under the 36.6 us standalone all-gather, so
overlap IS already recovering ~11 us of it. It is partial overlap, not none.

### Against the prediction

The plan above predicted ~93 us and a 77.6 us roofline. Neither held, and the reason is in the third row:
the roofline treated the gathered activation as free once it stops round-tripping DRAM, but 22.7 us says
L1-to-L1 fabric traffic is not free -- it contends with in1 streaming for NoC, and the origin's own shard
read (0.66 MB) is still DRAM. The byte-count claim itself is sound and is now structural rather than
estimated: under `DIRECT_L1` the writer never touches the staging accessor at all (it is `(void)`-cast, and
`in0`/`stage_acc` are unused), so the 5.25 MB/device staging round-trip is gone by construction, not by
tuning. It was not independently confirmed with a DRAM byte counter.

What direct-L1 did deliver: -22.7 us against Phase 1, and the fused path went from +28.3 us WORSE than the
unfused composition to +5.6 us worse than it. It is still 39% above the gate.

### What the 25.7 us is NOT

- **It is not per-packet overhead.** The mux slot holds header THEN payload, and `kMuxChannelBufferBytes`
  was hardcoded to 4096 -- BELOW the fabric's own 4352-byte max payload -- so direct-L1 was capped at ONE
  2 KiB bf16 tile per packet. Deriving the sizes from the fabric
  (`get_tt_fabric_max_payload_size_bytes` / `get_tt_fabric_channel_buffer_size_bytes`) fits two tiles.
  **Measured: 120.7 -> 120.2 us, nowait 101.7 -> 101.3.**

  That is the RIGHT magnitude, not a null result, and it is worth being precise about because the intuition
  "pack the packet, get utilisation" is correct in direction. The channel buffer is 4400 B for a 4352 B
  payload, so the header is 48 B: wire overhead goes from 48/2096 = 2.3% to 48/4144 = 1.2%. Payload bytes on
  the wire do not change -- only the header amortises -- so the predicted win is ~1.1% of the term, ~0.3 us
  against ~0.5 us measured. Packing helps by exactly the header fraction and no more. What this rules out is
  the much larger effect you would see if per-packet PROCESSING (mux slot handoffs, credit round-trips, NoC
  transaction issue rate) were the limit.

  Verified in effect rather than assumed: `TT_REGIME_A_LOG_CFG=1` prints `packet=4096B ... channel=4400B`.
- **It is not fixable with more links.** `num_links=3` and `4` both fail with *"Requested link index 2 is out
  of bounds. 2 ethernet channels available"*: **2 links is the hardware maximum on this axis**, and the
  measurements already use it.

**CORRECTION.** An earlier version of this section divided 1.92 MB by 25.7 us, called it ~75 GB/s, and
concluded the fabric was saturated and 25.7 us was a hard floor. That inference is invalid: 25.7 us is the
MARGINAL exposed cost (the wall-clock difference between fabric present and absent), not the transfer time.
If any part of the transfer already hides under compute -- and the fused path's 25.7 us against a 36.6 us
standalone all-gather says some does -- then the true wire time is larger than 25.7 us and the exposed
portion is a scheduling property, not a bandwidth one. Bytes/marginal-cost is not a bandwidth.

### TRIED AND REVERTED: deferring the fabric drain past the on-chip ring

The obvious attack on the 25.7 us, and it makes things worse. The sender does
`open -> payload -> credit -> write_barrier -> atomic_barrier -> close` and only THEN enters the on-chip
ring, so every sending core blocks on its own 32 KB draining before doing any ring or compute work.
`close()` does not have to happen there -- slot 0 is the send source and is never rewritten, and no packet
headers are allocated after the prologue -- so the drain can be deferred until after the ring, where the
ring's own `noc_async_write_barrier()` covers it.

Measured (medium/tp4/ring/2 links, 4 blocks):

    immediate drain (committed)   118.1 us   block medians 116.7 / 118.5 / 120.0 / 117.8
    deferred drain                127.3 us   block medians 118.7 / 127.3 / 139.5 / 146.2   (min 109, max 159)
    deferred, nowait              100.6 us   vs 101.3 immediate

So with nothing waiting it is marginally BETTER (100.6 vs 101.3 -- the core does get on with its own work
sooner), and with the real dependency it is much worse AND much less stable. That combination points at
contention rather than scheduling: deferring moves the egress into the window where the on-chip ring is
running, the ring sits on the arrival chain, and delaying it amplifies through every downstream core.

**The ring is the dominant NoC consumer, by an order of magnitude.** Each core forwards `shard_bytes` once
per ring step: `(G-1) * 32 KB = 224 KB` per core, and at the picked Pk=10 / 80-core config that is
**~17.9 MB of L1-to-L1 traffic per device against 1.92 MB of fabric traffic** -- roughly 9x. So the fabric
egress is not filling idle NoC; it is competing with the ring for it, and the current code is already doing
the right thing by draining before the ring rather than inside it.

That also retires the framing that the 25.7 us is "additive because it is not scheduled well". It is
additive because it is real NoC occupancy in a program that already saturates the NoC on-chip. (Consistent
with the measurement, not proven by it -- the 17.9 MB is derived from the config, not counted.)

### What is left

1. **The waiting term, 19.6 us** (`full 120.38 - nowait 100.79`). Fix: **consume in availability order, not
   ring order** -- see "What this bound does NOT rule out" above, and appendix B of the spec for the phasing.
   Not per-wave rings, and not a topology change: same traffic, same unicast ring, only the order moves.
   Phases 0-2 (committed) already made the schedule host-controlled and single-sourced; phase 3 is the
   schedule function.
2. **The payload occupancy, 24.3 us**, if the gate is to be reached at all. This is bytes over links at
   `num_links=2`, the hardware maximum on this axis, so it is a floor unless the byte count changes.

The on-chip ring is NOT on this list. It moves ~17.9 MB, 9x the fabric's 1.92 MB, but the floor measurement
bounds its entire cost at <=3.2 us (75.24 measured vs a 72.8 us DRAM roofline), because it is L1-to-L1 and
overlaps compute. An earlier version of this section had it as item 2 on the theory that the fabric was
queueing behind it; that was inferred from byte counts, contradicted by the floor, and is withdrawn.

Arithmetic worth keeping in view: floor 76.0 + the 25.7 us exposed fabric cost = 101.7, still above the
86.9 us gate. **So per-wave rings alone does not reach the gate** -- it targets the 19.0 us, and 101.7 is
what is left when that is gone. Reaching 86.9 for this shape needs the fabric's 25.7 us to come down too,
and the deferred-drain experiment above says that will not come from rescheduling it; it needs either fewer
bytes on the NoC or less competition from the on-chip ring.

## Scope limits of the implementation

Both refusals are TT_FATAL at program creation, never a silent fallback to the staged path -- a silent
fallback would let a Phase-1 measurement be reported as a Phase-2 one.

- **`Ns > 1` refused** (as designed above). Note this bites more than the pinned `ns2` test: the auto-picker
  chooses `Ns=2` for the `small` shape (32x2048x2048), so those configs are refused too.
- **More than 64 mux channels on one mux refused.** One stream register per channel, 64 per Tensix worker,
  and a mux binds exactly one link -- so `num_links` is the only lever.

  **This does not fire at the production `num_links=2`, so it is not a limit anyone meets in practice.**
  Measured, same binary, `TT_AGMM_DIRECT_L1=1`:

      num_links=1   28/40 pass   12 refused (8x Ns>1, 4x channel cap on LINE: 70/70/72/84 channels)
      num_links=2   32/40 pass    8 refused (8x Ns>1 -- the cap does not fire at all)

  Dealing the same clients over two muxes halves the per-mux count (84 -> 42), which clears every case. The
  cap binds at one link only because every consumer core is a client and, on a LINE, the client count is
  both ~2x the ring's (a stripe must fan out both ways from its origin, so the origin drives two muxes) and
  rank-dependent. Ring never hit it at either link count.

  The correctness suite now runs at `NUM_LINKS = 2`, matching production and matching every measurement
  above -- previously it hard-coded 1 link, which is how this got written up as a scope limit in the first
  place. **`Ns>1` is therefore the only genuine scope limit of this change.** The `TT_FATAL` stays: it is
  the right error for a real per-worker hardware ceiling, and it names `num_links` as the lever if anyone
  runs a LINE at one link.
- **Mux channel DEPTH is sized down to fit L1** (8 -> 4 -> 2 -> 1 buffers/channel) rather than the packet
  size, since the spec asks to optimise for the default 4 KiB packet. Without this, 48 channels x 8 buffers
  x 4 KiB = 1.5 MB against a 1.5 MB worker L1.
- **Both ablations work.** `nowait` is hooked into the direct-L1 arrival wait. `nogather` ablates the fabric
  on the HOST -- it clears the send flags in the stream plan, so the client counts come out 0 and no mux is
  deployed at all. Removing the sends from the KERNEL while leaving the muxes deployed is what hung
  (reproduced twice): mux v2 self-terminates by counting `close()` calls, so a client that never opens or
  closes leaves the forwarder RISC spinning. Deleting the mux is the only way to ablate its traffic.

Test status with `TT_AGMM_DIRECT_L1=1`, at the suite's `NUM_LINKS = 2`: **32/40 pass and the other 8 are
`Ns>1` refusals -- zero correctness failures and zero hangs.** The staged path is 40/40, at 1 and 2 links
both.

## Trap list for this area (from the branch's own history)

- Mux v2 self-terminates by counting `close()` calls against a compile-time channel count; a mismatch is a
  **hang**, not an error. Has produced silent hangs three separate times.
- Multicast rectangles must be handed corners in the issuing NOC's traversal order; wrong order aims at an
  inverted rectangle nobody receives. Invisible at Pk=1 (1x1 rectangles make the swap a no-op).
- Two different partitions of M with nothing synchronising them caused both the PCC 0.984 and PCC 0.79-0.98
  rounds.
- Anything not patched in `override_runtime_arguments` is correct on invocation 1 and stale from 2 on.
- Hardcoded compile-time accessor indices: adding CT args displaced every `TensorAccessorArgs`, built
  cleanly, and silently failed all 40 tests on PCC.
- Credit atomics must use NOC0 coords (`my_x[0]`, not `my_x[noc_index]`); the packet-header setter
  re-encodes with its own mirroring. Invisible on Blackhole, a hang on Wormhole.
- **`run host ID` in the device-profiler CSV is PER DEVICE**, not per mesh op: a tp=4 op appears as 4
  separate runids (verified: 4 devices x 30 ops = 120 runids, each on exactly one device). Grouping by runid
  and taking `max` over cores -- what the single-chip worker does -- therefore times ONE device and silently
  under-reports the makespan. Group per device, then max across devices. Cross-device *timestamps* are also
  not comparable (independent cycles-since-reset); only durations are.
- **Setup work lands in the same CSV.** `from_torch` writes and the zeroed persistent buffers add ops ahead
  of the measurement loop (measured: 2 per device), so a demux that asserts `runids == calls` fails and one
  that infers ops-per-call from the ratio silently mis-groups. Anchor on the tail with a known
  ops-per-call.
- The CSV column is spelled `run host ID`. Matching it as `run host id` yields zero runids, which reads
  exactly like "the profiler was not enabled".
