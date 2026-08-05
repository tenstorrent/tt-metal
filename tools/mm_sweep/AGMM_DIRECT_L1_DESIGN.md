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

Fixing the stall while keeping ring delivery means per-wave rings (one ring per arrival wave instead of one
ring over the whole gathered K), giving `T_ready_max + T_mm/tp`. Not in this change.

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

Phase 2 decomposes cleanly:

    79.0  matmul on the mesh
    22.7  gather traffic the matmul contends with even when nothing waits on it   (nowait 101.7 - 79.0)
    19.0  pure dependency stall                                                   (full 120.7 - nowait 101.7)
    ----
   120.7

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

### What is left, in priority order

1. **The ring's zero-overlap bound, 19.0 us.** Unchanged and exactly as analysed above: `makespan >=
   T_ready_max + G*delta`, so `fused >= T_gather + T_matmul`. Per-wave rings (one ring per arrival wave
   instead of one ring over the whole gathered K) is the fix, giving `T_ready_max + T_mm/tp`. Still not in
   this change.
2. **The 22.7 us of contention.** Now the larger term. Direct-L1 moved the activation off DRAM but not off
   the NoC, and every consumer core is a fabric client, so ingress is spread over the whole grid rather
   than concentrated where it can be scheduled around in1.

## Scope limits of the implementation

Both refusals are TT_FATAL at program creation, never a silent fallback to the staged path -- a silent
fallback would let a Phase-1 measurement be reported as a Phase-2 one.

- **`Ns > 1` refused** (as designed above). Note this bites more than the pinned `ns2` test: the auto-picker
  chooses `Ns=2` for the `small` shape (32x2048x2048), so those configs are refused too.
- **More than 64 mux channels on one mux refused.** One stream register per channel, 64 per Tensix worker,
  and a mux binds exactly one link -- so `num_links` is the only lever. With every consumer core a client
  this binds on a LINE, where the client count is both ~2x the ring's (a stripe must fan out both ways from
  its origin, so the origin drives two muxes) and rank-dependent. At `num_links=1`, tp=8 line needs 70-84
  channels. Ring is unaffected at every shape tested.
- **Mux channel DEPTH is sized down to fit L1** (8 -> 4 -> 2 -> 1 buffers/channel) rather than the packet
  size, since the spec asks to optimise for the default 4 KiB packet. Without this, 48 channels x 8 buffers
  x 4 KiB = 1.5 MB against a 1.5 MB worker L1.
- **`TT_AGMM_ABLATE=nogather` refused on this path.** With payload and credit removed the senders open and
  immediately close their mux channels, which HANGS (reproduced twice) rather than measuring a floor.
  `nowait` does work and is hooked into the direct-L1 arrival wait.

Test status with `TT_AGMM_DIRECT_L1=1`: **28/40 pass, 12 refused by the two scope limits above, zero
correctness failures and zero hangs.** The staged path remains 40/40.

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
