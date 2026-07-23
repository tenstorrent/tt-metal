# AGMM Task 3 (redo) — DRAM-staged streaming ring: design

Replaces the whole-shard source-to-all Phase-A gather with a neighbor store-and-forward streaming ring,
transport-chunk-granular readiness, progressive local+remote publication, credit/lifetime correctness, and
local-first compute. Source-to-all and full-wait are retained ONLY as hashed diagnostics.

## Transport chunk
- `C=1`: one kb K-block. A chunk is `Mt × kb` bf16 tiles (contiguous per-row set of kb K-tiles across all Mt
  rows for one originating shard). `blocks_per_shard = Kt_local / kb` chunks per shard.
- Global K-block id `b ∈ [0, Kt/kb)`; block b belongs to shard `b / blocks_per_shard`. For D ∈ {2,4,8} the
  regime_a 8-way ring chunks align to shard boundaries (D | 8), so per-block readiness maps cleanly to the
  reader's per-ring-position DRAM reads.

## Fabric (one forward mux per device, credit wraps)
- Each device connects a mux v2 client toward its +1 (forward) ring neighbor. Data travels forward 1 hop.
- Credits travel to the backward neighbor via the SAME forward mux with `num_hops = D-1` (ring wrap). For D=2
  forward==backward (1 hop). No second mux.

## Relay kernel (one worker core; replaces the injector)
L1: `transport_slots` (>=2) receive slots of `Mt*kb*2048` B; `recv_sem` (backward nbr increments per chunk);
`credit_sem` (forward nbr increments when it frees a slot we sent it); packet header CB.

Ring all-gather over rounds `r = 0..D-1` (unidirectional forward). In round r this device handles shard
`s = (device_index - r) mod D`:
- r==0 (own shard, as SOURCE): for each block: read own in0 block from DRAM into an L1 slot; store to local
  gather DRAM at shard s's global offset; **publish block s-ready** (per-block readiness); if D>1 forward to
  d+1 (write payload to d+1's slot + inc its recv_sem) respecting credit availability.
- r>0 (relay/RECEIVER): for each block: wait recv_sem (chunk in an L1 slot); store to local gather DRAM at
  shard s's global offset; **publish block s-ready**; if r<D-1 forward to d+1 (this device is not the last
  recipient of shard s); return a credit to the backward neighbor (inc its credit_sem via num_hops D-1) once
  store+forward are source-safe.

Ordering/lifetime (point 5):
```
payload stored (noc_async_write to DRAM, barrier) -> publish local readiness (sem inc, fanned to compute)
forward issued (fabric write) -> flush before the L1 send-slot is reused
credit returned to backward nbr only after store + forward are source-safe
drain payload writes + non-posted atomics before kernel exit
```
Slots: `transport_slots` bounds in-flight chunks; sender waits credit_sem before reusing a slot. Test with
`blocks_per_shard >> transport_slots` to force wraparound.

## Readiness (transport-chunk granular)
- Per-shard block-progress GlobalSemaphore `blk_ready[s]` (D of them), fanned out to all compute cores. When a
  block of shard s is stored+published, `blk_ready[s]` is incremented; it reaches `blocks_per_shard` when shard
  s is fully present. The in0 reader gates each per-ring-position DRAM read of a shard-s block on
  `blk_ready[s] >= (blocks_read_of_s so far)+1` (monotonic within a shard, since a shard's blocks are stored in
  order along the ring). This is finer than the old whole-shard gate (one event per kb-block, not per shard).
- `shard_landed[s]` (D) drive the relay hand-off (recv/credit). Total GlobalSemaphores: D blk_ready + D
  shard_landed + slot recv/credit (program-local on the relay core). Reset per launch.

## Local-first compute (point 6) — via ring-position ordering, no reader-internals rewrite
- regime_a's in0 8-bank ring reads each core's OWN shard-chunk from DRAM at step 0, then forwards; the in1
  reader reads in the SAME ring order (`s=(ring_pos+G-step)%G`). Correctness is order-independent (commutative
  K-sum), so the ONLY change needed for local-first is the ring-position assignment.
- For the fused op, assign `ring_pos` per Pk band so the band's chunks that fall in the LOCAL shard get the
  early ring positions (local-first) — replacing the PARETO hop-distance objective (F5) for D>1 (local-first
  dominates for overlap). The in1 reader already follows `ring_pos`, so it reads the same local-first order.
- Combined with per-block readiness, each core reading a local chunk computes immediately; cores reading remote
  chunks wait for that block. First local matmul thus precedes local-shard transmit completion, and first
  remote matmul precedes full-gather completion.

## Profiler markers (point, gates)
Custom kernel zones (kernel_profiler `DeviceZoneScopedN` / timestamp markers) on the compute/reader:
- `AGMM_FIRST_LOCAL_MM` at the first local-block compute; `AGMM_LOCAL_TX_DONE` on the relay when the local
  shard's last block has been forwarded; assert marker(first_local_mm) < marker(local_tx_done).
- `AGMM_FIRST_REMOTE_MM` at the first remote-block compute; `AGMM_GATHER_DONE` when all D shards are present;
  assert marker(first_remote_mm) < marker(gather_done). Parsed from the profiler CSV per device.

## Diagnostics (hashed)
`transport_mode ∈ {ring_stream (default), source_to_all, full_wait}` as a hashed op attribute (env
TT_AGMM_TRANSPORT). ring_stream = this design; source_to_all = the previous whole-shard path; full_wait =
source_to_all + reader waits all shards. Interleaved multi-relaunch A/B over the three.

## Bring-up order
D=2 (no relay: r=0 source + r=1 receiver, forward==backward) → validate chunk-granular + local-first + credits
+ markers + wraparound; then D=4/D=8 ring relay. Gates: PCC>=0.999 fresh/cached/fresh-sems, Pk=1 and Pk>1,
Ns/Sm; watcher-clean forced wraparound; markers prove both overlaps; A/B reported with all runs/median/spread.

## Results (implemented)
The general-D relay (`dm_in0_ring.cpp`) is ONE interleaved per-chunk loop (send+drain), NOT sequential rounds
(which deadlock past num_slots). Received chunks are forwarded FROM gather DRAM (round r reads shard (d-r) stored
in round r-1) so there is no L1-slot lifetime across rounds.

Correctness (`test_all_gather_regime_a_matmul_async.py`): PCC>=0.999 fresh+cached+fresh-sems on every device;
D=2/4/8; Pk=1 and Pk>1; Ns>1, Sm>1, auto-picker; forced slot wraparound ((D-1)*bps >> num_slots=2);
watcher-clean D=4.

Overlap PROVEN (`agmm_markers.py`, D=4, 4 devices x 11 relaunches, all True): first_local_MM < local_tx_done
(lead ~57-66us) and first_remote_MM < gather_done (lead ~156-163us). So compute on the local shard starts before
the local shard finishes transmitting, and compute on a remote shard starts before the all-gather finishes.

Local-first (point 6): achieved via round-0 local publication + per-kb-block readiness (verified by the markers
above), NOT via an explicit ring_pos reorder — the validated F5 PARETO placement is left intact, since the
markers already show local-first holds and the ring is not the faster transport anyway (below).

A/B (`agmm_ab.py`, three transports interleaved per round, 7 scored rounds, total device-span median):
- D=4  32x6144x3072: ring_stream ~211-231us, source_to_all ~135-143us, full_wait ~136-146us (ring 1.55x slower)
- D=4  256x6144x768:  ring_stream ~668-682us, source_to_all ~388-399us, full_wait ~398-407us (ring 1.71x slower)
- D=8  32x6144x3072: ring_stream ~251-271us, source_to_all ~186-210us, full_wait ~203-226us (ring 1.29x slower)
All PCC 0.99999. ring_stream is CORRECT and overlaps as intended but is currently SLOWER than the source_to_all
diagnostic: the single relay core serially does DRAM-read (forward-from-DRAM) + fabric-fwd + fabric-recv + DRAM-
write per chunk, whereas source_to_all reads each shard once and fires D-1 non-blocking unicasts. The ring is
fabric-bandwidth-optimal (each link carries the gather once) but relay-core / DRAM round-trips dominate at these
D. The gap narrows with D (1.55x@D4 -> 1.29x@D8). Task-4 perf levers: forward-from-L1 (drop the relay re-reads),
multiple relay cores/links, and confirming the crossover at larger D.

## Corrections (fix round)
1. **Transport modes** are now `ring_store_forward` (0), `source_to_all` (1), `full_wait` (2), hashed via
   `TT_AGMM_TRANSPORT`. **Default = `source_to_all`**, chosen on the A/B evidence (fastest correct transport at
   D=4/8), NOT on the planned-hypothesis ring. `ring_store_forward` and `full_wait` remain as hashed A/B modes.
2. **Per-band overlap markers** (`AGMM_MM_LOCAL`/`AGMM_MM_REMOTE`, `DeviceTimestampedData` tagged with the Pk band
   `kk`; relay `AGMM_LOCAL_TX_DONE`/`AGMM_GATHER_DONE` zones). `agmm_markers.py` proves, D=4, all 4 devices x 11
   runs, all True: `bands_with_local_begin_local_first` (every band that owns local K begins a local block before
   its first remote block), `first_local_before_local_tx_done`, `first_remote_before_gather_done`. Bands that own
   no local K for a device correctly emit no local marker and overlap their remote compute with the gather.
3. **Local-first (point 1)** is thus PROVEN per band, not just via a single earliest-matmul marker. The regime_a
   8-core ring gives each core exactly one own-shard DRAM read (the only gated read) consumed at ring step 0, and
   the fabric gathers the local device-shard first (round 0), so each band's local-owning cores begin local
   immediately. An explicit in-ring block reorder is unnecessary here and cannot help without changing the
   physical PARETO ring (forbidden): for the pinned config W=1 (one kb-block per ring shard), so the consumption
   order is fixed by ring position; the markers confirm no local-owning band ever begins remote-before-local.
4. **Direct-slot forwarding experiment** (forward a received chunk straight from its L1 recv slot while the DRAM
   store is in flight): implemented and REJECTED. For D>2 a relayed chunk is received a full shard-round (bps
   blocks) before it would be forwarded, so holding it in L1 needs O(bps) slots — far exceeding the bounded
   `num_slots` — and a sequential-round variant deadlocks (observed: D=2 hang). At D=2 it degenerates to the
   reread path (own shard forwarded from the send slot; no relay hop). Per "adopt only if clean", it is NOT
   adopted; `ring_store_forward` uses the gather-DRAM re-read, which is exactly the trade that keeps L1 bounded.
5. **A/B with injector span** (`agmm_ab.py`, total + fabric-relay/injector span, per run/median/spread, PCC):
   - D=4 32x6144x3072: ring total ~209-230us (inj ~164-175us), source_to_all ~134-141us (inj ~56-66us),
     full_wait ~136-147us (inj ~56-66us). Ring ~1.5x slower; injector ~3x.
   - D=8 32x6144x3072: ring total ~250-268us (inj ~196-222us), source_to_all ~188-210us (inj ~121-146us),
     full_wait ~202-224us (inj ~121-146us). Ring ~1.25-1.3x slower.
   All PCC 0.99999. The injector span confirms the relay-core serial DRAM round-trips (not fabric links) are the
   ring's bottleneck.
6. **Gates**: 23/23 tests pass (D=2/4/8 x {ring_store_forward, source_to_all}; Pk=1 & Pk>1; Ns>1; Sm>1;
   auto-picker; fresh program + cached replay with fresh semaphores; forced wraparound (D-1)*bps >> num_slots=2).
   Watcher-clean D=4 ring stress (zero watcher trips). No source_to_all regression.
