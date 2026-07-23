# AGMM Task 5 — Phase B: direct remote-L1 streaming (design blueprint)

Goal (plan Task 5): receive remote A transport chunks into bounded **L1** slots and feed regime_a compute
directly, instead of staging them through the DRAM gather buffer. Task 4 showed the fused compute span
(~143 µs at 32x6144x3072, D=4) is ~55 µs above single-chip regime_a (~88 µs) because the gather buffer is a
**second in0 DRAM read** competing with in1 bank BW. Removing that round-trip is the measured ceiling for
Phase B. Keep the staged-DRAM path (Phase A) behind a selector for the A/B (Task 6).

## What stays vs changes

- **Stays:** the fabric mux v2 injector reading the local shard once and unicasting to peers; the per-shard
  `shard_landed` GlobalSemaphores + monotonic `gather_ready` sequencing; the regime_a compute engine, in1
  reader, and 8-bank in0 ring; correctness oracle + tests.
- **Changes:** remote A no longer lands in DRAM. The injector (sender) targets **remote L1 slots**; a receiver
  path deposits chunks into the consumer's in0 ring L1 with credit-based flow control; the copied in0 reader
  reads local shard from DRAM (as today) but consumes remote K-blocks from the L1 slots rather than the DRAM
  gather buffer.

## Direct-L1 dataflow (per receiving device)

```
fabric ingress (mux client) -> bounded L1 slot ring (transport_slots deep, C*kb*Mt tiles/slot)
   -> publish slot's kb subblocks progressively to the in0 ring (local NoC)  -> compute
   -> return a credit to the sender once the slot is consumed AND any onward forward is source-safe
```

- **Slots:** `transport_slots` (start 1/2/4) L1 buffers, each `C*kb*Mt` bf16 tiles (the plan's transport chunk;
  the host plan already sizes `transport_l1_bytes = slots * C*kb*Mt*2048` and caps at ~L1/3). Reserve on the
  ingress/worker cores, NOT on the in1 BRISC.
- **Credits:** a sender may reuse slot s only after the receiver signals slot-consumed (semaphore) AND the
  local flush/barrier proves the source L1 is no longer read by onward forwarding. Readiness != source
  lifetime (plan decision principle). Test with more chunks than slots (forced wraparound).
- **Payload-before-readiness** on the same NoC/fabric route, as in Phase A.

## Key implementation choices to decide (experiments, per plan)

1. **Direct ring-core injection vs dedicated ingress/egress worker.** Compare feeding the 8-bank in0 ring
   directly from the fabric ingress vs a dedicated ingress worker that hands off over local NoC. Do NOT reread
   local A from DRAM for forwarding (read-once: first local DRAM read feeds both compute and forward).
2. **How remote K-blocks enter the in0 ring.** The Phase-A reader reads all K (local+remote) from the DRAM
   gather buffer via the 8-bank ring. For Phase B, remote K-blocks must be seeded into the ring from L1 slots.
   Cleanest: a compile-gated `AGMM_DIRECT_L1` variant of the copied in0 reader where, for K-blocks owned by a
   remote shard, it waits the slot-ready credit and reads from the L1 slot address instead of DRAM; local
   K-blocks still read DRAM. Preserve the existing ring forward for on-chip distribution to Ns/Sm consumers.
3. **Selector.** Add an internal `transport = {staged_dram, direct_l1}` knob to the op attributes (hashed,
   not public) so Task 6 can A/B the exact same config. Default stays staged_dram until Task 6 adopts.

## Correctness/ordering checklist (plan §Direct-L1 flow control)

- bounded slots + explicit credits; sender advances only after consume + source-safe.
- payload ordered before readiness on the same route.
- retain write flush/barrier before popping/overwriting a source slot.
- drain payload writes AND non-posted semaphore atomics before kernel termination.
- watcher-clean (use `TT_METAL_WATCHER_DISABLE_ETH=1`), fresh+cached, more chunks than slots.

## Measurement (Task 6, reuse `tools/mm_sweep/agmm_profile.py`)

Interleaved multi-relaunch A/B staged_dram vs direct_l1 over the corpus + D=2/4/8; report per-RISC compute
span, injector span, total device span, slot-depth/credit-stall stats. Adopt direct_l1 only on a stable win
with no correctness/hang/watcher/control regression; a hybrid needs a single explainable predictor
(e.g. required L1 capacity or transport-to-in1 time ratio).

## Risk notes

- L1 pressure: slots compete with regime_a CBs; the host plan's ~L1/3 cap is the guard — validate against the
  actual per-core CB footprint at each config.
- The overlap is ALREADY near-complete at Mt=1 corpus shapes (Task 4), so Phase B's win is mostly the removed
  DRAM read, not more overlap; expect the biggest gains on compute/in1-bound shapes, and re-check narrow-N.
