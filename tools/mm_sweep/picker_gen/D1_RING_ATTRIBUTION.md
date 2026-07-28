# D1 — in0 ring-forward cost attribution: distance vs per-core injection

Question: the in0 ring forward costs 18–31% of the wall on the deep golden shapes, ~2.5x more than its
ideal serial transfer time. Is that **hop distance / link occupancy+contention**, or **per-core injection /
L1-source-port bandwidth**? The two hypotheses invert the ranking of every candidate optimization, so
settle it before building anything.

## Method

Two new compile-gated, program-cache-hashed diagnostic bits (`diag_mask`, `TT_REGIME_A_DIAG_MASK`). Both
PERTURB the forward instead of removing it; the readiness semaphore still targets the TRUE ring successor,
so step count, dependency chain and CB protocol are byte-for-byte the baseline's:

- **bit6 (64) `FWD_NEAR`** — identical bytes and transaction count, but the payload is written to the
  NEAREST other program core on this core's writer NoC (host-computed via
  `get_worker_noc_hop_distance`, directed, ~1 hop with 96/110 cores occupied) instead of the ring
  successor. Removes distance, keeps everything else.
- **bit7 (128) `FWD_HALF`** — true destination, half the payload bytes. Byte-linearity probe.

Reference points: mask 0 (baseline) and mask 4 (`SKIP_IN0_RING_FORWARD`, payload removed entirely = the
full cost of the forward). Mask 0 PCC 1.00003 / 1.0001 + cached replay match on both shapes. Diagnostic
outputs are intentionally invalid (both perturbations deliver garbage in0). Mask-0 binaries unchanged.

Harness: existing `ablation_matrix_worker.py`, one persistent session per relaunch, 2 warmup + 12 timed
resident-input iterations per mode, kernel wall via run-host-id demux; 2 relaunches with **reversed mode
order** (`0,4,64,128` and `128,64,4,0`). BH p150b, 1.35 GHz, fw 19.5.0.

```
export TT_METAL_DEVICE_PROFILER=1 ARCH_NAME=blackhole
python3 tools/mm_sweep/picker_gen/ablation_matrix_worker.py 512 6144 2304 2 6 1 2 1 0,4,64,128 1
python3 tools/mm_sweep/picker_gen/ablation_matrix_worker.py 512 6144 4608 2 6 1 4 1 0,4,64,128 1
```

## Result (median us; fwd/rev = the two reversed-order relaunches)

### 512x6144x2304  cfg (Ns,Pk,Sm,kb,nsb)=(2,6,1,2,1)

| mode | fwd | rev | gain us | gain % | share of full forward cost |
|---|---|---|---|---|---|
| baseline | 169.59 | 169.84 | – | – | – |
| mask 4 `SKIP_IN0_RING_FORWARD` | 117.84 | 117.38 | +51.8 / +52.5 | 30.6% | 100% (definition) |
| mask 64 `FWD_NEAR` (same bytes, ~1 hop) | 132.28 | 133.00 | +37.3 / +36.8 | 22.0% | **71%** |
| mask 128 `FWD_HALF` (half bytes, true dst) | 137.91 | 137.95 | +31.7 / +31.9 | 18.7% | 61% |

### 512x6144x4608  cfg (Ns,Pk,Sm,kb,nsb)=(2,6,1,4,1)

| mode | fwd | rev | gain us | gain % | share of full forward cost |
|---|---|---|---|---|---|
| baseline | 223.95 | 224.05 | – | – | – |
| mask 4 `SKIP_IN0_RING_FORWARD` | 182.66 | 182.64 | +41.3 / +41.4 | 18.5% | 100% |
| mask 64 `FWD_NEAR` | 195.66 | 195.48 | +28.3 / +28.6 | 12.7% | **69%** |
| mask 128 `FWD_HALF` | 193.27 | 193.61 | +30.7 / +30.4 | 13.7% | 74% |

Relaunch agreement is tight (<=0.7 us on every mode of both shapes), so the ordering is not a
warm-up/order artifact.

## Conclusions

1. **The ring forward is DISTANCE-bound, not injection-bound.** Sending the same bytes with the same
   transaction count and the same dependency chain, but ~1 hop instead of ~3.6 hops average, recovers
   **69–71% of the entire forward cost**. Under the per-core-injection / L1-source-port hypothesis this
   perturbation would have changed almost nothing. It is worth ~21% of the wall on 512x6144x2304 and ~13%
   on 512x6144x4608.
2. `FWD_NEAR` is if anything a *lower* bound on the distance share: collapsing every edge onto near
   neighbours concentrates the receiving L1 writes on a few cores (some cores are the nearest peer of
   several senders), which penalizes the perturbed mode. So distance >= 70% of the cost.
3. **The residual ~30% is the floor for any topology-only fix.** No placement or ring-order change can go
   below it; only moving fewer bytes (cross-Ns dedup, multicast, bisection-minimal gather) can.
4. **Cost is super-linear in bytes**: halving the payload recovers 61%/74% of the full cost, not the 50% a
   linear bandwidth model predicts. That is a queueing/congestion signature — the ring is operating past
   the knee, so *relieving* load pays more than proportionally. Consistent with (1).
5. Directionally this validates the placement/topology family (region-local ring re-partitioning,
   link-load-balanced ring ordering, bisection-minimal two-phase gather) and de-prioritizes levers whose
   only effect is reducing per-core transaction count or injection rate (e.g. dual-NoC forwarding), which
   can at best attack the ~30% residual.

### Caveat on sizing the realizable win

`FWD_NEAR`'s ~1 hop is not achievable by a real 8-core ring: a ring confined to one bisection region still
has a Hamiltonian cycle of ~15–20 hops (avg ~2.0–2.5 hops/edge) versus ~29 hops (avg 3.6) today. A purely
linear reading of that gives ~31% of the distance component, i.e. ~11 us / -7% on 512x6144x2304. The
super-linearity in (4) plus the elimination of *all* bisection crossings argues for more than the linear
estimate, so the honest expectation for host-only region-local re-partitioning is **-5% to -12%** on the
deep Ns>=2 shapes, with the bigger ceiling reserved for the byte-reducing ideas.
