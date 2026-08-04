# FIXED (2026-08-03): reduce-scatter non-finite output — c_2 circular-buffer WRAP MISALIGNMENT

## Root cause

`c_2` (out_cb) was sized `2 * out_blk_tiles`, but under reduce-scatter both the producer (compute) and the
consumer (writer) move **`max_chunk = ceil(out_blk_tiles / Pk)` tiles per sub-block**, not whole sub-blocks.
When `2*rs_T` is not a multiple of `max_chunk`, a push eventually STRADDLES the circular-buffer wrap:
compute packs `max_chunk` tiles at `get_write_ptr` when fewer than that remain before the buffer end,
corrupting a partial block.

**Fix** (`plan.hpp`, `compute_cb_sizes`): size `c_2` in `max_chunk` units under reduce-scatter --
`cb2_tiles = rscatter ? 2*max_chunk : 2*out_blk_tiles`. Two chunk slots suffice (compute pushes one chunk per
sub-block, the writer pops one), and this is SMALLER than before: it frees **24-96 KB of L1 per core** on
reduce-scatter shapes rather than costing anything. Verified to move ZERO picks despite the looser feasibility.

## Why this single mechanism explains every observation

| observation | explanation |
|---|---|
| needed `Nbpc >= 11` | with rs_T=16, Pk=6, max_chunk=3 into a 32-tile CB, the first misaligned push is #11; Nbpc=10 never reaches it |
| corruption GREW with Nbpc (244 @ 12, 1188 @ 16) | more sub-blocks = more misaligned wraps |
| independent of Kt (fails at 192, 96, 48) | Kt does not enter the CB sizing |
| `Pk=4` and `Pk=5`/rem=1 clean | max_chunk=4 divides 32 exactly |
| `nsb=4` clean at identical rs_T/Pk/rem | Nbpc=6, below the wrap |
| only 4-5 bad elements per TILE | partial-block write at the wrap boundary, not a whole bad tile |
| shipped `(5,1,2,4,3)` safe (40/40 iters) | max_chunk=3 divides its 24-tile CB, and Nbpc=1 |

Note the same invariant was applied CORRECTLY to the ring's own CBs (`c_8`/`c_9` are exactly `2*max_chunk`,
and the code comments say so) -- `c_2` was simply missed. This is the third CB-granularity bug in this
op's reduce-scatter work, after the fused-epilogue push/pop desync (`max_chunk` vs `nt`) and the
`use_reduce`/`c_7` coupling.

## Verification

All 6 known-failing configs clean (`256x6144x6144`, `512x3072x6144` x2, `512x2304x6144`, `256x1536x8192`
which was the worst at 1188 non-finite, `224x1536x6144`), all controls and the shipped config unchanged.
Suites: 111 correctness + 40 audit + 10 golden perf pass. The 5 repros are now permanent regression tests
(`test_audit_reduce_scatter_cb_wrap_alignment`), asserting `isfinite()` as well as PCC.

## HISTORICAL: how it was found and the hypotheses that failed


Found 2026-08-03 by the exhaustive Tier 1/worst-util config sweeps (BH p150b, HEAD @ 12-entry table update).
NOT yet fixed. No shipped or proposed config triggers it (verified) -- but the picker is being steered
toward this region, so it should be fixed or gated before that changes.

## Symptom

The op returns silently-wrong output containing NaN and Inf. Example, 256x6144x6144 at (6,1,1,2,2):

* 229 NaN + 13 Inf out of 1,572,864 elements (0.015%)
* spread over **112 distinct tiles**, only **4-5 corrupted elements per tile**
* all 8 M row-tiles affected
* column tiles cluster at the **END of every bank's 24-tile N band**: 22-23, 46-47, 70-71, 94-95,
  116-119, 141-143, 165-167, 190-191 -- i.e. the LAST sub-block(s) of each band

**Partial-tile corruption implies a RACE, not uninitialised memory** (uninitialised data would corrupt
whole tiles). The location says the slip happens in the ring's final sub-block drain.

### Trigger characterisation (superseded by the root cause above)

An earlier version of this document claimed `Pk >= 5 AND (rs_T mod Pk) >= 2 AND Nbpc > 1`. **That is
REFUTED**: at Pk=6, rs_T=32, rem=2 the config is CLEAN with nsb=4 (Nbpc=6) but FAILS with nsb=2 (Nbpc=12).
All four known-failing configs have `N_sub == 2`, but N_sub==2 alone is not sufficient either (Pk=4 and
Pk=5/rem=1 are clean at N_sub=2). Do not rely on a predicate derived from the table below; treat the listed
configs as the known repros and re-derive the trigger with instrumentation.

Measured evidence:

| Pk | rs_T mod Pk | Nbpc | result |
|---|---|---|---|
| 4 | 0 | 12 | clean |
| 4 | 2 | 12 | clean  <- refutes "uneven partitions are broken" |
| 5 | 1 | 12 | clean  <- refutes "Pk>=5 is broken" |
| 5 | 2 | 12 | **NON-FINITE** |
| 6 | 2 | 12 | **NON-FINITE** |
| 6 | 4 | 12 | **NON-FINITE** |
| 5 | 2 | 1  | clean (40/40 iterations) <- refutes "Pk>=5 + uneven is broken" |
| 6 | 0 | 8  | clean (nsb=3, rs_T=24) |
| 6 | 2 | 6  | **clean (nsb=4, rs_T=32)** <- SAME Pk/rs_T/rem as a FAILING nsb=2 case |

9 configs found failing across 3 shapes: 256x6144x6144 (6,1,1,{1,2,4},2), 512x2304x6144 (5,1,1,{1,2},2),
512x3072x6144 (6,1,2,{1,2},2) and (6,1,1,{1,2},2).

## What was RULED OUT

* **Chunk-size protocol mismatch between writer and compute.** Both derive the partition identically
  (`base = rs_T/P`, `rem = rs_T - base*P`, `csize(c) = base + (c<rem)`, `coff(c) = c*base + min(c,rem)`).
* **Ring-position disagreement.** compute gets `cp.rs_pos = p`; the writer gets `cp.rs_own_chunk = (p+1)%Pk`
  and derives `rs_pos_local = (own-1)%P == p`. They agree.
* **Send/receive size mismatch.** The writer sends `csize((rs_pos_local + P - t) % P)` at round t; the
  receiver's compute reads `csize((rs_pos - t - 1) % P)`, and since receiver = sender's next
  (`rs_pos_recv = rs_pos_send + 1`), these are the same chunk. Verified algebraically.
* **A harness artifact.** The fp32 reference is computed once per process and other configs in the SAME
  process return PCC ~1.0; PCC is only computed after a successful call returns.

The remaining suspect is the epoch/credit accounting (`g`, the *global* epoch whose parity selects the
double-buffered slot) as it carries across sub-block iterations -- consistent with the corruption appearing
only when there are multiple sub-blocks (Nbpc > 1) and only when several chunks differ in size.

## Why our tests never caught it

1. `test_regime_a_matmul_audit.py`'s uneven-partition cases are `uneven_9over4` (rs_T=9, Pk=4) and
   `uneven_6over4` (rs_T=6, Pk=4) -- **both Pk=4, the one value that is clean.** There is no Pk=5 or Pk=6
   uneven case anywhere in the suites.
2. It only surfaces as a PCC failure because NaN poisons the whole correlation. Had the corruption been
   finite garbage at 0.015% density, PCC would have stayed ~0.9999 and passed every threshold we use --
   the same false-pass mode as the fp32-subblock DST bug earlier in this campaign.

### Fix attempts that FAILED before the real cause was found (all reverted)

1. **Arrival-ordering of the credit.** The sender issued `noc_async_write(payload)` then
   `noc_semaphore_inc(credit)`, relying on write->atomic ordering, with `noc_async_writes_flushed()` only
   AFTER the inc. Replacing that with a full `noc_async_write_barrier()` before the inc: 229 -> 227 NaN.
   NOT the cause.
2. **Variable transfer size.** The sender wrote `csize(chunk)` bytes into a `max_chunk`-sized slot. Sending a
   constant `max_chunk_bytes` (safe: the receiver only reads its `dn`-tile prefix) removes every
   size-dependent path from the transfer: 229 -> 231 NaN. NOT the cause.
3. **DST overflow in the accumulate.** Ruled out by inspection: `rs_add_chunk` does
   `acquire_dst/add_tiles/pack_tile/release_dst` per tile at dst index 0, so only one DST tile is live
   regardless of chunk size.

Also verified consistent by inspection (so NOT the cause): the writer's and compute's chunk partitions
(`base`/`rem`/`csize`/`coff` are computed identically); the ring position (compute gets `rs_pos`, the writer
derives `(rs_own_chunk-1)%P` and the factory sets `rs_own_chunk=(rs_pos+1)%P`); per-round send vs receive
chunk identity; cb_recv slot parity vs the sender's `g&1` offset (capacity is exactly 2*max_chunk);
cb_send push/pop balance (P-1 each).

### Lessons

1. **A CB whose push/pop granularity does not divide its capacity is a latent wrap bug.** Audit every CB
   against the granularity its producer/consumer actually use, per code path -- the rs path uses a different
   granularity for c_2 than the chain does.
2. **Assert `isfinite()`, not only PCC.** The corruption was 0.015% of elements; finite garbage at that
   density leaves PCC at ~0.9999 and passes every threshold we use. NaN only tripped PCC because it poisons
   the whole correlation.
3. **Uneven-partition coverage must span Pk.** The suite had two uneven cases, both Pk=4 -- a value where
   max_chunk happens to divide the CB. That is why this shipped undetected.
4. Minimising the repro was what cracked it: showing Kt was irrelevant and that Nbpc had a THRESHOLD
   (clean at 10, broken at 12, worse at 16) pointed straight at a wrap, after three plausible
   race/size/DST hypotheses had each been refuted on hardware.
