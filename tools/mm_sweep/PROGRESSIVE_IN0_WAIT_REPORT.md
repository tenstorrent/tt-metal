# Progressive cumulative in0 waits for resident Regime-A matmul — report

Change: in the resident-in0 compute path (`IN0_KSLICE_RESIDENT`), replace the single full-K-slice startup
barrier with per-K **cumulative** `cb_wait_front`s during the first N-sub-block, so the first matmul begins
as soon as the first ring shard lands instead of after all G=8. CB0 stays resident and is reused across the
remaining N-sub-blocks with no further waits; popped once at the end. Shipping as the DEFAULT; the old
barrier is retained as a compile-gated A/B baseline (`DIAG_FULL_IN0_WAIT`, mask `1<<10`, compute-only,
program-cache-hashed, absent from the public API).

## 1. Previous startup serialization
The writer already delivers in0 incrementally — one ring step pushes `W` compute blocks (`cb_push_back(in0_cb,
W*in0_blk)`), G steps for the full `K_num_blocks = G*W`-block k-slice, in block-major order. But compute did:
```cpp
cb_wait_front(in0_cb, K_num_blocks * in0_block_num_tiles);  // whole slice, before ANY matmul
```
So the first matmul waited for **all G ring shards** (the core's own DRAM read + G-1 nearest-neighbor
forwards) even though block 0 (the core's own shard) is available after a single direct DRAM read. The G-1
serial forwards sat fully in front of compute.

## 2. New cumulative-wait schedule
```cpp
// startup barrier removed (kept only under DIAG_FULL_IN0_WAIT)
for (k_block = 0; k_block < K_num_blocks; ++k_block) {
    if (first N-sub-block) cb_wait_front(in0_cb, (k_block+1) * in0_block_num_tiles);  // cumulative
    cb_wait_front(in1_cb, in1_block_num_tiles);                                       // unchanged
    matmul_blocks(..., k_block * in0_block_num_tiles);  // same resident offset
}
// later N-sub-blocks: CB0 already complete -> reused, no waits
// pop K_num_blocks * in0_block_num_tiles ONCE after all reuse
```
Cumulative counts (strictly increasing, no intervening pop) satisfy the CB API contract. Only the first
resident traversal waits (`M_blocks_per_core==1`); once the slice is complete it stays resident. Arbitrary
`W` is handled: the writer pushes W blocks at a time, so a wait for a mid-batch boundary is simply satisfied
when that W-batch lands. in1 was already streamed per block (own bank shard read directly from DRAM), so the
first matmul is now gated only on in0 block 0 + in1 block 0 — both direct DRAM reads, no ring hop.

## 3. Correctness
- Public suite (progressive default): **20/20 pass** — Pk=1, split-K, Ns>1, Sm>1, Mt=1/2/4/8, balanced
  tails, cache replay, column-dependent layout.
- Diag gtest `Correctness`: all masks {0,32,64,128,256,512,1024} **PCC 0.99999** vs CPU f32 golden, fresh
  AND cached-program.
- Diag gtest `ProgressiveVsFullWait`: progressive (mask 0) vs old full-wait (mask 1024) are **BIT-IDENTICAL**
  (`ab_maxdiff = 0.0`) — matmul accumulation order is unchanged, only the wait placement moves — across
  W=1 (both Mt=8 targets), W=4, and N_bpc = 1/2/3. Both also pass PCC vs golden.

## 4. Measured overlap and speedup (same-binary compile-gated A/B, median of 3 relaunches)
Device-profiler kernel µs. `util512` = ideal(logical bytes / 512 GB/s) / measured. Raw:
`regime_a_progressive_bench.json` (every relaunch + per-RISC).

| shape | group | cfg (Ns,Pk,Sm,kb,nsb) | full-wait µs | progressive µs | Δ | util512 |
|---|---|---|---|---|---|---|
| 256×2048×1024 | target | 1,4,2,2,2 | 30.3 `[30.1,30.3,30.3]` | **28.7** `[28.7,28.7,28.9]` | **−5.2%** | 37.2→39.3% |
| 256×6144×768 | target | 1,12,1,2,1 | 54.3 `[53.9,54.3,55.3]` | **53.3** `[52.9,53.3,53.5]` | **−1.9%** | 46.7→47.6% |
| 256×6144×2304 | control | 1,12,1,2,1 | 93.6 `[93.3,93.6,94.4]` | 91.4 `[91.2,91.4,91.8]` | −2.4% | 68.1→69.7% |
| 256×6144×4608 | control | 1,12,1,2,1 | 154.1 `[153.6,154.1,154.7]` | 152.2 `[151.4,152.2,152.4]` | −1.2% | 78.8→79.7% |
| 32×6144×4608 (Mt1) | control | 1,12,1,2,1 | 118.7 `[118.6,118.7,118.8]` | 118.2 `[118.1,118.2,118.8]` | −0.4% | 94.3→94.7% |
| 64×6144×4608 (Mt2) | control | 1,6,1,4,2 | 122.3 `[121.8,122.3,122.4]` | 119.5 `[119.3,119.5,119.7]` | −2.3% | 92.6→94.8% |
| 128×6144×4608 (Mt4) | control | 1,12,1,2,1 | 131.2 `[130.9,131.2,131.5]` | 129.7 `[129.4,129.7,129.9]` | −1.2% | 88.4→89.4% |

**All 7 shapes improve; zero regressions** — a strict Pareto win (bit-identical output, faster-or-equal
everywhere). No control regresses (bound was ≤3%; worst control −2.4%, i.e. faster). Biggest win on the
narrow-N small-K target (−5.2%), smallest on deep-K/large-N where the reduction chain dominates.

### The compute RISC begins useful work earlier (per-RISC spans, median µs)
| shape | mode | wall | TRISC (compute) | BRISC | NCRISC |
|---|---|---|---|---|---|
| 256×2048×1024 | full | 30.3 | 26.4 | 26.0 | 25.2 |
| 256×2048×1024 | **prog** | 28.7 | **23.1** | 22.4 | 22.4 |
| 256×6144×768 | full | 54.3 | 45.9 | 45.4 | 45.5 |
| 256×6144×768 | **prog** | 53.3 | **44.8** | 44.4 | 44.2 |

The matmul work is bit-identical, yet the compute (TRISC) **span shrinks 26.4→23.1 µs** on the small-K
target: the removed startup barrier (which blocked every matmul) was on the critical path and is now hidden
behind the ring; the whole pipeline shifts ~3 µs earlier in lockstep. This is direct confirmation the
intended overlap is genuinely present, not merely inferred from the incremental writer pushes.

## 5. Why the gain is real but modest — the new first bottleneck
The realizable gain (−2 to −5%) is well below the `skipfwd` ablation's −25/−30% upper bound because
`skipfwd` also removed the mid-loop forwarding that was *already* hidden behind compute; only the **serial
startup** (~G-1 ring hops before the first matmul) was actually exposed, and that is exactly what progressive
recovers (~3 µs on the small-K target). After the change the first matmul is gated only on the two own-shard
direct-DRAM reads (in0 block 0 + in1 block 0) — verified: no full-slice wait precedes the first matmul (the
barrier is `#ifdef DIAG_FULL_IN0_WAIT`-only, and the A/B is bit-identical proving the compute is otherwise
unchanged). The residual is the **compute-feed + split-K reduction/ring-sync floor**: narrow-N util is still
<50% (compute floor 1.3–1.8× the DRAM ideal, established in DIAG_ABLATION_REPORT.md), and on 768/2304/4608
the linear Pk=12 reduction chain dominates. That floor — a dedicated tiny-shape Mt=8 compute/reduction path —
is the next target, unchanged from the prior conclusion; in0 delivery is no longer the exposed cost.

## 6. Follow-up: did the best factorization change? (progressive re-sweep)
Re-ran the top-10 pre-change configs for each primary under progressive (mask 0), 3× relaunch. Raw:
`regime_a_progressive_resweep.json`.
- **256×2048×1024:** winner **moves** `(1,4,2,2,2)` 29.1 µs → **`(1,4,2,2,4)` 28.0 µs, a stable −3.8%**
  (relaunch bands `[27.9,28.0,28.6]` vs `[28.5,29.1,29.2]`). The old full-wait sweep ranked nsb=2 ahead of
  nsb=4, so the flip is **progressive-attributable**: nsb=4 ⇒ N_bpc=1 (single N-sub-block, no cross-subblock
  reuse), so the one-time startup overlap is a larger fraction of the shape's work.
- **256×6144×768:** winner **unchanged** `(1,12,1,2,1)` 53.3 µs.

**Picker recommendation (NOT applied here):** update the 256×2048×1024 pick to nsb=4 for a further ~3.8%.
Deferred because the spec gates a picker change on re-running the six-shape parity + affected FLUX/LTX corpus
subset (the picker is a validated artifact: 100% on 20 prod shapes + cost-model fallback), which is a
separate re-validation effort disproportionate to a 3.8% single-shape move. Recommend doing it together with
the next picker refresh.

## 7. Status
Progressive cumulative in0 waits is correct, bit-identical to the old schedule, measurably faster on all 7
shapes with no regression, and shipping as the default. The old barrier remains as the `DIAG_FULL_IN0_WAIT`
A/B baseline. Diagnostic infrastructure and refuted delivery variants are retained (cleanup deferred).
