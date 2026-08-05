# MiniMax-H3 transformer block performance log

Append-only. One entry per measured change, newest at the bottom. Never rewrite an entry: the point
is to be able to see what a change actually bought, including the ones that bought little.

The commit column is the one thing here that may need rewriting: it was refreshed once when the
bringup commits were squashed, which rebased every later commit and changed its hash. If you rewrite
this branch's history again, refresh it again -- a dangling hash makes the entry unbisectable.

All numbers are **device time for one transformer block**, from
`test_transformer_block_perf_minimax_h3.py` under `--profile`, analysed with
`project_block_perf.py` (which reproduces `tt-perf-report`'s totals exactly). 768P (768x1344) on
4x8 Blackhole Galaxy, ring, 2 links, TP=4 / SP=8. Per-device sequence lengths 4768 / 9216 / 13632.

Per denoise step = block x 50 layers. Per video = step x 50 denoise steps. Both cover the block stack
only -- refiner, input projections, `norm_out` and the output heads are excluded, a few percent on top.
Device time is a lower bound on wall clock; see the `device + op gap` note in `project_block_perf.py`.

Reproduce:

```
scripts/run_safe_pytest.sh --profile \
    models/tt_dit/tests/models/minimax_h3/test_transformer_block_perf_minimax_h3.py -k 5s_768p
python models/tt_dit/tests/models/minimax_h3/project_block_perf.py 5s=<csv>
```

| # | date | change | 5s | 10s | 15s | commit |
|---|------|--------|----|-----|-----|--------|
| 1 | 2026-08-03 | baseline: the transformer as first brought up | 26.03 ms | 57.65 ms | 98.72 ms | `3329c4834ee` |
| 2 | 2026-08-03 | addcmul gates + adaLN scale/shift folded into the fused norm | 25.61 ms | 57.19 ms | 97.63 ms | `bb4bff6beb0` |
| 3 | 2026-08-04 | ring SDPA chunk sizes tuned from a measured sweep | 24.80 ms | 57.19 ms | 97.63 ms | `ffd1617cae7` |
| 4 | 2026-08-04 | QK-norm + head split + RoPE fused into one op | 18.53 ms | not measured | not measured | `23022b34768` |
| 5 | 2026-08-04 | all-gather folded into the matmuls (AGMM), placeholder block sizes | 18.20 ms | 43.22 ms | 78.26 ms | `0928969663b` |
| 6 | 2026-08-04 | AGMM block sizes from a measured sweep | 17.79 ms | 42.43 ms | 77.01 ms | `9587b239154` |
| 7 | 2026-08-04 | attention gate addcmul folded into the to_out AGMM epilogue | 17.58 ms | 42.31 ms | 76.96 ms | `989f329d0c9` |
| 8 | 2026-08-04 | ff2 + reduce-scatter + gated residual fused, blocking swept | 17.19 ms | 41.12 ms | 75.33 ms | `f7f1795a7e3` |

Cumulative at 5s: **26.03 -> 17.19 ms, -34.0%**; 1.30 -> 0.86 s per 50-layer step, 65.1 -> 43.0 s per video.
At 10s: 57.65 -> 41.12 ms (-28.7%). At 15s: 98.72 -> 75.33 ms (-23.7%).

## Notes per entry

**1. Baseline.** Unfused all-gather + matmul; rotary embedding decomposed into slices, permutes and
concats because the 48-channel half boundary is not tile-aligned. 100 device ops.

**2. addcmul + adaLN fusion.** Removed 6 elementwise ops (BinaryNg 16 -> 10 calls). Only ~1%: these
were never expensive ops, and the norm absorbed 0.08-0.30 ms of the work. No dispatch-overhead claim
is made -- the gap-inclusive totals moved +3.5% / -2.5% / -13.2%, no consistent direction.

**3. SDPA chunk sizes.** 5s moved to (320, 384) -- SDPA 8.43 -> 7.56 ms, FLOP util 50.8% -> 56.7%.
10s and 15s were already optimal at (256, 512). The winner needs k=384, which is neither a power of
two nor in the {256, 512} set an initial search covers; L1 rejects (320, 512) but accepts (320, 384).

**4. Fused RoPE.** The largest win so far. The whole 36-op rotary tail collapses into the two QK-norm
ops; 96 -> 60 device ops. Works because MiniMax-H3's half-split rotary is the fused op's interleaved
rotary under a permutation of Q/K channels, and partial rotary needs only cos=1 / sin=0 padding.

10s and 15s are recorded as not measured rather than guessed: those two profiled runs failed -- the
perf test was still passing raw rotary_dim-wide tables and the new width check rejected them -- and
`--profile` masks pytest's exit code, so they reported PASS with a CSV holding only the ops that ran
before the exception. What caught it was the numbers being impossible: 15s "faster" than 5s. Check the
pytest summary line, not the exit code, after any profiled run. Entry 5's 10s/15s figures therefore
cover entries 4 and 5 together.

**5. AGMM.** Folding the TP all-gather into the matmul that consumes it. Only -1.8% at 5s, because
the block sizes are legality-driven placeholders, not swept: `K_block` must divide
`K_tiles_per_device` and `N_block` must be even, and the generic 8x8x8 default satisfies neither. The
sweep is the follow-up; these numbers are the floor, not the result.

**6. AGMM block sizes.** 811 combos over the three shapes with
`models/tt_dit/utils/sweep_mm_block_sizes.py`. to_qkv (8,7,8)->(8,7,12) -13.4%, ff1
(8,7,8)->(8,3,14) -11.7%, to_out already optimal. Neither winner was reachable by guessing: N_block
12 and 14 sit outside the 8-or-16 range one would try, and ff1's K_block swings 11.7% between two
legal divisors of 42. The three AGMM ops fall 5.02 -> 4.58 ms at 5s. The (K, N) keying transferred --
the same block shapes were swept at M=4768 and hold at 10s and 15s.

**7. Attention gate fusion.** Ternary ops 2 -> 1 (0.29 -> 0.14 ms) with the AGMM growing 0.06 ms.
Real but marginal, and honestly at the edge of what this measurement resolves: op-level accounting
predicts -0.085 ms while the block moved -0.21 / -0.12 / -0.05 ms, against run-to-run variance of
around 0.7%. Treat entry 7 as "a small win, not a measurable one" at 15s.

**8. Fused MM+RS+addcmul, with a swept blocking.** The negative result below, resolved. Same fusion,
same three ops into one, now **-25.0% on that stage at 5s** (1.76 -> 1.33 ms) where before it was +45%.
Nothing about the fusion changed; only the blocking did.

The sweep's first axis is not the block shape but the *core grid split*. In `FusedMMRSConfig`,
`compute_with_storage_grid_size` is the matmul grid and the reduce-scatter workers occupy the rows
between it and the full device grid, so `num_workers_per_link = ((device.y - mm.y) * device.x) / (2 *
links) - 1`. The matmul's cores and the collective's bandwidth are therefore in direct competition, and
the optimum is interior:

    12x7   84 mm cores, 8 RS workers/link   M=6 K=4 N=16 sb(2,2)   1.373 ms
    12x8   96 mm cores, 5 RS workers/link   M=4 K=8 N=14 sb(2,2)   1.313 ms   <- best
    12x9  108 mm cores, 2 RS workers/link   M=6 K=2 N=8  sb(2,2)   1.487 ms

12x9 starves the reduce-scatter, 12x7 the matmul. The default's 8x7 = 56 cores at subblock 1x1 is worse
than either end of that range, which is the whole of the earlier regression.

Only M=4768 was swept; 9216 and 13632 reuse its blocking. K and N are architecture-fixed and M only
sets how many blocks a core walks, while warmup compiles one program per combo and compile time grows
with M -- M=9216 alone is ~75 min against ~9 min at M=4768. The reuse is validated where it matters,
by the measured per-duration block times above, and the fused op tracks the sweep closely at 5s
(1.33 ms measured in-model vs 1.313 swept).

One bug worth recording because it produced a *silent no-op* rather than a failure. The gate that
decides whether to fuse initially required `M_block` to divide the M tile count. 4768/32 = 149 and
13632/32 = 426 are not divisible by 4, so **5s and 15s quietly kept the unfused path** while 10s fused,
and the first profiled run showed a win only at 10s. The constraint was invented, not real: a partial
trailing block along M is fine (unlike along K, where the ring delivers fixed-size chunks), as the
sweep itself proved by measuring M=4768 at `M_block=4`. The op list is what caught it -- 5s and 15s
still listed `ReduceScatterMinimalAsync` and `Ternary`. **After any fusion, check the profiled op list
for the ops that were supposed to disappear**; a PCC test cannot tell you the fused path never ran, and
here the block correctness test was passing at 99.9995% against an unfused model.

## Measured negative results

Kept so nobody re-attempts these blind. A negative result with a diagnosis is worth as much as a win.

**Fused matmul + reduce-scatter + addcmul for the feed-forward: 45% regression on that stage.**
Replacing ff2 + reduce-scatter + the gate-MLP addcmul with one
`minimal_matmul_strided_reduce_scatter_async` turned 1.75 ms (0.94 + 0.67 + 0.15) into 2.55 ms --
**+3.3% on the block at 5s, +2.6% at 10s, +2.3% at 15s.** Three ops became one and it got slower.

The cause is blocking, not the fusion. The unfused path reaches `get_matmul_config(..., default_block_size)`
and so uses our swept sizes, running ff2 at 64.5% FLOP util; the fused path goes through
`get_fused_mmrs_config`, a *separate* table keyed on `(M, K, N)` that holds only Wan's
`(9472, 3456, 5120)` and `(2368, 3456, 5120)` on the 12x10 grid, so ours takes
`default_fused_mmrs_config`.

It misses **silently**: that helper warns only when the *grid* is unknown, and 12x10 is registered. When
AGMM had the same problem an illegal `K_block` produced a hard TT_FATAL and it was fixed in minutes;
here there was no diagnostic at all and only the measurement caught it. Anyone reasoning "three ops
into one must be faster" would have committed the regression.

Prerequisite before re-trying: swept fused-MMRS entries, one per duration since M varies
(4768 / 9216 / 13632). `sweep_mm_block_sizes.py` covers `mm` and `agmm` but not MMRS, so the sweep
needs extending first. The hooks are in place -- `ParallelFeedForward.forward_fused_addcmul` now takes
`default_block_size` -- but the fused path routes block sizes through `get_fused_mmrs_config`, not
that argument, so wiring is still required.

General lesson, now twice observed: **fusing an op without tuning its blocking can be worse than not
fusing it.** AGMM was +1.8% before its sweep and +2.3% after; MMRS is -3.3% before a sweep it has not
had.

> **RESOLVED by entry 8** (`f7f1795a7e3`), which is where the numbers now live. The diagnosis above
> held exactly: with a swept blocking the same fusion is -25.0% on the stage instead of +45%, and
> nothing but `get_fused_mmrs_config`'s table changed. Two corrections to the prerequisites as written
> above -- the entries do *not* need to be one per duration (the blocking carries across M, and
> `mmrs_config.has_mmrs_config` registers it on demand for any M), and `default_block_size` was never
> the missing piece. The "silently" point stands and `get_fused_mmrs_config` now warns on an unknown
> shape, not just an unknown grid. This block is kept unedited because the reasoning is what made
> entry 8 findable.

## Where the time goes, from the 5s profile after entry 6

54 ops, 17.78 ms. Measured shares:

| target | share | note |
|--------|-------|------|
| ring SDPA | 42.8% (7.61 ms) | chunk-tuned (entry 3); util 56.7% at 5s, 67.9% at 15s |
| the three AGMM matmuls | 25.7% (4.58 ms) | block sizes swept (entry 6); ~0.9 ms of gather still exposed |
| 6 modulation row gathers | 10.2% (1.81 ms) | 294-305 us each |
| `ff2` matmul + reduce-scatter | 9.1% (1.61 ms) | ff2 is the one matmul at healthy util (64.5%) |
| adaLN table build | 3.8% (0.68 ms) | `adaln_proj` at 2.4% FLOP util (M=32, 2 real rows) + 25 tiny layout ops |
| 2 fused norm+RoPE ops | 1.5% (0.27 ms) | entry 4 |
| gate addcmuls (2) | 1.6% (0.29 ms) | |

## Opportunities

Ordered by expected value. "Measured" means taken from a profile; "est." is a projection from
measured shares and should be treated as a hypothesis, not a result.

### A. Reachable from existing building blocks

1. ~~**Fused matmul + reduce-scatter + addcmul for the feed-forward.**~~ DONE, entry 8, after being
   attempted, reverted as a 45% regression, and re-landed once the blocking was swept. The ~1.6 ms
   estimated here was close: -0.43 ms at 5s, -1.19 at 10s, -1.63 at 15s.
2. ~~**Fold the attention gate into the `to_out` AGMM.**~~ DONE, entry 7.

### B. Dispatch -- the largest wall-clock lever, and not a device-time item

3. **Trace capture in the pipeline.** Device time is 17.78 ms but the gap-inclusive total has run
   3-5x that: at 5s the machine is idle for most of the wall clock. Every other item on this list
   optimises the 17.78 ms; this one attacks the other ~40 ms. Nothing else here can outweigh it at
   short durations.

### C. Ring SDPA -- 42.8% of the block

4. **Try `exp_ring_joint_scaled_dot_product_attention`.** Wan enables it only for (tp=4, sp=32), and
   notably configures it on the *full* grid rather than reserving a CCL column -- 120 cores against
   our 110. +9% cores on 43% of the block, for a config change. Unknown whether it holds at sp=8.
5. **Revisit the TP/SP split.** Everything tuned so far assumes TP=4 / SP=8. The 5s util deficit
   (56.7% vs 67.9% at 15s) is fundamentally that 4768 rows/device is too little work to fill 110
   cores. SP=4 / TP=8 doubles per-device sequence length at 5s at the cost of more TP traffic, which
   AGMM now partly hides. The one item that specifically targets short durations.

   **Ruled out: lowering K/V precision (e.g. bfloat8_b).** Considered and explicitly declined -- do
   not re-propose. The remaining SDPA work is grid/parallelism/kernel, not numerics.

### D. AGMM overlap -- ~0.9 ms of gather still exposed at 5s

Measured overlap efficiency (gather hidden behind matmul compute): 46% at 5s, 78% at 10s, 59% at 15s.
Perfect overlap would have been worth -9.1% of the block at 5s; fusion captured -1.8%. ~1.43 ms/block
still on the table there.

6. **Sweep `num_workers_per_link` and `num_buffers_per_channel`.** Both are currently derived
   (`full_grid.x // num_links` = 6) or hardcoded (24 on Blackhole) and have never been swept for
   these shapes -- only block sizes were. They govern how finely the ring streams.
7. **Treat `K_block` as streaming granularity, not just a compute parameter.** ff1 landed on 3 of 42
   (14 ring chunks), qkv on 7 (6 chunks). The sweep minimised total time without isolating overlap,
   so it may have chosen good compute at the cost of streaming. Sweeping K_block against measured
   overlap efficiency would show whether the 1.43 ms is reachable or a kernel limit.

### E. adaLN modulation -- 14% combined

8. **Cache all modulations for every block up front.** The timestep schedule and the modality tags are
   known before sampling starts, so every `(step, block, param)` modulation table can be computed once
   per video and reused -- eliminating the `adaln_proj` matmul and the 25-op table build from the
   per-block path entirely (0.68 ms/block measured). Note this is *not* caching across steps of a
   recomputed value: `temb` changes every step, so the win comes from precomputing the whole known
   schedule, not from reuse within it. Cost is memory: 50 steps x 50 blocks x 6 params x 6 rows x
   hidden_local in bf16 is roughly 240 MB/device -- affordable, but worth confirming against the
   weight footprint before committing.
9. **Remove the row gathers by broadcasting inside the consuming ops.** 1.81 ms measured. The fused
   norm kernel already has a per-batch adaLN broadcast mode (`[batch, 1, H]`, broadcast over seq); it
   does not fit per-row modulation directly, but this model's packed sequence is contiguous runs per
   (timestep, modality), so a segment-wise broadcast is closer to reachable than a general gather-free
   path would be.

### F. Small, and only worth doing while nearby

10. ~~**Reduce-scatter runs on 12 cores**~~ against 112-120 everywhere else -- 673 us on ~10% of the
    machine. Largely moot now that item 1 has landed: the feed-forward's reduce-scatter is inside the
    fused op, where the worker count is set by the grid split and was swept. The attention path's
    reduce-scatter is gone via AGMM, so no standalone `ReduceScatterMinimalAsync` remains in the block.
11. **The fused RoPE rotates 4 tiles where 3 would do.** The pass-through tile is rotated then
    discarded by `sin=0` (see `prepare_rope_tables`). A fraction of 0.27 ms -- noted for completeness,
    not worth chasing.
12. **The block-size sweep's N range may be clipping the optimum.** The 12x7 MMRS winner sat at
    `N_block=16`, the top of the swept range, and both AGMM winners came in at 12 and 14 -- the useful
    values are consistently at the high end. The chosen 12x8 config is interior (`N=14`), so nothing is
    known to be lost, but raising the sweep's ceiling would cost one run and settle it.

Two measurement habits this log exists to enforce:

- After a `--profile` run, check the pytest summary line. `--profile` masks the exit code, so a failed
  run still prints PASS and leaves a truncated CSV.
- Do not quote the gap-inclusive total as a result. It has moved in both directions across changes
  that only ever removed ops, and one sample per config cannot separate it from noise.
