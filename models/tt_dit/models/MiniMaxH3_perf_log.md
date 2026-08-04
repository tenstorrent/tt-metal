# MiniMax-H3 transformer block performance log

Append-only. One entry per measured change, newest at the bottom. Never rewrite an entry: the point
is to be able to see what a change actually bought, including the ones that bought little.

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
| 1 | 2026-08-03 | baseline: block + attention as first brought up | 26.03 ms | 57.65 ms | 98.72 ms | `12c0e27c922` |
| 2 | 2026-08-03 | addcmul gates + adaLN scale/shift folded into the fused norm | 25.61 ms | 57.19 ms | 97.63 ms | `1739353a695` |
| 3 | 2026-08-04 | ring SDPA chunk sizes tuned from a measured sweep | 24.80 ms | 57.19 ms | 97.63 ms | `acfc6885b12` |
| 4 | 2026-08-04 | QK-norm + head split + RoPE fused into one op | 18.53 ms | not measured | not measured | `9c9f54d877d` |
| 5 | 2026-08-04 | all-gather folded into the matmuls (AGMM), placeholder block sizes | 18.20 ms | 43.22 ms | 78.26 ms | `368722dcf48` |
| 6 | 2026-08-04 | AGMM block sizes from a measured sweep | 17.79 ms | 42.43 ms | 77.01 ms | `45489e2b6ba` |
| 7 | 2026-08-04 | attention gate addcmul folded into the to_out AGMM epilogue | 17.58 ms | 42.31 ms | 76.96 ms | `9d4a5d85c2d` |

Cumulative at 5s: **26.03 -> 17.58 ms, -32.5%**; 1.30 -> 0.88 s per 50-layer step, 65.1 -> 43.9 s per video.
At 10s: 57.65 -> 42.31 ms (-26.6%). At 15s: 98.72 -> 76.96 ms (-22.0%).

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

1. **Fused matmul + reduce-scatter + addcmul for the feed-forward.** ATTEMPTED AND REVERTED -- a 45%
   regression on that stage. See "Measured negative results". The op exists and the arithmetic is
   right; what is missing is a swept fused-MMRS blocking for our shapes. Still worth ~1.6 ms if that
   is done, so it stays on the list, but it is no longer a free win.
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

10. **Reduce-scatter runs on 12 cores** against 112-120 everywhere else -- 673 us on ~10% of the
    machine. May be inherent to the ring. Moot if item 1 lands.
11. **The fused RoPE rotates 4 tiles where 3 would do.** The pass-through tile is rotated then
    discarded by `sin=0` (see `prepare_rope_tables`). A fraction of 0.27 ms -- noted for completeness,
    not worth chasing.

Two measurement habits this log exists to enforce:

- After a `--profile` run, check the pytest summary line. `--profile` masks the exit code, so a failed
  run still prints PASS and leaves a truncated CSV.
- Do not quote the gap-inclusive total as a result. It has moved in both directions across changes
  that only ever removed ops, and one sample per config cannot separate it from noise.
