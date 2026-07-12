# Speech encoder device-perf optimization plan (mel 4096, 1×4 BH-QB)

## Baseline profile (before any change)

`tt-perf-report` stacked report from `test_speech_encoder.py::...max_seq_pcc` (S=4096, 24 conformer
layers, TP=4 → 4 local heads). Total device-time sum ≈ **1.78 s**.

| Op (in0 buffer) | Total % | Device time | Count | Bucket |
|---|--:|--:|--:|---|
| **GatherDeviceOperation** (dram) | **81.58 %** | **1,449,556 µs** | 192 | rel-pos bias |
| MatmulDeviceOperation (l1) | 3.57 % | 63,450 µs | 1568 | compute |
| MatmulDeviceOperation (dram) | 3.56 % | 63,295 µs | 576 | compute |
| BinaryNg (dram) | 2.71 % | 48,203 µs | 491 | add/mask |
| **AllGatherDeviceOperation** (dram) | 2.64 % | 46,937 µs | 75 | TP collective |
| Softmax (dram) | 1.37 % | 24,374 µs | 192 | attention |
| LayerNorm (dram) | 0.76 % | 13,583 µs | 150 | — |
| Conv2d (height-sharded) | 0.57 % | 10,083 µs | 24 | depthwise |
| everything else | < 0.6 % each | | | |

**One op is 82% of the model.** The `ttnn.gather` in `_relative_logits_fused` builds the conformer
relative-position bias `rel[b,h,q,k] = q_scores[b,h,q, idx[q,k]]` over the full `[B,H,Qc,S=4096]`
grid — 192 calls (24 layers × 8 query-blocks), ~7.5 ms each. It is tagged **"Other"**, ~90× slower
than DRAM bandwidth for that volume — i.e. the gather op itself is pathologically slow at this shape,
not merely bandwidth-bound.

## Rank-ordered plan

### 1. Band-window the relative-position gather  ✅ DONE (the whole ballgame)

`idx[q,k] = clip(k − q, −left_max, +right_max) + left_max` with `left_max=64, right_max=8`
(vocab=73). The index is **constant** outside the diagonal band `k ∈ [q−64, q+8]`: keys well to the
left all map to `idx=0`, keys well to the right all map to `idx=vocab−1`. For a query block
`[q0, q1)` the whole block's band fits a key window `[q0−64, (q1−1)+8]` — ~`Qc+72` ≈ 584 columns
at `Qc=512`, vs the full 4096.

So gather only that tile-aligned window and rebuild the two constant regions by broadcasting the two
clamp columns of `q_scores` (`ttnn.repeat` of `q_scores[...,0:1]` on the left, `q_scores[...,-1:]` on
the right), then `concat`. Reconstruction is **bit-exact** (validated on-device, `max_abs_err = 0`).

Isolated micro-bench (`scratchpad/gather_bench.py`, per-device H=4):

| | full gather (S=4096) | window gather (W≈600) | full-rel rebuild (concat) |
|---|--:|--:|--:|
| time | 7.6 ms | **0.58 ms (13×)** | 1.88 ms (4×, bit-exact) |

Implemented as `_add_windowed_relative_logits`, wired into `_chunked_relative_attention_matmul`
(S ≥ 3072). First cut rebuilt the full-width bias with `ttnn.repeat`+`concat` — but that `repeat`
(+ its `tilize`) then profiled as the *new* #1 op (185k µs). Fixed by adding the bias **directly onto
`scores` region-by-region**: a real add on the window slice and a **width-1 broadcast `ttnn.add`** on
each constant slice (no materialization). Toggle via `SE_WINDOWED_REL=0` to fall back to the fused
full gather for A/B.

Measured device-time-sum (1×4, mel 4096), gather bucket first:

| stage | gather bucket | total device time | speedup |
|---|--:|--:|--:|
| baseline (fused full gather) | 1,449,556 µs (82%) | 1,776,600 µs | 1.0× |
| windowed via repeat+concat | 108,961 µs | ~644,000 µs | 2.76× |
| windowed via broadcast-add | **108,961 µs (23%)** | **~470,000 µs** | **3.78×** |

Bit-exactness of the rel tensor was verified against the fused full gather (`scratchpad/rel_compare.py`
and `scores_compare.py`, `max_abs_err = 0.0` at every query block). **PCC 0.9925 — matches the fused
baseline (0.99241).**

**Bug found & fixed along the way (`ttnn.gather` aliases its input).** The first integrated version
regressed end-to-end PCC to 0.9872 despite the rel being provably bit-identical in isolation.
Bisection (dump post-conformer → per-layer → layer-0 attention → per-query-block → pre-softmax scores →
`q`) showed: `q` was never mutated, yet the **later query blocks (q0 ≥ S/2)** had scores that differed
*uniformly across all key columns* — i.e. the `c_left`/`c_right` constant columns were corrupt. Cause:
`_add_windowed_relative_logits` sliced `c_left`/`c_right` from `q_scores` **after** calling
`ttnn.gather(q_scores, …)`, and `ttnn.gather` clobbers/aliases its input buffer in some DRAM states
(only triggered once several blocks had run, hence "second half only" and "fine in isolation"). Fix:
**extract the clamp columns before the gather.** Bit-exact and PCC-clean thereafter.

Follow-ups (not yet done):
- Apply the same windowing to `_chunked_relative_attention_matmul_f32_softmax` (mel-2048 path).
- Softmax-shift-invariance variant: drop the larger constant region entirely (adding a per-row
  constant across the **whole** row is a no-op through softmax), leaving only the window gather + the
  smaller region — approaches the 0.58 ms window-only cost and removes a slice+add+concat per block.
  Not bit-exact, so gate on PCC (this path is ASR-precision-sensitive).

### 2. TP collective — AllGather (now 9.75%, 45.8 ms, 75 ops)

After #1, `GatherDeviceOperation` (23%) and the two matmul buckets (~27% combined) lead; the TP
collective is the top *non-compute* bucket. Speech encoder uses `all_gather + sum` (not `all_reduce`)
per README (stability history). Try `num_links=2` (BH-QB max) and ring topology on the gather, as was
done for the text encoder — **gate on PCC + long-seq speech stability** (README explicitly warns
against unifying to `all_reduce` without re-running those tests).

The rel bias' own reassembly (`Concat` 5.87% + extra `Slice` 5.57%) is the residue of the region-add
approach; the softmax-shift-invariance follow-up above removes most of it.

### 3. BinaryNg adds 2.71% (48 ms, 491 ops)

Mostly residual/mask adds. After #1, revisit whether the windowed-rel path can fold its constant-region
add into fewer dispatches, and whether the fused residual+LN (already used at long seq) covers all adds.

### 4. Re-profile and re-bucket

After #1 lands, the profile shape changes completely (gather no longer dominant). Re-run the stacked
report and re-rank before spending effort on #2/#3 — matmul (now ~7% combined) or the collective may
become the next real target.

## Verification

- Gate: `test_speech_encoder.py::..._max_seq_pcc` (threshold 0.99), mel 4096 on 1×4.
- Re-profile with the `useful_command.txt` tracy line; compare stacked device-time-sum + op count.
- Also check mel 512 / 2048 shapes (the f32-softmax and short paths take different branches).
