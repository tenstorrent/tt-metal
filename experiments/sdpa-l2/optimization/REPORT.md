# FP32-destination SDPA: accuracy/performance experiments

September 9, 2026. Base: `2ba6fc2339d53300ae87c5202f335ef56492cfb3`.

## Outcome

The best low-overhead candidate found so far substantially removes the
long-context drift, but **does not reach 0.5% relative L2**. It keeps HiFi2
QK/PV matmuls and the fast approximate logit exponential. On normal inputs at
256K, L2 drops from 8.83% to 1.94%; short-query runtime increases about 3%.
Full causal attention drops from 8.72% to 1.95% with about 0.08% measured runtime
increase. This is an experimental candidate, not a general accuracy guarantee
or proof that the patch is globally minimal.

The tracked source changes and `final-hybrid.patch` contain this candidate.
Earlier patches are experiment snapshots, not alternative production fixes.

## Setup and metric

- New reservation 214149: `yyzo-bh-26`, one Blackhole P100A, firmware 19.12.0,
  KMD 2.9.0. These are **not Galaxy measurements**, and no NVIDIA comparison was run.
- Container: `yyzo-bh-26-special-cglagovich-for-reservation-214149`.
- Remote checkout: `/localdev/cglagovich/tt-metal-blackhole-20260908`.
- BF16 Q/K/V; reference uses those same rounded inputs with FP64 blockwise
  online softmax. Relative L2 is `100 * ||output - reference|| / ||reference||`,
  not squared error. The reference is checked against dense FP64 softmax.
- Unless indicated: B=1, H=1, Q=128, D=128, Q chunk=128, K chunk=512, seed=1234,
  noncausal, `fp32_hifi2`: FP32 destination, HiFi2, both approximation flags true.
  This selects the standard kernel, not `compute_streaming`.
- Full causal runs compute the entire square attention with H=4; accuracy
  normally samples the last 128 query rows per head. Timing covers the entire
  operation, not just the sampled rows. Spread-row checks are identified separately.
- Timing: one captured SDPA operation, two warm trace replays, then median of
  7 or 9 blocking replay wall times. Compilation, transfers and reference are
  excluded. Traced and ordinary outputs must match exactly. This is not a
  device-profiler cycle measurement; very small differences are timing noise.

## Main comparisons

All rows below use the same card, inputs, precision and public approximation
settings. The candidate changes internal formats/arithmetic, not Q/K/V precision.

| Workload | Main L2 % | Candidate L2 % | Main ms | Candidate ms |
|---|---:|---:|---:|---:|
| Normal, Q=128, K=4K | 2.6352 | 2.1945 | 0.2733 | 0.2823 |
| Normal, Q=128, K=256K | 8.8327 | 1.9415 | 15.1892 | 15.6560 |
| Full causal, H=4, S=4K, tail rows | 2.6245 | 2.1534 | 0.4924 | 0.4901 |
| Full causal, H=4, S=256K, tail rows | 8.7205 | 1.9527 | 1583.9723 | 1585.2513 |
| Outliers, Q=128, K=4K | 2.1203 | 1.8548 | 0.2739 | 0.2821 |
| Outliers, Q=128, K=32K | 2.0204 | 0.8676 | 1.9498 | 2.0093 |
| Outliers, Q=128, K=256K | 2.4597 | 1.1096 | 15.1886 | 15.6777 |

Sources: `bh26-base.jsonl`, `base-causal.jsonl`, `base-outliers.jsonl`,
`hybrid-fused-normal.jsonl`, `hybrid-fused-outliers.jsonl`, `final-causal.jsonl`.
Outliers are normal samples plus an independent 0.1% probability of an added
normal sample scaled by 10, in each of Q/K/V. The global L2 improvement does not
mean every row improves: outlier row-p95 L2 remains around 6–7%.

Performance is configuration-dependent. At 256K with seed=1235:

| K chunk / number of chunks | Main L2 % | Candidate L2 % | Main ms | Candidate ms | Overhead |
|---|---:|---:|---:|---:|---:|
| 128 / 2,048 | 54.0765 | 1.9538 | 20.9019 | 25.8130 | 23.5% |
| 256 / 1,024 | 19.6245 | 1.9551 | 16.6353 | 19.0846 | 14.7% |
| 512 / 512 | 8.7083 | 1.9538 | 15.1680 | 15.6824 | 3.4% |

For D=64/H=4 at 256K, main is 8.8084% / 10.9392 ms versus candidate
1.9582% / 11.8584 ms (+8.4%). Thus the patch meets a small-overhead objective
for the original D=128/K-chunk=512 workload and the measured full causal case,
**not universally**. Small chunks amortize the extra denominator work poorly.
Sources: `final-baseline-seed-chunks.jsonl`, `final-seed-chunks.jsonl`,
`final-baseline-d64.jsonl`, `final-d64.jsonl`.

## What the candidate changes

1. **Preserve recurrent state.** Promote running output CBs to FP32. Sum CBs
   were already FP32, but explicitly unpack them directly to FP32 destination;
   default Float32-CB unpack otherwise passes through TF32 SrcA/B.
2. **Treat denominator and numerator differently.** Rescale the positive
   denominator in SFPU and add to the current chunk with FP32 L1 accumulation.
   For the larger numerator, use `old + old * (correction - 1) + chunk`, with
   the leading old term retained in FP32 L1. When the maximum is unchanged,
   the correction term is zero, avoiding repeated rounding of the old state.
   The product and new chunk still have SrcA/B rounding; this is not a claim
   of end-to-end full FP32 arithmetic.
3. **Fuse the small correction work.** Calculate the max-change exponential
   with the full scale; store the correction CB in FP32. During the denominator
   update, reuse the already loaded scale to form the numerator delta. This
   avoids a separate correction read/transform pass.
4. **Improve inexpensive normalization.** Use the nonlegacy reciprocal with
   two Newton refinements; use HiFi4 only for the small elementwise correction
   product and final normalization. Large QK/PV matmuls remain HiFi2.
5. **Respect buffer formats/lifetimes.** Explicitly reconfigure correction
   packing. Reuse the consumed correction CB for the final denominator's one
   FPU reduction, not a partially advanced QK buffer. Column broadcast needs
   SrcB configured as well as SrcA.

Scope is gated to Blackhole, BF16 Q/K/V, FP32 destination, no attention sink,
in this factory. BF16-destination streaming is unchanged. Additional output
and correction CB payload is about 72 KiB/core for Q chunk=128, D=128.
Other dtype/architecture/sink paths are not enabled by this specialization.
Masks, unusual shapes, exhaustive adversarial data and all operator features
have not received a full regression campaign.

## Error attribution and rejected shortcuts

**Recurrent rounding is the long-context problem.** Merely enabling FP32
destination does not preserve precision when state is packed as BF16 or
re-enters through TF32. The candidate's seed-1235 result at 256K is 1.9538%,
1.9551%, 1.9538% for K chunks 128, 256, 512 (2,048, 1,024, 512 iterations).
Its uniform-attention/constant-V control returns exactly 1 at 4K and 256K.

**Do not apply the delta identity blindly to the denominator.** An earlier
near-zero-overhead candidate reached 1.9476% on normal 256K inputs, but worsened
4K outlier L2 from 2.1203% to 6.6995%. FP32 correction storage plus HiFi4
multiplication only reduced that regression to 3.2659%. When a maximum jumps,
`old + rounded(old) * (correction - 1)` cancels large quantities, leaving a
residual significant relative to the new denominator. Direct denominator
rescaling removes that observed regression. The numerator still uses the
identity and therefore deserves further adversarial validation.

**Reciprocal is useful but insufficient.** A hardware-estimate-only reciprocal
experiment changed 256K L2 from 8.8327% to 9.4172%; it cannot fix recurrent
state loss. Adding SFPU sum arithmetic without full-width unpack also failed
(9.2776%). Correct ingress is necessary, not just a change of arithmetic unit.

**The remaining normal-input error is mostly not a uniform gain error.** At
256K the candidate has gain 0.9971 and gain-corrected residual L2 1.9197%,
versus total L2 1.9415%. A reciprocal or output rescaling tweak cannot remove
that residual. The hard-coded fast logit exponential remains an important
source of shape error; the public false approximation flag is not honored
there on unmodified main.

**Cheap exp coefficient adjustment was insufficient.** Changing the fast
exp intercept from 32500.818359375 to 32512 on an earlier delta candidate gave
2.1256%, versus 2.1131% before the change, with essentially identical time.
This is one tested coefficient change, not an exhaustive optimization proof.

**Accurate exp and matmul fidelity show a path to the target, at high cost.**
On the earlier delta candidate's normal 256K ablation, forcing accurate logit
exp gave 0.7610% with HiFi2 at 46.2474 ms; adding HiFi4 matmuls gave 0.2436%
at 52.6237 ms (3.46x main). Public approximation flags remained true in those
rows; the saved source patch forces accurate exp for the ablation. These are
not results of the final hybrid patch and are not production recommendations.
BF16 rounding of the reference alone is 0.1676% for this input.

Earlier full-state accurate/quadratic/bit-polynomial prototypes cost roughly
2.6–3.2x on the short-query case. They do not satisfy the performance objective.
The next substantive opportunity is a much cheaper, better-shaped exponential
approximation, followed by targeted investigation of the HiFi2/TF32 residual;
changing a reciprocal alone is not enough to close the remaining gap.

## Validation and reproducibility

The final candidate builds with:

```sh
CMAKE_BUILD_PARALLEL_LEVEL=12 cmake --build build_Release --target install
```

Run in the remote checkout with `TT_METAL_HOME` set to that checkout,
`ARCH_NAME=blackhole`, and `PYTHONPATH=ttnn:tools:.:/opt/venv/lib/python3.10/site-packages`:

```sh
python_env/bin/python tests/ttnn/unit_tests/operations/sdpa/repro_sdpa_l2.py \
  --variants fp32_hifi2 --kv-lens 4096 262144 --benchmark-iters 9 \
  --max-l2-pct 2.5 --label hybrid --output /tmp/hybrid-normal.jsonl

python_env/bin/python tests/ttnn/unit_tests/operations/sdpa/repro_sdpa_l2.py \
  --variants fp32_hifi2 --kv-lens 4096 262144 --causal --heads 4 \
  --benchmark-iters 9 --max-l2-pct 2.5 --label hybrid \
  --output /tmp/hybrid-causal.jsonl
```

The 2.5% threshold above is a regression check for this candidate, **not** the
desired 0.5% acceptance threshold. `--max-l2-pct 0.5` correctly fails the normal
case. The CLI supports distributions, seed, D, head count, K-chunk sweeps and
spread query sampling; every row records the configuration and timing samples.

Additional completed checks: D=64/H=4 at 4K/256K (2.1924%/1.9582% L2),
causal spread rows at 4K/32K (1.3003%/0.5082%), finite-output checks, and
bitwise traced/ordinary-output equality. Spread aggregate L2 is weighted by
reference norm and can be dominated by early rows; its low value must not be
substituted for tail-row long-context accuracy. Those runs' median row L2
remains about 2%.

Formatting: Black 23.10.1 and git-clang-format 19.1.4. No commits or PR were made.

The final restored-main sentinel reproduced 2.63516764% at 4K and 8.83266446%
at 256K. BF16-destination streaming at 4K/32K had exactly matching recorded
accuracy metrics before/after the patch (2.508108%/2.836425% L2); see
`final-baseline-streaming.jsonl` and `final-streaming.jsonl`. This is a metric
comparison, not a saved-output bitwise comparison across builds.

After restoring the final candidate and rebuilding again, the smoke test
reproduced 2.19452005% at 4K and 1.94151951% at 256K, passing the 2.5% check.
Its 256K trace median was 15.6798 ms. See `final-restored-smoke.jsonl` and
the remote `build-final-restored.log`; the remote checkout is left on this
built candidate, not main.

## Invalid exploratory runs

Files prefixed `invalid-stale-host-` mixed restored main device headers with
a stale candidate host factory: `git archive` timestamps caused Ninja to skip
recompiling the factory. They are excluded. Forced rebuilds reproduced the
original baseline exactly. Always touch the restored factory and inspect the
build log before comparing a restored baseline.

Early `direct-state*` prototypes also failed broader causal/multiquery checks;
do not treat their short-query results as validation of a general implementation.
`hybrid` and `hybrid-v2` had a SrcB column-broadcast configuration mismatch and
are invalid. `hybrid-v3` corrected it; `hybrid-fused` and `final-*` validate the
fused candidate. Early diagnostic files may deliberately contain a numerator
or partial sum rather than an SDPA output. Empty/failed files and these
diagnostic runs are not acceptance results.
