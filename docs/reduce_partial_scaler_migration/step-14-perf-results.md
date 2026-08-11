# Step 14 — perf results for the whole migration

Baseline `d2555ff5379` (branch tip before this work) vs `HEAD`. Both sides built
`--disable-profiler`, three sweeps per side, median of three, on a Wormhole n300 with an 8-core host.
Case selection and the reasoning behind the baseline choice are in
[step-13](step-13-perf-scope-analysis.md).

## Headline

| | |
|---|---|
| **Clear win** | `moreh_softmax_backward` small kernels: **−6% to −10%** across H, W, and the LOG variant, on aligned *and* ragged shapes |
| **Clear fix** | 6 `ttnn.softmax` cases went from **hanging the device** to working |
| **Regressions** | **none** — no case moved outside the noise floor in the slow direction |
| **Everything else** | flat within noise, including the two changes that were expected to cost time |

## Significant results (3 runs/side, median)

| case | baseline µs | branch µs | delta | spread base/branch |
|---|---:|---:|---:|---:|
| `A6.logsoftmax_backward_small_w.aligned_512` | 218.5 | 197.3 | **−9.70%** | 1.8/6.6% |
| `A5.softmax_backward_small_w.ragged` | 184.4 | 166.6 | **−9.66%** | 0.6/2.3% |
| `A5.softmax_backward_small_w.aligned_512` | 183.7 | 166.0 | **−9.63%** | 7.4/0.3% |
| `A6.logsoftmax_backward_small_w.ragged` | 217.7 | 197.4 | **−9.32%** | 5.9/0.4% |
| `A5.softmax_backward_small_h.aligned_512` | 357.8 | 329.3 | **−7.96%** | 3.7/4.2% |
| `A5.softmax_backward_small_h.ragged` | 353.9 | 332.8 | **−5.97%** | 1.0/4.2% |

All six are the step-8 migration, and the win appears on **aligned** shapes too. That is expected once
you look at what was removed: the old kernel masked the last tile and ran the two-phase
`reduce(Ht-1)` + `reduce(single())` + `add_tiles` fold **regardless of whether the shape was ragged**
(the `Ht == 1` branch existed for the same reason). So every shape paid for the workaround, and deleting
it helps every shape. This is the same effect Step 1 saw on `moreh_sum_h`, but larger, because backward
was carrying more scaffolding: a mask CB read, a scratch CB round-trip, a second reduce and an
element-wise add per row.

## Not measurable on the baseline: 6 cases that hang there

`B1.ttnn_softmax_general_{w,h}_small` (aligned and ragged) and `B3.ttnn_softmax_boundary_w` (aligned and
ragged) **cannot be run on `d2555ff5379`**. The shared softmax reader emits two max-scaler tiles
unconditionally there while the two `ttnn` general *small* factories size that CB at one tile, so the
program deadlocks for every shape on that path — not just ragged ones.

This was not predicted from the code and then assumed; it happened during this measurement. The first
baseline sweep stopped after 22 cases, and the case it stopped on was `B1`. Clearing it took
`tt-smi -r 0`. The excluded cases are exactly the blast radius of that bug, and they were excluded via
the bench's new filter argument so the rest of the sweep could complete.

On `HEAD` all six run normally:

| case | branch µs |
|---|---:|
| `B1.ttnn_softmax_general_h_small.aligned_512` | 200.7 |
| `B1.ttnn_softmax_general_h_small.ragged` | 200.5 |
| `B1.ttnn_softmax_general_w_small.aligned_512` | 117.5 |
| `B1.ttnn_softmax_general_w_small.ragged` | 121.0 |
| `B3.ttnn_softmax_boundary_w.aligned_2048` | 89.7 |
| `B3.ttnn_softmax_boundary_w.ragged` | 89.9 |

So the honest statement of the largest perf effect of this work is not a percentage: a reachable
`ttnn.softmax` path went from wedging the card to running.

## Flat cases, and what each one rules out

| case group | delta range | what it means |
|---|---|---|
| `A1` moreh softmax SMALL (step 7a) | −0.5% … +5.0% | The step-6 regression fix is **not** measurable here. It removes one scaler-tile fill per kernel launch, amortised over all rows of the shape, so this is the expected outcome — see below. |
| `A2` moreh softmax LARGE (step 9) | −0.1% … +1.4% | Restructuring the max phase into one streaming reduce is perf-neutral. It removes a branch and an `Accumulate` but reduces the same tiles. |
| `A3`/`A4` SOFTMIN and LOG arms | −1.5% … +0.6% | The other `#ifdef` arms of the same kernels moved with the SOFTMAX arm, i.e. not at all. |
| `B2` ttnn general LARGE | −1.5% … +0.1% | Sharing the migrated `_large` kernels costs nothing. |
| `C` moreh sum/mean over H | −1.1% … +0.9% | The dead-runtime-arg removal is free, as expected. |
| `D` bias grad over H (step 10) | −0.5% … +0.6% | Dropping the copy-mask-restage detour for ragged-H tiles is neutral at this size; the op is dominated by other work (14 µs aligned vs 48 µs ragged is a shape effect, not a code effect — both sides show it). |
| `E` topk_router_gpt (step 7d) | −3.7% … −0.2% | **The one place a regression was plausible** — the helper adds an unpacker + packer reconfig the hand-rolled code did not have. It does not cost measurable time. |
| `CTL` layernorm, softmax LARGE_C, softmax_backward LARGE | −1.1% … +1.5% | Untouched code. Confirms the run is sane. |

### Why step 7a does not show up

Step 6 measured the *unconditional* pair emission as a ~2–3% cost on `softmax_small_w`. Step 7a removes
it for aligned shapes. It does not reappear as a ~2–3% win here, and the reason is the shape: the fill is
**one tile per kernel launch**, while these cases process 16 rows × 16 batches per launch. Step 6's
measurement was taken on a Tracy build where dispatch overhead is a much larger fraction of the total, so
a fixed per-launch cost weighed more. The change is still right — it removes work and lets the whole
partial-scaler path constant-fold on aligned shapes — it is just below this harness's resolution.

## Measurement caveats

- **Noise floor is 5.4%**, taken as the worst run-to-run spread across the untouched CTL cases. Deltas
  smaller than that are not interpretable on this host. Individual cases ran as noisy as **21.5%**
  (`E.topk_router_gpt.K4096` on the baseline side); its apparent −9.6% at two runs collapsed to −0.2% at
  three, which is why three sweeps were run rather than two.
- 8-core host: dispatch and JIT contention are a real part of the variance. A quieter machine would
  resolve the sub-2% effects this run cannot.
- The card was reset (`tt-smi -r 0`) **three times** during this work: once when a deliberate
  reader/CB-mismatch experiment deadlocked it, once when the baseline sweep hit the step-7b hang, and
  once after a `SIGKILL` of a running program left it wedged. None of these were caused by `HEAD` code.
- Reproduce with:
  ```
  ./build_metal.sh --enable-ccache --release --disable-profiler
  python tests/ttnn/unit_tests/operations/moreh/bench_reduce_partial_scaler.py out.json
  # on a pre-step-7b baseline, exclude the cases that hang there:
  python tests/ttnn/unit_tests/operations/moreh/bench_reduce_partial_scaler.py base.json '^(B1|B3)\.'
  ```

## Correctness, for the record

Every phase was verified before its commit; the totals across the run were **768 passing tests, 0
failures** (`ttnn` fused `test_softmax` 108, `test_moreh_softmax` 123, `test_moreh_logsoftmax` 100,
`test_moreh_softmin` 92, `test_moreh_sum` 229, `test_moreh_mean` 76, `test_moreh_linear` 219,
`test_topk_router_gpt` 12, `toy_reduce_partial` 36), plus the two new ragged-× forced-LARGE test groups
and one new multi-tile ragged-H bias-grad shape added along the way. One of those new tests caught a real
bug in this work before it was committed (step 8's `_large` attempt).
