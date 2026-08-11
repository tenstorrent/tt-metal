# Step 13 — which perf tests the migration actually needs

Derived from `git diff d2555ff5379..HEAD` rather than from memory, so that nothing touched is measured
by accident and nothing untouched is measured as if it mattered.

## Method notes that change the numbers

**Baseline is `d2555ff5379`, not `main`.** `main` has advanced ~85 commits since this branch's merge
base, and it has diverged *on these very files* (the softmax readers moved to the named-compile-time-arg
API there). A branch-vs-main comparison would attribute unrelated main changes to this work. Comparing
against the branch tip before this work isolates it. Steps 1–3 were already measured against `main` in
step 6.

**Both sides must be built `--disable-profiler`.** `build_metal.sh` enables Tracy by default. The
earlier A/B attempt in step 7 was run on a Tracy build and produced ±8% run-to-run spread — enough to
show `softmax_small_w.aligned` "+7.7%" and `ragged` "−8.9%" from a change that only removes work in the
aligned case. Absolute numbers from a Tracy build are also 10–20% slower than step 6's, so they are not
comparable to it either.

**Two runs per side**, so the noise floor is established before any delta is believed (step 6's
practice).

## The affected surface

`reduce_helpers_compute.hpp` is in the diff but the change is **comment-only** (verified: no non-comment
lines), so no kernel it feeds can change. Everything else maps to an op as follows.

| # | Changed files | Op / path reached | Why it can move |
|---|---|---|---|
| A1 | `moreh_softmax_{h,w}.cpp`, their readers, `softmax_{h,w}_small` | `moreh.softmax` SMALL_H / SMALL_W | step 7a: aligned shapes stop emitting + selecting a second scaler tile |
| A2 | `moreh_softmax_{h,w}_large.cpp`, their readers, `softmax_{h,w}_large` | `moreh.softmax` LARGE_H / LARGE_W | step 9: max phase became one streaming reduce; the `Wt==1`/`Ht==1` branch and an `Accumulate` are gone |
| A3 | same kernels, `SOFTMIN` define | `moreh.softmin` | different `#ifdef` arm of the same code |
| A4 | same kernels, `LOG` define | `moreh.logsoftmax` | different arm again; LOG uses a different reduce/epilogue |
| A5 | `moreh_softmax_backward_{h,w}.cpp`, readers, `softmax_backward_{h,w}_small` | `moreh.softmax_backward` SMALL_H / SMALL_W | step 8: two-phase split + mask + fold → one reduce |
| A6 | same, `LOG` define | `moreh.logsoftmax_backward` SMALL | step 8's LOG arm, the largest structural change in that step |
| B1 | `softmax_program_factory_general_{h,w}_small` | `ttnn.softmax` dim=-1 / dim=-2, rank 3, fits L1 | shares A1's kernels; step 7b fixed a hang here |
| B2 | `softmax_program_factory_general_{h,w}_large` | `ttnn.softmax` dim=-1 / dim=-2, too big for L1 | shares A2's kernels |
| B3 | `softmax_device_operation.cpp` | `ttnn.softmax` **strategy selection** | the L1-fit estimate now counts the scaler pair and the sum-scaler tile it had always omitted, so a borderline shape can fall back `*Small` → `*Large` |
| C | `reader_moreh_{sum,mean}_h.cpp` + factories | `moreh.sum(dim=H)`, `moreh.mean(dim=H)` | runtime-arg removal only — expected exactly flat, measured to confirm |
| D | `moreh_bias_backward_multi_core_h.cpp`, reader, factory | `moreh.linear_backward` bias grad | step 10: ragged-H tiles no longer take the copy-mask-restage detour |
| E | `topk_router_gpt/kernels/compute.cpp` | `ttnn.experimental.topk_router_gpt` | step 7d: the helper adds an unpacker + packer reconfig the hand-rolled code did not have. **The one place this work could plausibly cost time.** |

## Deliberate controls

Measured and expected flat. If a control moves, the run is noise and no delta in it should be believed.

- `ttnn.layer_norm` — step 3 was a pure reader refactor; untouched since.
- `moreh.softmax` LARGE_C (`moreh_softmax_c_large.cpp`) — never touched.
- `moreh.softmax_backward` LARGE_H / LARGE_W — attempted in step 8 and **reverted**, so the code is
  identical to baseline. A non-zero delta here is a direct read of the noise floor on a
  softmax-backward-shaped workload.
- C (sum/mean over H) doubles as a control.

## Shape choices

Every case runs **aligned and ragged**. The aligned variant is not decoration:

- For A1/B1 the *aligned* case is where the step 7a win must appear (that is the shape that stopped
  paying for a second scaler tile). A ragged-only measurement would miss the entire point.
- For A2 the ragged case is where the deleted branch used to run.
- For A5/A6 the ragged case is where the mask/fold used to run.

Sizes are chosen so device work dominates dispatch: the step 6 lesson was that small shapes put
everything in a 33–39 µs band that hides real differences. B2 needs a W large enough to fail the L1
estimate so that `GeneralWLarge` is actually selected, and B3 needs a shape near the estimate's boundary.

## Not measured, and why

- `moreh_{sum,mean}_w`, `reduce_*_neg`, the 2-D kernels, `numeric.h`/layernorm-sharded — **no code
  change** (steps 11, 12, and the out-of-scope list). Nothing to compare.
- `moreh.linear` forward, `moreh.norm`, `moreh.group_norm`, `moreh.layer_norm` — untouched kernels; their
  suites were run for correctness but there is no perf hypothesis to test.
- Model-level benchmarks — this work changes per-op kernel structure; op-level measurement localises any
  regression, and no model in-tree depends on these moreh ops in its hot path.
