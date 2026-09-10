# KDA offset performance analysis

Date: 2026-09-10
Revision: `1bc289c7b2c` (`mvasilijevic/kda-wrap-runtime`)
Hardware: eight P150b Blackhole devices, LoudBox, SP2xTP4, Fabric 1D

## Verdict

The slowdown has two regimes:

- At `T=1280`, `C=640`, baseline and split both use one recurrence group. A
  split costs a nearly offset-independent **7.92--7.94%**, or about
  **0.28--0.30 ms** under sustained load. There is no single bad matmul. The
  fixed cost is spread across three small affine matmuls, three selectors,
  expanded convolution-carry packing, and a final-state all-gather and slice.
- At `T=5120`, `C=2560`, a split also forces recurrence from four groups to one.
  A drift-balanced run measures **15.37--15.60%** split overhead. A controlled
  counterfactual shows group collapse alone costs **12.73%** (about 1.47 ms),
  explaining roughly **83%** of the total slowdown. Fixed split plumbing adds
  the remaining **2.68 percentage points**, about 0.30 ms.

The earlier 5K figures of +19.2% to +22.7% came from a run whose samples drifted
by about 18%. Preconditioning and rotating measurement order reduced within-case
spread to about 2%; the defensible current-HEAD result is about +15.4%.

## Wall-time results

Every trace was warmed and then preconditioned with 50 executions. Ten rounds
timed ten trace executions per case. The case order rotated each round, so every
case occupied every order position twice. Overhead is the median of per-round
comparisons with that round's baseline.

| Sequence / local rows | Offset case | `actual_start` | Median wall | Paired overhead |
| --- | --- | ---: | ---: | ---: |
| `T=1280`, `C=640` | Baseline | 0 | 3.493 ms | -- |
| | Rotation only | 640 | 3.500 ms | +0.05% |
| | Smallest split | 32 | 3.788 ms | +7.93% |
| | Midpoint split | 320 | 3.791 ms | +7.92% |
| | Largest split | 608 | 3.777 ms | +7.94% |
| `T=5120`, `C=2560` | Baseline | 0 | 11.504 ms | -- |
| | Rotation only | 2560 | 11.494 ms | +0.07% |
| | Smallest split | 32 | 13.307 ms | +15.60% |
| | Midpoint split | 1280 | 13.290 ms | +15.43% |
| | Largest split | 2528 | 13.282 ms | +15.37% |

The split positions are flat within 0.02 percentage points at `C=640` and 0.23
points at `C=2560`. Offset-dependent row volume is therefore not the dominant
factor in the runtime-wrap implementation.

## Which matmuls and data movement cost time at `C=640`

A callsite-resolved profile compared `S=0` with the largest split, `S=608`.
The profiling harness synchronized after every TTNN operation to associate
device programs with Python callsites. These device durations are attribution
evidence; traced wall time above remains the performance result.

| Split-only work | Exact callsite | Device-time delta |
| --- | --- | ---: |
| Final recurrent-state gather | `ttnn.all_gather`, `recurrence.py:661` | +63.9 us |
| Published-transform and tail-seed selection | 3 x `ttnn.where`, `recurrence.py:69` | +45.0 us |
| Tail-seed affine matmul | `ttnn.matmul`, `recurrence.py:156` | +32.4 us |
| Compose affine A | `ttnn.matmul`, `recurrence.py:127` | +16.7 us |
| Compose affine B | `ttnn.matmul`, `recurrence.py:135` | +17.3 us |
| Final recurrent-state slice | `ttnn.slice`, `recurrence.py:668` | +9.8 us |
| Move head/tail summaries to L1 | 4 conversions, `recurrence.py:114-115` | +14.2 us |
| Convert selected summaries for transport | typecast/move, `recurrence.py:580` | +11.3 us |
| Second range-summary call | `summarize_chunk_recurrence`, `recurrence.py:312` | +14.6 us |
| Expanded convolution-carry packing | `convolution.py:42-100` | about +66 us |

The three matmuls are now pinned down: the two in `_compose_affine`
(`recurrence.py:127,135`) and the one in `_apply_affine`
(`recurrence.py:156`). Together they cost **66.3 us**. Their two associated
adds cost another 10.8 us. The three `where` calls cost 45.0 us.

The broad data-movement bucket is also identifiable:

- `convolution.py:42-100` publishes two fragment ends instead of one. Extra
  padding, layout conversion, concatenation, and slicing cost about **66 us**.
  The fused `qkv_causal_conv1d_silu` kernel changed by only +2.7 us, consistent
  with noise; the cost is carry packing, not convolution computation.
- `recurrence.py:111-116,579-582` moves the two head/tail affine pairs through
  L1 and the BF16 transport boundary, costing about **25.5 us**.
- `recurrence.py:661-673` gathers all final states and slices the boundary
  state, costing about **73.8 us**.

The main scan changed by only +1.8 us at `C=640`. The callsite profile's summed
per-program maxima changed from 3.278 ms to 3.566 ms, a +288 us delta, agreeing
with the sustained traced-wall delta of roughly 0.28--0.30 ms.

## Why `T=5120` is slower

At `C=2560` there are 80 local 32-row chunks. The normal path uses
`group_chunks=20`, hence four groups. The split path assigns
`group_chunks=geometry.num_chunks`, hence one group, at
`recurrence.py:519-531`. This avoids needing a second intra-chip prefix chain
after the wrap, but sacrifices recurrence parallelism.

A controlled wall counterfactual captured three traces from the same layer and
inputs:

| 5K counterfactual | Median wall | Paired overhead vs four groups |
| --- | ---: | ---: |
| Normal `S=0`, four groups | 11.438 ms | -- |
| Normal `S=0`, forcibly one group | 12.912 ms | +12.73% |
| Largest split, necessarily one group | 13.209 ms | +15.42% |

The one-group baseline reproduces 1.47 ms of the split's 1.77 ms penalty while
executing no offset path. The remaining split-specific difference is 0.297 ms.
This directly establishes group collapse as the dominant 5K cause.

The callsite profile locates the resulting kernel slowdown:

| 5K callsite | Baseline | Largest split | Delta |
| --- | ---: | ---: | ---: |
| `recurrent_chunk_scan`, `recurrence.py:361` | 372.1 us | 1032.1 us | **+660.0 us** |
| `summarize_chunk_recurrence`, `recurrence.py:312` | 300.8 us / 1 call | 876.9 us / 2 calls | **+576.0 us** |
| Final-state gather and slice, `recurrence.py:661-673` | -- | 87.0 us | +87.0 us |
| Three affine matmuls, `recurrence.py:127,135,156` | -- | 66.7 us | +66.7 us |
| Three selectors, `recurrence.py:69` | -- | 44.9 us | +44.9 us |
| Expanded convolution carry, `convolution.py:42-100` | machinery present | two-slot machinery | about +80 us |
| `affine_exclusive_scan`, `recurrence.py:635` | 70.9 us | 18.8 us | -52.1 us |
| `reduce_affine_transforms`, `recurrence.py:605` | 40.8 us | 27.4 us | -13.4 us |

The summary and scan increases total 1.236 ms before smaller prefix savings.
They result from changing `groups_per_head` from four to one; they are not three
new scan calls. The fixed orchestration remains close to 0.30 ms in absolute
terms, as it was at `C=640`.

## Optimization order supported by the measurements

1. Recover multiple groups across a wrap at 5K. This is the only target with
   evidence for a double-digit relative win. It requires entry states for the
   post-wrap groups from a second intra-chip prefix chain.
2. Replace or narrow the final-state all-gather. It costs 64--77 us plus about
   10 us to slice on SP2 and will move more data on Galaxy SP8.
3. Fuse the affine micrograph. The three matmuls, two adds, and three selectors
   cost about 122 us and also introduce launch gaps.
4. Compact convolution-carry exchange. Publishing only the history rows in their
   final layout could avoid 66--80 us of packing and unpacking.

These are optimization hypotheses, not speedup claims; each needs an
implementation counterfactual and traced measurement.

## Validation and limitations

All hardware commands ran outside the filesystem sandbox. The authoritative
wrapper result was `SAFE_PYTEST_RESULT: PASS` for:

```text
MPLCONFIGDIR=/tmp/kda-offset-mpl scripts/run_safe_pytest.sh --run-all -s -q \
  models/demos/deepseek_v3_d_p/tests/kda/perf/test_offset_perf.py \
  --tt-arch blackhole

MPLCONFIGDIR=/tmp/kda-offset-mpl scripts/run_safe_pytest.sh --run-all -s -q \
  models/demos/deepseek_v3_d_p/tests/kda/perf/test_offset_callsite_profile.py \
  --tt-arch blackhole

MPLCONFIGDIR=/tmp/kda-offset-mpl scripts/run_safe_pytest.sh --run-all -s -q \
  models/demos/deepseek_v3_d_p/tests/kda/perf/test_offset_stable_perf.py \
  --tt-arch blackhole
```

The temporary callsite, stable-timing, and counterfactual harnesses were used
only to collect this report and then removed. One initial counterfactual attempt
completed the workload but failed during harness cleanup before printing; it is
excluded. The corrected rerun passed.

This is SP2xTP4 LoudBox evidence. `C=640` matches Galaxy's per-SP-rank compute
shape, but neither fabric scale nor final-state collective cost has been
measured on Galaxy SP8xTP4. The performance workloads do not themselves compare
outputs with a golden reference; numerical accuracy evidence is documented in
`kda_offset_prototypes_briefing.md`.
