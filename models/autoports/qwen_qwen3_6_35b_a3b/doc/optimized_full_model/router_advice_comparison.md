# Isolated FP32-router advice comparison

Real layer3 router weights at the pinned revision; six synthetic BF16 inputs
(four normal-scale, one scaled0.05 and one zero/tie control). These are not
recorded model activations and do not decide a precision policy. The unchanged
HiFi4/FP32 router, exact selector and BF16 score path are compared at fixed dtype.
Each variant has a separate clean timing process and tracking1/skip0/Watcher10
acceptance process. All192 per-run comparisons pass: six changed inputs × two
strategies × four tensors (logits/values/IDs/scores) × four ranks. Program caches
stay stable; the selected whole-model implementation is unchanged.

| Variant | Baseline ms | Candidate ms | Change |
|---|---:|---:|---:|
|l1|0.052364922|0.081232396|+55.128%|
|n2|0.052359470|0.052516150|+0.299%|
|l1_n2|0.052339894|0.054374424|+3.887%|

`l1` includes explicit DRAM→L1 input copy inside the trace and the implicit
matmul; `n2` uses four cores/per-core N2/subblock1×2 with block2 and DRAM input;
`l1_n2` combines them. All include projection, selector, softmax/BF16 scores and
four persistent diagnostic-output copies. The common copies and isolated
allocation/cache/activation conditions make absolute numbers different from
native production matmul work; paired component comparisons remain valid.
Median of three500-replay samples; no per-replay fence/upload/readback. All
samples are in the original JSON. Instrumented acceptance timings are excluded.
L1 and combined candidates lose; subblock2 has no demonstrated improvement
(about0.3% slower). Retain baseline. This is not a rejection of every wider
grid/subblock, an already-L1 producer family or natural-activation topology.

The initial probe's tracked run rejected one live trace-local score buffer.
`for x,d in zip(out,dest)` left `x` alive after `del out`; releasing both aliases
before replay fixes its lifetime. No corruptible-allocation acknowledgment or
tracker exemption was added. The original failed `router_l1_tracked.json` and
log remain; initial untracked timing rows are superseded by all six v2 runs.
The failure closed devices normally before any trace replay; no reset was
needed. The failed Watcher file and each final Watcher directory are preserved
under `watcher_router_*`; no ERROR/ASSERT/CORRUPT/FATAL match in final logs.

Reproduce from the configured checkout with lane environment/PYTHONPATH=., a
free1×4 mesh, and the installed wait_for plus bounded180s command:

```bash
python -u models/autoports/qwen_qwen3_6_35b_a3b/doc/optimized_full_model/router_geometry_probe.py --run-device --mode timing --variant l1 --output <fresh-result.json>
```

Repeat variants `n2` and `l1_n2`; timing uses tracking0, no Watcher/profiler.
Acceptance uses `--mode acceptance`, tracking1/skip0 and Watcher10, no profiler.
Per-result command/environment/source identity and exact fixtures are retained;
`router_advice_comparison.csv` links every accepted result. No production or
frozen full-model test file changed, so prior cumulative gates remain valid.
