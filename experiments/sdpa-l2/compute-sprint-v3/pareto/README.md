# Matched accuracy / resident throughput plot

Output: `sdpa_pareto_l2_distribution.png` (3040 × 1710 pixels).

The plot contains only D, C, B, A, E, G. Throughput comes from the exact final
resident measurements in the parent report. Accuracy was measured freshly on
the same original inputs for every variant, avoiding the unequal case mixes
of earlier per-agent qualification suites.

## Accuracy data

`matched-v1.json`: 126 completed runs on bh-lb-08, reservation 224379. Each
case has one head, 256 query rows, D128, noncausal Q256/K512. KV length varies;
these are rectangular query samples, not full square self-attention timings.
Original inputs are BF16 and reference attention is FP64. Seed 20260919.

- Broad panel: six distributions × three KV lengths (4096, 32768, 262144).
  Normal; normal clipped to ±2 on Q/K/V; Q/K scaled by 0.25; Q/K scaled by 2;
  sparse outliers; uniform attention (Q=0 with normal K/V).
- Separate stress panel: common Q, K, or V offset +32, each at 32768 KV.
  These are diagnostic extremes, not part of the broad-suite box statistics.

The six variants share original-input hashes for every case. Two actual trace
replays match eager output bits in all 126 runs. Original/prepared inputs and
selected source hashes are unchanged. No compute kernels were edited. C/D use
the final early FP32 fusion; B/E/G use final validity-tracked group-two state;
A is unchanged. Canonical E/G device preprocessing is retained. The existing
host libraries were reused and the selected device kernels compiled via JIT.

The data include L2, PCC, row errors and maximum absolute error. The plot shows
global relative L2 only. In particular, small global common-V error does not
imply accurate small variations around that offset.

## Interpretation

Broad-suite boxes are descriptive quartiles of 18 equally weighted case L2
errors; the center is the median and whiskers are the actual minimum/maximum.
They are not confidence intervals or estimates of a production distribution.
Dots are individual case errors at the true measured throughput (no x jitter).
Stress dots use Q/K/V-specific markers and show range/median only.

Throughput is useful QK+PV FLOPs per core on resident repeated KV with no
recurring input DM, measured separately from these accuracy cases. It excludes
preprocessing and is not model speedup or measured chip throughput. All six
selected choices are plotted, even where one is dominated on these two axes.
In particular, G has smaller KV storage/communication, an additional axis not
represented here. Do not infer a universally ordered accuracy frontier from
median errors alone.

A's broad-suite maximum (96.54%) is uniform attention at 256K, not normal QKV.
D/C broad-suite maxima are 0.2903%/0.4126%; common-K stress remains outside
those bounds. Thus the separate stress panel is important.

## Reproduce

Run `collect.py --output NEW_FILE.json` only through the shared device-lock
wrapper. It writes partial progress and marks `complete=true` only at the end;
do not plot partial files. The plotting script reads `matched-v1.json` and the
existing resident evidence, and writes the PNG plus `plot_summary.json` with
quantiles, exact throughput and evidence-file hashes.

Local render command (isolated temporary plotting environment):

```sh
/tmp/sdpa-pareto-plot-venv/bin/python experiments/sdpa-l2/compute-sprint-v3/pareto/plot.py
```

Both Python files compile locally; final PNG was visually inspected. Parent
qualification sources and production sources were not changed.
