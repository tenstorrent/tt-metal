# E-family KV precision ablation

Output: `sdpa_pareto_kv_precision.png` (3230 × 2090 pixels).

E_bfp8 is the previous E; E_bfp4 is the previous G. E_bf16 is a new storage
instantiation of exactly the same compute algorithm. No production or frozen
experimental compute/dataflow source was changed for this experiment.

## Controlled numerical recipe

All three use Q RNE7 stored BF16, LoFi QK/PV, BF16 destination, the same
approximate exponential and normalization, and the final `group2_valid`
compensated-state implementation. Q256/K512/D128, one core, the reader/writer,
CB tile counts, and two K/V input slots are identical. Only K/V preprocessing,
storage data format, and consequent CB page sizes differ.

| Variant | K and V preprocessing/storage | Bytes per 32×32 KV tile | Total CB allocation |
| --- | --- | ---: | ---: |
| E_bf16 | Per-value RNE5, stored BF16 | 2048 | 1,333,248 |
| E_bfp8 | Per-value RNE5, then native BFP8 packing | 1088 | 1,087,488 |
| E_bfp4 | RNE directly onto the shared-exponent BFP4 grid, with saturation | 576 | 956,416 |

RNE5 and RNE7 count significant bits, including the leading bit, not fraction
bits alone. RNE5 prepares K/V for LoFi's effective right-operand precision;
using unrounded BF16 instead would reintroduce truncation bias. Existing device
preprocessing supports BF16 output, so no new preprocessing kernel was needed.
The prepared tensors were compared exactly against independent CPU models.
This experiment does not claim an exhaustive search over other rounding schemes.

## Results

Measured 2026-09-21 on bh-lb-08, reservation 227098, device 0, grid 12×10,
firmware 19.13.1, KMD 2.9.0. Existing libraries were reused; the new BF16-KV
device kernel instantiations compiled and ran via JIT. Every device invocation
used the existing exclusive device-lock / dirty-marker wrapper.

| Variant | Resident TFLOP/s/core | Broad median L2 | Broad L2 range | Normal 256K L2 | Common-K +32 L2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| E_bf16 | 2.122001 | 3.0213% | 1.6600–7.8448% | 3.5853% | 46.9110% |
| E_bfp8 | 2.090987 | 3.0868% | 1.6342–7.9750% | 3.6462% | 46.9229% |
| E_bfp4 | 2.090106 | 16.2753% | 11.1903–41.1341% | 16.4053% | 79.8916% |

BF16 KV is about 1.48% faster than BFP8 KV in this resident test. Normal-input
L2 decreases by only 1.7–2.2% relative across the three lengths (about
0.061–0.070 percentage points absolute). Across all 21 cases BF16 has lower L2
in 17, but not all: two uniform-attention cases, scaled Q/K at 32K, and common V
are slightly worse. It does not fix common-K stress. Thus removing shared-
exponent BFP8 storage yields a modest change, not a different accuracy tier.
The observation is consistent with other errors in this LoFi/BF16 recipe
dominating ordinary BFP8 storage loss, but it is not a separate causal ablation
of every error source.

BFP8 and BFP4 resident throughput is effectively equal. Relative to BF16,
their physical tile bytes are 53.125% and 28.125%, respectively, including
shared-exponent overhead. Those storage and ring-communication advantages
remain important and are not represented by compute-only throughput. The
small BF16 resident advantage is not evidence of a model or ring speedup.

## Accuracy methodology and checks

`accuracy-v1.json`: 63 runs, 21 per E-family variant, seed 20260919. Same original
BF16 Q/K/V as the previous matched suite, FP64 attention reference, one head,
256 query rows, D128, noncausal. The query samples are rectangular, not full
square self-attention timings.

- Broad suite: normal, clipped ±2 Q/K/V, Q/K ×0.25, Q/K ×2, sparse outliers,
  uniform attention (Q=0); each at 4096, 32768, and 262144 KV tokens.
- Stress suite: common Q, K, or V +32, at 32768 KV tokens. Separate from broad
  quartiles and ranges. Low global common-V error can hide error in small
  variations around the offset.
- Two actual trace replays match eager output bits for every run; original and
  prepared inputs are unchanged. All selected source hashes are unchanged.
- All 42 E_bfp8/E_bfp4 accuracy outputs reproduce old E/G output hashes exactly.
  Original-input hashes match the previous suite for all 63 new runs.
- `smoke-v1.json`: 15 additional runs, covering first/second/odd-final chunks,
  multiple query blocks, normal/uniform/constant/zero V. The packed-format
  variants also match the frozen builder's outputs and original CB metadata.

## Throughput methodology

`perf-v1.json`: same one-core resident repeated-KV benchmark for all three,
Q256/K512/D128, 16 query repeats × 512 KV chunks. Useful FLOPs count QK and PV
only: `4 * 256 * 512 * 128 * 16 * 512 = 549755813888`.

Inputs are loaded once, with no recurring input data movement; preprocessing
is outside timing. Each variant has nine timed warmups, then 12 measured trace
replays, interleaved in rotated/reversed orders. TFLOP/s/core is useful FLOPs
divided by median blocking trace replay time. This is not a chip-throughput or
end-to-end model measurement. Host launch/completion overhead is included.

| Variant | Median ms | Min–max ms, 12 samples |
| --- | ---: | ---: |
| E_bf16 | 259.0743 | 259.0617–259.0945 |
| E_bfp8 | 262.9169 | 262.8885–263.0142 |
| E_bfp4 | 263.0277 | 263.0201–263.0764 |

The plot preserves D/C/B/A accuracy and throughput from the previous plot's
frozen evidence; these four were not remeasured. E-family timings are all new
and contemporaneous. No x jitter or fitted frontier curve is used. Boxes are
case quartiles with min–max whiskers, not statistical confidence intervals.
All seven choices are shown even when dominated on these two axes.

## Reproduce

From the remote repository, run each mode with a fresh output filename:

```sh
bash experiments/sdpa-l2/compute-sprint-v1/run_locked.sh \
  experiments/sdpa-l2/compute-sprint-v3/kv-precision-v1/bench.py \
  --mode smoke --output experiments/sdpa-l2/compute-sprint-v3/kv-precision-v1/smoke-NEW.json
```

Repeat with `--mode perf` and `--mode accuracy`, each with its own new output.
Do not plot partial reports. `plot.py` intentionally reads the final v1 files:

```sh
/tmp/sdpa-pareto-plot-venv/bin/python experiments/sdpa-l2/compute-sprint-v3/kv-precision-v1/plot.py
```

The plot script validates completion, integrity flags, matched inputs, and
legacy packed outputs before plotting, and saves source-data hashes and exact
summary statistics in `plot_summary.json`. The PNG was visually inspected.
Python files pass local compilation. Existing tracked changes and prior plots
were preserved.
