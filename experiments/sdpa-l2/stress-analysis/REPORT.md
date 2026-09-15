# Stress-error attribution: FP32 streaming SDPA

September 11, 2026. Base `2ba6fc2339d53300ae87c5202f335ef56492cfb3` plus the
retained improved implementation. Diagnostic experiment, not a new release
qualification or performance claim.

Follow-up: [HiFi4 on both matmuls](BOTH_HIFI4.md), retaining the same approximate
exp and using a matching HiFi4 denominator reduction.

## Conclusion

The scaled-Q/K and outlier failures have a substantial, identifiable source in
our implementation: the six-fraction-bit effective Q required by HiFi2 QK.
FP32 destination accumulation does not recover its missing products.

An FP64 Q-quantization-only model reproduces most of the stress error, including
the same worst rows. A matched device ablation confirms causality: retain the
streaming algorithm, HiFi2 PV, approximate exponential, effective-weight
denominator, reciprocal and FP32 online state, but restore QK's four fidelity
phases and feed original BF16 Q with the original attention scale.
The selected stress heads improve from 0.84–0.93% L2 to 0.34–0.38%.

This is not evidence that the preprocessing should simply be removed under
HiFi2: preprocessing mitigates the six-bit grid's error. The diagnostic removes
that precision restriction as well as its workaround. A QK-HiFi4 control that
keeps preprocessing is bit-identical to the retained implementation on all six
full outputs, demonstrating that added phases cannot recover information
already discarded in preprocessing.

We should not attribute these failures to NVIDIA or relax the stress thresholds
before addressing this tradeoff. No FA3/FA4 measurements were made.

## Matched measurements

All numbers are percentages. Each row represents one deliberately selected
head, with all 512 qualification query positions, not an aggregate across heads
or a new suite-wide pass rate. For each distribution/length we selected the
case/head with the largest previously measured sampled-row error across H=5/10
and the five qualification seeds. Full device attention still runs on the
original full shape. These are failure-directed diagnostics, not holdouts.

| N | Distribution | Retained device L2 | Q-only FP64 model L2 | QK-HiFi4/original-Q device L2 | Retained worst row | QK-HiFi4 worst row |
|---:|---|---:|---:|---:|---:|---:|
| 32768 | Normal | 0.495150 | 0.344476 | 0.394478 | 0.797395 | 0.521571 |
| 32768 | Scaled Q/K | 0.857172 | 0.815590 | 0.353255 | 3.605401 | 1.016110 |
| 32768 | Outliers | 0.851759 | 0.817299 | 0.339161 | 4.535149 | 1.688531 |
| 262144 | Normal | 0.493544 | 0.345430 | 0.394192 | 0.709366 | 0.491316 |
| 262144 | Scaled Q/K | 0.838897 | 0.797029 | 0.357768 | 3.829646 | 0.984096 |
| 262144 | Outliers | 0.933626 | 0.817696 | 0.383428 | 6.887660 | 1.487808 |

QK-HiFi4 device PCC is 0.99999213–0.99999425 on these six heads. All six head
L2 values pass even the original 0.5% target. However, the 256K outlier head
still misses the original p99-row gate: **1.179555% versus 1%**. Its worst row
is below the 2% gate. This is not a claim of complete numerical qualification.

Selected cases (head numbering is zero-based):

- Normal 32K: H=10, seed=1238, head=9.
- Scaled 32K: H=5, seed=1234, head=3.
- Outliers 32K: H=10, seed=1235, head=9.
- Normal 256K: H=5, seed=1234, head=2.
- Scaled 256K: H=5, seed=1238, head=2.
- Outliers 256K: H=10, seed=1234, head=3.

## Theory and attribution

For BF16 operands, HiFi2 covers phases 0 and 1: all seven fraction bits of
SrcA, but only six of SrcB. SDPA's QK operand order puts Q in SrcB. This is
independent of destination precision. The local ISA source is pinned at
`5287a62727350bcef35f7b411d1b8a706172ec4c`; see
[the fidelity phase table](../../../../tt-isa-documentation/WormholeB0/TensixTile/TensixCoprocessor/SrcASrcB.md)
and this repository's Blackhole `llk_math_matmul.h` phase increment.

The retained path effectively uses:

```
Q_eff = bitceil6(Q_BF16) / 1.0027
S = Q K^T / sqrt(D)
delta S = (Q_eff - Q) K^T / sqrt(D)
```

For a single attention row, first-order output sensitivity is:

```
delta O = sum_j p_j * delta S_j * (V_j - O)
```

Thus small Q errors can change the mixture of competing high-weight keys.
Scaling both Q and K by two scales the unperturbed logits by four; it also
scales this Q-grid perturbation by four for these ordinary BF16 inputs. A
constant relative Q error therefore does not imply constant attention error.
Outlier entries can amplify particular score errors still more. Large relative
output errors can also reflect cancellation among V contributions.

The FP64 model implements the equation directly. At 256K, the worst scaled row
has Q-only error **3.8211%**, first-order prediction **3.8734%**, and retained
device error **3.8296%**. The worst outlier row has Q-only error **6.1669%**,
first-order prediction **6.3122%**, and device error **6.8877%**. Both models
identify the same worst row as the device in these cases. The latter row has
max |Q|=13.0625, leading attention probability 0.6253, and an
attention-weighted score-error standard deviation of 0.06488.

The cosine between Q-only and device error vectors is 0.896–0.959 on the four
stress heads. This is strong attribution evidence, not an additive partition
of squared error: probability rounding and other errors interact.

### Other contributions

- Ideal six-bit probability rounding, without Q quantization, yields
  **0.150–0.213% L2** on these stress heads, including BF16 output rounding.
  It cannot alone explain the original ~0.9% head errors or ~7% worst row.
- In a global-max FP64 model, imposing the approximate-exp cutoff at -21.45
  contributes at most **0.000738% incremental head L2**, or **0.003223% worst
  row**, across these cases. This makes the cutoff an unlikely dominant cause
  here. It is not a full range proof for the online hardware algorithm.
- Normal-input error is different: Q-only L2 is ~0.345%, P6-only ~0.371%, and
  the combined idealized model ~0.479%. The retained ~0.49% result is already
  near that model's precision budget; adjusting exp coefficients alone cannot
  be expected to remove the Q component.
- Residual device error after restoring Q precision remains meaningful. The
  approximate exp/grid, TF32 ingress of intermediate scores, P quantization,
  and state arithmetic have not been individually ruled out. The 1.18% outlier
  p99 remains an explicit open item.

The CPU P models use global-max normalization and FP64 matmuls/reductions,
with ideal RNE-six-bit weights. The streaming device instead uses its biased
cubic, per-chunk running maxima and effective truncated weights. These models
are attribution controls, **not bit-accurate hardware emulators or FA3/FA4
surrogates**. All references consume the original already-BF16 Q/K/V; original
input rounding is not charged to the operator, but Q preprocessing is.

## Engineering implications and the PR comparison

The next precision/performance experiment should target **QK separately from
PV**. QK-HiFi4 here is a diagnostic, not a proposed final default: its latency
was not benchmarked. Potential cheaper candidates include QK-HiFi3 (restore
phase 2 while leaving only the low-by-low phase 3 product out) or a Q residual
correction. Neither has been measured in this diagnostic round. A different
single six-bit rounding grid may help statistically but cannot restore the
missing information for arbitrary inputs.

The eventual PR should compare four frozen variants on the same inputs:

| Variant | Purpose |
|---|---|
| Main BF16 streaming | Existing throughput baseline |
| Main FP32 destination | Existing accuracy-mode baseline |
| Improved BF16 compensated streaming | Fast tier with reduced drift |
| Final improved FP32 streaming | Accurate tier with justified extra work |

For every advertised workload, report per-head L2/PCC, row p99/max, device
latency, useful effective TFLOP/s and overhead versus its corresponding main
baseline. For noncausal full attention, useful FLOPs are `4*B*H*S*S*D` for
QK+PV; report useful FLOP/s rather than crediting added correction arithmetic
as more useful throughput. Count preprocessing in a separate measured column
and in any end-to-end claim. Record chunk sizes, buffering, actual dispatch,
precision, software snapshot and measurement method. Unsupported cases and
fallbacks must remain visible. Do not compare a sampled-Q timing with a
full-prefill baseline or mix old wall-clock and device-counter timings.

The present results establish *why* the stress tradeoff exists and a targeted
way to reduce it. They do not fill that four-way performance table, establish
model quality, or qualify Galaxy. The retained BF16 implementation is unchanged
and was not rerun in this FP32 attribution experiment.

## Reproduction and verification

Reservation 216406, `yyzo-bh-26`, Blackhole P100A. Remote checkout:
`/localdev/cglagovich/tt-metal-blackhole-20260908`. Python 3.10.19,
torch 2.11.0+cpu, 16 CPU threads. Q/K chunks 128/1024, D=128,
noncausal B=1, original input buffering unchanged.

Use the retained improved build and the existing repository environment:

```bash
export TT_METAL_HOME=$PWD ARCH_NAME=blackhole
export PYTHONPATH=ttnn:tools:.:/opt/venv/lib/python3.10/site-packages
python_env/bin/python experiments/sdpa-l2/stress-analysis/analyze.py \
  --device --lengths 32768 262144 --output /path/to/new-baseline.jsonl
git apply --check experiments/sdpa-l2/stress-analysis/qk-hifi4.patch
git apply experiments/sdpa-l2/stress-analysis/qk-hifi4.patch
CMAKE_BUILD_PARALLEL_LEVEL=12 cmake --build build_Release --target install
python_env/bin/python experiments/sdpa-l2/stress-analysis/analyze.py \
  --device --experimental-qk-hifi4 --original-q --output /path/to/new-original-q.jsonl
python_env/bin/python experiments/sdpa-l2/stress-analysis/analyze.py \
  --device --experimental-qk-hifi4 --output /path/to/new-retained-q-control.jsonl
git apply -R experiments/sdpa-l2/stress-analysis/qk-hifi4.patch
CMAKE_BUILD_PARALLEL_LEVEL=12 cmake --build build_Release --target install
```

Use fresh output paths: the diagnostic refuses to overwrite results.
The experimental flag records an external patch; it does not itself switch
fidelity. The device compute config still says HiFi2 because only the
experimental QK LLK calls override it. JIT compilation and execution are
required after applying the patch.

Raw results and logs are adjacent to this report. `summarize.py` cross-checks
all input/position provenance, the six bit-identical retained-Q controls, and
the restoration probes. Numerical models self-test quantization, the FP64
reference and first-order sensitivity. The retained source hashes are recorded
in `../qualification-v1/RESTORED-SHA256.txt`; the experimental patch is kept as
an artifact, not applied to the local retained kernels.

Verification completed: both the experimental and restored Release builds
succeeded, and the experimental device kernels compiled and executed. All
18 matched device diagnostics passed provenance checks; all six retained-Q
HiFi4 controls matched the qualification full-output hashes exactly. After
reversing the diagnostic patch, all four retained operator-source hashes
matched, and the 32K normal plus 256K outlier restoration probes reproduced
their original full outputs exactly. Python formatting and the read-only
summary cross-check passed. No production kernel changes or commits were
retained, no qualification thresholds were changed, and no performance or
model-quality claims are made from these diagnostic runs.
