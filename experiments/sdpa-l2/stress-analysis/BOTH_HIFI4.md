# HiFi4 on both QK and PV, with unchanged approximate exp

September 11, 2026. Follow-up to [QK-only attribution](REPORT.md). This is a
matched correctness diagnostic, not a latency benchmark or suite qualification.

## Outcome

HiFi4 PV further improves normal-input accuracy, but its incremental benefit is
smaller for the selected stress heads after QK precision is restored. It does
not remove the remaining 256K outlier p99-row failure.

All values below are relative L2 percentages on exactly the same selected heads
and 512 sampled query rows used in the preceding experiment. Full device
attention runs on the original complete inputs. The cases were selected by
worst previously observed sampled-row error, not held out.

| N | Distribution | Retained QK2/PV2 + Q preprocessing | Original Q, QK4/PV2 | Original Q, QK4/PV4 |
|---:|---|---:|---:|---:|
| 32768 | Normal | 0.495150 | 0.394478 | 0.237896 |
| 32768 | Scaled Q/K | 0.857172 | 0.353255 | 0.319625 |
| 32768 | Outliers | 0.851759 | 0.339161 | 0.305752 |
| 262144 | Normal | 0.493544 | 0.394192 | 0.233993 |
| 262144 | Scaled Q/K | 0.838897 | 0.357768 | 0.318737 |
| 262144 | Outliers | 0.933626 | 0.383428 | 0.378522 |

The both-HiFi4 result in more detail:

| N | Distribution | PCC | Row p99 L2 % | Worst sampled row L2 % |
|---:|---|---:|---:|---:|
| 32768 | Normal | 0.999997177 | 0.282939 | 0.330573 |
| 32768 | Scaled Q/K | 0.999995373 | 0.693071 | 0.793390 |
| 32768 | Outliers | 0.999995338 | 0.854806 | 1.547020 |
| 262144 | Normal | 0.999997298 | 0.282876 | 0.293762 |
| 262144 | Scaled Q/K | 0.999995212 | 0.730148 | 0.784592 |
| 262144 | Outliers | 0.999992882 | **1.140383** | 1.482808 |

All six selected heads pass the original 0.5% head-L2 and 0.99998 PCC gates;
the 256K outlier head still misses the 1% row-p99 gate. This says nothing about
unmeasured heads/cases passing: it is not a full rerun of qualification.

## What changed, and what did not

- Original BF16 Q/K/V, no Q preprocessing, attention scale `1/sqrt(128)`.
- FP32 destination and recurrent state; compute streaming; noncausal B=1,
  D=128, Q/K chunks 128/1024; same input buffering and core grid.
- QK and PV use HiFi4. The probability-times-ones denominator reduction also
  uses HiFi4, so it consumes the same effective probability bits as PV.
- P remains stored in the existing FP32 circular buffer and enters the matrix
  unit through its existing TF32 path. This is not full-FP32 operand matmul;
  original Q/K/V and final output remain BF16.
- The fast-exp grid, cubic coefficients, range cutoff, max-change correction,
  reciprocal and FP32 state update code are unchanged.
- In particular, the exp polynomial still contains the half-six-bit-ULP bias
  designed for the retained HiFi2 probability truncation. It was deliberately
  not retuned: this tests the requested fidelity change, not the best possible
  HiFi4-specific exp implementation.

Simply switching PV while leaving the denominator LoFi would be inconsistent:
PV would consume probability bits omitted from the denominator, creating a
normalization mismatch. The matched-denominator change is essential to this
experiment's interpretation.

The temporary [patch](both-hifi4.patch) sets the actual compute kernel's math
fidelity to HiFi4 after the existing FP32-streaming dispatch and geometry have
been selected. The Python compute config remains HiFi2 to select that retained
experimental dispatch; the patch overrides the emitted compute descriptor.
Both QK and PV therefore actually compile with HiFi4, unlike merely recording
a different metadata label. It also replaces the denominator's explicit LoFi
override with the kernel fidelity. The earlier QK-only patch is not applied.

## Interpretation

The large normal-input improvement is consistent with the earlier probability
quantization model: six-bit P precision was a substantial part of the normal
error budget. For these stress cases, most of the original error was instead
removed by fixing QK precision; increasing PV precision buys much less.

The near-unchanged 256K outlier result (0.3834% to 0.3785%) and p99 tail
(1.1796% to 1.1404%) mean that PV precision is not the dominant remaining
limitation for this head. Intermediate score precision and the retained
approximate-exp/grid behavior are candidates for further isolation, not
proven causes. This experiment does not establish an inherent BF16 accuracy
floor or predict FA3/FA4 results. We have not changed the exp bias, tested
accurate exp in this streaming ablation, or ruled out the remaining arithmetic.

No latency or effective TFLOP/s benchmark was collected. The elapsed times in
the diagnostic JSON include input generation, transfers, CPU reference work
and potentially JIT compilation; they must not be used as device timings.
Whether both-HiFi4 is a worthwhile production tradeoff remains unmeasured.

## Reproduce

Use reservation 216406 on `yyzo-bh-26`, or an equivalent Blackhole setup, with
the retained improved checkout and its existing Python/toolchain environment.
Base commit and input/selection provenance are in [REPORT.md](REPORT.md).

```bash
export TT_METAL_HOME=$PWD ARCH_NAME=blackhole
export PYTHONPATH=ttnn:tools:.:/opt/venv/lib/python3.10/site-packages
git apply --check experiments/sdpa-l2/stress-analysis/both-hifi4.patch
git apply experiments/sdpa-l2/stress-analysis/both-hifi4.patch
CMAKE_BUILD_PARALLEL_LEVEL=12 cmake --build build_Release --target install
python_env/bin/python experiments/sdpa-l2/stress-analysis/analyze.py \
  --device --experimental-both-hifi4 --original-q --output /path/to/new-results.jsonl
git apply -R experiments/sdpa-l2/stress-analysis/both-hifi4.patch
CMAKE_BUILD_PARALLEL_LEVEL=12 cmake --build build_Release --target install
```

Use a fresh output path; the runner refuses overwrites. The experimental flag
records the applied patch and does not implement a runtime fidelity switch.

Artifacts: [results](both-hifi4.jsonl), [execution/JIT log](both-hifi4.log),
[build log](both-hifi4-build.log). The read-only `summarize.py` compares all
three configurations, checks identical original-input hashes and sampled-row
positions, and checks the two post-experiment restoration outputs.

Verification completed: experimental and restored Release builds succeeded;
all six device diagnostics compiled/executed and passed provenance checks.
The temporary patch was reversed, all four retained operator-source hashes
matched, and 32K/256K outlier restoration probes reproduced the original full
output hashes exactly. Python formatting, syntax and summary checks passed.
No production kernel changes, qualification changes or commits were retained.
