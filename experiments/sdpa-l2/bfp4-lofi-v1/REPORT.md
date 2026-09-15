# BFP4 / LoFi attention — first experimental checkpoint

September 15, 2026 UTC. Blackhole P100A, yyzo-bh-08; initial reservation
219611, completed work on reservation 220027. Base research checkpoint:
637d956c887. No production kernel, dispatch, or selected-option changes.

## Conclusions

1. **The operand-width hypothesis works on silicon.** BFP8-left/BFP4-right
   and BFP4/BFP4 give bit-identical LoFi and HiFi4 outputs in the tested
   single-product, QK, and PV geometries. Reversing the mixed operands does not.
2. **Quantization, not LoFi arithmetic, dominates initial attention error.**
   Q8/K4/P8/V4 has approximately 16% normal-input L2 with host-quantized QKV,
   or 32% with the tested native device QKV conversion. This is not yet a
   competitive-accuracy replacement for any of the four locked modes.
3. **Host and device BFP4 conversion are materially different.** A device-
   validated model is essential; an ideal nearest-rounding model is optimistic.
4. **Centering plus Hadamard helps outliers/common modes, not ordinary Gaussian
   representation error.** No evidence here establishes model-level quality.

## Measurement contract

- Original BF16 inputs; B1/H1/D128, 128 query rows, N distinct K/V rows;
  noncausal, K blocks fixed to 512. These are short-query accuracy diagnostics,
  not full square-N attention calls or Galaxy measurements.
- `device_attention.py` runs QK and PV on Blackhole with **LoFi, FP32 DST,
  FP32 matmul output, packer L1 accumulation disabled**. P conversion also
  runs on device. Softmax, maximum tracking, and recurrent numerator/denominator
  updates run on the CPU in FP32; final output is explicitly rounded to BF16.
  Preprocessing/corrections are CPU computations, not optimized TT kernels.
- `numerics.py` isolates quantization using FP64 arithmetic and a validated
  FP32-input quantizer. It algebraically reproduces K512 online maximum
  updates/rescaling; this is not a finite-precision recurrence emulator.
- Reference is FP64 attention on the **original** BF16 inputs. Global relative
  L2 means `100 * ||output-reference|| / ||reference||`; PCC is also retained.
- Both original-P and represented-P (matched) denominators are recorded.
- **No fused LoFi SDPA or performance result is claimed.** Script runtimes
  include CPU work/transfers/JIT and are not device-time measurements.

## 1. Packer behavior: established for the tested typecast configurations

Final probe compares threshold sweeps and random blocks, both signs,
multiple exponents, BF16/FP32 sources, and BFP4/BFP8 outputs: 655,360 output
values per conversion route. Both final host and device models match exactly.
Subnormal, exceptional-value, and overflow behavior is not covered.

| Conversion route | Observed/modelled behavior |
| --- | --- |
| Host BFP4/BFP8 | Mantissa alignment, then target-width nearest-even rounding and saturation |
| Device BFP8 | Shared-exponent nearest-away rounding |
| Device BFP4 | Per-datum E8M6 nearest-away rounding, shared-exponent BFP8 nearest-away rounding, then truncation to three magnitude bits |

The first BFP4 rounding can change the group's exponent. For example, with
group values 1.75 and 1.9921875, the device can produce 1.5 and 2.0;
host conversion produces 1.75 and 1.75. This is not captured by simply
rounding to shared-exponent BFP8 and truncating.

Host alignment also discards sticky bits before rounding. A rare transformed-
FP32 threshold exposed a one-element mismatch against ideal mathematical
RNE; the final host model reproduces the integer alignment implementation.

These observations describe the current **typecast/packer configuration**, not
an assertion that every possible Blackhole packer configuration must behave
identically. A custom kernel must revalidate its chosen pack configuration.

Evidence: [probe-final.jsonl](probe-final.jsonl), `probe.py`, and the local
ISA `Packers/FormatConversion.md` early/late conversion descriptions.

## 2. LoFi fidelity isolation

FP32-DST matmuls compared against HiFi4 on exactly the same host-quantized
operands. QK uses stored K plus `transpose_b=True`, preserving K's native
16-channel exponent groups; PV uses native V rows.

| Left / right formats | LoFi-vs-HiFi4 QK L2 | LoFi-vs-HiFi4 PV L2 |
| --- | ---: | ---: |
| BFP4 / BFP4 | 0%, bit-identical | 0%, bit-identical |
| **BFP8 / BFP4** | **0%, bit-identical** | **0%, bit-identical** |
| BFP4 / BFP8 | 1.558% | 1.692% |
| BFP8 / BFP8 | 1.559% | 1.693% |
| BF16 / BF16 | 2.590% | 2.630% |

The single-product control gives the same qualitative result without a
dot-product accumulation confound. These findings support wider Q/P on SrcB,
narrower K/V on SrcA. They do not prove equal end-to-end throughput.

## 3. Attention accuracy

Normal N(0,1), seed1240, no preprocessing. QKV use the host conversion; P
uses the measured device conversion. Entries are **L2 % / PCC**.

| K/V length | All4, original denominator | All4, matched denominator | Q8/K4/P8/V4, matched denominator | Evidence |
| --- | ---: | ---: | ---: | --- |
| 4,096 | 31.201 / 0.951983 | 45.466 / 0.951742 | 16.434 / 0.986643 | Device-backed diagnostic |
| 32,768 | 31.029 / 0.953462 | 43.646 / 0.953279 | 16.023 / 0.987224 | Device-backed diagnostic |
| 262,144 | 31.339 / 0.952289 | 44.566 / 0.952169 | 16.944 / 0.985824 | Quantization-only model |

Second seed1241 at 32K gives mixed-format **16.583% / 0.986483**, confirming
the ordinary-input band in this small check. For perspective, our historical
same-geometry normal-32K sweep gave main BF16 2.660–2.718%, FAST 2.479–2.529%,
balanced 0.383–0.387%, and ACCURATE 0.178–0.180%. Those four are historical
measurements, not newly rerun hardware baselines in this investigation.

For the complete seed1240 32K device-backed set, model and device global L2
differ by at most **0.000133 percentage points**. This agreement isolates the
large errors to representation/quantization rather than matmul arithmetic.

### Where the mixed-format error comes from

Normal32K, seed1240, host QKV quantization, device P conversion, accurate
arithmetic, no preprocessing:

| Quantized parts | L2 % |
| --- | ---: |
| Q8/K4 only; exact P/V | 11.768 |
| P8/V4 only; exact Q/K | 11.070 |
| Both | 16.023 |
| Neither; BF16 output rounding only | 0.166 |

These errors are not additive bounds; the ablations locate both major sources.
Simply improving exp or increasing fidelity cannot recover discarded K/V bits.

### Quantization route matters

Normal4K, all QKV converted by the current **device** path:

| Variant | Original-denominator L2 % | Matched-denominator L2 % |
| --- | ---: | ---: |
| All4 | 48.807 | 38.082 |
| Q8/K4/P8/V4 | 32.855 | 32.855 |

Thus a host-prequantized result cannot stand in for fused on-device
preprocessing. Ideal RNE for P is also an ablation, not an implemented
efficient device solution: with host QKV it lowers normal32K all4 matched L2
from 43.646% to approximately25.783%, still far from our existing FAST mode.

## 4. Preprocessing and normalization lessons

Measured32K, host QKV, device P, matched denominator:

| Input / variant | No preprocessing, seeds1240/1241 | Center Q/K + randomized Hadamard, seeds1240/1241 |
| --- | ---: | ---: |
| Normal, all4 | 43.646 / 44.761% | 43.881 / 44.694% |
| Normal, mixed | 16.023 / 16.583% | 16.156 / 16.605% |
| Sparse outliers, all4 | 45.808 / 25.185% | 18.600 / 18.548% |
| Sparse outliers, mixed | 47.785 / 24.120% | 16.655 / 13.958% |
| Common K+32, mixed (seed1240) | 78.011% | 16.106% |

Q centering restores the higher-precision `q_mean @ centered_K.T` correction
before softmax. Hadamard acts on the feature dimension after centering and
preserves the unquantized dot product. Its CPU cost is excluded from any
performance claim because there is no such claim yet.

Rotation alone can catastrophically worsen common-mode cases; the observed
ordinary-outlier benefit does not authorize applying it blindly. Common-V
global L2 can be misleadingly tiny after BF16 output rounds to a constant.

Native P4 packing keeps only about69.4% of the unquantized row mass and zeros
approximately49.1% of weights in the normal32K host-QKV case. P8 zeros about
0.27% and preserves nearly all row mass. Matching the denominator restores normalization and
constant represented V, but it cannot restore relative weights and can
increase L2 on zero-mean V. Conversely it helps some device-QKV cases. Both
normalizers must remain ablations, not an assumed universally superior choice.

## Next experiments

1. Retain Q8/P8 with K4/V4 as the useful native-format starting point; all4
   remains an accuracy/cost control, not the default candidate.
2. Investigate less-biased device BFP4 conversion and residual correction for
   K/V. Host RNE already shows the limits of fixing rounding alone.
3. Add a BFP8-only LoFi control with right-operand pre-rounding to the observed
   five-significant-bit width. It can distinguish BFP4 storage loss from what
   one-phase compute could achieve with a wider storage format.
4. Test actual model QKV and downstream quality before deciding whether a
   10–20% synthetic operator-error band is useful. Sage-style model-quality
   claims do not imply a sub-0.5% operator-L2 target.
5. Implement/fuse and measure resident-core plus end-to-end throughput only
   after identifying a numerical scheme worth optimizing. No change to the
   locked four-option frontier is justified by these results.

## Reproduction and verification

In the configured Blackhole container, with a writable local JIT cache:

```bash
export TT_METAL_HOME="$PWD" ARCH_NAME=blackhole
export PYTHONPATH=ttnn:tools:.:/opt/venv/lib/python3.10/site-packages
export TT_METAL_CACHE="$PWD/experiments/sdpa-l2/bfp4-lofi-v1/.jit-cache/"
python_env/bin/python experiments/sdpa-l2/bfp4-lofi-v1/probe.py --label NEW_PROBE
python_env/bin/python experiments/sdpa-l2/bfp4-lofi-v1/device_attention.py \
  --label NEW_DEVICE --length 32768 --distributions normal outliers common_k \
  --preprocessing none center_qk_hadamard
python_env/bin/python experiments/sdpa-l2/bfp4-lofi-v1/numerics.py \
  --label NEW_MODEL --qkv-round host --p-round device
```

`validate.py` checks the saved evidence without torch/device access. Python
syntax and Black checks passed; device matmul/pack kernels JIT-built and ran.
No host build was required for these Python-only changes. The existing
selected-variant manifest still validates all14 pinned sources and136 repeats.

Primary evidence: `probe-final.jsonl`, `device-attention-host-qkv-v1.jsonl`,
`device-attention-device-qkv-v1.jsonl`, `device-attention-host-qkv-32k-v2.jsonl`,
`device-attention-host-qkv-32k-seed1241.jsonl`, `numerics-host-final.jsonl`,
`numerics-host-256k-final.jsonl`. Other completed numerical runs are ablations.

Incomplete historical attempts are **not** qualification evidence:
`probe-v1` stopped on an NFS-cache ELF mapping error after pack checks;
`probe-v2` stopped on cache-directory permissions; `device-attention-host-qkv-32k-v1`
stopped on the rare host-model threshold mismatch fixed above. Their partial
records remain for auditability. Large pack tensor dumps and logs are ignored.
