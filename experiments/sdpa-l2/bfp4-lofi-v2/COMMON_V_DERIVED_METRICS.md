# Common-V stress: residual error and the BF16 output floor

## Conclusion

The large common component makes aggregate L2 small, while final BF16 output
rounding alone can erase much of the small attention-dependent signal.
These are distinct from additional kernel error. At 32K, the HiFi2 native/LUT
records reach the regenerated BF16 nearest-output **error norm**; LoFi
native/LUT error norms are **13.41–13.56 times that floor**. Exp refinement
barely changes this common-V result.

This is a [CPU-derived 16-row ledger](common-v-derived-metrics-v1.jsonl),
produced by [common_v_derived_metrics.py](common_v_derived_metrics.py) from
stored hardware L2 and regenerated original-input FP64 references. It does
not reload actual output tensors, newly verify device replay, or establish
an actual-output/floor tensor hash match. Equal aggregate error norms are
reported as such, not promoted to a full-output bitwise claim.

## Definition: preserve the error, expose its scale

Inputs are the pinned REPRO generator's original BF16 Q/K/V: normal Q/K,
`V=BF16(normal+32)`, H2/D128, seeds 1240/1241, noncausal. Let R be the original
FP64 reference and A the recorded BF16 device output. Let mu be the FP64
token mean of original BF16 V, independently per head and value channel.
Subtract the **same** mu from A and R:

```text
||(A-mu) - (R-mu)|| = ||A-R||
derived centered L2 % = stored original L2 % * ||R|| / ||R-mu||
```

No change is made to the original reference or error numerator. Centered L2
is an additional signal-scale diagnostic, not replacement acceptance criteria.
Centered PCC cannot be recovered from these scalar records and is not inferred.

The floor is the independently regenerated `BF16(R)` output's error norm,
using the original reference. The actual-error/floor ratio compares norms;
it is not an additive or orthogonal decomposition of kernel and cast errors.

## BF16 output floor

Tables are generated from the JSON; full-precision raw values are authoritative.
N1024 covers all 1024 Q rows/head. N32768 covers the recorded 128 sampled Q
rows/head and every KV token, not all 32768 output rows.

| N | Seed | Residual reference RMS | BF16-floor original L2 % | BF16-floor centered L2 % | BF16-floor error norm |
|---:|---:|---:|---:|---:|---:|
| 1,024 | 1240 | 0.040843 | 0.134 | 104.854 | 21.926 |
| 1,024 | 1241 | 0.041323 | 0.136 | 105.298 | 22.278 |
| 32,768 | 1240 | 0.007373 | 0.029 | 125.688 | 1.678 |
| 32,768 | 1241 | 0.007292 | 0.029 | 125.961 | 1.663 |

The original reference RMS stays near 32. As context grows, the residual
shrinks far below BF16 output spacing around that common level. A floor
centered L2 above 100% therefore does **not** imply an avoidable kernel bug.
Even nearest BF16 rounding has that error relative to this residual norm.
Preserving a much smaller signal on top of such a common component may require
a wider output representation or a changed representation of the downstream
computation; simply increasing internal precision cannot beat the fixed BF16
nearest-output floor.

## Stored hardware error, re-expressed

The four controls are LoFi Q7/KV5 in BFP8 storage and HiFi2 Q7/BF16 KV, each
with native or LUT exp; all use FP32 DST/state and BF16 output. Differences
between LoFi and HiFi2 are not an isolated single-instruction ablation.

N1024, all output Q rows:

| Seed | Variant | Original L2 % | Recorded PCC | Derived centered L2 % | Error norm / BF16 floor |
|---:|---|---:|---:|---:|---:|
| 1240 | LoFi native | 0.348 | 0.731781 | 273.016 | 2.604 |
| 1240 | LoFi LUT | 0.348 | 0.732617 | 272.910 | 2.603 |
| 1240 | HiFi2 native | 0.134 | 0.631063 | 104.890 | 1.000348 |
| 1240 | HiFi2 LUT | 0.134 | 0.632540 | 104.875 | 1.000203 |
| 1241 | LoFi native | 0.343 | 0.742925 | 265.854 | 2.525 |
| 1241 | LoFi LUT | 0.343 | 0.743428 | 265.693 | 2.523 |
| 1241 | HiFi2 native | 0.136 | 0.637114 | 105.335 | 1.000351 |
| 1241 | HiFi2 LUT | 0.136 | 0.638607 | 105.319 | 1.000196 |

N32768, 128 recorded Q rows/head:

| Seed | Variant | Original L2 % | Recorded PCC | Derived centered L2 % | Error norm / BF16 floor |
|---:|---|---:|---:|---:|---:|
| 1240 | LoFi native | 0.388 | 0.038507 | 1685.417 | 13.410 |
| 1240 | LoFi LUT | 0.388 | 0.038507 | 1685.417 | 13.410 |
| 1240 | HiFi2 native | 0.029 | Undefined | 125.688 | 1.000 |
| 1240 | HiFi2 LUT | 0.029 | Undefined | 125.688 | 1.000 |
| 1241 | LoFi native | 0.389 | 0.021296 | 1707.650 | 13.557 |
| 1241 | LoFi LUT | 0.389 | 0.031421 | 1707.608 | 13.557 |
| 1241 | HiFi2 native | 0.029 | Undefined | 125.961 | 1.000 |
| 1241 | HiFi2 LUT | 0.029 | Undefined | 125.961 | 1.000 |

At 32K the LoFi error norms are 22.496–22.542 versus floor norms 1.663–1.678.
Thus BF16 output rounding explains the irreducible floor, but cannot by itself
explain the much larger LoFi error norm. Which input-quantization, matmul,
normalization or other mechanism creates that excess needs its own ablation;
this derived metric does not identify the cause.

**PCC alone reverses the practical comparison at 1K:** LoFi's 0.732–0.743
exceeds HiFi2's 0.631–0.639 despite LoFi having 2.52–2.60 times the nearest-output
error norm and HiFi2 only 1.0002–1.00035 times it. At 32K, the nearest-BF16
reference itself has undefined PCC, as do the recorded HiFi2 outputs. A tiny
nonzero LoFi PCC should not be treated as superior fidelity to a floor-limited
result. Conversely, small original L2 alone conceals the excess LoFi error
relative to the residual signal.

## Independent audit and limits

All 16 expected length/seed/variant combinations are present, and all 18 pinned
source/evidence hashes match current files: this driver, REPRO and 16 original
hardware JSON records. The exact pinned `make_inputs`, `reference` and `metrics`
function ASTs execute unchanged without importing the REPRO device main.
Every source record pins the same REPRO bytes.

All eight HiFi2 original Q/K/V hashes match regenerated BF16 inputs. LoFi
records lack original input hashes; their linkage is by the same pinned
generator, seed, shape, sampled rows and matching reference RMS, not a new
input-bit verification. All 16 stored reference RMS values match regenerated
values within `rtol=1e-12, atol=1e-12`. Per length/seed, all four derived rows
share the same regenerated original-reference hash. The direct FP64 reference
on `V-mu` agrees with `R-mu` (maximum absolute discrepancy 1.43e-13).

All derived norm formulas, stored L2/PCC correspondence, source immutability
and original CPU-input checks pass. The run completed in 0.7 seconds using
four CPU threads and one interop thread, with zero device jobs. This is not
a new device qualification, benchmark, all-output verification at 32K, or
model-quality claim. It complements the original stress records' existing
evidence and limitations; it does not repair missing replay/hash fields.
