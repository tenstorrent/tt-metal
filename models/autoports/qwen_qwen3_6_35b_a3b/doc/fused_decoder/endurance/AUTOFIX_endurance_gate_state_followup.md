# Saved-state follow-up: gate contribution versus inherited recurrent error

CPU/source investigation only, 2026-11-12. Read `../endurance_state_probe.json.pos*.pt` as they became available, with frozen `../mlp_norm_shard_dram.py.snapshot` as the device implementation. No TTNN imports, devices, runtime edits, or test edits. Parent reports that the old42ac 8192-token control also fails; therefore the latest memory-read matmul and post-norm changes are not necessary causes.

## Finding

The remaining attention discrepancies are predominantly **inherited recurrent-state differences amplified by a few sensitive heads**, not bad current-token recurrence arithmetic. The new snapshots allow exact one-step CPU replay, unlike the earlier final-output-only artifact. At position3979, replacing just the incoming TT state with HF state improves reconstructed attention PCC from **0.995922 to0.999983**. Replacing all current inputs with HF inputs while retaining TT state leaves PCC **0.995982**. Current-token gate replacement barely moves it. A gate fix must therefore run over the complete preceding trajectory, not just the failure token.

A/B/beta rounding remains a plausible contributor to that history. The earlier isolated 8192-token CPU gate experiment caused ~1.2% state error and attention PCC0.995616 at3979. However current snapshots also show upstream input-normalization/projection differences, and rounded TT beta is not universally closer to HF. The evidence supports the isolated gate-boundary device A/B described in `AUTOFIX_endurance_gates_hypothesis.md`, not a claim that it alone repairs every failure.

## Exact CPU one-step replay

Loaded saved normalized Q/K/V, scalar log gates, beta, and incoming states. Expanded TT's16 key/query heads consecutively to32, then evaluated the canonical FP32 recurrence:

```text
D = incoming_state * exp(g)
delta = beta * (v - sum_key(D*k))
new = D + k*delta
core = sum_key(new*q) / sqrt(128)
```

HF CPU core, rounded BF16, reproduces the saved HF core exactly at128/1024/2048/5011/7551/7668 and to approximately1e-8 relative L2 at3979/6595. This validates the interpretation of the saved inputs and incoming states.

CPU replay on all TT inputs matches saved device new state within **1.4e-5–2.7e-5 relative L2** and saved device core within **5.6e-6–3.4e-5**, all PCCs rounding to1.0 at nine decimal places. The small residual includes the current matmul-based memory read and device arithmetic; it is orders of magnitude below the inherited state error. This does not exclude a tiny error accumulating over time, but rules out a large instantaneous recurrence failure at these positions.

Incoming-state relative L2 to HF is0.010509 at128 (immediately after prefill),0.016391 at1024,0.017441 at2048,0.016598 at3979,0.016977 at5011,0.018104 at6595,0.018228 at7551, and0.017419 at7668. It is not steadily growing at the failing tokens.

For attribution, each CPU core was passed through the **same canonical HF downstream**: BF16 core boundary, FP32 unweighted RMSNorm, BF16 normalized activation and weight multiplication, FP32 SiLU of canonical HF z, BF16 gate product, and checkpoint output projection. These attention PCCs compare counterfactual CPU outputs, not a complete device-layer rerun:

| Position | All TT recurrent inputs | HF incoming state + TT current inputs | TT incoming state + HF current inputs | Only fix current TT A/B/beta rounding |
|---|---:|---:|---:|---:|
| 128 | 0.999906414 | 0.999939308 | 0.999971753 | 0.999905936 |
| 1024 | 0.999825434 | 0.999961926 | 0.999834545 | 0.999827314 |
| 2048 | 0.999936496 | 0.999989750 | 0.999941707 | 0.999936126 |
| 3979 | 0.995922195 | 0.999982818 | 0.995982068 | 0.995924273 |
| 5011 | 0.995165078 | 0.999984060 | 0.995071137 | 0.995173316 |
| 6595 | 0.995484254 | 0.999916237 | 0.996801667 | 0.995488126 |
| 7551 | 0.997846669 | 0.999969554 | 0.997765069 | 0.997847328 |
| 7668 | 0.999415753 | 0.999974357 | 0.999409971 | 0.999414348 |

The corrected current-gate column rounds A/B from the actual saved TT FP32 packed projection to BF16, widens A before dt_bias/softplus, and computes sigmoid from rounded B with a final BF16 beta round. It leaves all historical state unchanged. Its near-zero effect at3979 does not refute the historical gate hypothesis.

## Why small global core differences become visible attention errors

The saved core differences concentrate in particular low-magnitude heads after RMS normalization and z gating. With canonical downstream applied identically to HF and TT saved cores:

| Position | Dominant head | Share of squared gated-norm error | That head's core relative L2 | HF head core RMS |
|---|---:|---:|---:|---:|
| 3979 | 25 | 98.628% | 0.13590 | 0.0003802 |
| 5011 | 23 | 99.380% | 0.35082 | 0.0017936 |
| 6595 | 23 | 96.916% | 0.33683 | 0.0016788 |
| 7703 | 15 | 91.204% | 0.09299 | 0.0007164 |
| 8157 | 15 | 55.018% | 0.07145 | 0.0014165 |

Thus global core PCC can remain high while a head materially affects attention and expert selection. This is a numerical sensitivity of the genuine equations; changing RMS epsilon or thresholds would mask it rather than repair the upstream discrepancy.

## Earliest remaining gate mismatch is not only output rounding

At3979, rounding the saved TT packed projections to BF16 still disagrees with HF in20/32 A elements and23/32 B elements. A CPU BF16 linear projection from **TT input-normalized hidden states** disagrees with HF in21/32 A and25/32 B elements. Comparing TT-rounded projection directly against that same-input CPU projection leaves7 A differences and11 B differences. Consequently, much of the remaining projection mismatch is already explained by the slightly different input norm, with an additional TT projection contribution. Gate-output rounding alone cannot produce bit-identical HF gates on these different inputs.

For beta at3979, current TT versus HF relative L2 is0.001964; canonical rounding applied to the saved TT projection changes it to0.002387 (worse at this token). Conversely at5011 it improves0.001660→0.000765, and at6595 it improves0.001853→0.001691. A projection-rounding repair should be judged over the entire trajectory, not accepted or rejected from a single current beta comparison. The A/g boundary is closer at3979 (relative g error0.0000705→0.00000989), but its immediate state contribution is small for this token.

## Minimal repair experiment and next discriminator

Run the previously specified **decode-only A/B/beta BF16-boundary patch** over the entire8192-token stream, keeping the BF16 consumed-cache behavior and all other frozen-F operations unchanged. This is the smallest source-supported semantic repair for the assigned hypothesis. Preserve thresholds and compare failures/global minima, then compare incoming-state error specifically for heads25,23,15 at3979/5011/7703. Do not substitute HF state in production or change normalization epsilon.

If gate rounding does not materially improve the full trajectory, the next localized investigation should target historical Q/K/V/input-norm precision rather than another recurrence rewrite: same-input CPU replay already tracks the device recurrence to about1e-5, while state substitution almost removes the failing attention discrepancy. The nonzero initial state error at128 also warrants a controlled canonical-prefill seed test, but that is a diagnostic separating prefill from decode accumulation, not an authorized production fallback.

No device result for the proposed gate patch is claimed. All new numerical results in this memo are CPU calculations from saved device/HF tensors and checkpoint weights.
