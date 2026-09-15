# Four-point frontier: compact distinct-input accuracy sweep

September 14, 2026 (local); device run September 15 UTC. Blackhole P100A,
yyzo-bh-08, reservation 219611.

## Conclusion

The two FP32 variants retain narrow regular-input accuracy bands. FAST prevents
most long-context recurrent drift but remains distribution-dependent at a few
percent. Main BF16 does not retain a stable band at long context.

Thirty regular cases: five distributions × three K/V lengths (4096, 32768,
262144) × two new seeds (1240, 1241). Four additional stress cases at 262144
keys use seed 1240. All four algorithms execute on exactly the same original
BF16 input tensors and FP64 reference per case: 136 executions total.
The initial sweep took 21.30 seconds inside the opened-device test loop,
including input generation, references, kernel JIT/execution and metrics;
setup, smoke cross-checks and analysis are additional.

## L2 (%) ranges across lengths and seeds

| Input | Main BF16 | FAST | FP32 QK4/PV2 | ACCURATE |
| --- | ---: | ---: | ---: | ---: |
| Normal N(0,1) | 2.531–18.470 | 2.479–3.243 | 0.383–0.398 | 0.178–0.180 |
| Q/K std 0.5; V std 1 | 2.173–34.675 | 2.132–3.437 | 0.351–0.393 | 0.174–0.178 |
| Q/K std 2; V std 1 | 3.893–6.394 | 3.882–5.534 | 0.303–0.388 | 0.182–0.208 |
| Uniform [-sqrt(3), sqrt(3)] | 2.509–19.304 | 2.454–3.279 | 0.376–0.397 | 0.176–0.178 |
| Sparse outliers (0.1%, additive 10x) | 1.334–5.251 | 1.319–3.977 | 0.175–0.435 | 0.131–0.279 |

## Normal-input length dependence (two seeds)

| K/V length | Main BF16 L2 % | FAST L2 % | QK4/PV2 L2 % | ACCURATE L2 % |
| --- | ---: | ---: | ---: | ---: |
| 4,096 | 2.531–2.536 | 2.489–2.510 | 0.384–0.385 | 0.178–0.179 |
| 32,768 | 2.660–2.718 | 2.479–2.529 | 0.383–0.387 | 0.178–0.180 |
| 262,144 | 17.904–18.470 | 3.229–3.243 | 0.397–0.398 | 0.179–0.179 |

## Regular-case summary

| Variant | Global L2 % range | Minimum PCC | Worst per-case row-p95 L2 % |
| --- | ---: | ---: | ---: |
| Main BF16 | 1.334–34.675 | 0.995408670 | 50.737 |
| FAST (compensated BF16) | 1.319–5.534 | 0.998486447 | 11.927 |
| FP32 QK4/PV2 | 0.175–0.435 | 0.999990976 | 0.760 |
| ACCURATE (QK4/PV4) | 0.131–0.279 | 0.999996109 | 0.480 |

QK4/PV2 and ACCURATE are each below 0.5% global L2 in all 30 regular cases.
This is not an every-row guarantee: QK4/PV2 row-p95 can exceed 0.5% for scaled
Q/K and outlier inputs. Aggregate L2, row-p95, PCC and raw maximum relative
error are all retained in the JSONL. Normal-input ACCURATE is near the BF16
output-rounding floor (approximately 0.166% here), but that floor changes
with the output distribution.

## Separate stress diagnostics (256K, one seed)

| Input | Main BF16 L2 % | FAST L2 % | QK4/PV2 L2 % | ACCURATE L2 % |
| --- | ---: | ---: | ---: | ---: |
| Uniform attention (Q=0) | 95.256 | 2.005 | 0.186 | 0.184 |
| Q + 32 | 34.196 | 34.201 | 1.254 | 0.446 |
| K + 32 | 26.807 | 18.021 | 1.084 | 0.744 |
| V + 32 | 15.552 | 0.952 | 0.010 | 0.010 |

Large common Q/K break the ordinary bands; neither FP32 algorithm should be
claimed universally below 0.5%. In particular, common-Q behavior is seed
dependent: the earlier apparently easy common-Q result does not generalize.
For V+32, both FP32 outputs round to constant BF16 values and PCC is undefined.
Their small global L2 equals the output-rounding floor, not proof that they
preserve small centered variations. Uniform attention remains a useful
recurrent-state stress case: main has very large gain/drift error despite
high PCC.

## Contract and scope

- Noncausal, batch 1, one head, D128. Each case executes all 128 query rows
  against genuinely distinct K/V, not a repeated resident K/V block.
- All four use Q128/K512; BF16 retains two K/V slots and FP32 one. BF16 input
  and output, no Q preprocessing. No padding, masks, causal test, model
  evaluation, or multi-device/Galaxy qualification.
- Exact inputs are shared between variants, and their hashes are recorded.
  FP64 blocked attention uses the already-rounded BF16 inputs. The reference
  self-tests include dense/blocked and common-mode identities.
- Main uses the frozen main streaming headers. FAST uses frozen retained
  paired-denominator headers, with both numerator and denominator
  compensation. QK4/PV2 uses the cheap-subtraction, matched-denominator
  implementation; ACCURATE uses HiFi4 on both matmuls and full-FP32 subtraction.
- Streaming is explicitly instantiated by the test wrapper: no fallback.
  At 4K, FAST is tested as an algorithm although the current production
  compensation guard only enables it from 32K. This is not a dispatch change.
- This accuracy sweep is Q128, not the Q256 resident performance plot. No
  new performance measurements or cross-geometry throughput claims are made.

## Harness checks and reproducibility

The earlier experimental Q256 live-input FAST reader/compute configuration
had failed; its results are not used here. This Q128 wrapper uses the retained
FAST headers and matches production FAST bit-for-bit on normal 32K, seed 1240
(`smoke-v2.jsonl`). An initial 4K production comparison was invalid because
the production guard selects main there; that smoke assertion is retained
in `smoke-v1.log`, not counted as a numeric failure of FAST.

All 136 initial executions produced finite outputs and finite references.
A second complete sweep after Black/clang-format reproduces all 136 output
hashes and L2 values exactly (`repeat-v1.jsonl`). The FAST production smoke
cross-check is also exact. Black, clang-format checks and git diff --check pass.
`source-sha256.txt` records the final wrapper and frozen kernel source hashes.
Device kernels compile through JIT; no production C++/dispatch files were
modified. This does not fix or explain the older Q256 wrapper failure.

```bash
# In the configured allocated container:
python_env/bin/python experiments/sdpa-l2/frontier-accuracy-v1/run.py \
  --smoke --label fresh-smoke
python_env/bin/python experiments/sdpa-l2/frontier-accuracy-v1/run.py \
  --label fresh-sweep
```

Use fresh labels; the runner refuses to overwrite an existing JSONL.
Raw records: `sweep-v1.jsonl`, `repeat-v1.jsonl`, logs and per-execution provenance.
