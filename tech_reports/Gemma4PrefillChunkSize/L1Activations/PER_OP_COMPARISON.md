# Per-op comparison: chunk 2048 vs 4096 vs 8192

`tt-perf-report` Device Time, **ms per layer**, on `mmanzoor/svuckovic/gemma4-L1-activations`
@ `9dc8e32a2` (+ multi-hop halo), `GEMMA4_PREFILL_L1_ACT=1`, mesh 8x4, ctx 262,144.

Two depths per width, at **matched prior context 57,344 tokens** — chunk index 28 / 14 / 7 for
2048 / 4096 / 8192. Matching prior context (not chunk index) is what makes the prefix work
exactly proportional to chunk size.

**`r2048` = chunk-2048 time / chunk-8192 time. Ideal is 0.250** (a quarter of the tokens should
be a quarter of the work). Anything well above 0.250 does not scale with chunk width.

The single-cell tables these are built from are in `reports/` — see the index in the summary.


## Sliding layer (x50 in the model)

| op | n | 2048 first | 2048 depth | 4096 first | 4096 depth | 8192 first | 8192 depth | `r2048` | cores 2048→8192 |
|---|---|---|---|---|---|---|---|---|---|
| `MatmulDeviceOperation _x5376x5376` | 3 | 0.567 | 0.567 | 0.862 | 0.862 | 0.636 | 0.635 | **0.891** ⚠️ | 84 → 96 |
| `AllGatherDeviceOperation` | 3 | 0.159 | 0.161 | 0.313 | 0.314 | 0.564 | 0.566 | **0.282** | 10,2 → 34 |
| `LayerNormDeviceOperation` | 7 | 0.462 | 0.463 | 0.468 | 0.468 | 0.523 | 0.522 | **0.884** ⚠️ | 64,32,8 → 120,32 |
| `ReduceScatterDeviceOperation` | 2 | 0.144 | 0.145 | 0.255 | 0.257 | 0.483 | 0.484 | **0.299** | 18 → 34 |
| `RingJointSDPADeviceOperation` | 1 | 0.403 | 0.402 | 0.421 | 0.443 | 0.450 | 0.461 | **0.895** ⚠️ | 114 → 112 |
| `MatmulDeviceOperation _x5376x4096` | 1 | 0.178 | 0.178 | 0.121 | 0.121 | 0.226 | 0.227 | **0.789** ⚠️ | 64 → 96 |
| `GatherCodegenDeviceOperation` | 2 | 0.072 | 0.072 | 0.125 | 0.125 | 0.193 | 0.193 | **0.375** | 64 → 120 |
| `MatmulDeviceOperation _x2048x5376` | 1 | 0.086 | 0.086 | 0.135 | 0.135 | 0.164 | 0.164 | **0.521** ⚠️ | 84 → 96 |
| `BinaryNgDeviceOperation` | 4 | 0.042 | 0.041 | 0.079 | 0.078 | 0.149 | 0.147 | **0.280** | 120 → 120 |
| `TilizeDeviceOperation` | 1 | 0.057 | 0.056 | 0.090 | 0.090 | 0.122 | 0.123 | **0.466** ⚠️ | 84 → 84 |
| `NlpCreateHeadsDeviceOperation` | 1 | 0.056 | 0.056 | 0.057 | 0.057 | 0.060 | 0.059 | **0.939** ⚠️ | 8 → 32 |

- **chunk 2048: layer total 2.329 ms first chunk → 2.330 ms at depth** (+0.1%)

- **chunk 4096: layer total 3.041 ms first chunk → 3.063 ms at depth** (+0.7%)

- **chunk 8192: layer total 3.716 ms first chunk → 3.724 ms at depth** (+0.2%)

## Global layer (x10 in the model)

| op | n | 2048 first | 2048 depth | 4096 first | 4096 depth | 8192 first | 8192 depth | `r2048` | cores 2048→8192 |
|---|---|---|---|---|---|---|---|---|---|
| `RingJointSDPADeviceOperation` | 1 | 0.189 | 4.249 | 0.356 | 4.410 | 1.314 | 9.426 | **0.144** | 114 → 114 |
| `MatmulDeviceOperation _x5376x5376` | 3 | 0.567 | 0.567 | 0.862 | 0.862 | 0.635 | 0.635 | **0.893** ⚠️ | 84 → 96 |
| `AllGatherDeviceOperation` | 3 | 0.159 | 0.161 | 0.315 | 0.314 | 0.567 | 0.569 | **0.281** | 10,2 → 34 |
| `LayerNormDeviceOperation` | 6 | 0.462 | 0.462 | 0.470 | 0.470 | 0.521 | 0.520 | **0.886** ⚠️ | 64,8 → 120,32 |
| `ReduceScatterDeviceOperation` | 2 | 0.144 | 0.145 | 0.260 | 0.265 | 0.499 | 0.506 | **0.288** | 18 → 34 |
| `GatherCodegenDeviceOperation` | 5 | 0.176 | 0.176 | 0.213 | 0.213 | 0.374 | 0.362 | **0.472** ⚠️ | 32 → 120 |
| `MatmulDeviceOperation _x4096x5376` | 1 | 0.152 | 0.152 | 0.234 | 0.233 | 0.280 | 0.280 | **0.545** ⚠️ | 84 → 96 |
| `MatmulDeviceOperation _x5376x4608` | 1 | 0.186 | 0.185 | 0.283 | 0.283 | 0.249 | 0.248 | **0.746** ⚠️ | 72 → 96 |
| `SliceDeviceOperation` | 5 | 0.208 | 0.210 | 0.210 | 0.210 | 0.219 | 0.220 | **0.948** ⚠️ | 120,32 → 120 |
| `BinaryNgDeviceOperation` | 5 | 0.044 | 0.045 | 0.081 | 0.081 | 0.149 | 0.150 | **0.296** | 120 → 120 |
| `TilizeDeviceOperation` | 1 | 0.057 | 0.057 | 0.090 | 0.090 | 0.122 | 0.122 | **0.471** ⚠️ | 84 → 84 |

- **chunk 2048: layer total 2.548 ms first chunk → 6.611 ms at depth** (+159.5%)

- **chunk 4096: layer total 3.597 ms first chunk → 7.651 ms at depth** (+112.7%)

- **chunk 8192: layer total 5.195 ms first chunk → 13.299 ms at depth** (+156.0%)


## How to read it

**⚠️ marks `r2048 > 0.45`** — ops that are nowhere near scaling with chunk width. Those are the
per-chunk floor, and they are why `a(2048)` is 126 ms rather than 205/4 = 51 ms.

**The sliding layer barely moves between "first chunk" and "at depth" at any width** — that is
the control. A windowed layer cannot see history, and it doesn't. It makes the global layer's
growth credible rather than a harness artifact.

**The global layer's growth is one op**, `RingJointSDPADeviceOperation`, ~100% of the delta.
Growth ratios 0.490 / 0.501 / 0.978 at 2048 / 4096 / 8192 against occupancy's predicted
0.500 / 0.500 / 1.000 — so 2048 and 4096 cost the *same* for prefix work.

**On the `Cores` column:** trustworthy for LayerNorm (8 / 16 / 32 = `(chunk/CP)/32` exactly —
the defect is visible directly) and for the matmuls (84 vs 96). **Not** trustworthy for the
SDPA, which reads 114 at every width because idle cores run padded handshake iterations
instead of being dropped.
