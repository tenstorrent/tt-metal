# Roofline: nomic-embed-text-v2-moe on one Blackhole chip

Theoretical time per op and per layer at 8x512 (B = 8, S = 512, 4096 tokens) on one p300c chip,
against the measured device kernel time of the first complete port (baseline, `249160a48a6`) and of
the code of 2026-10-01 (current).

## Setup

| | baseline | current |
|---|---|---|
| measured, 8x512 | 119.78 ms, 66.8 sequences/s | 26.63 ms, 300 sequences/s |
| weights | bf16 | bf16, both expert banks bf8_b |
| fidelity | HiFi4 everywhere | HiFi3: QKV, out_proj, fc2, SDPA; HiFi2: fc1, w1; LoFi: w2; HiFi4: router |
| GELU | Accurate (erf), a separate op | tanh, fused into fc1 and w1 |
| experts | all 8 on every token | all 8 on every token |

| hardware | value | source |
|---|---|---|
| cores | 110 (11x10) | `compute_with_storage_grid_size()` |
| clock | 1.35 GHz | `CHIP_FREQ[MHz]` in the device profiler log |
| FPU peak | 4096 FLOP/cycle/core x 110 cores x 1.35 GHz / f, f = 1 to 4: 608 (LoFi), 304 (HiFi2), 203 (HiFi3), 152 (HiFi4) TFLOP/s; ridge 1188, 594, 396, 297 FLOP/B | `MatmulDeviceOperation::create_op_performance_model` |
| DRAM | 512 GB/s | `ttnn/core/operation.cpp` |
| SFPU, per 32x32 tile | GELU 2270 cycles (Accurate), 1800 (tanh); exp 70; softmax 204 per q tile | GELU: Tensix counters on this chip; exp, softmax: #55760 |

```
compute = FLOP / FPU peak at the op's fidelity + SFPU cycles / (110 cores x 1.35 GHz)
DRAM    = (inputs + outputs + weights) / 512 GB/s     every op reads and writes DRAM, bf16 activations
theory  = max(compute, DRAM)
target  = max(compute / 0.70, DRAM / 0.60)            70% of compute peak, 60% of DRAM peak
FLOP    = 2 x M x K x N per matmul; SDPA 4 x B x 12 x S^2 x 64
```

## Per layer

ms. Model = 6 dense layers + 6 MoE layers + the embedding.

| | config | theory | target | measured | measured / target |
|---|---|---|---|---|---|
| Dense layer | baseline | 0.85 | 1.26 | 4.25 | 3.4x |
| Dense layer | current | 0.68 | 1.04 | 0.95 | 0.9x |
| MoE layer | baseline | 4.06 | 5.89 | 15.70 | 2.7x |
| MoE layer | current | 2.45 | 3.65 | 3.48 | 1.0x |
| MoE layer, top-2 routed | current precision | 0.94 | 1.44 | - | - |
| **Model** | baseline | 29.51 | 43.00 | 119.78 | 2.8x |
| **Model** | current | 18.78 | 28.23 | 26.63 | 0.9x |
| **Model, top-2 routed** | current precision | 9.72 | 14.95 | - | - |

- **The current code meets the target of its formulation per layer, not per op.** Above target:
  dispatch and combine (409 against 108 us), routing (299 against 21), w2 (591 against 501), SDPA
  (114 against 82), QKV, out_proj and fc2 (13% to 17% over). Below: w1 and fc1, which run the GELU
  beside the matmul (w1 1630 against 2429), and the head ops, which run from L1.
- **Both ports run all 8 experts on every token, 4x the expert work the model needs.** With top-2
  routing at the current precision the model target is 14.95 ms; the current 26.63 ms is 1.8x
  that. The absolute floor, top-2 routed with activations never leaving the chip, is 6.49 ms.

## Utilization

Measured, 8x512.

| | compute, baseline | compute, current | DRAM, baseline | DRAM, current | SFPU busy, baseline | SFPU busy, current |
|---|---|---|---|---|---|---|
| Dense layer | 10% | 30% | 15% | 11% | 7% | 16% |
| MoE layer | 14% | 26% | 15% | 27% | 11% | 35% |
| **Model** | 13% | 26% | 15% | 24% | 10% | 31% |

- **Compute:** the FPU time of the FLOP the ops do, at each op's fidelity, over measured time. The
  MoE layer counts all 8 experts; counting the model's top-2 only, it is 9% and the model 14%.
- **DRAM:** the bytes each op really moves (its DRAM-resident inputs and outputs, tile-padded,
  each counted once) over measured time, against 512 GB/s: 9.29 GB a forward at baseline, 3.23 GB
  now.
- **SFPU busy:** Tensix counters. The GELU is SFPU work, which the compute column does not count.

## Per op

8x512, one call, us. Bound is the larger term of theory, baseline / current.

| op | calls | GFLOP | theory, baseline | measured, baseline | theory, current | measured, current | bound |
|---|---|---|---|---|---|---|---|
| embedding, mask, rotary tables | 1 | 0 | 49.2 | 130.2 | 49.2 | 77.9 | DRAM / DRAM |
| QKV | 12 | 14.50 | 95.3 | 604.1 | 71.5 | 119.0 | FPU / FPU |
| head split | 12 | 0 | 73.7 | 99.4 | 73.7 | 47.1 (L1) | DRAM / DRAM |
| rotary q, k | 12 | 0 | 49.2 | 173.1 | 49.2 | 81.3 (L1) | DRAM / DRAM |
| SDPA | 12 | 6.44 | 56.1 | 1132.6 | 49.2 | 114.3 | FPU / DRAM |
| head concat | 12 | 0 | 24.6 | 37.1 | 24.6 | 16.7 (L1) | DRAM / DRAM |
| out_proj | 12 | 4.83 | 31.8 | 243.2 | 26.9 | 52.6 | FPU / DRAM |
| add + norm1 | 12 | 0 | 36.9 | 84.3 | 36.9 | 52.6 | DRAM / DRAM |
| fc1 | 6 | 19.33 | 127.1 | 705.7 | 212.5 (with GELU) | 259.4 (with GELU) | FPU / SFPU |
| GELU, dense | 6 | 0 | 187.8 | 210.5 | in fc1 | in fc1 | SFPU / - |
| fc2 | 6 | 19.33 | 127.1 | 870.8 | 95.3 | 153.5 | FPU / FPU |
| add + norm2 | 12 | 0 | 36.9 | 84.0 | 36.9 | 58.8 | DRAM / DRAM |
| router + top-2 | 6 | 0.05 | 12.3 | 394.0 | 12.3 | 298.9 | DRAM / DRAM |
| expert w1 | 6 | 154.62 | 1016.8 | 5114.0 | 1700.0 (with GELU) | 1629.5 (with GELU) | FPU / SFPU |
| GELU, experts | 6 | 0 | 1502.7 | 1656.0 | in w1 | in w1 | SFPU / - |
| expert w2 | 6 | 154.62 | 1016.8 | 5483.3 | 300.3 | 590.5 | FPU / DRAM |
| dispatch + combine | 6 | 0 | 110.6 | 591.4 | 64.5 | 409.1 | DRAM / DRAM |

- **The GELU costs more than the matmul that feeds it:** 1800 SFPU cycles a tile (tanh) against
  768 FPU cycles for the HiFi2 tile at K = 768, so fc1 and w1 are SFPU-bound.
- **(L1):** the current code keeps these operands in L1, so the DRAM term does not apply. SDPA's
  operands are in L1 too, where its floor is its compute, 45.5 us.
- **w2 at LoFi cannot be compute-bound with its input in DRAM:** its activation intensity, 1157
  FLOP/B with a bf8_b input and output, is below the LoFi ridge of 1188.

## Where an op turns DRAM-bound

Current precision, top-2 routed experts. Below these token counts the op is bound by reading its
weights; in parentheses, the same with its activations in DRAM.

| op | tokens |
|---|---|
| QKV | 396 (1267) |
| out_proj | 396 (never compute-bound) |
| fc1 + GELU | 253 (431) |
| fc2 | 396 (1114) |
| expert w1 + GELU | 539 (726) |
| expert w2 | 2525 (never compute-bound) |
| **model** | about 450 |

Each expert sees T / 4 tokens but all 8 experts' weights are read, so the experts turn compute-bound
latest. At 1x128 the floor is the 354 MB weight read, 0.69 ms, against 3.96 ms measured.

## Assumptions

- Verified: the hardware table, the model dimensions (`../reference/config.json`), the dtypes and
  fidelities (`../tt/model_config.py`), the measured times (tracy, one warmed forward, random ids,
  no padding) and the SFPU busy fractions (Tensix counters).
- Every op runs on all 110 cores; T and S are multiples of 32; every expert gets a token.
- Theory runs FPU and SFPU work one after the other, and SDPA's exp after its matmuls.
- Not counted: elementwise FPU work (rotary, norms, residual adds; about 0.13 ms at 8x512), op
  launch and host dispatch.
