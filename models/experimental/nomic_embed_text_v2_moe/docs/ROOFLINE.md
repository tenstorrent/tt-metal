# Roofline: nomic-embed-text-v2-moe on one Blackhole chip

The lowest device time the model's arithmetic and memory traffic allow on one p300c chip, per op
and end to end, against the port as measured on 2026-10-01: 26.63 ms, 300 sequences/s of kernel
time at 8x512.

## Result

| shape | floor | FPU / SFPU / DRAM work | dense-expert floor | measured | measured / floor |
|---|---|---|---|---|---|
| 1x128 | 0.70 ms | 0.10 / 0.09 / 0.70 ms | 0.70 to 0.78 ms | 3.96 ms | 5.7x |
| 8x384 | 3.92 to 4.77 ms | 2.65 / 2.11 / 0.71 ms | 8.80 to 11.37 ms | 19.94 ms | 4.2x to 5.1x |
| 8x512 | 5.35 to 6.49 ms | 3.62 / 2.85 / 0.72 ms | 11.86 to 15.29 ms | 26.63 ms | 4.1x to 5.0x |

- **Floor:** the model as defined (top-2 routing) at the port's dtypes and fidelities, every op at
  peak, activations kept on chip. The low end runs each GELU beside the matmul that feeds it, the
  high end after it. At 8x512 that is 1230 to 1500 sequences/s.
- **Dense-expert floor:** the same for the port's formulation, which runs all 8 experts on every
  token.
- **Measured:** device kernel time of one warmed forward (random ids, no padding), 2026-10-01.

## Hardware

| | value | source |
|---|---|---|
| cores | 110 (11x10) | `compute_with_storage_grid_size()` |
| clock | 1350 MHz | `CHIP_FREQ[MHz]` in the device profiler log |
| FPU | 4096 FLOP/cycle/core at LoFi; HiFi2, HiFi3 and HiFi4 take 2, 3 and 4 passes | `MatmulDeviceOperation::create_op_performance_model`, `tech_reports/matrix_engine/matrix_engine.md` |
| FPU peak | 608.3 (LoFi), 304.1 (HiFi2), 202.8 (HiFi3), 152.1 (HiFi4) TFLOP/s | |
| DRAM | 512 GB/s | `ttnn/core/operation.cpp` |
| L1 | 1.46 MB a core for buffers, 160.7 MB a chip | `cb_limit` of the device info |
| SFPU, per 32x32 tile | tanh GELU 1800 cycles; approximate exp 70; softmax 204 per q tile per key chunk | GELU: 171 us for 12,288 tiles at 87% SFPU busy (Tensix counters); exp, softmax: #55760 |

The ridge point, peak FLOP/s over DRAM bandwidth, is 1188 FLOP/B at LoFi, 594 at HiFi2, 396 at
HiFi3 and 297 at HiFi4. Ops that only stream DRAM reach 81% to 92% of 512 GB/s on this port
(tt-npe), so a realistic DRAM floor is 1.09x to 1.23x the one here.

## Work

Per token at sequence length S, top-2 routed:
- 226.6 MFLOP of projections, 2 per weight of the 113 M parameters a token uses outside the
  embedding, plus 36,864 x S of attention: 245.4 MFLOP at S = 512, 1005 GFLOP for 8x512. Matmul
  FLOP are 2 x M x K x N at logical sizes, SDPA's 4 x B x 12 x S^2 x 64.
- 18 x 3072 GELU elements: 3072 in each of the 6 dense layers, 2 x 3072 in each of the 6 MoE
  layers.

Per forward, whatever the batch, 354 MB of weights at the port's dtypes (566 MB in bf16): 0.69 ms
at 512 GB/s. They do not fit in L1, so every forward reads them. Running all 8 experts on every
token, as the port does, takes 2397 GFLOP and 3x the GELU work at 8x512.

## Per op

8x512, one call. "act" is the op's activations in and out (bf16, the expert intermediate bf8_b)
if they go through DRAM. The last column is the token count above which the op stops being bound
by its weights, then the same with its activations in DRAM.

| op | calls | GFLOP | fidelity | FPU us | SFPU us | weights us | act us | bound, act on chip | bound, act in DRAM | leaves DRAM above |
|---|---|---|---|---|---|---|---|---|---|---|
| QKV | 12 | 14.50 | HiFi3 | 71.5 | | 6.9 | 49.2 | FPU | FPU | 396 (1267) |
| SDPA | 12 | 6.44 | HiFi3 | 31.8 | 13.7 | | 49.2 | FPU | DRAM below S = 576 | |
| out_proj | 12 | 4.83 | HiFi3 | 23.8 | | 2.3 | 24.6 | FPU | DRAM | 396 (never) |
| fc1 + GELU | 6 | 19.33 | HiFi2 | 63.6 | 148.9 | 9.2 | 61.4 | SFPU | SFPU | 253 (431) |
| fc2 | 6 | 19.33 | HiFi3 | 95.3 | | 9.2 | 61.4 | FPU | FPU | 396 (1114) |
| router | 6 | 0.05 | HiFi4 | 0.3 | | 0.05 | 12.3 | - | DRAM | never |
| expert w1 + GELU | 6 | 38.65 | HiFi2 | 127.1 | 297.9 | 39.2 | 76.8 | SFPU | SFPU | 539 (726) |
| expert w2 | 6 | 38.65 | LoFi | 63.6 | | 39.2 | 76.8 | FPU | DRAM | 2525 (never) |
| rotary, norms, routing, dispatch, combine, embedding | | 0 | | | | | 12 to 49 | - | DRAM | never |

- **fc1 and w1 are SFPU-bound wherever they are not weight-bound.** The tanh GELU's 1800 cycles a
  tile are 2.3x the 768 FPU cycles of the HiFi2 tile it is applied to (24 tile products at
  K = 768).
- **out_proj and w2 cannot be compute-bound with their activations in DRAM, and SDPA only above
  S = 576.** Their activation intensity, 384, 983 and S / 2 FLOP/B, is below the ridge of their
  fidelity, 396, 1188 and 396; SDPA's softmax adds enough SFPU time to cross at S = 576 rather
  than 792.
- **The experts stay weight-bound longest:** all 8 experts' weights are read while each sees T / 4
  tokens on average.
- With bf16 weights at HiFi4, every dense projection would cross at 297 tokens.
- The whole model turns compute-bound at about 450 tokens. At 1x128 every op with weights is
  bound by them.

## Where the measured time goes

8x512, per forward, ms. Each floor is the sum over ops of max(FPU + SFPU, DRAM).

| | model, act on chip | model, act in DRAM | dense experts, act in DRAM | measured |
|---|---|---|---|---|
| QKV | 0.86 | 0.86 | 0.86 | 1.43 |
| SDPA | 0.55 | 0.59 | 0.59 | 1.37 |
| out_proj | 0.29 | 0.32 | 0.32 | 0.63 |
| fc1 + GELU | 1.28 | 1.28 | 1.28 | 1.56 |
| fc2 | 0.57 | 0.57 | 0.57 | 0.92 |
| expert w1 + GELU | 2.55 | 2.55 | 10.20 | 9.78 |
| expert w2 | 0.38 | 0.70 | 2.08 | 3.54 |
| router and routing | 0.00 | 0.07 | 0.07 | 1.79 |
| dispatch and combine | 0 | 0.22 | 0.66 | 2.45 |
| residual adds, norms, output | 0.01 | 0.90 | 0.90 | 1.34 |
| rotary | 0 | 0.59 | 0.59 | 0.98 |
| head split and concat | 0 | 0 | 0 | 0.77 |
| embedding | 0.01 | 0.05 | 0.05 | 0.08 |
| **total** | **6.49** | **8.69** | **18.17** | **26.63** |

- **The dense experts set the port's ceiling.** The MoE layers take 17.57 of the 26.63 ms; at the
  model's floor they need 2.17 to 2.93. The port's w1 + GELU already sits between its
  formulation's floors (7.15 ms with the GELU beside the matmul, 10.20 after it) and w2 is at
  1.7x, so no tuning takes the dense formulation much below 12 ms at 8x512. Routed dispatch is the
  change that moves the floor.
- **The GELU is the other large term.** It is 2.68 of the 6.49 ms, SFPU work that no matmul
  tuning removes. FastLut is 4.4x cheaper than tanh as a separate op (38.8 against 171.2 us at
  8x512) but fails five accuracy tests; with a free GELU the floor would be 3.81 ms.
- **The projections and SDPA run at 1.2x to 2.5x of their floors:** fc1 + GELU 1.2x, fc2 1.6x,
  QKV 1.7x, out_proj 2.2x, SDPA 2.5x. SDPA's math units are busy 46% of its time (counters).
- **The routing, dispatch, combine, norm, rotary, head and embedding ops take 7.41 ms (28%) for
  0.05 GFLOP.** Their floor as separate DRAM-to-DRAM ops is 1.83 ms (2.27 with dense experts), and
  zero when fused into their neighbours.
- **At 1x128 the gap is latency.** The kernels take 3.96 ms against a 0.70 ms floor. The norms,
  routing and head ops take 1.47 ms for 0.03 ms of traffic, mostly on 4 of 110 cores, and the
  expert matmuls stream their weights at 51% (w1) and 27% (w2) of 512 GB/s. The host needs about
  as long again to enqueue the 236 ops, so without trace this shape is host-bound.

## Utilization now

8x512 per op, then the forward at each shape.

| op | measured ms | FLOP util | FPU busy | SFPU busy | DRAM util | bound |
|---|---|---|---|---|---|---|
| QKV | 1.43 | 60% | 69% | 0% | 16% | COMPUTE |
| SDPA | 1.37 | 28% | 32% | 14% | 0% | LATENCY |
| out_proj | 0.63 | 45% | 58% | 0% | 5% | LATENCY |
| fc1 + GELU | 1.56 | 24% | 26% | 53% | 8% | LATENCY |
| fc2 | 0.92 | 62% | 80% | 0% | 6% | COMPUTE |
| expert w1 + GELU | 9.78 | 31% | 32% | 73% | 16% | COMPUTE (SFPU) |
| expert w2 | 3.54 | 43% | 44% | 0% | 51% | LATENCY |
| router and routing | 1.79 | 0% | 0% | 1% | 10% | LATENCY |
| dispatch and combine | 2.45 | - | 2% | 0% | 66% | LATENCY 60%, DRAM 40% |
| residual adds, norms | 1.34 | - | 7% | 2% | 50% | LATENCY |
| rotary | 0.98 | - | 4% | 0% | 0% | LATENCY |
| head split and concat | 0.76 | - | 0% | 0% | 0% | LATENCY |
| embedding | 0.08 | - | 5% | 1% | 63% | LATENCY 59%, DRAM 41% |
| **forward, 8x512** | **26.63** | **26%** | **30%** | **31%** | **24% (3.23 GB)** | **LATENCY 51%, COMPUTE 46%, DRAM 4%** |
| **forward, 8x384** | **19.94** | **26%** | **29%** | **31%** | **25% (2.51 GB)** | **LATENCY 48%, COMPUTE 47%, DRAM 5%** |
| **forward, 1x128** | **3.96** | **5%** | **6%** | **7%** | **21% (0.43 GB)** | **LATENCY 91%, COMPUTE 7%, DRAM 3%** |

```
FLOP util = T_ideal / T_measured,   T_ideal = FLOP / peak
FLOP      = sum of 2 x M x K x N over the op's products, at logical sizes
peak      = 4096 FLOP/cycle x 1.35 GHz x 110 cores / f,   f = 1, 2, 3, 4 for LoFi, HiFi2, HiFi3, HiFi4
DRAM util = (I/O bytes / T_measured) / 512 GB/s
I/O bytes = DRAM-resident inputs and outputs, tile-padded, bf8_b at 1088 B a tile, each counted once
```

- **f is the op's own fidelity** (`../tt/model_config.py`). Against the LoFi peak (f = 1 for every
  op) the 8x512 forward reads 14.8% instead of 26%.
- **110 cores for every op**, not the cores it runs on (tt-perf-report's FLOPs % uses those) nor
  the 120 in the profiler header.
- **FLOP per layer:** QKV, out_proj, SDPA's two products per sequence and head
  (4 x B x 12 x S^2 x 64), then fc1 and fc2, or the router, w1 and w2. The expert products count
  all 8 experts, since the port computes them. A quarter of that is the model's: its own FPU work
  is 14% of the 8x512 forward (3.62 of 26.63 ms).
- **A row's util** is the sum of T_ideal over the sum of T_measured of every op in it: the routing
  ops count in the router's time, the fused GELU in fc1's and w1's.
- **DRAM I/O is a floor:** an operand an op re-reads counts once, and L1 tensors not at all, which
  is why attention reads 0%. The embedding counts only the rows it gathers.
- **FPU, SFPU busy** are not computed: they are Tensix counter busy cycles over the op's cycles,
  rescaled from the profiler's 120-core basis to 110.
- **Bound:** per op, tt-perf-report's rule: DRAM or COMPUTE at 65% or more of either, else
  LATENCY. Only QKV, fc2 and the expert w1 (on the SFPU) reach a roofline.

## Assumptions

Verified: the hardware table, the model dimensions (`../reference/config.json`), the port's dtypes
and fidelities (`../tt/model_config.py`) and the measured times.

Assumed:
- Every op runs on all 110 cores at the full FPU or SFPU rate and reads DRAM at 512 GB/s. T and S
  are multiples of 32.
- Top-2 routing computes 2T (token, expert) pairs, every expert gets at least one token (true at
  all three shapes), and the cores share the work evenly whatever the per-expert load.
- "On chip" means activations never reach DRAM. The largest at 8x512, the 26.7 MB routed expert
  intermediate, fits in L1.
- SDPA's exp runs after its matmuls, as on its fp32-accumulating kernel; its softmax cost assumes
  one key chunk per sequence.
- Not counted: elementwise FPU work (rotary, norms, residual adds; about 0.13 ms at 8x512), the
  router's softmax and top-k, op launch and host dispatch.
