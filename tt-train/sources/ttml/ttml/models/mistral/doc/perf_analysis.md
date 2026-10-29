# Mistral-Small-24B-Base-2501 LoRA training on one Blackhole chip: performance analysis

Setup for every number unless stated: chip 0 of the QB2 box (110 cores, 31.83 GiB), seq 1024, batch 1, LoRA r16/a32 on all four block linears, AdamW, bf16, no recompute. MFU numerator: 96.37 TFLOP per 40-layer step (`doc/plan.md` §3; `useful_flops_per_step`), i.e. 2.3406 T per layer + 2.749 T fixed; denominator 148.5 TFLOP/s. The full model does not fit one chip (`doc/plan.md` §4), so MFU runs are at 8 of 40 layers and the 40-layer figures are projections from the depth fits below, labelled as such. Plan target: 1.62 s/step, 40.0 % MFU (±10 %), stretch 1.26 s / 51.6 % with a memory-bound LoRA path.

## MFU history

| Configuration | depth | ms/step (wall, no profiler) | MFU | projected 40 layers | evidence |
|---|---|---|---|---|---|
| composite baseline (frozen-weight dW skipped, frozen table untilized once, 5-op `LoraLinear`) | 8 of 40 | **434.0** (best 432.4), warm-chip rerun 453.4 | **33.32 %** | **1947 ms / 33.3 %** (wall fit) | `composite/evidence/train_8layers.log`, `train_8layers_run2.log` |

Wall-clock at the four depths, same command with `--random-init --random-data` (timing does not depend on the data): 1: 102.5, 2: 151.4, 4: 245.4, 8: 434.0 ms; MFU 33.4 / 33.1 / 33.2 / 33.3 %. Fit `wall = 56.1 + L x 47.27 ms`.

## Tracy, four doubling depths (2026-10-29)

Block = one decoder layer (homogeneous model). Runs: `python -m tracy -r -p --op-support-count 40000 -o evidence/tracy_{L}layers train.py --layers L --steps 4 --warmup-steps 2 --random-init --random-data`; device 0, steps 3-4 (steps 1-2 are warm-up: one-time untilize of the frozen table and program compiles). Analysis `composite/evidence/tracy_analysis.txt` from `sources/examples/mistral_train/analyze_tracy.py`.

| layers | ops/step | device ms/step (steps 3, 4) | fit | residual |
|---|---|---|---|---|
| 1 | 94 | 88.52 (88.88, 88.17) | 89.08 | -0.55 ms (-0.6 %) |
| 2 | 175 | 129.38 (130.06, 128.70) | 129.32 | +0.06 ms (0.0 %) |
| 4 | 337 | 210.70 (210.83, 210.57) | 209.81 | +0.89 ms (+0.4 %) |
| 8 | 661 | 370.40 (371.01, 369.80) | 370.79 | -0.39 ms (-0.1 %) |

Fit: **device ms = 48.83 + L x 40.245**; ops = 13 + 81 L. Every residual is within 0.6 % (tolerance 5 %). Projection to 40 layers: **1658.6 ms device, 3253 ops**.

Validation against the non-profiled wall-clock: the gap (wall - device) is 14.0 / 22.0 / 34.7 / 63.6 ms at 1 / 2 / 4 / 8 layers = 13.6-14.7 % of the wall at every depth, fit **gap = 7.2 + L x 7.03 ms** (96-149 µs per op, falling with depth as the fixed part amortises). Projected 40 layers: device 1658.6 + gap 288 = **1947 ms**, the same as the wall fit. The gap is ~6x the 16 µs/op the plan assumed: it is the host/dispatch class and is a loop item (see below).

Per class, fitted and projected to 40 layers (`analyze_tracy.py` classes; the 5-op LoRA chain's scale and add are glue, its matmuls are math):

| class | fixed ms | per layer ms | at 40 layers | share of device |
|---|---|---|---|---|
| useful math (matmul, SDPA, norms, SwiGLU, RoPE, CE, AdamW, embedding) | 47.3 | 36.63 | 1512.5 | 91.2 % |
| glue (LoRA scale and add, gradient accumulation, loss-grad scale) | 1.6 | 3.33 | 134.6 | 8.1 % |
| layout (create/concat heads, transposes, concat) | 0.0 | 0.29 | 11.5 | 0.7 % |
| wasted | 0 | 0 | 0 | the gating and the frozen-table path removed it before this baseline |
| host/dispatch (wall - device) | 7.2 | 7.03 | 288 | 14.8 % of wall |

## Ranked cost table, projected to 40 layers

From `tracy_analysis.txt` (per-item fit over the four depths; ops per step at full depth) with the attribution from the one-step op sequence at 1 layer (`composite/evidence/op_sequence_1layer.txt`, op numbers `#` below refer to it). Multiplicity is already in the ops/step column (forward + backward mirror; no recompute). The Tracy CSV reports a stale `[1024,6144]` shape for every `BinaryNg` launch, so the glue rows are keyed by sequence position, not shape.

| ms/step | ops/step | class | op | model code | note |
|---|---|---|---|---|---|
| 424.9 | 40 | math | Matmul `[1024,5120] x [65536,5120]^T` (110 cores) | `LlamaMLP.forward` `w_gate_up` (#19) | 10.5 ms per call = 65 TFLOP/s = **44 % of peak**; the dX of the same weight (next row) runs at 68 %. Top math item. |
| 273.5 | 40 | math | Matmul `[1024,65536] x [65536,5120]` (88 cores) | `w_gate_up` dX, `linear_op.cpp` (#56) | 6.8 ms = 100 TFLOP/s, 68 % |
| 137.8 | 40 | math | Matmul `[1024,32768] x [5120,32768]^T` (88) | `w2` forward (#25) | 3.4 ms = 100 TFLOP/s, 67 % |
| 121.6 | 40 | math | Matmul `[1024,5120] x [5120,32768]` (110) | `w2` dX (#48) | 3.0 ms = 113 TFLOP/s, 76 % |
| 70.6 | 480 | glue | `BinaryNg`, two inputs | LoRA `update * scaling` is the single-input row below; this row: `base + update` for the 4 adapters (#6, #16, #23, #29), the residual adds (#17, #30), the LoRA backward grad accumulations into the shared input (#49, #57, #59, #66, #83, #85) | the gate_up add alone is 0.81 ms per call (#23) |
| 54.4 | 40 | math | `SDPABackwardKV` `[1,32,1024,128]` | `scaled_dot_product_attention` backward (#70) | |
| 45.1 | 40 | math | `SwigluPackedBw` `[1024,65536]` | `swiglu_packed` backward (#50) | 1.15 ms vs 0.66 ms roofline |
| 35.3 | 40 | math | `SDPAForward` | (#10) | 0.88 ms; 17.2 GFLOP full-square -> 20 TFLOP/s counted (causal kernel does half) |
| 34.7 | 40 | math | `SDPABackwardQ` | (#69) | |
| 33.5 | 160 | glue | `BinaryNg`, scalar | LoRA `update * scaling` forward (#5, #15, #22, #28); the gate_up one is 0.67 ms | ideal 0: fold into the B matmul or `addalpha` |
| 33.0 | 40 | math | Matmul `[1024,32] x [65536,32]^T` (110) | gate_up LoRA B forward (#21) | 0.82 ms for 134 MiB written: 164 GB/s |
| 30.8 | 40 | math | Matmul `[1024,65536] x [65536,32]^T` (33 cores) | gate_up LoRA `dh = g @ B` (#53) | 0.77 ms on 33 cores |
| 29.7 | 81 | math | `RMSNormBackward` (32 cores) | `RMSNormLayer` backward (#42, #58, #84) | 0.37 ms on 32 cores vs 0.06 ms roofline |
| 29.3 | 160 | glue | `BinaryNg`, scalar, `[1024,5120]` | gradient scale/accumulate on the residual stream in backward (#43, #51, #60, #77) | #51 is 0.59 ms (the `[1024,32768]`-wide one, stale shape) |
| 26.9 | 40 | math | Matmul `[1024,6144] x [6144,5120]` | `qkv_linear` dX (#82) | 0.67 ms = 96 TFLOP/s |
| 25.9 | 40 | math | Matmul `[1024,5120] x [6144,5120]^T` | `qkv_linear` forward (#2) | 0.65 ms = 100 TFLOP/s |
| 22.8 | 40 | math | Matmul `[1024,65536]^T x [1024,32]` | gate_up LoRA `dB` (#52) | |
| 20.2 | 1 | math | Matmul `[1024,5120] x [131072,5120]^T` | `fc` (LM head) forward (#32) | 21.0 ms = 66 TFLOP/s, 44 %, same shape class as gate_up forward |
| 18.6 + 17.1 | 80 | math | `out_linear` forward (#12) / dX (#65) | | 0.43-0.47 ms, 92-100 TFLOP/s |
| 16.9 + 15.1 + 7.3 | 120 | math | w2 LoRA A forward `[1024,32768]x[32,32768]^T` (#26, 0.38 ms on 33 cores), `dh` (#47), `dB` (#46) | | |
| 16.2 | 40 | math | `SwigluPackedFw` | (#24) | 0.40 ms = roofline |
| 13.6 | 1 | math | Matmul `[1024,131072] x [131072,5120]` | LM head dX (#41) | 13.6 ms = 101 TFLOP/s |
| 12.4 | 81 | math | `RMSNormForward` (32 cores) | (#1, #18, #31) | 0.15 ms vs 0.04 roofline |
| 7.3 + 3.9 | 2 | math | `CrossEntropyBackward` / `Forward` (32 cores) | `cross_entropy_loss` (#39, #33) | plus 1.38 ms loss-grad scale `BinaryNg [1024,131072]` (#40, glue) and 4 `MorehMean` + `Full` (#34-38) |
| 7.4 x 2 + 6.2 x 2 + 3.9 + 3.5 + 4.3 + 3.7 + 2.5 + 2.3 | ~600 | math | the remaining LoRA matmuls of qkv / out / w2 (A, B, dA, dB, dh, dx) at 33-99 cores, 40-110 µs each | `LoraLinear.forward` and its autograd | each 1-5 % of peak |
| 5.8 | 80 | math | `RotaryEmbeddingLlama` q and k, forward and backward | `rope` (#8, #9, #71, #72) | |
| 5.2 | 160 | math | `AdamW` on the 8 adapter tensors per layer | (#86-93) | 12-84 µs each |
| 3.0 + 2.5 + 2.1 + ~2 | 240 | layout | `NLPConcatHeads` (#11, #73-75), `Concat` (#76), `NlpCreateHeads` (#7), transposes in `heads_fusion` backward (#67-68) | `heads_creation` / `heads_fusion` | 0.29 ms per layer in total |
| 0.07 | 1 | math | `Embeddings` on the row-major frozen table | `FrozenEmbedding` (#0) | the per-step untilize is gone |

Sum of the table ≈ the projected 1659 ms device.

## Where the baseline stands against the plan

| | plan: achievable (existing kernels) | measured / projected |
|---|---|---|
| big matmuls per layer | 24.7 ms (62 %) | 26.2 ms: gate_up forward at 44 % is the outlier, the rest at 67-76 % |
| LoRA adapters per layer (matmuls + chain) | 10.5 ms | ~9.9 ms (matmuls ~5.4 + glue ~4.5) |
| SDPA per layer | 0.69 ms | 3.11 ms (fwd 0.88, bwd 2.23) |
| memory-bound ops per layer (SwiGLU, norms, RoPE) | 1.9 ms | 3.4 ms (RMSNorm bwd 1.1 on 32 cores, SwiGLU bwd 1.15) |
| fixed | 35.5 ms | 48.8 ms device (LM head forward 21.0 at 44 %, CE 11.2, loss-grad scale 1.4) |
| dispatch | 58 ms (16 µs/op) | 288 ms (88 µs/op at full depth) |
| **step** | **1620 ms / 40.0 %** | **1947 ms / 33.3 %** (projected), 434 ms at 8 layers measured |

Ranked next items for the loop (class first, then cost, at 40 layers): host/dispatch gap 288 ms (fewer ops: the LoRA chain is 5 launches per adapter per pass, 40 of the 81 ops per layer are LoRA or glue); glue 135 ms (scale+add chain -> `LoraApply` from `umales/qwen38`, grad accumulation adds); gate_up forward matmul at 44 % (-170 ms if it reached its dX's 68 %); LM head forward, same class (-7 ms); RMSNorm backward on 32 cores (-25 ms); SDPA backward (2.23 ms vs 0.23 roofline); LoRA skinny matmuls (~210 ms vs 41 ms memory roofline); layout 11.5 ms.
