# Mistral-Small-24B-Base-2501 LoRA training on one Blackhole chip: performance analysis

Setup for every number unless stated: chip 0 of the QB2 box (110 cores, 31.83 GiB), seq 1024, batch 1, LoRA r16/a32 on all four block linears, AdamW, bf16, no recompute. MFU numerator: 96.37 TFLOP per 40-layer step (`doc/plan.md` §3; `useful_flops_per_step`), i.e. 2.3406 T per layer + 2.749 T fixed; denominator 148.5 TFLOP/s (110 cores x 1.35 TFLOP/s, the `ttml.common.performance` convention, i.e. the peak at the nominal 1350 MHz). The full model does not fit one chip (`doc/plan.md` §4), so MFU runs are at 8 of 40 layers and the 40-layer figures are projections from the depth fits below, labelled as such. Plan target: 1.62 s/step, 40.0 % MFU (±10 %) at the nominal clock, stretch 1.26 s / 51.6 % with a memory-bound LoRA path; see "Clock" below for the delivered clock and the target expressed at it.

**Clock, and how to read the Tracy numbers.** The chip does not run at 1350 MHz under this load. From the raw cycle counters in the Tracy CSVs (`composite/evidence/clock_analysis.txt`, `sources/examples/mistral_train/clock_analysis.py`): step period in device cycles over the host step period gives **1187 / 1171 / 1146 / 1130 MHz at 1 / 2 / 4 / 8 layers**, and step cycles over the non-profiled wall-clock 1194 / 1185 / 1172 / 1160 MHz; the idle-dominated step 1 gives 1349.2-1349.9 MHz at every depth (the control: the counter runs at AICLK, and the chip is at 1350 MHz only when idle). `tt-smi -s` during an 8-layer run: `composite/evidence/clock_trace_8layers_tt-smi.txt` (summarised in the README). Tracy converts cycles to ns at the clock it read at device open (`CHIP_FREQ[MHz]: 1350` in every `profile_log_device.csv`), so **every Tracy "ms" in this document is a cycle count divided by 1.35 GHz**: it is the kernel's cost in cycles, and the per-cycle efficiency ("% of peak") it implies is the fraction of the *delivered* peak. Wall time = Tracy ms x 1350 / f_delivered. With f = 1.16 GHz (8 layers) that factor is 1.164.

## MFU history

| Configuration | depth | ms/step (wall, no profiler) | MFU | projected 40 layers | evidence |
|---|---|---|---|---|---|
| composite baseline (frozen-weight dW skipped, frozen table untilized once, 5-op `LoraLinear`) | 8 of 40 | **434.0** (best 432.4), warm-chip rerun 453.4 | **33.32 %** | **1947 ms / 33.3 %** (wall fit) | `composite/evidence/train_8layers.log`, `train_8layers_run2.log` |
| same code, **full model, TP=4** over the box (user-authorised; 594 TFLOP/s denominator), random init | 40 of 40 | **598.7** (best 594.7) | **27.10 %** | measured, not projected | `optimization/evidence/full_tp4_baseline_random.log`, tt-smi `full_tp4_baseline_random_tt-smi.txt` |
| same, real weights, 10 steps on Shakespeare (loss 0.45 -> 0.14-0.46) | 40 of 40 | 641.9 (best 639.6; chips warm from the preceding run) | 25.27 % | measured | `optimization/evidence/full_tp4_baseline_real.log`, `.json` |
| **iteration 1**: 3-op LoRA apply (`LoraApply`: A matmul, B matmul, `addalpha`; scale folded; `dh` all-reduced before `dA` under TP), from `umales/qwen38` | 8 of 40 | 438.7 (best 435.8) at a delivered 1101 MHz (baseline 434.0 at 1160 MHz); cycle-time 357.1 vs 370.4 ms (-3.6 %) | 32.96 % | 1592.5 ms cycle-time (-66 ms), 1858-1908 ms wall at 1160-1130 MHz | `optimization/evidence/iter1_loraapply_*` |
| iteration 1, full model TP=4, random init, cool chips (10 min idle) | 40 of 40 | **574.3** (best 571.3) vs 598.7 baseline (-4.1 %) | **28.25 %** | measured | `optimization/evidence/iter1_loraapply_mfu_tp4.log`, tt-smi per chip `..._tt-smi.txt`; 4-layer TP profile 70.9 vs 73.4 ms (`iter1_loraapply_tracy_tp4_4layers`, per-chip check `tp_analysis.py`) |
| **iteration 2**: frozen base weights stored transposed (`[in, out]`, plain matmul forward on a tuned grid, `(W_t @ g^T)^T` input gradient) for qkv, out, gate_up and the LM head | 8 of 40 | **380.0** (best 376.8), cool chip (10 min idle), ~1.1 GHz; cycle-time 313.7 vs 357.1 ms (-12.2 %) | **38.06 %** | **1398.7 ms cycle-time** (-193.8), 1628-1671 ms wall at 1160-1130 MHz | `optimization/evidence/iter2b_transposed_*` |
| iteration 2, full model TP=4, random init, cool chips (7 min idle) | 40 of 40 | **564.5** (best 561.5) vs 574.3 iteration 1 (-1.7 %) | **28.74 %** | measured | `optimization/evidence/iter2c_transposed_tp_mfu_tp4.log`, per-chip tt-smi; 4-layer TP profile `iter2c_transposed_tp_tracy_tp4_4layers` (kernels 63.9 -> 58.8 ms per chip, chips equal) |

Wall-clock at the four depths, same command with `--random-init --random-data` (timing does not depend on the data): 1: 102.5, 2: 151.4, 4: 245.4, 8: 434.0 ms; MFU 33.4 / 33.1 / 33.2 / 33.3 %. Fit `wall = 56.1 + L x 47.27 ms`.

Full-model convention from here on (plan.md scope change): the 40-layer MFU and loss runs are TP=4 (weights column/row-sharded over 4 chips, vocab-parallel embedding and LM head, collectives inside every linear), reported against 4 x 148.5 TFLOP/s; the single-chip projection stays as the kernel-work yardstick (no collectives, one chip's clock). The TP=4 full model at 598.7 ms is 4 x 598.7 = 2395 chip-ms per step against 1933-1985 projected single-chip wall-ms: the difference (~20 %) is the collectives (all-reduce/all-gather per linear, replicated vocab-parallel logic) plus whatever the per-chip clock does in the mesh; it is attributed in the loop's first TP profile.

## Tracy, four doubling depths (2026-10-29)

Block = one decoder layer (homogeneous model). Runs: `python -m tracy -r -p --op-support-count 40000 -o evidence/tracy_{L}layers train.py --layers L --steps 4 --warmup-steps 2 --random-init --random-data`; device 0, steps 3-4 (steps 1-2 are warm-up: one-time untilize of the frozen table and program compiles). Analysis `composite/evidence/tracy_analysis.txt` from `sources/examples/mistral_train/analyze_tracy.py`.

| layers | ops/step | device ms/step (steps 3, 4) | fit | residual |
|---|---|---|---|---|
| 1 | 94 | 88.52 (88.88, 88.17) | 89.08 | -0.55 ms (-0.6 %) |
| 2 | 175 | 129.38 (130.06, 128.70) | 129.32 | +0.06 ms (0.0 %) |
| 4 | 337 | 210.70 (210.83, 210.57) | 209.81 | +0.89 ms (+0.4 %) |
| 8 | 661 | 370.40 (371.01, 369.80) | 370.79 | -0.39 ms (-0.1 %) |

Fit: **device ms = 48.83 + L x 40.245**; ops = 13 + 81 L. Every residual is within 0.6 % (tolerance 5 %). Projection to 40 layers: **1658.6 ms device, 3253 ops**.

Validation against the non-profiled wall-clock. Naively, wall - Tracy device = 14.0 / 22.0 / 34.7 / 63.6 ms at 1 / 2 / 4 / 8 layers (14-15 % of the wall, fit 7.2 + 7.03 L ms); the first draft of this document called that the host/dispatch gap. It is not: it is the clock conversion. In cycles (`clock_analysis.txt`, steady step 3): at 8 layers the step spans 501.75 M cycles, the kernels sum to 500.87 M (**device busy 99.8 % of the span**), and the next step's first kernel starts 1.65 ms later; the host enqueues the whole step in ~45 ms and then waits. At 1 / 2 / 4 layers the in-step idle is 0.5 / 1.1 / 0.2 % and the inter-step idle 1.5 / 1.6 / 2.2 ms. So wall = Tracy ms x 1350 / f_delivered + ~2-4 ms, with f from a measurement independent of that wall: f = 1130 MHz (the profiled run's cycles over its own host period) gives 370.4 x 1350/1130 + 2.5 = 445.1 ms against the warm-chip 443.8 ms (+0.3 %) and the cool-chip 434.0 (+2.5 %); the tt-smi in-run mean of 1151 MHz gives 437.1 ms (-1.5 % / +0.7 %). The delivered clock itself moves 1130-1160 MHz with the thermal state, which is the 434 vs 444-453 ms spread of the MFU runs. Projected 40 layers: 1658.6 ms of cycle-time, **1933 ms (1160 MHz) to 1985 ms (1130 MHz) wall, 33.6-32.7 % MFU**, with the wall fit's 1947 inside. The **host/dispatch class is ~2-4 ms per step, < 1 %**, far below the plan's 58 ms estimate, and is not a loop target.

Per class, fitted and projected to 40 layers (`analyze_tracy.py` classes; the 5-op LoRA chain's scale and add are glue, its matmuls are math):

| class | fixed ms | per layer ms | at 40 layers | share of device |
|---|---|---|---|---|
| useful math (matmul, SDPA, norms, SwiGLU, RoPE, CE, AdamW, embedding) | 47.3 | 36.63 | 1512.5 | 91.2 % |
| glue (LoRA scale and add, gradient accumulation, loss-grad scale) | 1.6 | 3.33 | 134.6 | 8.1 % |
| layout (create/concat heads, transposes, concat) | 0.0 | 0.29 | 11.5 | 0.7 % |
| wasted | 0 | 0 | 0 | the gating and the frozen-table path removed it before this baseline |
| host/dispatch (in-step idle 0.2-1.1 % + 1.5-2.2 ms between steps) | ~2 | ~0 | ~3 | < 1 % |
| delivered clock | | | | 1.13-1.19 GHz; wall = cycle-time x 1.16-1.19 |

## Ranked cost table, projected to 40 layers

From `tracy_analysis.txt` (`analyze_tracy.py --warmup 2 --steps 2 --full-depth 40 --top 45`; per-item fit over the four depths; ops per step at full depth) with the attribution from the one-step op sequence at 1 layer (`composite/evidence/op_sequence_1layer.txt`, op numbers `#` below refer to it). Multiplicity is already in the ops/step column (forward + backward mirror; no recompute). The Tracy CSV reports a stale `[1024,6144]` shape for every `BinaryNg` launch, so the glue rows are keyed by sequence position, not shape; the loop must recover the real shapes from the op sequence before costing the glue chain. All ms are cycle-time (cycles / 1.35 GHz, see "Clock"): multiply by 1.16-1.19 for wall time; every "TFLOP/s" and "% of peak" is per cycle, i.e. relative to the delivered peak (at 1.16 GHz the delivered peak is 127.6 TFLOP/s and the gate_up forward's 65 TFLOP/s-per-cycle is 56 TFLOP/s of wall-clock throughput, 38 % of the nominal 148.5). Ranks and ratios between kernels do not depend on the clock.

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
| dispatch | 58 ms (16 µs/op) | ~3 ms (device busy 99.8 % of the step; the host enqueues a step in ~45 ms) |
| clock | 1350 MHz assumed | 1.13-1.19 GHz delivered: x 1.16-1.19 on every row above for wall time |
| **step** | **1620 ms / 40.0 %** at 1350 MHz (1562 kernel + 58 dispatch); 1885 ms / 34.4 % at the delivered 1.16 GHz | **1659 ms cycle-time** of kernels + ~3 ms dispatch (= 1933-1985 ms wall at 1160-1130 MHz) / 33.6-32.7 % MFU (projected); 434 ms wall at 8 layers measured |

In cycle-time the baseline's kernels take 1659 ms against the target's kernel budget of 1562 ms (the 1620 ms target includes 58 ms of planned dispatch that does not exist): 6 % above, inside the ±10 % margin. The kernel work still ahead of the loop is the 1659 -> ~1200 ms stretch (memory-bound LoRA) and the items below, and the wall-clock sits 1.16-1.19x above the cycle-time because of the delivered clock, which no kernel change moves (see the plan's §3 calibration note for the exit condition at the delivered clock). Every MFU rerun in the loop carries its own clock measurement (tt-smi sampled alongside the run, as `composite/evidence/device_chain2.sh` does) or a cool-chip protocol, since the thermal state alone moves the wall-clock by 4 %.

Ranked next items for the loop (class first, then cost, at 40 layers, cycle-time): glue 135 ms (the LoRA scale+add chain -> `LoraApply` from `umales/qwen38` folds the scale into an `addalpha` and removes 2 launches per adapter per pass; grad-accumulation adds); gate_up forward matmul at 44 % of the delivered peak vs its dX at 68 % (-170 ms if it reached the same efficiency; LM head forward is the same class, -7 ms); LoRA skinny matmuls (~210 ms vs 41 ms memory roofline, the plan's stretch item); RMSNorm backward on 32 cores (-25 ms); SDPA backward (2.23 ms vs 0.23 roofline per layer, 89 ms); layout 11.5 ms. Not a target: host/dispatch (~3 ms).

## Iteration 1 (2026-10-29): LoRA scale+add chain -> 3-op `LoraApply`

Class: glue (and 2 of the 5 launches per adapter per pass). Rung: reuse (`umales/qwen38`'s `LoraApply` / `_LoraUpdateAdd` ported into `ttml/modules/lora.py`; under TP the column-parallel adapter all-reduces `dh` before `dA`, which also makes the replicated `lora_A` train identically on every chip: the baseline TP path silently diverged there). Verification: PCC harness 11 passed (`optimization/evidence/iter1_loraapply_harness.log`); Tracy at 1/2/4/8 layers (`iter1_loraapply_tracy_analysis.txt`, residuals ≤ 0.6 %) and the diff against the baseline (`diff_tracy.py`):

| depth | device ms (cycle-time) before -> after |
|---|---|
| 1 | 88.52 -> 86.55 |
| 2 | 129.38 -> 125.97 |
| 4 | 210.70 -> 202.44 |
| 8 | 370.40 -> 357.12 |
| fit | 48.83 + 40.245 L -> 48.25 + 38.606 L; projection 1658.6 -> **1592.5 ms** (-66.1) |
| class projection | glue 134.6 -> 72.9, layout 11.5 -> 11.4, math 1512.5 -> 1508.2 |

Targeted ops gone: the scalar `multiply` on the update (160/step, 33.5 ms) and the scalar grad scale on `[1024,5120]` (160/step, 29.3 ms); the two-input `BinaryNg` row shrinks 480 -> 160 ops (70.6 -> 38.6 ms); new: 320 tiny scales on `[1024,32]` / `[out,32]` (3-23 µs each; 31.2 ms is the stale-shape row, the real tensors are the rank-32 ones) and the LoRA A/B matmuls re-keyed by the raw `ttnn.matmul` shapes (+7.3 / +4.1 / -4.3 / -3.6 / -3.5 / -3.4 / +2.2 ms, net ≈ 0). Nothing else changed. Ops per step unchanged (94 / 175 / 337 / 661): the chain lost 2 launches per adapter per forward and gained the 2 tiny scales per adapter in backward.

Single-chip wall: 438.7 ms vs 434.0 baseline, but the delivered clock was 1101 vs 1160 MHz (`iter1_loraapply_clock_analysis.txt`; the chain ran the harness and four profiles first); at equal clock the 3.6 % cycle-time gain is 418 ms. TP=4 full model: **574.3 ms (28.25 %) vs 598.7 (27.10 %)** on cool chips.

Anomaly (resolved): the first TP measurement of this iteration read 816.9 ms/step and a 140 ms 4-layer profile (`iter1_loraapply_mfu_tp4_hot.log`, `..._tracy_tp4_4layers_hot/`). Per chip, chip 3 ran every matmul at 1.68x the cycles of the other chips on the same core counts (e.g. the `[1024,16384]x[16384,5120]` dX 11.72 vs 6.95 ms), so chips 0-2 waited in every collective (the 64 KB `dh` reduce-scatter 522 vs 14-54 µs, the 10 MB one 1123 vs 187-209 µs, 31 % idle); the run followed a 13-minute TP run without cool-down. Control: the same code after 10 min idle (`tp_analysis.py` on `iter1_loraapply_tracy_tp4_4layers`: all four chips within 3 % on every matmul shape, idle 1.7-2.0 %), and the baseline profile taken in between (73.4 ms, chips equal). Subsystem: thermal/power throttling of one chip. Resolution: measurement protocol changed (cool-down before TP runs, tt-smi per chip alongside, per-chip matmul ratio check before any TP number is used). The ported code change itself costs 16 x 54 µs = 0.9 ms per 4 layers of small reduce-scatters (9 ms at 40 layers) for the correct `dA`.

Remaining glue after iteration 1 (per layer, from `iter1_loraapply_op_sequence_1layer.txt`): the `addalpha` on the gate_up output (#20, 783 µs: 134 MiB x 3 streamed, at DRAM roofline), the accumulation of the base and LoRA `dx` into the SwiGLU-output gradient (#46, 374 µs on `[1024,32768]`), the three small `addalpha`s and two residual adds (~60 µs each), and the fixed loss-gradient scale on the logits (#36, 1.36 ms) plus the four `MorehMean` ops of the mean reduction, which the `cross_entropy_bw` kernel's `scaler` argument could absorb. Only a fusion of the LoRA update into the base matmul (K-concatenation `[x | s·h] @ [W | B]^T`, the forward half estimated at -1.5 ms/layer net of the activation concats) removes the first; it is the next glue item.

Next by class, then cost: glue 73 ms (K-concat forward fusion, est. -60 ms; loss-grad scale -1.4 ms), layout 11.4 ms (heads create/concat), then math: the gate_up forward matmul at 44 % of the delivered peak vs its dX at 68 % (-170 ms if matched; LM-head forward is the same class, -7 ms), LoRA skinny matmuls (~200 ms vs 41 roofline), SDPA backward (89 vs 9), RMSNorm backward on 32 cores (30 vs 5). Because the glue remainder cannot go to zero without the fusion work and the gate_up forward is a program-config question on an existing kernel, iteration 2 takes the matmul (a few hours of experiments, 2.5x the remaining glue in cost) and iteration 3 the fusion; recorded as the stated reason for the order.

## Iteration 2 (2026-10-29): base-linear matmuls on a transposed frozen weight

Class: useful math (the biggest kernels of the step). Rung: reuse (the same `ttnn.matmul` kernel with a different weight layout, core grid and gradient route; no new kernel). Attribution: the gate_up forward `[1024,5120] x [65536,5120]^T` ran at 44 % of the delivered peak while its dX ran at 68 % (baseline ranked table); microbenchmarks of every base-linear shape (`optimization/evidence/iter2_matmul_bench{,2,3,4}.log`, `sources/examples/mistral_train/matmul_bench*.py`) showed that (a) `transpose_b` costs 14-28 % on the wide-N shapes, (b) ttnn's default program config splits N over the grid's 11 columns and 65536/32 = 2048 tiles do not divide by 11 but do by 8, so an 8x11 grid is 10 % faster again for gate_up (and catastrophic for N = 131072), (c) the input gradient is 7-17 % faster as `(W_t @ g^T)^T` than as `g @ W` for every shape except w2 (`32768 -> 5120`, where the `[32768, 1024]` transpose costs more than it saves), and (d) without an explicit `core_grid` ttnn's default config is 1.5-1.8x slower on these shapes, which is what the first profile of this iteration showed (`iter2_transposed_*`: gate_up dX 274 -> 358 ms) before the fix.

Implementation: `ttml/modules/frozen_linear.py` (`FrozenLinearT` autograd function, `transpose_frozen_weights(model)` converting every frozen linear whose local `(in, out)` is in the measured allow-list once after loading and LoRA injection, tuned `FORWARD_CORE_GRID`), honoured by `LinearLayer`, the column/row-parallel linears and the three LoRA wrappers; `train.py` applies it unless `--no-transposed-weights`. A trainable weight is never transposed (optimizer and checkpoint formats expect `[out, in]`).

Verification: harness 13 passed with the two new transposed-path cases (`optimization/evidence/iter2c_harness.log`), same PCC and relative errors as the untransposed path. Tracy at 1/2/4/8 layers (`iter2b_transposed_tracy_analysis.txt`, residuals ≤ 0.6 %) against iteration 1 (`diff_tracy.py`):

| depth | device ms (cycle-time) iteration 1 -> 2 | ops/step |
|---|---|---|
| 1 | 86.55 -> 76.31 | 94 -> 98 |
| 2 | 125.97 -> 110.12 | 175 -> 182 |
| 4 | 202.44 -> 177.87 | 337 -> 350 |
| 8 | 357.12 -> 313.66 | 661 -> 686 |
| fit | 48.25 + 38.606 L -> 42.33 + 33.911 L; projection 1592.5 -> **1398.7 ms** (-193.8, -12.2 %) | 13 + 81 L -> 14 + 84 L |
| class projection | math 1508.2 -> 1310.3, glue 72.9 -> 71.8, layout 11.4 -> 16.6 (+3.9 ms of `[5120,1024]` transposes in the dX route, 81/step) |

Per kernel at 40 layers (cycle-time): gate_up forward 424.3 -> 293.1 (7.33 ms per call = 94 TFLOP/s per cycle, 63 % of the delivered peak, from 44 %), gate_up dX 273.5 -> 218.9 (5.47 ms, 85 %), qkv dX 26.9 -> 22.1, out dX 18.6 -> 14.3, LM head forward 19.1 -> 18.1 and dX 13.7 -> 10.9; qkv/out forwards and w2 unchanged (w2 left untransposed by measurement). Nothing else changed. Single-chip MFU at 8 of 40 layers: 380.0 ms, 38.06 % (baseline 434.0 / 33.32 %, iteration 1 438.7 at a lower clock).

TP=4 (`transpose_frozen_weights` converts the local shards whose shape is listed: gate_up, w2 and the LM head shards, 5 linears at 2 layers, 121 at 40): full model **564.5 ms, 28.74 %** vs 574.3 / 28.25 % (iteration 1, cool chips both). The 4-layer TP profile (`tp_analysis.py`): kernel time per chip 63.9 -> 58.8 ms (matmuls 36.8 -> 33.2), collectives 5-9 ms, idle 1.5 %, all four chips equal on every matmul shape; the wall at 4 layers stayed 70.5 vs 70.9 ms because that profile ran straight after the MFU run and the chips sat at 1.13-1.22 GHz instead of 1.21-1.28. The TP shards gain less than the single chip (the local gate_up `[1024,5120]x[5120,16384]` forward was already even over the grid), which is why the single-chip cycle-time is the yardstick and the TP full model the report.

Rejected within this iteration: HiFi2 math fidelity (another 15-25 % on the big matmuls in the microbenchmark, relative error 4.7e-3 against HiFi4) is a numerics policy change, not a layout one; listed as a separate candidate to be taken only with the PCC harness and the loss comparison rerun. Storing LoRA B transposed halves `dB` (0.59 -> 0.39 ms for gate_up) and is queued with the LoRA skinny-matmul item.

Next by class, then cost: glue 71.8 ms (the gate_up `addalpha` 31 ms needs the K-concat fusion; the loss-gradient scale and mean chain 1.4 ms fixed), layout 16.6 ms (the dX transposes 3.9 ms are the price of a faster matmul; heads create/concat 11-12 ms), then math: LoRA skinny matmuls (~200 ms at 1-5 % of peak vs 41 ms memory roofline; B transposed, program configs), gate_up forward still at 63 % (-70 ms to its dX's 85 %), SDPA backward (89 vs 9 ms), RMSNorm backward on 32 cores (30 vs 5), w2 forward at 67 %.
