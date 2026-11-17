**Matmul shape set and roofline**

Stock TTNN models on `main` @ 8e28a65, QuietBox 2 (2 × P300c, 4 chips), Tracy device kernel time per chip. Covers #65142's scope: every Qwen3.8 projection and the LM head, FLUX.2's DiT projections and text encoder, PDL's 1×1 convs and bilinear-upsample matmuls. Roofline per chip = max(compute, DRAM): 2·M·K·N at 70% of 110 cores × 4096 FLOP/cycle ÷ fidelity passes × 1.35 GHz (426 TFLOPS LoFi, 213 HiFi2), and the bytes of DRAM-resident operands at 90% of 512 GB/s (spec), dims padded to 32. Ethernet is not in the roofline because each matmul runs on one chip, but Qwen3.8 and FLUX.2 do move data over Ethernet in the collectives around their matmuls: *Collective* marks a separate all-gather (AG), reduce-scatter (RS) or all-reduce (AR) next to the matmul on QB2; its time is not included, and fusing it is #65143. All accumulate in fp32 except PDL's bilinear upsample.

| Model | Per | Calls | Measured / roofline | Share of model device time |
|---|---|---:|---|---|
| Qwen3.8-27B (1×4, TP4) | decode token | 260 | 10.14 / 8.17 ms (1.24×) | 45% |
| | prefill, M = 128 / 512 / 1k / 2k / 4k | 256 | 1.93 / 2.07 / 1.49 / 1.37 / 1.34× | 35 / 35 / 34 / 33 / 32% |
| FLUX.2-dev (2×2, SP2 × TP2) | denoising step | 215 | 317.9 / 303.1 ms (1.05×) | 34% |
| | image (prompt encoding) | 201 | 69.6 / 53.7 ms (1.30×) | 60% of text encoder |
| PDL (1 chip) | inference | 55 | 0.99 / 0.33 ms (3.00×) | 17% |

### Qwen3.8-27B

Real weights; the demo's reduced profiling stack (one GDN + one full-attention layer) scaled to 48 + 16 layers. All bf16 × bfp4 → bf16, LoFi.

Decode, measured at batch 1, 16 and 32 (one 32-row tile; times within 1.1%): `ttnn.linear` DRAM-sharded; activation and output L1 width-sharded, weights DRAM width-sharded; all DRAM-bound.

| Matmul | Calls | Shape per chip | Collective | Measured µs | Roofline µs | Ratio |
|---|---:|---|---|---:|---:|---:|
| GDN in_proj (qkv/z/b/a) | 48 | `[1 × 5,120] · [5,120 × 4,608]` (4,120 used) |  | 37.5 | 28.8 | 1.30× |
| Full-attn qkvg (q/gate/k/v) | 16 | `[1 × 5,120] · [5,120 × 3,584]` |  | 29.2 | 22.4 | 1.31× |
| GDN out_proj, full-attn o_proj | 64 | `[1 × 1,536] · [1,536 × 5,120]` | AR after | 15.3 | 9.60 | 1.59× |
| MLP gate_up | 64 | `[1 × 5,120] · [5,120 × 9,216]` (8,704 used) |  | 63.3 | 57.6 | 1.10× |
| MLP down | 64 | `[1 × 4,352] · [4,352 × 5,120]` | AR after | 36.9 | 27.2 | 1.36× |
| LM head (vocab chunks) | 3 + 1 | `[1 × 5,120] · [5,120 × 16,384 / 12,928]` | | 127 / 105 | 102 / 80.8 | 1.24 / 1.29× |

Prefill, measured / roofline µs: M = 128 runs `ttnn.linear` 1D mcast (DRAM-bound), M ≥ 512 `minimal_matmul` (compute-bound, except out_proj/o_proj up to M = 1,024). Operands DRAM interleaved, except the GDN-layer activations in L1 at M = 128. The LM head runs once per prefill, as in decode.

| Matmul × calls | Shape per chip | Collective | M = 128 | M = 512 | M = 1,024 | M = 2,048 | M = 4,096 |
|---|---|---|---:|---:|---:|---:|---:|
| GDN in_proj (qkv/z/b/a) ×48 | `[M × 5,120] · [5,120 × 4,160]` |  | 63.6 / 28.3 | 109 / 51.2 | 139 / 102 | 256 / 205 | 508 / 410 |
| Full-attn qkvg (q/gate/k/v) ×16 | `[M × 5,120] · [5,120 × 3,584]` |  | 52.9 / 27.2 | 101 / 44.1 | 133 / 88.3 | 239 / 177 | 372 / 353 |
| GDN out_proj, full-attn o_proj ×64 | `[M × 1,536] · [1,536 × 5,120]` | RS after | 26.5–27.3 / 12.4–13.3 | 52.1 / 24.4 | 83.2 / 39.2 | 157 / 75.7 | 299 / 151 |
| MLP gate_up ×64 | `[M × 5,120] · [5,120 × 8,704]` |  | 105–113 / 59.2–62.1 | 214 / 107 | 307–315 / 214 | 565 / 429 | 1,148 / 857 |
| MLP down ×64 | `[M × 4,352] · [4,352 × 5,120]` | RS after | 58.7 / 30.0–32.5 | 112 / 53.6 | 157 / 107 | 283 / 214 | 535 / 429 |
| Ratio (all) | | | 1.93× | 2.07× | 1.49× | 1.37× | 1.34× |

### FLUX.2-dev

QB2 config of `test_performance_flux2.py` (SP2 × TP2, FSDP, `shard_prompt`), 1024², 512-token prompt. Random weights at the FLUX.2-dev config (the checkpoint is gated); 1 double + 2 single blocks and 2 text-encoder layers, scaled to 8 + 48 and 40. `minimal_matmul` (`ttnn.linear` for the M = 1 MLPs), HiFi2, operands DRAM interleaved, bf16 × bf16 → bf16, except fp32 activations on the timestep path and `time_mod`, and bfp8 text-encoder weights. FSDP all-gathers each block weight before its matmul. DRAM-bound: the M = 1 rows, the 256-row text stream, `context_embedder`, `x_embedder` and `proj_out`; the rest is compute-bound. On QB2 the DiT M is 256 / 2,048 / 2,304 (text / image / joint); #65142's 128 / 1,024 / 1,152 are Galaxy (SP4 × TP8) sizes.

| Matmul | Calls | Shape per chip | Collective | Measured µs | Roofline µs | Ratio |
|---|---:|---|---|---:|---:|---:|
| *Per denoising step (×50)* | | | | | | |
| Timestep / guidance MLP linear_1 | 1 + 1 | `[1 × 256] · [256 × 6,144]` |  | 13.1 / 10.9 | 8.60 / 7.72 | 1.52 / 1.41× |
| Timestep / guidance MLP linear_2 | 1 + 1 | `[1 × 6,144] · [6,144 × 6,144]` |  | 195 / 192 | 167 / 166 | 1.17 / 1.16× |
| time_mod (17 modulation sets, split output) | 1 | `[1 × 6,144] · [6,144 × 52,224]` + bias |  | 2,311 | 1,409 | 1.64× |
| x_embedder | 1 | `[2,048 × 128] · [128 × 3,072]` |  | 62.9 | 30.1 | 2.09× |
| Double QKV (img / txt), split ×3 | 8 + 8 | `[2,048 / 256 × 6,144] · [6,144 × 9,216]` | AG before | 1,216 / 453 | 1,089 / 263 | 1.12 / 1.72× |
| Double attn out (img / txt), + addcmul | 8 + 8 | `[2,048 / 256 × 6,144] · [6,144 × 3,072]` | AG before | 568 / 193 | 363 / 96.0 | 1.56 / 2.01× |
| Double FF1 (img / txt), + SwiGLU | 8 + 8 | `[2,048 / 256 × 6,144] · [6,144 × 18,432]` | AG before | 2,394 / 866 | 2,179 / 509 | 1.10 / 1.70× |
| Double FF2 (img / txt) | 8 + 8 | `[2,048 / 256 × 9,216] · [9,216 × 6,144]` | RS after | 1,259 / 456 | 1,089 / 263 | 1.16 / 1.74× |
| Single QKV, split ×3 | 48 | `[2,304 × 6,144] · [6,144 × 9,216]` | AG before | 1,245 | 1,226 | 1.02× |
| Single FF1, + SwiGLU (blocks 1–47 / final) | 47 + 1 | `[2,304 / 2,048 × 6,144] · [6,144 × 18,432]` | AG before | 2,398 / 2,394 | 2,451 / 2,179 | 0.98 / 1.10× |
| Single out, attn ‖ MLP inputs (blocks 1–47 / final) | 47 + 1 | `[2,304 / 2,048 × (3,072 + 9,216)] · [12,288 × 6,144]` | RS after | 1,689 / 1,641 | 1,634 / 1,453 | 1.03 / 1.13× |
| proj_out | 1 | `[2,048 × 3,072] · [3,072 × 128]` | RS after | 61.0 | 30.1 | 2.02× |
| *Per image* | | | | | | |
| context_embedder | 1 | `[256 × 15,360] · [15,360 × 3,072]` |  | 405 | 225 | 1.80× |
| Text encoder qkv_proj | 40 | `[512 × 5,120] · [5,120 × 3,072]` |  | 139 | 75.7 | 1.84× |
| Text encoder o_proj | 40 | `[512 × 4,096] · [4,096 × 2,560]` | AG before + after | 79.4 | 50.4 | 1.57× |
| Text encoder gate, up (separate) | 80 | `[512 × 5,120] · [5,120 × 16,384]` |  | 501 | 403 | 1.24× |
| Text encoder down | 40 | `[512 × 16,384] · [16,384 × 5,120]` | RS after | 509 | 403 | 1.26× |

### PDL

`test_tt_model.py -k panoptic_deeplab_110_cores` (CI perf test), 512×1024, Detectron2 weights; 5.75 ms/inference. All bfp8 × bfp8 → bfp8, LoFi, with bias (+ ReLU on reduce, ASPP and decoder convs); weights DRAM interleaved; activations L1 height-sharded (HS, 1D mcast) or block-sharded (BS, 2D mcast). Called as `ttnn.conv2d` 1×1 or `ttnn.matmul` (upsample). Compute-bound unless marked.

<details><summary>55 matmuls per inference</summary>

| Matmul | Calls | Shape per chip | Act | Measured µs | Roofline µs | Ratio |
|---|---:|---|---|---:|---:|---:|
| res2 shortcut | 1 | `[32,768 × 128] · [128 × 256]` | HS | 9.80 | 5.04 | 1.94× |
| res2 reduce | 1 | `[32,768 × 128] · [128 × 64]` | HS | 3.98 | 1.26 | 3.16× |
| res2 expand | 3 | `[32,768 × 64] · [64 × 256]` | HS | 8.72 | 2.52 | 3.46× |
| res2 reduce | 2 | `[32,768 × 256] · [256 × 64]` | HS | 5.01 | 2.52 | 1.99× |
| res3 shortcut, stride 2 (conv2d path, bf16 act) | 1 | `[8,192 × 256] · [256 × 512]` | HS | 13.1 | 5.04 | 2.59× |
| res3 reduce | 1 | `[32,768 × 256] · [256 × 128]` | HS | 9.36 | 5.04 | 1.85× |
| res3 expand | 4 | `[8,192 × 128] · [128 × 512]` | HS | 8.17 | 2.52 | 3.24× |
| res3 reduce | 3 | `[8,192 × 512] · [512 × 128]` | HS | 6.90 | 2.52 | 2.73× |
| res4 shortcut, stride 2 (conv2d path, bf16 act) | 1 | `[2,048 × 512] · [512 × 1,024]` | BS | 16.2 | 5.04 | 3.21× |
| res4 reduce | 1 | `[8,192 × 512] · [512 × 256]` | BS | 16.6 | 5.04 | 3.28× |
| res4 expand | 6 | `[2,048 × 256] · [256 × 1,024]` | BS | 13.6 | 2.52 | 5.40× |
| res4 reduce | 5 | `[2,048 × 1,024] · [1,024 × 256]` | BS | 9.40 | 2.52 | 3.73× |
| res5 shortcut | 1 | `[2,048 × 1,024] · [1,024 × 2,048]` | BS | 72.3 | 20.2 | 3.59× |
| res5 reduce | 1 | `[2,048 × 1,024] · [1,024 × 512]` | BS | 10.6 | 5.04 | 2.10× |
| res5 expand | 3 | `[2,048 × 512] · [512 × 2,048]` | BS | 24.2 | 10.1 | 2.40× |
| res5 reduce | 2 | `[2,048 × 2,048] · [2,048 × 512]` | BS | 16.1 | 10.1 | 1.60× |
| ASPP 1×1 branch | 2 | `[2,048 × 2,048] · [2,048 × 256]` | BS | 13.8 | 5.04 | 2.73× |
| ASPP image pool (bf16 act, DRAM-bound) | 2 | `[1 × 2,048] · [2,048 × 256]` | BS | 6.95 | 1.23 | 5.66× |
| ASPP projection | 2 | `[2,048 × 1,280] · [1,280 × 256]` | BS | 16.8 | 3.15 | 5.34× |
| Decoder skip projection | 2 | `[8,192 × 512] · [512 × 64]` | HS | 4.65 | 1.26 | 3.69× |
| Decoder skip projection | 2 | `[32,768 × 256] · [256 × 32]` | HS | 3.60 | 1.26 | 2.86× |
| Semantic logits (19 of 32 cols) | 1 | `[32,768 × 256] · [256 × 32]` | HS | 3.60 | 1.26 | 2.85× |
| Bilinear upsample, W pass | 3 | `[32×128 × 256] · [256 × 1,024]` | L1 interleaved | 15.4 | 5.04 | 3.05× |
| Bilinear upsample, H pass (DRAM-bound) | 3 | `[32×512 × 128] · 32×[128 × 1,024]` | DRAM | 123 | 43.5 | 2.83× |
| Center / offset heads (1, 2 of 32 cols) | 2 | `[32,768 × 32] · [32 × 32]` | HS | 3.46 | 0.16 | 22× |

</details>

### Findings

- Qwen decode is DRAM-bound at 1.10–1.36×, except out_proj/o_proj (K = 1,536) at 1.59×; headroom is ~2 ms/token. Prefill is 1.9–2.1× at M ≤ 512; from M = 2,048 it is 1.05–1.36×, except out_proj/o_proj, which stay at ~2× at every M.
- FLUX.2 matmuls are 318 ms of the ~924 ms step at 1.05× overall. The 2,048- and 2,304-row block projections are compute-bound at 0.98–1.16× (attention output with fused addcmul: 1.56×); the 256-row text stream is DRAM-bound at 1.70–2.01×. `time_mod` (M = 1, 642 MB of bf16 weights per chip) is 1.64×. Per image, the text encoder's projections take 69 ms at 1.24–1.84×.
- PDL: 48 of the 55 matmuls take under 17 µs, 1.6–5.7× over roofline, with a ~3.5 µs floor (22× on the 32-column heads). The upsample H pass is 37% of PDL's matmul time.
- Epilogues: SwiGLU is fused in FLUX.2 FF1 (`fuse_swiglu`) and runs as separate ops after Qwen's packed gate_up and the text encoder's gate/up. addcmul (residual) is fused in FLUX.2's double-block attention output and runs as a separate op after the reduce-scatter for FF2 and the single-block output. Bias and ReLU are fused where used.
- Coverage needed: M 1–32,768, K 32–16,384, N 32–52,224; activations bf16/fp32/bfp8, weights bf16/bfp8/bfp4; LoFi/HiFi2; DRAM interleaved, DRAM width-sharded, L1 width/height/block-sharded; split outputs (×3, ×17) and a two-input K (single-block output).

**Caveats:** FLUX.2 weights are random; shapes and timing don't depend on them. Tracy reports a cached `minimal_matmul`'s first-seen input shape for later calls with a different M (relevant to #65141), so M is taken from the model structure. One run per op; repeat runs (PDL, Qwen decode) matched within 1.8%.
