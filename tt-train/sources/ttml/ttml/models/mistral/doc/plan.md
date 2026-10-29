# Mistral-Small-24B-Base-2501 training bring-up: plan

| | |
|---|---|
| HF model | `mistralai/Mistral-Small-24B-Base-2501`, revision `b0a2e4ed093c26997495ae625528f81ea04b749f` (not gated) |
| Reference | `transformers` 5.12.1 `models/mistral/modeling_mistral.py` (`MistralForCausalLM`), torch 2.11.0+cpu, run on CPU in fp64 for PCC |
| tt-metal | branch `agobeljicTT/mistral-bringup` at `5ba149c8fcb` (main of 2026-10-29), tt-train built with `./build_metal.sh --build-tt-train --enable-ccache` (Release, Tracy on) |
| Hardware | host `e0118d75da98`, chip 0 of a QB2 box (2x p300c, 4 Blackhole chips, board `000004613193108e`); one chip: 11x10 = 110 Tensix compute cores, 8 DRAM banks, 31.83 GiB DRAM; tt-smi 4.1.2 |
| Recipe | **LoRA**: rank 16, alpha 32 (scaling alpha/rank = 2), dropout 0, adapters on every block's four linears (`qkv_linear`, `out_linear`, `w_gate_up`, `w2`); embedding, LM head, norm gammas and all base weights frozen. Sequence 1024, batch 1, AdamW (lr 1e-4, betas 0.9/0.999, eps 1e-8, weight decay 0), bf16 parameters, bf16 optimizer state. Next-token cross-entropy: every sample is a window of 1025 tokens, inputs `ids[0:1024]`, targets `ids[1:1025]`, so all 1024 positions are valid and the loss is the mean over them (ttml `cross_entropy_fw` has no `ignore_index`; the torch reference is fed the same shifted targets, not HF's `ForCausalLMLoss` with its internal shift, ). Per-module PCC compares against torch fp64 on bf16-rounded inputs; the loss-trajectory run compares ttml's loss per step against the torch reference's own loss on the same tokens. |

The HF model is Llama-architecture with one difference that matters here: `head_dim` = 128 is explicit and `num_attention_heads * head_dim` = 4096 ≠ `hidden_size` = 5120, so the attention projections are `5120 -> 4096 (q) / 1024 (k, v)` and `4096 -> 5120 (o)`. Everything else (RMSNorm eps 1e-5, SwiGLU with intermediate 32768, RoPE theta 1e8 without scaling, no biases, untied embeddings, vocab 131072, 40 layers, 32 query / 8 KV heads, `sliding_window: null`) maps one-to-one onto the ttml Llama stack, whose config currently rejects `head_dim * heads != hidden`.

**Why LoRA.** Full fine-tuning needs weights + grads + fp32 AdamW state = 16 B per parameter = 377 GB for 23.57 B parameters; the whole box has 4 x 31.8 GiB. Even bf16 weights alone (43.9 GiB) exceed one chip. The recipe is therefore parameter-efficient by necessity, and LoRA follows the team's precedent for the previous 27B bring-up on this box (`umales/qwen38`, rank 16 / alpha 32).

## 1. Kernel inventory

Reference forward, per `MistralForCausalLM` at B=1, T=1024, and the backward autograd emits for it. Layout contract: activations cross module boundaries as `[B, 1, T, W]` bf16 TILE with heads merged on the last dim; attention internally uses `[B, H, T, D]` (head axis on dim 1, not -2, so no 32-row padding of heads). Weights are `[1, 1, out, in]` bf16 TILE. All paths are relative to the repo root.

| # | Reference op (shapes at B=1, T=1024) | Forward kernel (exists) | Backward (exists) | Layout / dtype accepted | Missing / notes |
|---|---|---|---|---|---|
| 1 | `embed_tokens`: ids `[1,1024]` -> `[1,1024,5120]` | `ttnn.embedding` via `tt-train/sources/ttml/ops/embedding_op.cpp` (`ttml.ops.embedding.embedding`) | `ttnn.embedding_bw` in the same file | ids uint32 ROW_MAJOR `[B,1,1,T]`; table bf16 TILE, the op `untilize`s the whole table every call | Frozen under LoRA: backward not needed; the per-call untilize of the 1.34 GB table is wasted work and a 1.34 GB transient. The `umales/qwen38` branch has `FrozenEmbedding` (`models/qwen38/frozen_embedding.py`: untilize once, in place) to port. |
| 2 | `input_layernorm` / `post_attention_layernorm` / final `norm`: RMSNorm eps 1e-5 on `[1,1024,5120]` | `ttml.metal.rmsnorm_fw` (`tt-train/sources/ttml/metal/ops/rmsnorm_fw`) via `ops/rmsnorm_op.cpp`, `ttml.modules`-level `RMSNormLayer` in `models/llama/transformer.py` | `ttml.metal.rmsnorm_bw` (`metal/ops/rmsnorm_bw`), `compute_dgamma` already keyed on `gamma.requires_grad` | bf16 TILE `[B,1,T,W]`, gamma `[1,1,1,W]` | Nothing. Frozen gamma: dgamma already skipped. |
| 3 | `q_proj`, `k_proj`, `v_proj`: `[1024,5120] x [5120,4096/1024/1024]` | one fused `LinearLayer(5120, 6144)` -> `ttnn.linear` (`ttnn/cpp/ttnn/operations/matmul`) via `ops/linear_op.cpp`; rows `[Q | K | V]` with the loader's RoPE un-permute | `ttnn_linear_backward` in `ops/linear_op.cpp`: dX = dY·W, dW = dYᵀ·X as two `ttnn_fixed::matmul` | bf16 TILE, HiFi4, fp32 accumulate | **dW is computed even when the weight is frozen** (gradient dropped in `Tensor::add_grad`). `umales/qwen38` gates both matmuls on `requires_grad` (`ops/linear_op.cpp`, `ops/binary_ops.cpp`); port as the first composite commit. `GroupedQueryAttention` sizes `qkv_dim` from `hidden/num_heads`: needs an explicit `head_dim`. |
| 4 | reshape to heads `[1,32,1024,128]`, `[1,8,1024,128]` x2 | `ttnn.experimental.nlp_create_qkv_heads` via `ops/multi_head_utils.cpp::heads_creation` (derives `head_dim` = width / (H + 2G) = 6144/48 = 128) | `nlp_concat_heads` x3 + `concat` (same file) | bf16 TILE; `head_dim` must be a multiple of 32 | Layout class (ideal 0). Nothing missing for 128. |
| 5 | RoPE on q, k (theta 1e8, default type, no scaling) | `ttnn.experimental.rotary_embedding_llama` via `ops/rope_op.cpp::rope`; cos/sin/trans_mat from `build_rope_params(T, 128, 1e8)` | same kernel with the negated caches (same file) | bf16 TILE `[B,H,T,D]`, D even, caches `[1,1,T,D]` | Nothing. Reference RoPE is the HF rotate-half form; the loader's `unpermute_proj_rows` already handles the HF -> interleaved weight permutation for Llama; same for Mistral. |
| 6 | causal GQA SDPA, scale 128^-0.5, `sliding_window` null | `ttml.metal.sdpa_fw` (`metal/ops/sdpa_fw`) via `ops/scaled_dot_product_attention.cpp::scaled_dot_product_attention` with on-device causal mask (`mask=None`), returns logsumexp intermediates | `ttml.metal.sdpa_bw` (`metal/ops/sdpa_bw`, Q and KV passes), native GQA (dK/dV come out at G heads) | bf16 TILE `[B,H,T,D]`, S and D tile-aligned, H % G == 0 | Nothing. Composite fallback `scaled_dot_product_attention_composite` (repeat + matmuls + `moreh_softmax_backward`) exists for debugging. |
| 7 | concat heads `[1,32,1024,128]` -> `[1,1,1024,4096]` | `nlp_concat_heads` via `heads_fusion` | transpose + reshape + transpose (same file) | bf16 TILE | Layout class (ideal 0). |
| 8 | `o_proj`: `[1024,4096] x [4096,5120]` | `LinearLayer(4096, 5120)` as row 3 | as row 3 | | `GroupedQueryAttention.out_linear` is built as `(hidden, hidden)`: needs `num_heads * head_dim` input width. |
| 9 | residual adds (2 per layer) | `ttnn.add` via `ops/binary_ops.cpp::operator+` | pass-through grads (same file) | bf16 TILE | Glue class. Backward also pushes a grad into a frozen operand's `add_grad` (no-op) — `umales/qwen38` gates it. |
| 10 | `gate_proj`, `up_proj`: `[1024,5120] x [5120,32768]` x2 | one fused `LinearLayer(5120, 65536)`, rows `[gate | up]` | as row 3 | | |
| 11 | `silu(gate) * up` -> `[1024,32768]` | `ttml.metal.swiglu_packed_fw` (`metal/ops/swiglu_packed_fw`) via `ops/swiglu_packed_op.cpp` | `ttml.metal.swiglu_packed_bw` (`metal/ops/swiglu_packed_bw`) | bf16 TILE, packed `[.., 2I]` with I tile-aligned | Nothing. |
| 12 | `down_proj`: `[1024,32768] x [32768,5120]` | `LinearLayer(32768, 5120)` | as row 3 | | |
| 13 | `lm_head`: `[1024,5120] x [5120,131072]` | `LinearLayer(5120, 131072)` (vocab is already a multiple of 32: no padding slice) | as row 3 | | Frozen: dW (1.37 TFLOP per step) is wasted unless gated. |
| 14 | cross-entropy, mean over tokens, targets uint32 `[1,1024]` | `ttml.metal.cross_entropy_fw` (`metal/ops/cross_entropy_fw`) via `ops/losses.cpp::cross_entropy_loss` | `ttml.metal.cross_entropy_bw` (`metal/ops/cross_entropy_bw`) | logits bf16 TILE `[B,1,T,V]`, targets uint32 ROW_MAJOR `[B,T]` | Nothing. |
| 15 | LoRA adapters: `x·Aᵀ` `[1024,in]x[in,16]`, `h·Bᵀ` `[1024,16]x[16,out]`, `* scaling`, `+ base` | `ttml.modules.lora.LoraLinear` (`tt-train/sources/ttml/ttml/modules/lora.py`): two `ttml.ops.linear.linear` + `mul` + `add` (5 ops) | autograd of the above: dB, dA, dh, dx matmuls + scale | bf16 TILE; rank 16 is padded to a 32-wide tile | Exists and is correct, but slow: these shape classes run at 2-5 % of peak on this chip (`mm_eff.py`: `[1024,5120]x[5120,32]` 4.3 TFLOP/s, `[1024,32]x[32,5120]` 3.6, `[5120,32]x[32,1024]`-class 6.9, `[32,1024]x[1024,5120]`-class 7.7), 10.5 ms per layer fwd+bwd against a 1.0 ms memory roofline. `umales/qwen38` has `LoraApply` (`modules/lora.py`): A-matmul, B-matmul, `addalpha` = 3 ops, scale folded (removes the chain glue, not the matmul inefficiency); port in the loop. **Missing: a memory-bound rank-32 LoRA path** (a skinny-matmul program config that streams `x`/`g` once, or the A/B matmuls fused into the base matmul's epilogue) — new work, costed in §3 as the difference between the two targets. |
| 16 | AdamW on the LoRA parameters | `ttml.metal.adamw` via `optimizers/adamw.cpp` (`ttml.optimizers.AdamW`) | n/a | moments allocated like the parameter (bf16 for bf16 params); `AdamWFullPrecision` exists for fp32 master weights | Nothing. |
| 17 | dropout (attention 0.0, LoRA 0.0) | not emitted | | | |

Inference implementations under `models/`: `models/tt_transformers` (Llama/Mistral text decoders) and `models/experimental/mistral_24b` (the multimodal Mistral-Small-3.1 vision tower) exist, but they are inference-only ttnn graphs (sharded program configs, KV-cache decode paths) with no autograd; the forward kernels they use (`ttnn.linear`, SDPA, RoPE, RMSNorm) are the same ttnn/ttml kernels listed above, so nothing is taken from them.

**New code needed in the composite phase** (no new kernels): (a) decouple `head_dim` from `hidden_size / num_heads` in `LlamaConfig`, `GroupedQueryAttention` (qkv width, out-proj input width), `build_rope_params`, `flops.py` and the safetensors loader rules; (b) `models/mistral/`: `MistralConfig.from_hf` accepting `model_type: mistral` (rejecting `sliding_window != null`), the model (the Llama stack with the explicit head_dim), the loader (same HF key names as Llama: `model.layers.N.self_attn.{q,k,v,o}_proj`, `mlp.{gate,up,down}_proj`, `input_layernorm`, `post_attention_layernorm`, `model.norm`, `model.embed_tokens`, `lm_head`), FLOPs; (c) port the frozen-operand gating of `linear_op.cpp` / `binary_ops.cpp` and `FrozenEmbedding` from `umales/qwen38` so the freeze map below is honoured from day one; (d) the training example `sources/examples/mistral_train/`; (e) the PCC harness `tests/python/test_mistral.py`.

## 2. Freeze map

| Tensor | Shape | Count | Needs gradient under the recipe | Consequence |
|---|---|---|---|---|
| `tok_emb.weight` | `[131072, 5120]` | 1 | **no** | no `embedding_bw`; no per-step untilize (untilize once at load) |
| `blocks.i.attention_norm.gamma`, `mlp_norm.gamma`, `ln_fc.gamma` | `[1,1,1,5120]` | 81 | **no** | `rmsnorm_bw` with `compute_dgamma=false` (already) |
| `blocks.i.attention.qkv_linear.weight` | `[6144, 5120]` | 40 | **no** | linear backward computes dX only |
| `blocks.i.attention.out_linear.weight` | `[5120, 4096]` | 40 | **no** | dX only |
| `blocks.i.mlp.w_gate_up.weight` | `[65536, 5120]` | 40 | **no** | dX only |
| `blocks.i.mlp.w2.weight` | `[5120, 32768]` | 40 | **no** | dX only |
| `fc.weight` (LM head) | `[131072, 5120]` | 1 | **no** | dX only (saves a `[5120,1024]x[1024,131072]` matmul, 1.37 TFLOP/step) |
| biases | none in the checkpoint | 0 | n/a | |
| LoRA `lora_A` / `lora_B` on each of the 4 linears per block | A `[16, in]`, B `[out, 16]` | 160 adapters, 82.6 M parameters | **yes** | dA, dB every step; AdamW state for these only |
| **Module inputs** | | | | |
| token ids | `[1,1,1,1024]` uint32 | | no | |
| embedding output = block 0 input | `[1,1,1024,5120]` | | **no** (its only consumer is the frozen table) | with the frozen-table lookup the block-0 input carries no autograd node, so `add_backward_node` (`autograd/graph_utils.hpp`) prunes block 0's `attention_norm` dX and the qkv-base dX automatically; the roofline below still counts qkv dX for all 40 layers (0.43 ms overcount, kept for simplicity) |
| every other block input / residual stream, norm outputs, qkv, heads, rope outputs, SDPA output and lse, concat-heads output, gate_up, SwiGLU output, down output, final norm output, logits | as in the inventory | | **yes** (on the path from the loss to some LoRA parameter) | |
| causal mask | generated on device by `sdpa_fw` | | no | |
| RoPE cos/sin/trans caches | `[1,1,1024,128]` | | no (constants) | |

Every trainable gradient is a LoRA matrix gradient; every weight-gradient matmul of a base weight is the wasted class and must not be emitted.

## 3. Roofline target

Chip: 110 cores x 1.35 TFLOP/s = **148.5 TFLOP/s** bf16 peak (the number `ttml.common.performance.get_device_peak_tflops_bf16` reports and the previous bring-up on this box used as the MFU denominator), DRAM **512 GB/s** nominal (8 x GDDR6; to be checked against a measured untilize/copy in the loop).

**Useful FLOPs per step** (B=1, T=1024, counted the way the previous LoRA bring-up on this box counts them, i.e. `torch.utils.flop_counter` on the reference with `requires_grad` mirroring the freeze map: a frozen linear costs forward + dX, a trainable one forward + dX + dW; attention counted as the standard full-square `4·H·D·T²` forward and twice that backward; elementwise work not counted):

| Term | FLOPs |
|---|---|
| per layer, base linears forward `2·T·555.7M` | 1.138 T |
| per layer, base linears dX | 1.138 T |
| per layer, LoRA adapters (fwd + bwd = 3 x `2·T·16·Σ(in+out)`) | 12.7 G |
| per layer, attention fwd `4·32·128·1024²` + bwd 2x | 51.5 G |
| **per layer** | **2.3406 T** |
| 40 layers | 93.62 T |
| LM head fwd + dX `2 x 2·T·671M` | 2.749 T |
| **per step** | **96.37 TFLOP** (94.11 GFLOP per token) |

`MFU = 96.37e12 / step_seconds / 148.5e12`. This is a deliberate deviation from the phase skill's literal `6 x trainable params x tokens`, which for a LoRA would give 0.5 TFLOP and ignore the frozen-weight forward and dX matmuls that are the bulk of the required work; the full-fine-tuning convention (`6·N·T + 12·L·H·D·T²` = 146.9 TFLOP) is not used either, since a third of it is dW matmuls the recipe never runs. Later MFU runs must use 96.37 TFLOP per step (94.11 GFLOP per token) and not "correct" it to either.

**Per-op roofline**, `max(FLOPs / 148.5 TFLOP/s, bytes / 512 GB/s)`, one layer (script: `doc/plan/numbers.py`):

| Op (per layer) | class | compute ms | memory ms | roofline ms |
|---|---|---|---|---|
| qkv fwd / dX | math | 0.434 / 0.434 | 0.168 / 0.168 | 0.868 |
| out fwd / dX | math | 0.289 / 0.289 | 0.119 / 0.119 | 0.578 |
| gate_up fwd / dX | math | 4.628 / 4.628 | 1.593 / 1.593 | 9.256 |
| down fwd / dX | math | 2.314 / 2.314 | 0.807 / 0.807 | 4.628 |
| LoRA adapters, 4 linears, fused ideal (A and B matmuls, x and y streamed once more each, fwd + bwd) | math | 0.085 | 1.032 | 1.032 (existing path: 10.5, see below) |
| SDPA fwd / bwd (causal, GQA) | math | 0.116 / 0.231 | 0.066 / 0.098 | 0.347 |
| SwiGLU fwd / bwd | math | 0 | 0.393 / 0.655 | 1.048 |
| RMSNorm x2 fwd / bwd (no dgamma) | math | 0 | 0.082 / 0.123 | 0.205 |
| RoPE q,k fwd / bwd | math | 0 | 0.041 / 0.041 | 0.082 |
| residual adds fwd + grad accumulation bwd | glue | 0 | 0.246 | 0.246 (ideal 0 when fused into the matmul epilogue) |
| heads creation / fusion fwd + bwd | layout | 0 | 0.082 | 0.082 (ideal 0) |
| **per layer** | | | | **18.37** (math 18.04) |

Fixed costs per step: LM head fwd 9.26 + dX 9.26 (compute-bound, 1.37 TFLOP each), cross-entropy fwd 0.52 + bwd 1.05 (256 MiB logits streamed), embedding lookup 0.04, final norm 0.10, AdamW on 82.6 M parameters 2.26 (7 x 165 MB streamed) = **22.5 ms**.

**Roofline step time = 40 x 18.37 + 22.5 = 757 ms, i.e. 85.7 % MFU.** This is the physical floor at the stated peak and bandwidth.

**Achievable target.** Every efficiency below is measured on this chip or marked provisional:
- Big matmuls (`ttnn.linear`, HiFi4, fp32 accumulate): 60-68 % of the 148.5 TFLOP/s peak for the `[1024,5120]x[5120,4352]`-class shapes in the Qwen3.8 Tracy profile of 2026-10-21 (median 62 % over 48 calls per shape; `/home/ttuser/agobeljic/scratch/tracy_4layers_baseline/reports/2026_10_21_10_41_36/ops_perf_results_2026_10_21_10_41_36.csv`, re-derived with `doc/plan/mm_eff.py`). The Mistral `gate_up` (N = 65536) and `down` (K = 32768) classes sit next to the measured N = 62080 (61.6 %) and K = 62080 (67.7 %) rows. Applied at 62 %.
- LoRA adapters: priced as the existing `LoraLinear` path runs them, at the skinny-matmul rates measured in the same profile for the same M = 1024 and rank-32 shapes at K/N = 5120 and extrapolated to K = 32768 / N = 65536 (inventory row 15): **10.5 ms per layer** fwd + bwd including the scale/add chain, against the 1.0 ms fused-ideal roofline. This term is 28 % of the target and is the first thing the first Tracy run calibrates.
- SDPA kernels: 50 % of peak, provisional (no measurement at 32 heads x 128 on this chip yet; the first Tracy run calibrates it).
- Memory-bound ops (SwiGLU, norms, RoPE, CE, optimizer): 70 % of the 512 GB/s nominal, provisional until an untilize/copy bandwidth is measured in the first loop iteration. The glue and layout rows (0.33 ms/layer) are left at nominal bandwidth since their ideal is zero; at 70 % they would add 5.6 ms per step, inside the margin.
- Dispatch: the Qwen3.8 diary measured a host floor of ~25 µs per op on one chip (`host_dispatch_bench.py`) and ~16 µs per op *exposed* at full scale (the 4-layer baseline shows 10.7 µs). A layer here is dominated by 0.4-7 ms matmuls that hide the floor, so the exposed figure, 16 µs x ~90 ops per layer, is used: 58 ms per step.

| | per layer | x40 | fixed | dispatch | step | MFU |
|---|---|---|---|---|---|---|
| achievable with existing kernels | 38.2 ms | 1527 ms | 35.5 ms | 58 ms | **1620 ms** | **40.0 %** |
| stretch, if a memory-bound LoRA path lands (row 15) | 29.2 ms | 1166 ms | 35.5 ms | 58 ms | 1259 ms | 51.6 % |

**Target: 1.62 s/step, 40.0 % MFU, on one chip at seq 1024, batch 1, LoRA rank 16; margin ±10 % (1.46-1.78 s).** The stretch figure is conditional on the new LoRA kernel work and is not the loop's exit condition unless that work lands, in which case the target moves to 1.26 s ±10 %. The loop exits when the projected full-depth step time (§4 depths) lands inside the margin, or when every remaining item is a matmul or SDPA kernel at the stated fraction of its own roofline with a documented blocker. The loop's first targets are already visible in the inventory: frozen-weight dW matmuls (wasted, 40 x 1.138 T + 1.37 T = 46.9 TFLOP per step, 49 % of the useful work), the per-step embedding untilize (wasted), the 5-op LoRA chain per linear (glue), the LoRA skinny matmuls (math at 2-5 % of peak), heads creation/fusion (layout).

Profiling block: the model is homogeneous, so a block is one decoder layer. The four doubling depths are **1, 2, 4 and 8 layers** (all at real width, heads, sequence length and recipe), fitted to `fixed + B x per-layer` and projected to 40; 16 layers does not fit on one chip on the composite path (§4), so the deepest point is 8. If the loop makes 16 fit (LoRA chain removed and intermediates released), the depths move to 2/4/8/16 and the fit is redone.

## 4. Memory budget

Per chip 31.83 GiB. bf16 everywhere unless stated.

| Item | Size |
|---|---|
| base weights per layer (555.7 M) | 1.035 GiB |
| embedding + LM head (2 x 671 M) | 2.50 GiB |
| **full model weights (40 layers)** | **43.9 GiB — does not fit one chip** |
| LoRA parameters + grads + 2 AdamW moments (82.6 M x 4 x 2 B) | 0.63 GiB |
| activations per layer that backward *needs* (block input 10, norm out 10, q/k/v after RoPE 12, SDPA out 8 + lse 4, concat 8, residual 10, norm2 out 10, gate_up 128, SwiGLU out 64 MiB) | 264 MiB |
| activations per layer ttml autograd *retains* (every op output stays alive until its backward node runs: each grad closure captures `out`, `autograd/tensor.cpp` releases closures only as backward proceeds) with the 5-op `LoraLinear` chain: the above plus the qkv/out/gate_up/down base outputs, both residual sums, pre-RoPE heads, and per adapter `h`, update, scaled update and sum | **786 MiB** |
| same with `LoraApply` (3 ops: only `h` and the sum per adapter) | 466 MiB |
| logits bf16 + grad + fp32 cross-entropy intermediates (256 + 256 + 512 MiB) | ~1.0 GiB |
| workspace (matmul intermediates, in-flight backward tensors such as the 128 MiB gate_up output gradient and the 640 MiB gate_up dW transient until the frozen-weight gating lands, allocator slack) | ~1.5 GiB |

Depth that fits on one chip (weights + retained activations without recompute + logits/CE + LoRA state + 1.5 GiB workspace, against 95 % of 31.83 GiB = 30.2 GiB):

| layers | `LoraLinear` chain (composite path) | `LoraApply` | minimal set |
|---|---|---|---|
| 1 | 7.7 GiB | 7.4 | 7.2 |
| 2 | 9.5 | 8.9 | 8.5 |
| 4 | 13.1 | 11.8 | 11.0 |
| 8 | 20.3 | 17.8 | 16.2 |
| 12 | 27.5 | 23.8 | 21.4 |
| 16 | **34.7, does not fit** | 29.7 (no headroom) | 26.6 |
| 40 | 78.1 | 65.6 | 57.6, and the weights alone do not |

The retention figures come from reading each op's grad closure; the composite phase calibrates them with the allocator's peak after forward at 2 and 4 layers and corrects this table before the depths are final.

**Conclusions.**
- **Recompute policy: none.** At the depths used on one chip the retained activations (≤ 6.2 GiB at 8 layers) are small next to the weights, and recompute would add a full forward (1.14 TFLOP per layer, +50 % of the layer's math) to the loop's cost multiplicity. `RunnerType.MemoryEfficient` stays available but is not planned; releasing intermediates the backward does not need (the difference between 786 and 264 MiB per layer) is loop work, not recompute.
- **Optimizer state: bf16**, matching the parameters (`AdamW` allocates moments like the parameter). fp32 state would cost 0.33 GiB more and is affordable; it is not needed for a LoRA with lr 1e-4 and is left as a loss-match lever only if the composite loss trajectory diverges from torch.
- **The full model does not fit on one chip** at any sequence length: 43.9 GiB of frozen bf16 weights against 31.83 GiB. The composite phase's gate "full model runs training steps on one chip" is therefore unmeetable for this model, and this plan does not route around it with multi-device work. The single-chip substitutes are: (a) PCC harness per module and per layer kind at real shapes (hidden 5120, 32/8 heads x 128, intermediate 32768, vocab 131072, T = 1024) against a torch fp64 reference of the same module; (b) the loss trajectory over N steps at the deepest fitting depth, 8 layers, against a torch reference truncated to the same 8 layers and loaded with the same checkpoint tensors; (c) Tracy at 1/2/4/8 layers with the fit projected to 40; (d) the MFU wall-clock run at 8 layers, without the profiler, which validates the 8-layer point of the fit, and the 40-layer step time reported as the projection. Every number is labelled "N of 40 layers". A full-depth run would need the frozen weights spread over the box's four chips (the Llama stack has the tensor parallelism; ~11 GiB weights + ~4 GiB activations per chip at TP = 4); that is multi-device work outside this single-chip bring-up and is **not planned**; it is recorded here as a decision for the user, who can authorise it separately.
- The torch fp64 reference is built only at the truncated depths (8 layers = 35.6 GB of fp64 block weights + 10.7 GB embeddings/head, 46 GB of the 249 GB host RAM); a full-depth fp64 reference (188 GB) is not promised.
- Weight loading: the 47 GB safetensors checkpoint (10 HF shards; the repo also ships a `consolidated.safetensors` in Mistral's own key format, which the loader must be pointed away from) is read lazily per tensor; host RAM (249 GB) is not a constraint. Weights and the HF cache stay outside the repo.

### Calibration from the composite phase (2026-10-29)

Allocated DRAM read from the allocator (`train.py --mem-probe`, random init, seq 1024, batch 1, LoRA r16, `LoraLinear` path): weights + LoRA state 5.14 / 7.27 / 11.54 GiB at 2 / 4 / 8 layers (1.067 GiB per layer + 3.0 GiB fixed), after the forward 6.43 / 10.08 / 17.39 GiB, so the retained activations are **0.76 GiB = 778 MiB per layer** against the 786 MiB this plan derived. 16 layers would need 3.0 + 16 x (1.067 + 0.76) = 32.2 GiB: the 1/2/4/8 depths stand. Evidence: `doc/composite/evidence/mem_probe_*layers.log`.
