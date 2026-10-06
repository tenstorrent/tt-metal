# Why Gemma-4-26B-A4B can't reach 0.99 PCC in bf16 (and Qwen3-MoE / OLMoE can)

All experiments: Hugging Face Transformers 5.12.1 on CPU, 1023 tokens of tale-of-two-cities with BOS,
PCC over positions 511-1010 against an unrounded run of the same model. "fp32 compute" = weights stored
bf16 (checkpoint values) and every operation in fp32; validated against the true fp32 model at PCC 0.99989.
Scripts: this folder (see README.md). Data and logs: ~/benchmark-data/gemma4-pcc/ and its logs/ subfolder.

## 1. Which single bf16 rounding hurts Gemma 4 (gemma_precision_ablation.py)

| bf16 rounding added at ONE place, all else fp32 | PCC | positions <0.99 | experts flipped (mean over layers) |
|---|---|---|---|
| queries + keys | 0.975 | 175 | 15.9% |
| attention scores | 0.970 | 184 | 19.3% |
| attention probabilities | 0.983 | 137 | 12.0% |
| values + attention output | 0.981 | 158 | 13.0% |
| all matmul outputs | 0.980 | 170 | 15.6% |
| layer-norm outputs | 0.971 | 169 | 15.2% |
| residual stream (layer outputs) | 0.982 | 151 | 11.5% |
| router steps | 0.961 | 228 | 25.0% |
| expert steps | 0.981 | 145 | 14.9% |
| embedding output | 0.985 | 124 | 11.5% |
| ALL of the above (= a bf16 model) | 0.957 | 226 | 25.5% |

Any one rounding costs 0.015-0.04 PCC; every one of them works by flipping experts.

## 2. Same perturbation on three MoE models (cross_model_perturb2/6.py)

| | Gemma-4-26B-A4B | OLMoE-1B-7B | Qwen3-30B-A3B |
|---|---|---|---|
| 0.2% relative noise on the embedding output: PCC | 0.985 | 0.9997 | 0.9993 |
| same noise: experts flipped first -> last layer | 0.4% -> 28% | 1.0% -> 2.6% | 0.6% -> 6.3% |
| same noise, routing frozen to the reference: PCC | 0.99991 | 1.0 | 1.0 |
| one forced 8th<->9th swap in layer 0: PCC | 0.941 | 0.997 | 0.998 |
| HF-style bf16 router rounding (fp32 softmax): PCC | 0.960 | 0.9995 | 0.998 |
| Gemma-style bf16 router rounding (bf16 softmax): PCC | 0.965 | 0.9992 | 0.998 |

## 3. Per-layer growth of the difference (cross_model_perturb6.py, flip_cascade_model.json)

Per layer, the relative difference in the residual stream is multiplied by
(what the non-routing parts keep, measured with routing frozen) + (what expert flips add).

| | smooth parts keep, per layer | flips add, per layer | predicted | observed | layers |
|---|---|---|---|---|---|
| Gemma-4-26B-A4B | x1.106 | +0.153 | x1.259 | x1.299 | 30 |
| OLMoE-1B-7B | x0.934 | +0.177 | x1.111 | x1.145 | 16 |
| Qwen3-30B-A3B | x0.969 | +0.108 | x1.077 | x1.066 | 48 |

Components of "flips add": flips caused per 1% drift (Gemma 4.05, OLMoE 2.25, Qwen3 3.55) x
drift caused by flipping every position in a middle layer (Gemma 3.78%, OLMoE 7.87%, Qwen3 3.05%).

## 4. Things measured and ruled out as the distinguishing cause
- HF computing Gemma's router softmax in bf16 (Qwen3/OLMoE use fp32): Qwen-style rounding hurts Gemma as much (0.960 vs 0.965).
- Router near-ties: Qwen3 has about as many as Gemma (gap < 1 bf16 step: 16.1% vs 18.4% of decisions) and its first-layer flip rate under router rounding is higher (5.0% vs 3.6%).
- Attention without 1/sqrt(d) scaling: attention scores are similar in size (row max ~9.5 Gemma, 7.5 OLMoE, 10.3 Qwen3).
- Large per-layer updates / layer_scalar < 1: OLMoE's updates are larger relative to its residual (65% vs 56%) and it is stable.

## 5. Open
- Which part of Gemma's non-routing path makes it keep x1.106 per layer instead of < 1 (not yet isolated).
- Google's own JAX Gemma 4 disagrees with HF fp32 at PCC 0.63: these results describe HF's implementation (which the Tenstorrent port copies).

## 6. Per-layer / per-op PCC on the chips (2026-10-05, optimized tree 36b3922cc29, QB2 1x4)
tt_per_layer_pcc.py vs HF fp32 reference (hf_per_layer_ref.py), 512 book tokens; baseline hf_bf16_per_layer.py.
Results: tt_per_layer_pcc_optimized.json, hf_bf16_per_layer_pcc.json. Logs tt-per-layer-optimized-2.log, hf-bf16-per-layer.log.
- Isolated (each layer fed the reference input), mean over 30 layers, chip vs HF bf16:
  attention 0.99982 vs 0.99999, shared MLP 0.99982 vs 0.99999, experts 0.99682 vs 0.99935, layer out 0.99957 vs 0.99994,
  same 8 experts 83.3% vs 94.9% of tokens (0.17 vs 0.05 of 8 experts differ per token per layer).
- Accumulated prefill layer out: chip 0.9996 (L0) -> 0.904 (L26-28) -> 0.957 (L29); HF bf16 0.9999 -> 0.951 (L28) -> 0.977 (L29).
- Decode (prefill 480, decode 480-511): same shape, L28 0.924, L29 0.972. No single layer or op is a cliff.

## 7. Same per-layer measurement: unmodified code, and decode-only bf16 vs fp32 intermediates (2026-10-05)
Prefill: tt_per_layer_pcc.py on b809f097c06 (unmodified) -> tt_per_layer_pcc_unmodified.json: same as optimized
(isolated attn 0.99982, mlp 0.99984, experts 0.9968, out 0.99958, same-8 83.3%; accumulated L28 0.899, L29 0.955).
Decode-only (tt_per_layer_pcc_decode.py, Gemma4Generator paged KV, positions 0-511 one at a time) on 87505f06886,
switches off (bf16) vs GEMMA4_FP32_ACTIVATIONS=1 GEMMA4_ROUTER_TOPK_ON_SCORES=1 GEMMA4_FP32_ATTENTION=1 (fp32):
| isolated mean | attn | mlp | experts | out | same 8 experts |
| decode bf16 | 0.99972 | 0.99989 | 0.99735 | 0.99972 | 84.4% |
| decode fp32 | 0.99998 | 1.00000 | 0.99980 | 0.99998 | 98.6% |
| HF bf16 (CPU) | 0.99999 | 0.99999 | 0.99935 | 0.99994 | 94.9% |
Accumulated layer out L10/L20/L28/L29: decode bf16 0.991/0.946/0.815/0.887; decode fp32 0.998/0.988/0.961/0.981;
HF bf16 0.997/0.986/0.951/0.977. Decode bf16 accumulates worse than prefill bf16 (0.899/0.955) at similar isolated error: open.

## 8. Why only Gemma 4: where the extra growth comes from (2026-10-05)
Same 2^-9 nudge on the embedding output, HF fp32 arithmetic on CPU, 1023 book tokens, PCC over positions 511-1010
(cross_model/freeze_components.py). "Residual difference" = relative difference of the last layer's output.

| Same nudge, frozen to the clean run | Gemma 4: PCC / expert sets changed / residual difference | Qwen3-30B-A3B | OLMoE-1B-7B |
|---|---|---|---|
| nothing | 0.9846 / 11.9% / 7.6% | 0.9993 / 3.5% / 2.7% | 0.9997 / 1.6% / 1.5% |
| routing | 0.99991 / 0 / 0.27% | 1.0 / 0 / 0.03% | 1.0 / 0 / 0.09% |
| attention probabilities | 0.9981 / 4.8% / 2.4% | 0.9983 / 3.0% / 2.6% | 0.9999 / 1.2% / 0.7% |
| both | 1.0 / 0 / 0.03% | 1.0 / 0 / 0.03% | 1.0 / 0 / 0.07% |

- In every model, the PCC loss needs expert changes (frozen routing gives >= 0.9999).
- The nudge starts at 0.2%. With routing frozen, Gemma's layers grow it to 0.27%; Qwen3's shrink it to 0.03%, OLMoE's to
  0.09%. Freezing Gemma's attention probabilities as well removes that growth (0.03%, same as Qwen3). In Qwen3, freezing
  attention changes nothing (0.03% -> 0.03%). So Gemma's extra growth outside the routers is in how its attention
  probabilities respond, and that larger difference then reaches the routers and changes 3-7x more expert choices.
- Attention frozen in only some Gemma layers (FREEZE_SUBSETS=1): 5 global layers 0.9943, 25 sliding layers 0.9958,
  first / middle / last third 0.9925 / 0.9956 / 0.9909. Spread over all layers; per layer the global layers
  (K = V, 512-wide heads) do ~4x the damage of a sliding layer.

Inside one layer (cross_model/probe_layer_gain.py: one layer at a time, clean input vs input + 2^-9 nudge in the real
direction the difference takes; routing frozen; gain = relative change at that point / relative change of the input;
mean over layers):

| | input norm | queries | keys | attention probabilities | layer output | same, routing free | expert sets changed |
|---|---|---|---|---|---|---|---|
| Gemma 4 | 1.37 | 1.04 | 1.12 | 1.75 | 1.08 | 1.31 | 1.26% |
| Qwen3 | 1.01 | 0.65 | 0.64 | 0.61 | 0.97 | 1.10 | 0.92% |
| OLMoE | 0.93 | 0.53 | 0.52 | 0.44 | 0.94 | 1.14 | 0.59% |

Two Gemma-specific steps:
1. The input norm enlarges the difference (1.37x vs ~1x). Gemma's residual stream carries a few huge channels (top 4
   channels = 34-64% of its size, values ~25-76 vs a median channel of ~0.006-0.5), and the norm weights those channels
   almost to zero (e.g. layer 3: -0.0018) while giving large weights (median 4-42) to channels where the residual is tiny
   (cross_model/massive_channels.py). What attention reads is therefore the small remainder, and a difference measured
   against the whole residual is larger against that remainder. For a difference spread evenly over channels the norm
   multiplies it 40x in Gemma, 1.2x in Qwen3, 0.95x in OLMoE. Qwen3 also has huge channels (38% of the size) but its
   norm weights don't single out the tiny channels the same way.
2. Per 1% change of queries and keys, Gemma's attention probabilities change ~1.6% (Qwen3 0.95%, OLMoE 0.83%).
   Not the softmax temperature: the spread of the scores under the attention distribution is the same in all three
   (1.65 / 1.74 / 1.86, cross_model/attention_spread.py), although Gemma uses scaling 1.0 where the others use 1/sqrt(d).
   Largest in the global layers (probabilities 2.1x, keys 1.5x).

Both are properties of the trained weights (norm weights, the residual stream's layout, the attention heads), not of
any implementation: Hugging Face's own bf16 run loses the same way (0.958 vs HF fp32).

## 9. Attention settings sweep (2026-10-06, chip/attn_sweep.py, merged code 065d1b0167f)
Each setting patched at runtime, model built once. End to end: gate text (128 prefill + 128 decode) and book (512 + 499),
PCC vs HF fp32 (reference/hf_gate_fp32_ref.py, book_logits_hf_fp32.pt). Isolated (SWEEP_ISO=128): decode positions 0-127
with every layer fed HF's exact fp32 input; error = 1 - PCC averaged over the 30 layers, x1e-4. The chip is deterministic
(baseline run twice: identical to the last digit).

| Setting | Attention error | Layer-output error | Same 8 experts | Gate PCC vs fp32 | Book PCC vs fp32 |
|---|---|---|---|---|---|
| baseline | 2.557 | 1.611 | 92.66% | 0.97347 | 0.93876 |
| sdpa_hifi4_exact | 2.548 | 1.528 | 93.2% | 0.97058 | 0.94008 |
| sdpa_hifi2_exact | 2.562 | 1.643 | 92.73% | 0.97405 | 0.93639 |
| sdpa_hifi4_approx | 2.549 | 1.656 | 92.97% | 0.97384 | 0.93387 |
| sdpa_kchunk32 | 2.534 | 1.63 | 92.79% | 0.97465 | 0.94047 |
| sdpa_kchunk128 | 2.588 | 1.573 | 93.12% | 0.96411 | 0.93957 |
| sdpa_cores_per_head1 | 2.58 | 1.598 | 92.89% | 0.9698 | 0.94322 |
| sdpa_cores_per_head4 | 2.557 | 1.611 | 92.66% | 0.97347 | 0.93785 |
| qkv_hifi2 | 1.892 | 1.379 | 93.49% | 0.97284 | 0.93669 |
| qkv_hifi4 | 1.889 | 1.474 | 93.49% | 0.97636 | 0.94413 |
| qkv_default_config | 2.294 | 1.606 | 93.36% | 0.96784 | 0.93821 |
| qkv_hifi4_kblock1 | 1.889 | 1.477 | 93.39% | 0.96641 | 0.94184 |
| qkv_hifi4_kblock11 | 1.889 | 1.469 | 93.54% | 0.9736 | 0.94354 |
| qkv_lofi_kblock1 | 3.042 | 1.789 | 92.68% | 0.9693 | 0.94308 |
| oproj_hifi4 | 2.539 | 1.628 | 92.47% | 0.97084 | 0.94306 |
| oproj_hifi4_kblock1 | 2.539 | 1.628 | 92.47% | 0.9695 | 0.9343 |
| oproj_hifi4_kblock16 | 2.539 | 1.628 | 92.5% | 0.97154 | 0.94155 |
| headnorm_decode_fp32 | 2.239 | 1.441 | 93.75% | 0.96525 | 0.94475 |
| combo_qkv_hifi4_kchunk32 | 1.86 | 1.405 | 93.39% | 0.96972 | 0.94295 |
| combo_max_precision | 1.575 | 1.374 | 93.83% | 0.96781 | 0.94223 |
| combo_max_precision_kchunk32 | 1.548 | 1.438 | 93.46% | 0.969 | 0.9385 |

Settings run end to end only (pass 1): exp_approx_mode and the reduce-scatter + all-gather all-reduce give bit-identical
results to the baseline; decode k_chunk 256 and prefill k_chunk 256 exceed L1; prefill k_chunk 32: gate 0.97342, book 0.94219;
prefill q_chunk 32/128: gate unchanged, book 0.9357; prefill QKV HiFi4: gate 0.97438, book 0.93351.

- Real precision gains (isolated): QKV matmul off LoFi (sliding layers: tuned program config without a compute config makes
  ttnn pick LoFi) to HiFi2/HiFi4 + fp32 accumulation: attention error -26%; decode per-head Q/K/V norm with fp32
  accumulation: -12%; all four precision changes together: -38%. SDPA precision / chunking / core split: within +-1%.
- Order (K blocking) changes nothing once accumulation is fp32; with bf16 accumulation smaller K blocks are worse (3.04).
- End to end every change moves PCC by +-0.003-0.01 with random sign, including pure order changes; none reaches 0.99 and
  the most precise combination scores below the baseline on the gate text. Attention precision is not the limit: the
  layer-output error falls only 15% for a 38% attention gain and expert agreement only 92.7% -> 93.8%; the experts' own
  error (about 11e-4) is ~4x attention's.

## 10. Can any setting reach 0.99 on the accuracy gate? (2026-10-06)
The gate scores against Hugging Face bf16 (sdpa attention). Hugging Face itself, re-run on the gate's 129 positions with one
thing changed (reference/hf_gate_variants.py): identical settings or 1 CPU thread 1.00000 (bit-identical); **eager attention
instead of sdpa, same bf16: 0.98179** (52/129 positions below 0.99); fp32: 0.97328. Only a bit-for-bit copy of the reference's
CPU rounding reaches 0.99; an independent implementation of the same precision lands around 0.98.

Best chip results on the gate (vs HF bf16), chip/attn_sweep.py (results/attn_sweep_v3a/v3b/v4.jsonl):
| Configuration | Gate vs bf16 | Gate vs fp32 | Book vs fp32 |
|---|---|---|---|
| current code (065d1b0167f) | 0.96367 | 0.97347 | 0.93876 |
| mimic HF bf16 (every matmul / norm HiFi4 + fp32 accumulation, bf16 tensors, SDPA HiFi4 exact) | 0.97420 | 0.97072 | **0.95777** |
| all fp32 switches (fp32 intermediates, top-k on scores, fp32 manual attention) | **0.98180** | 0.97689 | |
| fp32 switches + prefill k_chunk 32 | 0.98024 | 0.97843 | |
| fp32 switches + prefill QKV HiFi4 | 0.97330 | **0.98045** | |
| fp32 switches, SDPA attention (fp32 attention off) | 0.97285 | 0.96896 | |

With the fp32 switches the chip equals HF's own eager-vs-sdpa agreement (0.9818). More precision moves the chip toward fp32
and away from the bf16 reference (prefill QKV HiFi4: fp32 0.9805, bf16 0.9733). Mimic HF bf16 is the largest real gain with
bf16 tensors: isolated layer-output error -39% (0.98e-4), experts error -35%, book +0.019; untraced 0.105 vs 0.101 s/step
(traced cost not measured). No attention setting or combination reaches 0.99 against the bf16 reference.
