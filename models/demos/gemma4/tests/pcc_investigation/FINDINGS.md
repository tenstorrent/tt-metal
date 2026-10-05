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
