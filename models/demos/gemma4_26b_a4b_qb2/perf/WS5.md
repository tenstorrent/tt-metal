# WS5 · Fused batch-1 decode (B, 16-17 Nov 2026)

Branch `ws5-fused`, gate commit `2d8342f5d2` (= `integration` f8cc89d0d7 (WS6 + plan) + WS5). Precision `bfp8_release_selected`.
Status: **done**: every gate item passes (gate run 2, 17 Nov 00:45-01:28 UTC).

## Result

| | integration (WS4 + WS6) | WS5 | Target |
|---|---|---|---|
| Batch-1 native step, 128 prompt (`decode_timing.json`) | 8.53 ms (WS6 gate; WS4: 8.44) | **4.995 ms** (200.2 tok/s/user) | ≤ 5 ms |
| Batch-1 native step, 4K prompt | 9.03 ms (WS4: 8.91) | 5.43 ms | |
| Batch 8 / 16 / 32 step | 17.64 / 20.20 / 22.55 ms | 17.55 / 20.08 / 22.37 ms (unchanged paths) | ≤ 3 % slower |
| 2-layer fused decode vs HF, 4 prompts x 8 steps (`fused_decode.txt`) | (control, `FUSED=0`) median 0.99822, 29 / 32 ≥ 0.995, min 0.9554 | **median 0.99841, 26 / 32 ≥ 0.995, min 0.9424** | median ≥ 0.995, every step ≥ 0.90 |
| Agreement vs integration, 8 prompts x 128 (`agreement.json`) | | 98.44 % (1008 / 1024) | ≥ 99 %, or 98-99 % with HF parity no worse |
| HF top-1 parity, same 1024 positions (`hf_agreement.txt`) | 996 / 1024 | **997 / 1024** | no worse than integration |
| GPQA Diamond first 10 | 9 / 10 (WS1), 8 / 10 (WS4) | **9 / 10** | ≥ 8 / 10 |

## What the batch-1 step runs now

New package `tt/fused_decode/` (Python `ProgramDescriptor` ops + kernels, gpt-oss `fused_decode` style), switched in by
`MultichipDecoder.fused_setup` / `fused_decode` and `Gemma4Model._fused_decode` for batch 1 (one active slot); batch 2-32,
prefill and the WS6 verify keep their paths. Per decode step:

| Program | What | Replaces |
|---|---|---|
| token entry (`boundary.py` mode 3) | the token's row of a replicated row-major embedding table, × embed scale = the residual; the first input norm | embedding, tilize, all-gather, scale, input norm, reshard (≈ 44 µs) |
| per layer: QKV matmul (ttnn) | unchanged | |
| `HeadsB1` (`heads.py`) | one head per core, packed into one tile; RMSNorm + NeoX RoPE (cos / sin rows read from the RoPE tables at the position) | `qkv_heads_norm_rope_b1` (3 cores, 13.4 / 23.3 µs) + the per-step RoPE row gathers |
| cache update, SDPA, concat heads, o_proj (ttnn) | o_proj writes this device's partial (no all-reduce) | |
| attn boundary (`boundary.py`) | fabric all-reduce of the o_proj partials (each device multicasts its flat partial; slots summed in slot order, bit-identical on every device) + post-attention norm + residual + the pre-FFN norm; writes the FP32 router input, the expert input (gammas folded) flat | RS + AG, 3 norms, add, typecast, reshard, 2 multiplies |
| `RouterB1` (`router.py`), in the attn boundary's program | FP32 projection split over K on 32 cores, partials summed in FP32 on the SFPU, integer top-8 / softmax / scale + the 3 shared slots; the matmul cores read their weights while the boundary runs and start when its sender raises their semaphore (router input packed) | FP32 linear (4 cores, compute bound) + `router_topk_b1` |
| merged MoE (`moe_decode` gate\|up, down) | the routed experts AND the shared MLP as 3 always-routed virtual experts (both pre-norm gammas folded into the BFP8 gate\|up rows); the down sums routed and shared slots into two outputs written flat into the boundary's partial buffer | shared MLP (3 programs) + its all-reduce |
| moe boundary | all-reduce of both partials + ff1 / ff2 / ff norms + residual + layer scalar + the next layer's input norm (or the final norm) | pack, RS, AG, unpack, 4 norms, add |
| `LmHeadGreedyB1` (`lmhead.py`), greedy requests | the vocab projection streamed per DRAM bank with the argmax in the same program (softcap tie classes: the lowest id among the softcap's BF16 ties, as the common sampler), merge, all-gather of one tile, pick into the token input | LM head (507-527 µs) + softcap + the sampler's top-k pipeline (~470 µs) |

Non-greedy requests (temperature > 0, penalties, log-probs, host sampling) take the standard LM head + softcap + common
sampler after the fused layers. Every change has an environment switch (`GEMMA4_26B_WS5_{FUSED, SHARED_B1, MERGED,
FLAT_MOE, ROUTER, JOIN_ROUTER, HEADS, TOKEN_ENTRY, GREEDY, LMHEAD, ROUND_N, BND_FAST}=0`).

Programs per sliding layer: 10 (QKV, heads, cache, SDPA, concat, o_proj, attn boundary + router, gate\|up, down, moe
boundary; WS4: 32); per step + 4 (token entry, LM head + argmax, all-gather, pick).

## Path (full-model step, 8-prompt agreement runs)

| Step | B1 step | agreement | HF parity |
|---|---|---|---|
| integration (WS4) | 8.44 ms | | 996 |
| fused layer boundaries (fabric all-reduce + norms) | 7.07 | 98.6 % | 996 |
| shared MLP through moe_decode; flat MoE inputs; MoE partials written into the boundary | 6.44 | 98.0 % | 997 |
| greedy terminal in the model trace (tokens identical to the stock sampler on 8 / 8) | 5.85 | 98.0 % | |
| fused router | 5.69 | 97.75 % | 997 |
| shared MLP merged into the routed experts' programs | 5.45 | 98.3 % | 1000 |
| HeadsB1 + token entry | 5.29 | 98.2 % | 1001 |
| logits in L1, readers with one barrier, RoPE rows from the tables | 5.18 | 98.2 % | |
| boundary format reconfigs, router tail in place | 5.03 | 98.2 % | |
| fused LM head + argmax | 4.905 | 98.05 % | 1000 |
| boundary + model L1 allocated first (the 4K prefill CBs clashed with the persistent buffers) | 5.054 (`decode_timing`) | | |
| attn boundary and router in one program | 4.995 (`decode_timing`) | 98.44 % | 997 |

The path's first rows are the `greedy_agreement` runs' step (`generate`, 8 prompts; 4.977 ms in the gate's run); the
last two `decode_timing`'s batch-1 128-prompt step, the gate's number.

Device profile of one step (`ops_profile.txt`): a sliding layer is 10 programs, 134.9 µs of kernel time (WS4: 32
programs, 224.4 µs); the attn boundary + router program takes 22.9 µs on 34 cores (separately 8.8 + 15.7 µs plus a
dispatch gap); the greedy LM head 394 µs.

Serving (`serve_bench.sh perf,gpqa`, vLLM, capacity 16):

| | integration (WS6 gate) | WS5 |
|---|---|---|
| 1 user, 128/128: TTFT, decode tok/s | 88.1 ms, 116.2 | 88.8 ms, **198.5** |
| 16 users, 128/128: TTFT, decode tok/s per user, total tok/s | 1596 ms, 62.7, 565 | 1592 ms, 63.1, 567 |
| GPQA Diamond first 10 (threshold 0.8, temperature 1.0: standard LM head + sampler after the fused layers) | 8 / 10 (WS4) | **9 / 10** |

Measured and dropped: the common sampler's force-argmax path (2.6x slower step: all-gather of the full logits); dense
projections through the moe_decode down program (k = 1: one RISC per column streams ~290 GB/s, slower than ttnn); a
DRAM-sharded QKV matmul (16 µs standalone, 20 µs in the model with the FP32 accumulation the release policy uses).

## Accuracy

Numerics (`numerics.txt`, `quick_check.json`, both pass): layer 0 / 5 traced decode PCC 0.99985 / 0.99991 (WS1 baseline
0.99984 / 0.99991), prefill unchanged. These run the unfused batch-1 layer; gate run 1 failed them (L5 decode 0.985)
because that layer fed the expert-gamma-scaled input to the merged weights (gamma folded in, so applied twice); it now
takes the plain normed input (commit 2d8342f5d2).

Unit tests (`unit_tests.txt`): `test_moe_decode.py` + `test_fused_boundary.py`, 14 passed.

**2-layer fused decode vs HF** (`tests/test_fused_decode.py`; layers 0 + 5 with real weights, the model's embedding,
final norm, tied LM head and softcap; 66-token fixture + 3 chat prompts, 8 teacher-forced steps each; logits after the
softcap). Same run, both paths; bold = below 0.995, ✗ = top-1 differs from HF (a 2-layer model's logits are nearly flat:
25 / 32 top-1 on both paths):

| step | fixture fused | fixture control | chat0 fused | chat0 control | chat1 fused | chat1 control | chat2 fused | chat2 control |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.9992 ✗ | 0.9989 ✗ | 0.9992 ✗ | 0.9990 ✗ | **0.9936** ✗ | 0.9971 ✗ | 0.9991 ✗ | 0.9989 ✗ |
| 1 | 0.9992 | 0.9992 | 0.9979 | 0.9984 | 0.9979 | 0.9982 | 0.9991 ✗ | 0.9989 ✗ |
| 2 | 0.9991 | 0.9987 | 0.9981 | 0.9985 | **0.9894** | 0.9982 | 0.9989 ✗ | 0.9983 ✗ |
| 3 | **0.9532** | **0.9554** | 0.9976 | 0.9980 | 0.9961 | 0.9967 | 0.9990 ✗ | 0.9988 ✗ |
| 4 | **0.9424** | 0.9981 | 0.9983 | 0.9982 | **0.9467** | 0.9982 | 0.9991 | 0.9990 |
| 5 | 0.9985 | 0.9982 | 0.9959 | **0.9935** | 0.9984 | 0.9952 | 0.9977 | 0.9980 |
| 6 | 0.9988 | 0.9982 | 0.9986 | 0.9981 | 0.9952 | 0.9952 | 0.9982 | 0.9976 |
| 7 | 0.9986 | 0.9981 | 0.9986 | 0.9982 | 0.9985 | 0.9985 | **0.9566** | **0.9613** |

Fused: median 0.99841, 26 / 32 ≥ 0.995, min 0.9424. Control (`GEMMA4_26B_WS5_FUSED=0`, the integration path): median
0.99822, 29 / 32, min 0.9554. The bar (median ≥ 0.995, every step ≥ 0.90) holds for both.

The 6 fused misses, each at a single top-8 expert flip against HF (the test reads the fused layer's routing row; HF-only
expert / device-only expert):

| prompt, step | PCC | flip |
|---|---|---|
| fixture 3 | 0.9532 | L0: 15 / 89 |
| fixture 4 | 0.9424 | L0: 15 / 25 |
| chat1 0 | 0.9936 | L5: 59 / 79 |
| chat1 2 | 0.9894 | L5: 58 / 74 |
| chat1 4 | 0.9467 | L0: 58 / 16 |
| chat2 7 | 0.9566 | L0: 44 / 16 and L5: 104 / 125 |

Every fused step without a flip is ≥ 0.995. Two of them (fixture 3, chat2 7) are also the control's two largest
misses, so the control most likely flips the same expert there (its unfused router does not expose the routing row to
the test). The fused path without the merged MoE (`GEMMA4_26B_WS5_MERGED=0`: shared MLP as its own moe_decode program)
gave 25-26 / 32 on the same prompts, so the misses do not come from folding the gammas into the BFP8 weights.

**Full model** (`agreement.json`, `hf_agreement.txt`): 8 prompts x 128 greedy tokens, teacher-forced against
integration's tokens: 98.44 % agreement (1008 / 1024; per prompt 96.1-100 %). Under the 98-99 % rule the teacher-forced
HF top-1 (HF bf16 on CPU, same histories) decides: WS5 997 / 1024, integration 996 / 1024, no worse.

## Proof files (`perf/results/WS5/`)

`decode_timing.json`, `numerics.txt`, `quick_check.json`, `unit_tests.txt`, `fused_decode.txt`,
`fused_decode_control.txt`, `agreement.json`, `hf_agreement.txt`, `gpqa/summary.json`, `serve_perf_c1.json`,
`serve_perf_c16.json`, `ops_profile.txt`.

Decode-path contract for other stages: `MultichipDecoder._prefill_continuation` and the `decode_forward` rows contract are
unchanged; the fused path is only taken for batch-1 decode (one active slot) and every piece has its switch.
