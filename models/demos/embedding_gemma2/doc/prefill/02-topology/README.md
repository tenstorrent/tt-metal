# Prefill stage 2: op topology (E9.2)

Text-encoder prefill of `google/embeddinggemma-2`, batch 1, masked path, 12 x 10 grid (120 cores),
precision `sdpa_nofp32acc`, P150a at the default 150 W TDP. Labels: **[M]** measured (run named),
**[I]** inferred. Run folder `tt-ops/runs/embeddinggemma-2/e9/02-topology/` (`$T`); commands in
`work_log.md`.

## Before / after (latency_ms per traced warmed forward, 12 x 10) [M]

Before = same-session control on the same code with the stage-1 ops restored (`sweep_ctl`; its gate
values equal the stage-1 baseline to every digit, see work_log). After = code defaults (`sweep_new`).
Both `EMBEDDING_GEMMA2_PREFILL_ALL=1` (gate, burst, host, sustained with telemetry), back to back,
records `$T/run_ab/sweep_{ctl,new}/prefill.jsonl`, summary `perf_summary.json`. Headline: sustained.

| bucket | burst before | burst after | change | sustained before (AICLK MHz) | sustained after (AICLK MHz) | change | gate PCC / cos after (before) |
|---|---|---|---|---|---|---|---|
| 128 | 4.918 | 4.879 | -0.81 % | 4.914 (1350) | **4.875** (1350) | **-0.81 %** | 0.99424 / 0.99834 (0.99424 / 0.99834) pass |
| 256 | 5.917 | 5.874 | -0.72 % | 5.913 (1350) | **5.871** (1350) | **-0.70 %** | 0.99432 / 0.99861 (0.99432 / 0.99861) pass |
| 512 | 7.260 | 7.214 | -0.63 % | 7.257 (1350) | **7.212** (1350) | **-0.63 %** | 0.99537 / 0.99912 (0.99537 / 0.99912) pass |
| 1024 | 11.741 | 11.648 | -0.79 % | 11.736 (1350) | **11.644** (1350) | **-0.79 %** | 0.99577 / 0.99939 (0.99577 / 0.99939) pass |
| 2048 | 25.351 | 25.103 | -0.98 % | 25.339 (1350) | **25.099** (1350) | **-0.95 %** | 0.99574 / 0.99952 (0.99574 / 0.99952) pass |
| 4096 | 44.703 | 43.920 | -1.75 % | 44.706 (1350) | **43.925** (1350) | **-1.75 %** | 0.99581 / 0.99961 (0.99581 / 0.99961) pass |
| 8192 | 103.871 | 100.319 | -3.42 % | 103.944 (1350) | **100.553** (1343) | **-3.26 %** | 0.99628 / 0.99965 (0.99628 / 0.99965) pass |
| 16384 | 224.675 | 211.796 | -5.73 % | 230.534 (1300) | **220.905** (1262) | **-4.18 %** | 0.99706 / 0.99967 (0.99706 / 0.99967) pass |
| 32768 | 567.674 | 520.161 | -8.37 % | 595.830 (1243) | **560.511** (1193) | **-5.93 %** | 0.99693 / 0.99965 (0.99693 / 0.99965) pass |
| 65536 | 1717.316 | 1558.909 | -9.22 % | 1775.430 (1200) | **1643.468** (1125) | **-7.43 %** | 0.99724 / 0.99958 (0.99724 / 0.99958) pass |

No bucket regresses. From 16384 up the sustained AICLK is 38-75 MHz lower after the change (less
light mask work between the heavy ops **[I]**), so the sustained gain there is smaller than the burst
gain. The control's numbers match the stage-1 baseline (`../01-baseline`: sustained 4.914 / 25.335 /
598.529 / 1785.044 ms at 128 / 2048 / 32768 / 65536).

Burst A/B per rewrite (same session, `chain_ab`, all 10 buckets each; drift ctl_end vs ctl_start
<= 0.03 % up to 32768, +0.14 % at 65536) **[M]**:

| bucket | control (start) | #1 key bias fused | #2 band + full | #3 per-layer PLE | #4 pooler | all kept | control (end) |
|---|---|---|---|---|---|---|---|
| 128 | 4.918 | 4.897 (-0.43 %) | 4.918 (-0.00 %) | 4.919 (+0.01 %) | 4.898 (-0.41 %) | 4.878 (-0.82 %) | 4.918 (+0.00 %) |
| 256 | 5.917 | 5.896 (-0.34 %) | 5.916 (-0.00 %) | 5.916 (-0.00 %) | 5.896 (-0.35 %) | 5.873 (-0.73 %) | 5.917 (+0.00 %) |
| 512 | 7.261 | 7.238 (-0.32 %) | 7.261 (+0.00 %) | 7.262 (+0.01 %) | 7.238 (-0.31 %) | 7.215 (-0.63 %) | 7.262 (+0.02 %) |
| 1024 | 11.742 | 11.735 (-0.06 %) | 11.691 (-0.43 %) | 11.738 (-0.03 %) | 11.715 (-0.23 %) | 11.647 (-0.81 %) | 11.740 (-0.02 %) |
| 2048 | 25.345 | 25.341 (-0.02 %) | 25.158 (-0.74 %) | 25.343 (-0.01 %) | 25.308 (-0.14 %) | 25.106 (-0.94 %) | 25.349 (+0.02 %) |
| 4096 | 44.704 | 44.702 (-0.01 %) | 43.981 (-1.62 %) | 44.703 (-0.00 %) | 44.655 (-0.11 %) | 43.918 (-1.76 %) | 44.709 (+0.01 %) |
| 8192 | 103.915 | 103.905 (-0.01 %) | 101.017 (-2.79 %) | 103.278 (-0.61 %) | 103.821 (-0.09 %) | 100.303 (-3.48 %) | 103.891 (-0.02 %) |
| 16384 | 224.692 | 224.724 (+0.01 %) | 213.195 (-5.12 %) | 223.474 (-0.54 %) | 224.564 (-0.06 %) | 211.752 (-5.76 %) | 224.750 (+0.03 %) |
| 32768 | 567.378 | 567.435 (+0.01 %) | 522.316 (-7.94 %) | 565.615 (-0.31 %) | 567.799 (+0.07 %) | 520.092 (-8.33 %) | 567.461 (+0.01 %) |
| 65536 | 1723.464 | 1723.688 (+0.01 %) | 1567.750 (-9.03 %) | 1721.384 (-0.12 %) | 1725.789 (+0.13 %) | 1560.868 (-9.43 %) | 1725.929 (+0.14 %) |

Gate PCC / cosine identical in all seven runs at every bucket (5 digits); last hidden bitwise equal
to the control in every run, embedding bitwise except #4 / all kept (<= 3e-8) (`$T/analysis/ab_bitwise.log`).

## Op sequence per layer (from profiles)

Before: stage-1 12 x 10 Tracy profiles (`a903d517c6`, one warmed eager forward, `$T/analysis/before/`,
`$T/scripts/op_seq.py`). 645 ops at 128-512, 646 at >= 1024 (the band mask add appears at S > 513).
Every layer has 26 ops, 2 of them data movement only (DM). One sliding layer at 2048:

| # | op | DM | device us (2048) | in -> out |
|---|---|---|---|---|
| 0 | Slice (PLE layer i of [S, 12288]) | DM | 9.7 | DRAM -> L1 [S, 512] |
| 1 | RMSNorm (per-layer input) | | 10.4 | |
| 2 | RMSNorm (input_layernorm) | | 10.6 | |
| 3 | Matmul QKV [512 -> 2048] | | 48.9 | |
| 4 | NlpCreateHeads (q, k, v) | DM | 22.1 | DRAM -> L1 heads |
| 5-7 | RMSNorm q / k / v (per head) | | 16.2 / 11.3 / 8.9 | |
| 8-9 | RotaryEmbedding q / k | | 51.2 / 31.8 | |
| 10 | SDPA (output_concat_heads) | | 317.7 | mask bfp4 DRAM |
| 11 | MinimalMatmul o_proj | | 36.0 | |
| 12-13 | RMSNorm post-attn, Add residual | | 10.5 / 9.9 | |
| 14-18 | RMSNorm pre-ff, Matmul up, Matmul gate (+gelu), Mul, MinimalMatmul down | | 10.6 / 48.5 / 77.2 / 57.8 / 51.0 | |
| 19-20 | RMSNorm post-ff, Add residual | | 10.5 / 6.3 | |
| 21-25 | Matmul PLE gate (+gelu), Mul, MinimalMatmul PLE proj, RMSNorm, Add (x layer_scalar) | | 30.5 / 7.8 / 27.9 / 10.4 / 6.5 | |

Outside the layers, before: prologue = mask build (Sub, Mul, Add full, Add sliding at S > 513),
Embedding, wide PLE matmul [512 -> 12288]; epilogue 16 ops = final RMSNorm, head matmul, pooler 14
ops (Matmul keep @ h, Typecast DM, Clone DM, FillPad DM, Reduce, clip, Div, then L2: Mul, Clone DM,
FillPad DM, Reduce, sqrt, clip, Div).

After (from the code paths and the A/B runs; **no after profile was captured**, the stage ended at
the person's deadline before the Tracy step) **[I]**:

| part | before ops | after ops | change |
|---|---|---|---|
| mask build (S > 513) | 4 (Sub, Mul, Add [S,1]+[1,S], Add band + bias row-broadcast) | 2 (Add [S,1] + act(keep), Add band + full) | -2 ops; the slow row-broadcast add replaced |
| mask build (S <= 513) | 3 | 1 | -2 ops |
| PLE, >= 8192 tokens | 1 wide matmul + 24 x (Slice DM) | 24 x [S,512]x[512,512] matmul | -1 op, -24 DM ops, +24 matmuls (net -1 op, no DM) |
| per layer, >= 8192 tokens | 26 (2 DM) | 26 (1 DM: NlpCreateHeads) | Slice -> Matmul |
| per layer, < 8192 tokens | 26 (2 DM) | 26 (2 DM) | unchanged |
| pooler | 14 (5 DM) | 8 (2 DM: the L2 Reduce's Clone / FillPad) | -6 ops (Typecast, Clone, FillPad, Reduce, clip, Div of the token count) |
| forward total | 645 / 646 | 637 at <= 512, 638 at 1024-4096, 637 at >= 8192 | -8 / -8 / -9 |

## Rewrite table (every catalogue class assessed)

| # | class | candidate | action | PCC per contract shape | burst wall (A/B vs control, same session) | result | reason / blocker |
|---|---|---|---|---|---|---|---|
| 1 | 1 mask | key bias `(keep - 1) * 1e9` as SFPU pre-activations of the full-mask add (`input_tensor_b_activations`) | applied | bitwise equal 10 / 10 | -0.43 / -0.34 / -0.32 % at 128 / 256 / 512; within drift >= 1024 | **kept** | 2 ops fewer per forward |
| 2 | 1 mask | sliding mask = band + full (dense + dense) instead of band + key-bias row broadcast | applied | bitwise equal 10 / 10 | -0.43 % (1024), -0.74 (2048), -1.62 (4096), -2.79 (8192), -5.12 (16384), -7.94 (32768), -9.03 % (65536) | **kept** | probe: same values, 4.3x faster build (8192 3.69 -> 0.83 ms, 16384 14.75 -> 3.44 ms) |
| 3 | 4 slice after matmul | narrow the PLE matmul per layer instead of slicing the [S, 12288] projection, >= 8192 tokens | applied (gated on tokens) | bitwise equal 10 / 10 | -0.61 / -0.54 / -0.31 / -0.12 % at 8192 / 16384 / 32768 / 65536; < 8192 unchanged | **kept** | removes 24 DM slices and the [S, 12288] tensor (1.6 GB at 65536); below 8192 the wide matmul is cheaper (profiles, work_log) |
| 4 | 1 / 2 pooler | L2-normalize the masked sum; drop the token count (Typecast, Clone, FillPad, Reduce on 1 core, clip, Div) | applied | last hidden bitwise; embedding max abs diff <= 3e-8, cosine 1.0 vs control; gate values unchanged to 5 digits | -0.41 % (128) ... -0.11 % (4096); within drift >= 8192 | **kept** | normalize(sum) = normalize(mean); Reduce of keep was 468 us at 65536 on 1 core |
| 5 | 5 lifetime | free each mask after the last layer of its kind, the PLE source after the last layer input; pooler / `l2_normalize` pre-clip temporaries freed (leaked until GC before) | applied | covered by 1-4 | - | **kept** | `TextModel.encode_embeds` (shared with `tt/model.py`) |
| 6 | 1 mask | full layers: windowed SDPA (`cu_window_seqlens = [0, L, S]`, on-device mask) instead of the dense mask | probed | bitwise equal on valid rows (12 / 12 probe cases) | SDPA 1.4-3.1x slower (hd 512 2048: 0.58 -> 1.18 ms; 8192: 4.14 -> 7.61 ms) | rejected | slower than the mask it removes (33.5 us at 2048, 26.5 ms at 65536); stage-4 item |
| 7 | 1 mask | sliding layers without a dense mask | not possible | - | - | rejected | `sliding_window_size` cannot combine with `attn_mask` (`sdpa_device_operation.cpp:142`) nor with windowed mode (`:477`); the mask must be [.., Sq, Sk] (`:178-179`), no row broadcast |
| 8 | 1 mask | mask built outside the trace, refreshed per padding length | not built | - | - | rejected | moves the build to every request with a new length; saves only for repeated lengths |
| 9 | 1 mask | no-mask path | not built | - | - | n/a | the contract has no full-length (`unmasked`) shape |
| 10 | 1 head split | attention reads heads from the QKV output (remove NlpCreateHeads, DM) | not possible with stock ops | - | - | rejected | per-head q / k / v RMSNorm needs [B, H, S, hd]; SDPA takes split heads; folding into a reader is stage 6. Cost left 0.45 ms at 2048 (1.8 %), 31.4 ms at 65536 (1.9 %) |
| 11 | 1 concat heads | SDPA writes [B, 1, S, H*hd] | already applied | - | - | n/a | `output_concat_heads=True` (before stage 2) |
| 12 | 1 typecast / reshape / permute / sharded | none in the layers; pooler typecast removed by #4 | - | - | - | n/a | profile: no Reshape / Permute / Typecast / reshard op in any layer |
| 13 | 2 add + norm | `rms_norm(residual_input_tensor=...)` | not applied | - | - | rejected | sandwich norms add the residual after the post-norm; fusing the next pre-norm still needs the sum materialized for the following residual: no op removed |
| 14 | 2 spelled-out norm | pooler L2 normalize as `rms_norm(x) * 768^-0.5` | probed | output norm 0.990-1.007 (max error 1.4e-3 vs 6e-9) | 3.3x faster per call (0.033 vs 0.109 ms host-timed) | rejected | precision (unit-norm contract); fp32-accumulation adaptation not tried (deadline). Cost left about 28 us at 2048 |
| 15 | 2 SDPA / softmax / activation | in use | - | - | - | n/a | SDPA, gelu_tanh fused in the matmul packer or the multiply's pre-activation |
| 16 | 3 pack | MLP gate + up as one [512, 4096] matmul + 2 slices + gelu-multiply | probed | PCC 0.999995 | 1.19-1.39x slower (M 512 / 2048 / 8192: 0.074 / 0.192 / 0.848 -> 0.100 / 0.266 / 1.008 ms) | rejected | no stock op consumes the packed output without the split (`ttnn.geglu` composite; `dit_minimal_matmul_addcmul_fused` takes a [1, N] multiplier) |
| 17 | 3 pack | Q / K / V | already packed (`wqkv`) | - | - | n/a | |
| 18 | 4 fold | bias / activation / transpose / scale / layer_scalar | already folded | - | - | n/a | no biases; gelu fused; scale 1.0 in SDPA; layer_scalar as the residual add's post-activation |
| 19 | RoPE | per-layer cos / sin slicing; fused Q+K rotary | not possible | - | - | rejected | no slice op in the profile (the op reads rows 0..S-1 of the table); `rotary_embedding_llama_fused_qk` is decode only (validation: "only supports decode mode with seq_len=1") |
| 20 | MLP multiply into the down matmul | fused epilogue / prologue | not possible | - | - | rejected | no stock matmul with a binary prologue / full-tensor epilogue |
| 21 | PLE hoist | per-layer slice + norm out of the loop | not applied | - | - | rejected | each layer reads a distinct segment: no repeated work; one 24-segment norm needs a DM reshape |

## Accuracy and gates at closure [M]

- Contract gate at 12 x 10 (traced replay vs HF fp32): 10 / 10 pass in `sweep_new`, values equal to
  the stage-1 baseline to 5 digits (last hidden bitwise equal; embedding within 3e-8, table above).
- Text suite + full model + trace + generator_vllm + precision_config (153), `test_text_model[b1_s32768]`
  and the demo pass at 12 x 10 and 13 x 10 (`gates_e92_*`).
- Soak 2000 replays at 1 x 2048 and 1 x 4096: 0 mismatches.
- b8 x 512 / b32 x 128 (perf.py, forward mean): 34.613 / 32.332 ms vs control 34.612 / 32.348 ms.

## Open / not done

- No after Tracy profile (deadline): the after op counts above are from code and A/B, not profiles.
- Sustained clock at >= 16384 is lower after the change (16384 / 32768 / 65536: 1262 / 1193 / 1125 vs 1300 / 1243 / 1200 MHz), so the sustained
  gain is smaller than the burst gain there; not attributed per op.
- Remaining mask cost (2 adds) and NlpCreateHeads (DM) are left for stages 4-6.
