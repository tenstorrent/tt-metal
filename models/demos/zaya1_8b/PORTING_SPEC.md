# ZAYA1-8B (Zyphra) — tt-metal Porting Spec

Target HW: Blackhole **P150a** single card first; scale to 2–4 cards for perf (TP).
Reference: HF `Zyphra/ZAYA1-8B` + Zyphra transformers fork branch `zaya`
(`pip install "transformers @ git+https://github.com/Zyphra/transformers.git@zaya"`).
Reference modeling code mirrored under `reference/zaya_hf/` (modeling_zaya.py / configuration_zaya.py).

License: Apache-2.0. Weights: bf16, 4 safetensors shards (~17.7 GB total) → fits one P150a (32 GB DRAM).

## 1. Top-level config (from HF config.json)

| field | value | notes |
|---|---|---|
| model_type / arch | `zaya` / `ZayaForCausalLM` | transformers 4.57.1 fork |
| hidden_size | 2048 | |
| num_hidden_layers | 80 | = len(zaya_layers); **alternating** ATT / MoE → 40 ATT + 40 MoE |
| num_attention_heads | **16** | ⚠️ config.json says **8** (SKEW) — must override to 16 (hidden/head_dim=2048/128). Confirmed at load: all weight shapes close at 16. |
| head_dim / kv_channels | 128 | |
| num_key_value_heads / num_query_groups | 2 / 2 | GQA |
| cca_num_q_heads | 8 (per ATT layer) | query compressed to 8 heads → o_proj input = (num_heads//2)*head_dim = 8*128 = 1024 |
| partial_rotary_factor | 0.5 | RoPE on first 64 dims of each head only |
| rope_theta | 5e6 | |
| num_experts | 16 | per MoE layer |
| moe_router_topk | 1 | top-1 routing |
| ffn_hidden_size | 4096 | gated swiglu → fc1 out 4096, glu halves to 2048, fc2 2048→2048 |
| zaya_mlp_expansion | 256 | router latent dim D |
| normalization | RMSNorm, eps 1e-5 | |
| residual_in_fp32 | true | residual stream kept fp32 |
| scale_residual_merge | true | per-channel learned residual scaling (see §4) |
| zaya_use_mod | true | Mixture-of-Depths via skip expert (see §3) |
| zaya_use_eda | true | cross-layer router-state carry (see §3) |
| tie_word_embeddings | true | lm_head shares embed_tokens weight |
| vocab_size | 262272 | |
| max_position_embeddings | 131072 | |

## 2. Layer topology

`config.zaya_layers = ['a', 16, 'a', 16, ...]` (length = num_hidden_layers = 80).
- `'a'` → **ZayaDecoderATTLayer** (CCA self-attention).
- `int N` (=16) → **ZayaDecoderMLPLayer** (MoE block with N experts).

Strict alternation: ATT, MoE, ATT, MoE, … → 40 attention + 40 MoE layers.

### Per-layer skeleton (BOTH layer types share this residual pattern — "delayed/parallel residual merge")
```
# at entry: (hidden_states, residual) from previous layer
if scale_residual_merge:        # ResidualScaling, §4
    residual, hidden = res_scale(residual, hidden)
residual = hidden if residual is None else (hidden + residual)   # MERGE happens here
hidden   = input_norm(residual)            # RMSNorm(2048)
hidden   = sublayer(hidden)                # CCA attention  OR  MoE block
return (hidden, residual)                  # residual carried forward, NOT added to sublayer out yet
```
After the last layer: `res_scale` → `residual = hidden + residual` → `final_norm` (RMSNorm) → `lm_head`.
Note: the sublayer output of layer L is merged into the residual at the **start of layer L+1**, not within L.

## 3. Novel components (no existing tt-metal building blocks)

### 3a. CCA — Compressed Convolutional Attention  (`CCA` + `ZayaAttention`)  ← HARDEST
Per ATT layer. Geometry (assuming num_heads=16, head_dim=128):
- `linear_q`: 2048 → latent_q = cca_num_q_heads(8)*128 = **1024**
- `linear_k`: 2048 → latent_k = num_kv_heads(2)*128 = **256**
- `val_proj1`: 2048 → latent_k/2 = **128**   (on current hidden hs)
- `val_proj2`: 2048 → latent_k/2 = **128**   (on time-shifted hs_d = hs shifted +1 along seq)
- `conv_qk`: `Sequential(Conv1d, Conv1d)` over **sequence**, on concat[q,k] (in_out_ch = 1024+256 = 1280):
  - conv0: kernel=cca_time0=2, groups=1280 (depthwise)
  - conv1: kernel=cca_time1=2, groups=(num_q+num_kv)=10
  - causal: left-pad total_padding=2 along seq before convs
- mean residual: `qk_mean_q = (q_head + k_head_repeated)/2`; conv output is added to these means.
- normalize: per-head L2-normalize query & key, scale by sqrt(head_dim); key also × learned `temp[kv_head]`.
- value: `cat([v1, v2])` → reshape to (kv_heads=2, head_dim=128).
- Then standard path: reshape q→(num_heads//2=8 heads,128), k/v→(2,128); **partial RoPE (50%)** on q,k;
  GQA repeat kv by `num_key_value_groups//2`; SDPA (causal); `o_proj`: 1024 → 2048.
- **KV/conv cache**: `ZayaDynamicCache` holds (a) standard K/V cache AND (b) `conv_states` [B, 1280, kernel=2]
  per ATT layer for streaming conv, plus (c) `prev_hs` [B, 2048] per layer for the v2 time-shift stream.
  Prefill vs decode have different conv handling (decode rolls the 2-wide conv window).

tt-metal implications: need causal depthwise/grouped Conv1d along sequence (Mamba-style conv state — see qwen36
DeltaNet op), per-head L2 norm + scale, custom 2-part KV+conv+prev_hs cache. SDPA core itself maps to
`ttnn.transformer.scaled_dot_product_attention` once q/k/v are built.

### 3b. ZAYA1 Router (MLP router) — `ZayaRouter`
Per MoE layer:
```
hs = down_proj(hidden)                       # Linear 2048 -> 256 (+bias)
if use_eda and prev_router_states: hs += prev_router_states * router_states_scale   # §3c
router_hidden_states_next = hs.clone()       # pre-norm, carried to next MoE layer (EDA)
hs = rmsnorm_eda(hs)                          # RMSNorm(256)
logits = router_mlp(hs)                       # 256->256 GELU ->256->256 GELU ->256->E (last no bias)
prob = softmax(logits)
biased = prob.detach().float() + balancing_biases      # bias affects choice only
choice = topk(biased, k=1)                    # E = num_experts (+1 skip expert if MoD)
route_prob = gather(prob, choice)
```
`balancing_biases`: zeros, except skip-expert (index E-1) = -1.0 when MoD.

### 3c. EDA — cross-layer router state (`zaya_use_eda`)
The 256-dim pre-norm router state is passed from each MoE layer to the next via `prev_router_hidden_states`,
added (scaled by learned `router_states_scale`) into the next router's down-proj output. Active on all MoE
layers except `layer_number == 1`. Implemented as a second residual stream in router (256-dim) space.

### 3d. MoD — Mixture of Depths (`zaya_use_mod`)
Router has an extra "skip" expert (E = num_experts+1 = 17). With top-1, tokens choosing the skip expert
(idx 16) **bypass all experts** (identity passthrough), then are scaled by their skip-route prob.
In `ZayaBlock.forward`: tokens sorted by expert; experts run on all but the skip group; skip group is
concatenated back unchanged; output `* route_prob`. → data-dependent token dropping (hard on static HW).

### 3e. Expert MLP — `MLP` / `SequentialMLP`
Each expert: `linear_fc1` 2048→4096, swiglu (chunk 2 → silu(a)*b → 2048), `linear_fc2` 2048→2048. No bias.
top-1 means each token hits exactly one expert (or skip). 16 experts/layer × 40 MoE layers.

### 3f. ResidualScaling — `scale_residual_merge`
Per layer (incl. a trailing one before final_norm): learned per-channel (2048) scale+bias on hidden, and
(except first layer) on residual: `hidden=(hidden+h_bias)*h_scale; residual=(residual+r_bias)*r_scale`.

## 4. Numerics
- residual stream in fp32 (`residual_in_fp32`); RMSNorm computes in fp32.
- mamba/conv cache dtype fp32.
- bf16 weights. On tt-metal: start bf16 activations / bfp8_b weights for matmuls, validate PCC, then tune.

## 5. Bring-up plan (reference-first, op-by-op PCC)
- **Phase 0**: HF reference on CPU (Zyphra fork) → dump per-module golden tensors (1–2 prompts, small seq).
  Confirm: num_attention_heads, zaya_layers list, all sub-tensor shapes, partial-RoPE dims.
- **Phase 1**: embedding, RMSNorm, partial RoPE, lm_head (tied) — PCC vs golden.
- **Phase 2**: MoE block (down_proj→eda→rmsnorm→router_mlp→softmax→top1; expert swiglu MLP; MoD skip; weighting).
  Reuse mixtral_moe patterns. Decide static dense-compute vs gather/scatter for top-1.
- **Phase 3**: CCA attention (proj + causal grouped conv + mean residual + L2norm/temp + value 2-stream + partial RoPE + SDPA + o_proj) and the 3-part cache.
- **Phase 4**: EDA cross-layer router stream + ResidualScaling + delayed-residual wiring; assemble one ATT+MoE pair, PCC.
- **Phase 5**: full 80-layer model + decode loop + tokenizer; end-to-end logits/greedy PCC & sample quality.
- **Phase 6**: single-P150a perf (sharding, dtype, op fusion). Per-segment recompile avoidance (cf. BH-QB note).
- **Phase 7**: scale 2–4 P150a via TP (experts / attention) using models/common tt_ccl.

## 6. Phase 0 RESOLVED (reference runs on CPU; golden tensors in reference/golden/)
1. **num_attention_heads = 16** (config.json's 8 is a SKEW). head_dim=128, kv_heads=2, kv_groups=8.
   CCA: q=8 heads, kv=2 heads, head_dim=128, latent_q=1024, latent_k=256. Verified vs checkpoint weight shapes.
2. zaya_layers (len 80) = ['a',16,'a',16,...] strict alternation → 40 ATT + 40 MoE.
   cca_num_q_heads=[8,0,...], num_query_groups_list=[2,0,...], ffn_hidden_size_list=[0,4096,0,4096,...].
3. lm_head.weight IS embed_tokens.weight (tied) — confirmed True at runtime.
4. Sanity: prompt "The capital of France is" → top-1 next token " Paris". Reference is correct.
5. Ground-truth layer weight shapes (per ATT layer / per MoE layer):
   - ATT: linear_q[1024,2048] linear_k[256,2048] val_proj1[128,2048] val_proj2[128,2048]
     conv_qk.0[1280,1,2](depthwise) conv_qk.1[1280,128,2](groups=10) temp[2] o_proj[2048,1024]
   - MoE: router.down_proj[256,2048]+bias router.rmsnorm_eda[256]
     router_mlp.0[256,256]+b .2[256,256]+b .4[17,256] balancing_biases[17]
     experts.local_experts.{0..15}.linear_fc1[4096,2048] linear_fc2[2048,2048]
   - both: input_norm[2048] res_scale.hidden_states_{scale,bias}[2048] (MoE also residual_{scale,bias}[2048])
   - global: embed_tokens[262272,2048] final_norm[2048]; total 2483 tensors.

## 7. ⚠️ Checkpoint↔fork config skew — REQUIRED patches to load the HF reference
The released config.json does not load with the `zaya` fork modeling as-is. Apply BEFORE constructing model:
- `num_attention_heads`: 8 → 16  (else CCA linear_q builds [2048,2048] ≠ checkpoint [1024,2048]).
- `zaya_mlp_expansion`: scalar 256 → per-layer list `[256 if isinstance(l,int) else 0 for l in zaya_layers]`
  (modeling indexes `config.zaya_mlp_expansion[layer_n]`).
See reference/dump_golden.py for the exact patch. Both are config-loading fixes only; the checkpoint tensors
themselves are correct.

## 8b. BRING-UP RESULTS (Phases 1-5, single P150a, device 1, bf16)
All op-level blocks validated against fp32 golden (PCC). Tests: `tests/run_phase{1,2,3,5}.py`.
- **Phase 1** (embedding/RMSNorm/partial-RoPE/lm_head): 6/6 PASS, pcc 0.9999-1.0.
- **Phase 2** (MoE: MLP router + softmax-17 + balancing-bias + top-1 one-hot gate + 16 dense
  swiglu experts + MoD skip): 4/4 PASS; expert_choice EXACT, moe_block 0.99996.
  Key on-device trick: top-1 via `gate = (biased == max(biased)) * prob` (no argmax op).
- **Phase 3** (CCA): 5/5 PASS; cca_q/k/v 0.9999, cca_attention 0.997. Key trick: the two causal
  depthwise/grouped Conv1d(k=2) compose into `conv = qk@Cm^T + shift1(qk)@Bm^T + shift2(qk)@Am^T + bias`
  with host-precomputed block-diagonal [1280,1280] matrices (conv_equiv pcc 1.0). Seq shifts via
  row-major pad/slice; head splits via slice/unsqueeze/concat (NOT reshape — reshape makes the small
  head dim a tile dim and breaks volume). SDPA via ttnn.transformer.scaled_dot_product_attention(is_causal).
- **Phase 5** (full 80-layer prefill): logits pcc 0.996, **last-token argmax CORRECT (9079 'Paris')**.
  Demo (`demo/demo.py`) generates "The capital of France is **Paris**" end-to-end on device.
  - Initial bring-up had a last-token routing flip (predicted 528 ' in'). Diagnosis: device-bf16 drifted
    from bf16-CPU reference (~0.5%/layer), first routing divergence at MoE layer 17, cascading flips of
    the last token. **FIX**: replaced ttnn SDPA with manual attention doing **fp32 softmax** (matching HF
    eager `softmax(dtype=fp32)`) + explicit causal mask. This cut per-attention-layer drift (cca_attention
    0.9973->0.9981) enough that the final prediction matches. residual stream also kept fp32.
  - Remaining: a few intermediate-layer per-token hidden PCCs vs the fp32 golden are low (0.90-0.94) where a
    token routes differently than fp32-ref — inherent to top-1 MoE under any precision delta; does NOT
    affect the (correct) final output. fp32 router (ttnn fp32 matmul currently misbehaves) would tighten
    these further — Phase 6 item.

## 8c. Phase 6 optimizations (device-resident, correctness preserved)
Verified: forward/generation path has ZERO ttnn host fallbacks (`tests/run_fallback_audit.py`,
throw_exception_on_fallback + fast_runtime off → all device-ok).
1. **Build-once / seq-agnostic**: model loads weights to device once and serves any seq length;
   CCA caches per-seq cos/sin/causal-mask (`_seq`/`_seq_pos`). (was: 17GB reload per token.)
2. **Tile-native seq shift**: `_shift_seq` uses tile slice+concat (no row-major round-trip).
3. **Device argmax + last-token-only lm_head** for greedy (only a token id leaves device).
4. **Incremental decode** (`tt/cache.py` ZayaCache + CCA `decode_forward`): per ATT layer caches
   conv_state(last-2 qk), prev_hs, and roped KV (grown by concat); prefill populates it. Validated
   token-exact vs full recompute (`tests/run_decode_consistency.py`: prefill+3 steps ALL PASS).
5. **Sparse top-1 MoE decode** (`ZayaMoEBlock.decode_forward`): runs ONLY the chosen expert; only the
   1 expert index leaves device (route-prob weighting stays on device). MoD skip == identity. Token-exact.
6. **Program cache** enabled (`device.enable_program_cache()`) — kernels compiled once, reused per step.
7. **Batched experts**: all 16 experts run as ONE batched matmul (weights stacked [16,2048,4096]/[16,2048,2048],
   gate-weighted-sum), ~13x fewer op dispatches than the per-expert loop, fully on-device. Token-exact (Phase 2 PCC unchanged).
8. **Dense > sparse for decode**: measured sparse top-1 (40 host argmax syncs/token) at 1389ms vs dense
   1404→ batched-dense **508ms**; the host syncs cost more than the extra tiny matmuls, so decode uses
   batched-dense (no host sync — also better for the no-host goal). Sparse kept as `decode_forward` option.

Decode latency progression (device 1, bf16, warm): naive full-recompute 3523 ms/tok → incremental(cache,
sparse) 1252 → +program-cache dense-loop 830 → **+batched experts 508 ms/tok (1.97 tok/s)** = ~7x over naive.
Token-exact at every step (`tests/run_decode_consistency.py`).

Still dispatch-bound (~5000 tiny op launches/token @ ~0.1ms each; compute itself is <1ms). **Next lever =
ttnn trace** (capture decode-step op graph once, replay → removes per-op host dispatch, expect ~10x). Trace
prerequisites: preallocated fixed-shape KV (ttnn.update_cache/fill_cache + masked attention over MAX, instead
of the growing concat) + persistent input/mask buffers + ttnn.begin/end_trace_capture/execute_trace. Also:
fuse per-head L2-norm / to_heads / rope (currently slice-heavy). Multi-card TP = Phase 7.

## 8. Reference reproduction
- venv: `/home/yito/work/zaya-ref-venv` (torch CPU, `transformers @ git+.../Zyphra/transformers@zaya`).
- weights: `HF_HOME=/home/yito/work/hf_cache` (model cached, ~17 GB).
- dump goldens: `HF_HOME=/home/yito/work/hf_cache /home/yito/work/zaya-ref-venv/bin/python reference/dump_golden.py`
- still TODO for Phase 1+: decode-path (use_cache) golden with ZayaDynamicCache (conv_states + prev_hs).
