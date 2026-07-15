# VibeVoice speech-frame decode — device-perf baseline + optimization plan

Box: Blackhole P150 (110 worker cores, 8 DRAM banks). Branch `ign/vibevoice1.5_exps`.
Capture: `VV_TRACE_SEGMENT=0 VV_PROFILE_SPEECH_FRAME=2 VV_PROFILE_SPEECH_FRAME_EXIT=1`,
`demo 1p_CH2EN --max_new_tokens 32` (prefill 478, profiled the 2nd speech frame).
Report: `lm/speech_frame_baseline.txt`. Bucketer: `scratchpad/bucket4.py`.

## Regime
Eager op stream is **host/dispatch-bound** (op-to-op gaps 60–1300 µs ≫ device kernels 1–65 µs).
The deployable path is the **whole-segment fused trace** (`VV_TRACE_SEGMENT=1`, ~4.4× on decode)
which hides dispatch, so **device-kernel time == wall time under trace**. Therefore the
optimization target is *device-kernel time per op*, and this eager report is the right instrument.
Diffusion head + tokenizers have NO trace-specific code path, so their eager ops ship verbatim.
The LM has a traced variant (`_transformer_layer_traced`) but it uses the **same 1-core
`ttnn.rms_norm` and `nlp_create_qkv_heads`** — so the norm/head inefficiencies below are real
in both paths.

## BASELINE — one speech frame = 41.06 ms device time (3122 ops)

| Phase | Device time | Share | What runs |
|---|---:|---:|---|
| **post** (acoustic decode + semantic encode + 2 connectors) | 12.23 ms | 29.8% | 68 convs + 107 RMSNorm + tokenizer FFN |
| **pos_lm** (28-layer Qwen2 decode + lm_head) | 10.21 ms | 24.9% | matmuls + 56 RMSNorm + heads + lm_head |
| **diffusion** (diffusion head, CFG batch-2 × 10 DPM steps) | 9.60 ms | 23.4% | 4608 matmuls + adaLN + 50 RMSNorm |
| **neg_lm** (28-layer Qwen2 decode, no lm_head) | 9.03 ms | 22.0% | matmuls + 57 RMSNorm + heads |

Remarkably balanced — no single phase dominates; wins must come from cross-cutting op classes.

### Op-class budget across the whole frame
| Op class | Device time | Share | Note |
|---|---:|---:|---|
| Matmul (all) | ~20.5 ms | 50% | LM FFN+attn, diffusion 4608, tokenizer FFN, lm_head |
| **RMSNorm (LayerNormDeviceOperation)** | **5.65 ms** | **13.7%** | **270 ops, all on 1 core @ ~24 µs** |
| Conv2d | 4.69 ms | 11.4% | post only; 19 convs >100 µs = 3.64 ms; K=14336 @ 189 µs, **6.6% FLOPs** |
| NlpCreateHeads + NLPConcatHeads | 2.58 ms | 6.3% | LM attention, 1 core |
| BinaryNg (elementwise) | 1.77 ms | 4.3% | mostly diffusion adaLN/CFG glue |
| Unary (silu) | 1.23 ms | 3.0% | |
| SdpaDecode | 0.93 ms | 2.3% | 17 µs × 56, fine |
| RoPE / I2S / Slice / Untilize / Concat | ~3 ms | 7% | glue |

### Sharp findings
1. **RMSNorm runs on 1 core everywhere** — 24 µs × 270 = 5.65 ms (13.7% of the frame). Highest-leverage cross-cutting win.
2. **Diffusion adaLN matmuls run FP32×BF16→FP32 @ HiFi4** — 60 ops, 2.21 ms, only ~21% FLOPs util. Wasteful precision.
3. **Post convs are data-movement bound** — K=14336 convs at 189 µs and only 6.6% FLOPs; 66 cores. Plus 1.5 ms of I2S/Untilize/Concat glue around them.
4. One diffusion 4608 matmul is tagged **SLOW** (left on auto program config).
5. Tokenizer Block1D FFN matmuls (8192×2048, 2048×8192) = 2.1 ms, likely auto config.
6. LM FFN gate/up (4.49 ms) at 67% DRAM BW (headroom); FFN down (3.66 ms) at 82% (near ceiling); lm_head 1.23 ms bf16.

## FINDINGS FROM EXECUTION (updates the plan below)

**Regime clarified:** the frame is ~3122 tiny ops (batch-1 decode / batch-2 diffusion). Nearly every
op is **fixed-overhead-bound** (tensors are 1 tile tall). Matmul configs, conv layout, and precision
were all already swept by the prior session (see code comments + git reverts). So per-op tuning and
precision are largely exhausted; the levers are **op-count/pipelining**, not per-op knobs.

**P1 (sharded RMSNorm) — REJECTED (measured).** Width-sharding the `[32,1536]` decode norm is
*slower*: 1-core 50 µs wall vs sharded 68–120 µs, and 140–216 µs with reshards. A 1-tile-tall tensor
can't parallelize the row reduction; the ~24 µs is fixed overhead, not DRAM-BW-on-1-core. Same logic
rules out sharding the equally-tiny head-create/concat ops. `scratchpad/norm_iso.py`.

**P2 (diffusion fp32→bf16) — DOWNGRADED.** The diffusion head runs fp32 deliberately; git log shows
bf16/HiFi4 diffusion changes were **reverted** for quality. High risk vs the hard no-degradation gate.

**NEW P0 — CFG-batched LM (validated premise, ~9 ms / 22% of frame, numerically exact).**
neg-LM (9.0 ms) + pos-LM (10.2 ms) run as two separate batch-1 forwards. Every LM matmul/norm/head
pads M=1→32 tiles, so a **row-stacked batch-2 forward (`[1,1,2,hidden]`, fuse_batch=True) costs the
SAME device time as batch-1** — measured: down 66.7µs(M1)==66.8µs(M2), gate 153≈144µs;
SDPA-decode runs batch-2 with per-batch `cur_pos`. So the 2nd LM stream is ~free.
*Complication:* neg-LM and pos-LM are offset by diffusion+post within a frame; fusing them requires
**pipelining the negative stream one frame ahead** (batch pos-LM(i) with neg-LM(i+1); both inputs are
ready mid-frame i), a dual-batch KV cache (row0=pos, row1=neg), per-batch RoPE (pos=absolute,
neg=segment-relative), per-batch cur_pos, and segment-boundary/first-frame special-casing. Interacts
with the whole-segment trace. Substantial refactor, but the highest-value lever by far and
zero quality risk (batching doesn't change math). `scratchpad/batch2_iso.py`.

## POST-WIRING MATMUL/GLUE ANALYSIS (exp5)
- **FFN-down decode-config fix (LANDED, −2.36 ms):** the batched forward's S=2 disabled the swept
  `_W2_DECODE_PROGCFG` (S==1 gate) → down_proj ran auto (142 µs, SLOW, 199 GB/s). Forcing decode=True
  (row-stacked M=2 = 1 tile) → 65 µs. Frame 34.14 → 31.79 ms; equiv PCC 0.9996 → 1.0.
- **Other matmuls are at/near their ceilings** (report tags, exp5): lm_head 1536x151936 75% DRAM BW;
  LM FFN down/gate 65-82%; tokenizer FFN down 8192x2048 **84%** (near ceiling), up 2048x8192 68% (bf8_b);
  diffusion 4608 65-74%. All prior-swept MultiCast1D. The report's blanket "try DRAM-sharded" would
  need per-weight DRAM-shard relayout for marginal gain over the tuned configs — not worth it now.
- **Post-phase glue = ~2.0 ms (16% of post):** InterleavedToSharded 0.84 ms ×69 (conv2d input shard,
  the expensive direction — S2I is only 0.08 ms), UntilizeWithUnpadding 0.44, Concat 0.25 (streaming
  cache), Slice/Copy/Halo small. Mostly intrinsic to ttnn.conv2d's internal shard/halo + the
  prior-tuned ROW_MAJOR streaming concat. A probe worth trying: L1-interleaved conv activations so the
  I2S is L1→L1 (not DRAM→L1) — numerically free, ~0.4 ms upside, but bounded by L1 fit. Marginal.
- **L1 vs DRAM I/O:** the LM decode already keeps residual/act in L1 (res_mc); the big conv activations
  can't go blanket-L1 interleaved (size), but selective L1 on the conv chain is the glue probe above.

## Campaign total: 41.06 → 31.79 ms device time (−22.6%), all PCC-gated
diffusion fusions −0.55, CFG-batched LM −6.73, FFN-down config −2.36.

## INCREMENTAL WINS LANDED (numerically-exact, low risk)
User chose "low-risk incremental wins first" over the big CFG-batched-LM refactor.

| Change | File | Effect | Gate |
|---|---|---|---|
| `silu(c)` computed once/step (was 5×) | ttnn_diffusion_head.py | −40 Unary ops | diffusion PCC ✓ |
| SwiGLU gate fused `activation="silu"` | ttnn_diffusion_head.py | −40 Unary ops | diffusion PCC ✓ |
| `ones_like`+add → scalar `+1.0` | ttnn_diffusion_head.py | −~50 alloc ops | diffusion PCC ✓ |
| **t_emb precomputed once** (fixed timesteps) | head + scheduler + generator | removes timestep embedder (sin/cos/concat + 2 matmuls) from every step; eager + traced paths | `test_diffusion_tembs_equiv.py` PCC **1.0** |

exp1 (first 3 fusions): frame 41.06 → 40.88 ms (Unary −0.27 ms; rest run-to-run noise ±0.05).
**exp2 (all 4 changes): frame 41.06 → 40.52 ms (−0.55 ms, −160 ops).** Matmul −0.245 (t_emb's
2 MLP matmuls/step removed), Unary −0.332 (silu fusions + sin/cos removed). Demo runs clean.
**Note:** `test_e2e_generate_pcc.py` cannot run here — pre-existing HF version drift in the *reference*
(`_prepare_generation_config` arity) breaks the golden path; unrelated to these TT changes. Functional
gate is the demo `generate` run (profile run doubles as it) + component PCC tests.
**adaLN weight-merge — TRIED, REGRESSED, REVERTED (exp3).** Merging the 4 head + final adaLN
matmuls (same input silu(c)) into one `1536x21504` matmul: 40.52 → 42.29 ms (**+1.77 ms**). The merged
matmul on auto config is 347 µs ×10 = 3.48 ms vs the tuned separate ones at 2.0 ms, plus +0.3 ms extra
slices. The per-layer adaLN matmuls are already well-tuned (`_MM_ADALN`/`_MM_FADALN`); merging loses the
tuned config and adds slice overhead. Numerically exact but a perf loss — reverted. This exhausts the
low-risk incremental vein.

**Lesson:** micro-fusion of 2–3 µs ops yields ~0.3 ms/batch — real but near the noise floor; and
fusing already-tuned matmuls loses more (config) than it saves (launch overhead). **Net banked:
41.06 → 40.52 ms (−0.55 ms), all numerically exact.** The real prize remains the CFG-batched LM (~9 ms).

## CFG-BATCHED LM — feasibility VALIDATED (scratchpad/cfg_sdpa_iso.py)
The two make-or-break primitives were validated in isolation:
- **Batched SDPA-decode** (combined cache `[2,n_kv,maxS,hd]`, per-batch `cur_pos_tensor=[p_pos,p_neg]`)
  == two separate batch-1 decodes: **PCC 1.000000** both rows.
- **Per-row RoPE** (different position per row, `[1,1,2,hd]` cos/sin via `_apply_rope_ttnn`):
  **PCC 0.999996** both rows.
- Matmul cost premise (M=1≡M=2 row-stacked, fuse_batch): validated earlier (scratchpad/batch2_iso.py).

### STATUS: WIRED into the eager path + measured (exp4)
`generate()` eager TT path (`VV_CFG_BATCHED=1`, default; traced + reference paths unchanged) now
runs the CFG streams as one row-stacked batch-2 forward with the combined cache, pipelining the
negative stream one frame ahead. **Speech frame 40.88 → 34.14 ms device time (−6.73 ms, −515 ops);
from the original baseline 41.06 → 34.14 ms (−16.8%).** Breakdown: Matmul −2.89, LayerNorm −1.38,
Nlp{Create,Concat}Heads −1.29, SDPA −0.38 (56→28 batch-2), RoPE −0.41; UpdateKVCache→PagedUpdateCache.
**Correctness:** frame-0 audio PCC **0.9997** (old vs new, seed 0; gate 0.90), test_lm_pcc intact
(prefill 0.995 / decode 0.9997), test_cfg_batched_lm_equiv + _prefill_equiv PASS. `VV_CFG_BATCHED=0`
falls back to the separate-cache path.
REMAINING: wire the whole-segment trace (deployed path) to the batched forward.

### batched decode CORE (3de97d74f4f)
`alloc_kv_cache_batched2`, `_rope_rows_from_pos_int2`, `_attention_decode_batched2`,
`_transformer_layer_batched2`, `forward_decode_batched2` — all in ttnn_vibevoice_lm.py, inert
until wired. `test_cfg_batched_lm_equiv.py`: batched == 2 separate decodes (pos_hidden 0.9996,
neg_hidden 0.9997, pos_logits 0.9995, token match 1.0).

### REMAINING: AR-loop integration (design locked)
- **Prefill → combined-cache row 0:** pass the `[2,…]` combined cache to the existing
  `prefill_embeds`; its fp32 prefill writes `fill_cache(batch_idx=0)` and reads `slice[0:1]`
  (B=1) → populates row 0 only. Works as-is (verify).
- **Neg reset → combined-cache row 1 (the trick):** reuse `forward_decode_batched2` itself —
  row0 = dummy embed at a *scratch* position (`maxS-32`, never read since pos_valid ≪ maxS →
  harmless write), row1 = speech_start embed at pos 0. Returns neg_start_hidden AND writes row 1
  pos 0. No new prefill/batch_idx surgery needed.
- **AR loop (pipeline neg one frame ahead):** keep `neg_hidden_pending` (init = neg_start_hidden).
  Per diffusion frame i: diffusion uses `neg_hidden_pending`; then ONE batched forward
  (row0 = pending_embeds @ pos_pos=prefill_len+step, row1 = speech_diffusion_id @ neg_pos) →
  logits+step_hidden (row0), `neg_hidden_pending` ← hidden2 row1; neg_pos++.
- **Boundaries:** segment start re-runs the neg reset; speech_end → the speculative neg row is
  discarded (reset wipes it). Non-diffusion tokens keep the single-stream pos decode.
- **Trace:** mirror in `_run_segment_frame_traced` (`_sf_*` persistent buffers, dev-RoPE rows via
  a batch-2 `_rope_rows_from_pos` gather; positions self-advance with `plus_one`).
Expected ~9 ms/frame (neg-LM ~free). Gate: test_lm_pcc + demo sanity + re-profile.

Integration steps (build additively so the working model is never broken):
1. `alloc_kv_cache_batched2` → combined cache `[2,n_kv,maxS,hd]` (row0=pos, row1=neg).
2. batched decode attention: row-stacked QKV `[1,1,2,2048]` → nlp_create_qkv_heads → permute to
   batch-2 → per-row RoPE (2 gathered position rows) → paged_update_cache(update_idxs=[p_pos,p_neg])
   → batched SDPA-decode(cur_pos_tensor) → concat heads → wo (row-stacked).
3. AR loop: pipeline neg one frame ahead — batch pos-LM(i) with neg-LM(i+1) (both inputs ready
   mid-frame i), assemble row-stacked input, one batched 28-layer forward, split outputs
   (pos→logits+hidden, neg→hidden cached for next frame). lm_head on the pos row only.
4. Segment-boundary + first-frame special-casing; wire into the whole-segment trace.
Expected ~9 ms/frame (neg-LM becomes ~free). Numerically exact → zero quality risk.

## OPTIMIZATION PLAN (prioritized by expected device-time win × confidence)

### Tier 1 — cross-cutting, highest leverage
**P1. Sharded RMSNorm (all 4 phases).** 5.65 ms @ 1 core → target ~5–8 µs/op via width-sharded
hidden dim + distributed reduction (tt_transformers decode RMSNorm pattern). Keep HiFi2 +
fp32_dest_acc_en=True. **Est. save 2–3 ms.** Reductions compound over 28 layers → gate at
FULL-model PCC: `test_lm_pcc` (prefill+decode), `test_diffusion_head_pcc`, tokenizer PCC tests.
Ref: normalization.md. Validate in isolation on [1,32,1536] first (tiny tensor — confirm sharding
overhead doesn't eat the win).

### Tier 2 — per-phase big buckets
**P2. Diffusion adaLN FP32→BF16.** 2.21 ms of fp32 matmuls (K=1536, N∈{1536,3072,4608}). Fold the
`silu(cond)` output to bf16 before the adaLN projection. **Est. save 0.8–1 ms.** Low effort;
gate `test_diffusion_head_pcc.py`. Ref: matmul-and-mlp.md + foundations §5.

**P3. Post convs (K=14336, 3.22 ms @ 6.6% FLOPs).** Data-movement bound. Investigate the
polyphase-transpose reshape + ROW_MAJOR↔TILE conversions and streaming-cache slice/copy around
them; tune act_block_h / sharding; keep tensors sharded across the conv chain to kill the
1.5 ms of I2S/Untilize/Concat glue. **Est. save 1–1.5 ms.** Medium effort. Gate acoustic+semantic
tokenizer PCC. Ref: data-movement-and-fusion.md conv-codec section.

**P4. Tokenizer Block1D FFN matmuls (2.1 ms).** Confirm on auto; sweep MultiCast1D config +
bf8_b weights. **Est. save 0.5–0.8 ms.** Gate tokenizer PCC. Ref: matmul-and-mlp.md.

### Tier 3 — LM refinements
**P5. Sharded decode head ops.** Replace eager `nlp_create_qkv_heads` + `nlp_concat_heads`
(2.58 ms, 1 core) with the `_decode` sharded variants (multi-core). **Est. save 1–1.5 ms.**
Medium (numerics re-validation, height-sharded L1 plumbing). Ref: decode-prefill-multidevice.md.

**P6. lm_head bf8_b weight.** 1.23 ms bf16 → ~0.7 ms. **Est. save 0.5 ms** (pos_lm only).
PCC-gate on argmax token-match (logits→greedy). Ref: matmul-and-mlp.md.

**P7. LM FFN gate/up DRAM-sharded program config.** 4.49 ms at 67% DRAM BW → push toward ~85%.
The report itself advises `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig`. **Est. save
0.5–0.7 ms.** Precision-neutral; sweep per methodology.md.

### Tier 4 — fusion / glue
**P8. Fuse diffusion adaLN modulation.** BinaryNg 1.23 + Unary 0.55 + Slice 0.33 ≈ 2.1 ms:
avoid the 3× slice to split shift/scale/gate; fuse `(1+scale)*norm + shift` and the gated
residual. **Est. save 0.5–0.8 ms.** Ref: data-movement-and-fusion.md fusion section.

## Realistic target
41.06 ms → **~30–33 ms** device time per frame (~20–27% faster traced frame), P1 (sharded
RMSNorm) being the single biggest lever. Loop: apply → PCC-gate → re-profile → re-bucket; the
next bucket promotes as each is fixed. Matmul BW ceilings (FFN down @82%) are near-irreducible.

## Verification gates (run after each change)
- LM: `tests/pcc/test_lm_pcc.py` (prefill≥0.996, decode≥0.999).
- Diffusion: `tests/pcc/test_diffusion_head_pcc.py`.
- Tokenizers: `tests/pcc/test_acoustic_tokenizer_pcc.py`, `test_semantic_tokenizer_pcc.py`, `test_connector_pcc.py`.
- Cross-module: `tests/pcc/test_e2e_generate_pcc.py` (frame-0 PCC ≥ 0.90 forced-token gate).
- Re-profile with the same command; compare `lm/speech_frame_expN.txt`.
