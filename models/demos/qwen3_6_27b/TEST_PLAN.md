# Qwen3.6-27B — TDD Test Plan (Test-First Relay)

> **Iron rule**: production code for any block in this plan **MUST** be preceded by a failing test that exercises that block's behavior. Watch it fail for the expected reason. Then write minimal production code to pass it. If you didn't watch it fail, you don't know it tests the right thing.

> Per-test spec: each entry has (a) **what** it tests, (b) **input** spec, (c) **expected RED** — the failure mode you must observe before writing production code, (d) **pass criterion** for GREEN, (e) explicit **dependencies** on earlier tests.

> Test path convention: `models/demos/qwen3_6_27b/tests/` for model-level; per-block kernels live under `tt_transformers/tt/tests/` or `models/experimental/.../tests/` as appropriate.

> **PCC threshold: > 0.99 unless otherwise noted**. Mandatory per CLAUDE.md.

---

## Phase 1 — Reference (PyTorch, CPU)

Standalone reference implementations + HF parity checks. Source: cherry-picked from `ssinghal/qwen3.5-27B/models/demos/qwen3_5/reference/`. New tests verify HF parity at our exact shapes.

### T1.1 — DeltaNet single-step decode == single-step from chunked prefill
**What:** the recurrent and chunked DeltaNet paths produce identical output at every step.
**Input:** `B=1, T=64, n_v=48, n_k=16, d_k=d_v=128, conv_k=4`. Random bf16 weights, fixed seed.
**RED:** initial reference has not been verified at our exact head counts; expect either a shape mismatch (n_v=48 vs default n_v=4 in the upstream FLA tests) or a > 1e-2 deviation between paths.
**GREEN:** torch.allclose(recurrent_out, chunked_out, atol=1e-3, rtol=1e-3) at every t ∈ [0, T).
**Depends:** none — first test of the relay.
**File:** `tests/reference/test_deltanet_consistency.py::test_recurrent_matches_chunked`

### T1.2 — DeltaNet vs HF reference
**What:** our standalone DeltaNet matches HF `Qwen3_5LinearAttention` from `transformers >= 4.57`.
**Input:** real Qwen3.6-27B layer-0 `linear_attn` weights via `convert_hf_to_meta_qwen3_5`. `B=1, T ∈ {1, 64, 1024}`.
**RED:** without the `wq_gate` split or with wrong K/V head fan-out, the output diverges at random in the first head; PCC ~0.5–0.8.
**GREEN:** PCC > 0.99 vs HF at all T.
**Depends:** T1.1.
**File:** `tests/reference/test_deltanet_hf_parity.py`

### T1.3 — Gated Attention vs HF reference (prefill)
**What:** Q+gate split, QK-norm, partial RoPE (rotary_dim=64), and output-gate produce HF-matching output.
**Input:** real layer-3 (full-attn) weights, `B=1, S=512`, attention mask = causal+image-bidirectional with synthetic `token_type_ids` marking positions 100–199 as vision.
**RED:** without `attn_output_gate` enabled, PCC ~0.6 (the `sigmoid(gate) * attn_out` factor is missing); without partial RoPE, PCC ~0.8 (RoPE applied to all 256 head dims instead of just 64).
**GREEN:** PCC > 0.99 vs HF.
**Depends:** none.
**File:** `tests/reference/test_gated_attention_hf_parity.py`

### T1.4 — Gated Attention prefill == incremental KV-cache decode
**What:** running prefill once vs running 1 prefill + (S-1) single-step decodes yields the same final hidden state.
**Input:** `B=1, S=128`, real weights.
**RED:** without K/V repeat into the 24-head Q space (GQA group=6), decode position interpretation breaks; PCC < 0.9.
**GREEN:** allclose(prefill_hidden, incremental_hidden, atol=1e-3).
**Depends:** T1.3.
**File:** `tests/reference/test_full_attention_kv_consistency.py`

### T1.5 — MRoPE [11,11,10] interleaved sections
**What:** the multimodal RoPE applied to (T,H,W) positions matches HF's `Qwen3VLRotaryEmbedding`.
**Input:** synthetic (T=4, H=8, W=8) image grid; head_dim=256; rotary_dim=64.
**RED:** flat causal RoPE (the user's branch's current behavior) gives a measurable angular drift on the (H, W) axes — PCC <0.97 even for short sequences.
**GREEN:** allclose(our_mrope_freqs, HF_freqs, atol=1e-5) over the full position range.
**Depends:** none.
**File:** `tests/reference/test_mrope.py`

### T1.6 — Vision encoder block vs HF
**What:** one ViT block (attention + MLP + 2 LayerNorms) matches HF on a single image.
**Input:** real `model.visual.blocks.0` weights, `pixel_values` from `Qwen2VLImageProcessorFast` on a 224×224 PIL image.
**RED:** the fused QKV split for `attn.qkv.weight [3456, 1152]` uses dim 0 splits, but if you treat 3456 as `3 × 16 × 72` head-padded instead of `3 × 1152`, K and V swap. PCC ~0.5.
**GREEN:** PCC > 0.99 on block output.
**Depends:** none.
**File:** `tests/reference/test_vision_block.py`

### T1.7 — Patch merger vs HF
**What:** `[4608] → [4608] → [5120]` via two-layer MLP matches HF.
**Input:** real `model.visual.merger.*` weights, random `[B=1, N_patches=64, 4608]` input.
**RED:** missing the `merger.norm` LayerNorm gives PCC ~0.6.
**GREEN:** PCC > 0.99.
**Depends:** none.
**File:** `tests/reference/test_merger.py`

### T1.8 — MTP block vs HF
**What:** the MTP single transformer block + pre-FC concat-norms + `fc` projection match HF.
**Input:** real `mtp.*` weights, a synthetic hidden state and next-token embedding pair.
**RED:** missing the concat-then-fc step (`[h_norm ‖ emb_norm] → H`) gives wrong shape; PCC undefined.
**GREEN:** PCC > 0.99.
**Depends:** T1.3 (MTP block shares the gated-attention shape).
**File:** `tests/reference/test_mtp.py`

### T1.9 — Hybrid 4-layer slice (3 DeltaNet + 1 Gated-Attn) vs HF
**What:** assemble layers 0..3 of Qwen3.6-27B as defined by `layer_types` and run forward.
**Input:** real layer-0..3 weights, B=1, S=256, causal mask.
**RED:** layer dispatch wrong (running layer 3 as DeltaNet) gives PCC ~0.4.
**GREEN:** PCC > 0.99 vs HF hidden state after layer 3.
**Depends:** T1.2, T1.3.
**File:** `tests/reference/test_hybrid_4layer.py`

### T1.10 — Full 64-layer text-only decoder vs HF
**What:** full text path on a short prompt (no image), greedy next-token logits comparison.
**Input:** real Qwen3.6-27B weights, prompt = "The capital of France is".
**RED:** missing the final pre-LM-head RMSNorm gives PCC ~0.95 on hidden but next-token argmax mismatches.
**GREEN:** top-5 token match with HF for first 16 tokens; PCC > 0.99 on logits.
**Depends:** T1.9.
**File:** `tests/reference/test_full_decoder.py`

### T1.11 — End-to-end VLM forward vs HF (image)
**What:** image + text prompt → first token argmax matches HF.
**Input:** real weights, sample image from `qwen3_vl/test_images/`, prompt "What is in this image?".
**RED:** wrong `image_token_id` placement or missing image-bidirectional mask drops PCC below 0.9.
**GREEN:** top-1 token match for first 8 tokens; PCC > 0.99 on logits.
**Depends:** T1.6, T1.7, T1.10.
**File:** `tests/reference/test_vlm_image.py`

### T1.12 — End-to-end VLM forward vs HF (video)
**What:** video + text → first token matches HF.
**Input:** sample video (4-8 frames), prompt "Describe what happens".
**RED:** frame markers (`vision_start`, `vision_end`) not marked as vision → ~30pp accuracy regression (per Molmo2 prior).
**GREEN:** top-1 token match for first 8 tokens.
**Depends:** T1.11.
**File:** `tests/reference/test_vlm_video.py`

---

## Phase 2 — TTNN per-block (single chip, BH)

Per CLAUDE.md: PCC > 0.99 mandatory. Single chip first; mesh in Phase 3.

### T2.1 — DeltaNet recurrent kernel PCC at Qwen3.6 shapes
**What:** `ign/deltnet_kernel_fusion`'s `recurrent_gated_delta_rule_ttnn` matches the reference at our shapes.
**Input:** weights from `convert_hf_to_meta_qwen3_5`, `B=1, T=1`, recurrent state initialized to zero, T-step loop from t=0..63.
**RED:** kernel was validated at K=128, V=256, 4 heads — at our K=V=128, 16/48 heads the matmul program configs may need re-tuning; expect either a hang (bad subblock size) or PCC ~0.95 from BFP8 vs FP32 mismatch in the SSM step.
**GREEN:** PCC > 0.99 vs T1.2 reference at every step.
**Depends:** T1.2.
**File:** `tests/ttnn/test_deltanet_recurrent_kernel.py`

### T2.2 — DeltaNet chunked kernel PCC at Qwen3.6 shapes
**What:** the chunked-prefill path matches the reference.
**Input:** same weights, `B=1, T ∈ {64, 256, 1024, 4096, 32768}`.
**RED:** the Neumann series for `(I-M)^-1` uses 5 iterations at cs=64; if numerical drift accumulates over T=32K, PCC may dip below 0.99.
**GREEN:** PCC > 0.99 at every T.
**Depends:** T1.2, T2.1.
**File:** `tests/ttnn/test_deltanet_chunked_kernel.py`

### T2.3 — Gated attention (single chip) PCC
**What:** lift the user's branch attention; PCC vs reference.
**Input:** real layer-3 weights, `B=1, S=128`, causal+image-bidirectional mask.
**RED:** the user-branch attention's WH compute kernel config is `WormholeComputeKernelConfig` — on BH, expect either a runtime error or PCC drop from wrong fidelity defaults.
**GREEN:** PCC > 0.99 vs T1.3.
**Depends:** T1.3.
**File:** `tests/ttnn/test_gated_attention_single_chip.py`

### T2.4 — Gated attention KV-cache decode step PCC
**What:** paged_update_cache + sdpa_decode against incremental reference.
**Input:** `B=1`, prefill S=128, then 16 decode steps.
**RED:** without partial-RoPE applied identically in prefill and decode paths (one off-by-one in rotary_dim slicing), per-step PCC ~0.95 with the gap growing over steps.
**GREEN:** PCC > 0.99 at every step.
**Depends:** T2.3, T1.4.
**File:** `tests/ttnn/test_gated_attention_decode.py`

### T2.5 — RMSNorm + DistributedNorm single chip
**What:** `tt_transformers/tt/distributed_norm.py` matches torch RMSNorm.
**Input:** H=5120, real layer-0 `input_layernorm` weight.
**RED:** already covered by tt_transformers; expect PASS — used as smoke test.
**GREEN:** PCC > 0.999.
**Depends:** none.
**File:** existing in `tt_transformers/tt/tests/`.

### T2.6 — MLP single chip
**What:** SwiGLU MLP with I=17408 matches reference.
**Input:** real layer-0 MLP weights.
**RED:** `intermediate=17408` is not a multiple of 24×32 — if `nearest_multiple` padding is wrong, hidden dim mismatch.
**GREEN:** PCC > 0.99.
**Depends:** none.
**File:** `tests/ttnn/test_mlp.py`

### T2.7 — Vision attention single chip
**What:** ViT attention (no RoPE, fused QKV with bias) matches reference.
**Input:** real `model.visual.blocks.0.attn.*` weights.
**RED:** the QKV bias is non-zero (`attn.qkv.bias` present); if biases aren't loaded, PCC ~0.85.
**GREEN:** PCC > 0.99.
**Depends:** T1.6.
**File:** `tests/ttnn/test_vision_attention.py`

### T2.8 — Vision block single chip
**What:** attention + MLP + 2× LayerNorm.
**Input:** real `model.visual.blocks.0.*`.
**RED:** Qwen3-VL ViT uses LayerNorm (with bias), not RMSNorm — must use `vision_layernorm.py`, not `rmsnorm.py`. PCC < 0.5 if confused.
**GREEN:** PCC > 0.99.
**Depends:** T2.7.
**File:** `tests/ttnn/test_vision_block.py`

### T2.9 — Patch merger single chip
**What:** `[4608]→[4608]→[5120]` with LayerNorm matches reference.
**Input:** real merger weights.
**RED:** merger uses GELU activation (Qwen3-VL); if SiLU is used (default), PCC ~0.7.
**GREEN:** PCC > 0.99.
**Depends:** T1.7.
**File:** `tests/ttnn/test_merger.py`

### T2.10 — MTP block single chip
**What:** the MTP transformer block (= gated attention layer) + 2 pre-fc norms + fc projection match reference.
**Input:** real `mtp.*` weights.
**RED:** if the MTP block's attention skips QK-norm (it shouldn't), PCC ~0.92.
**GREEN:** PCC > 0.99.
**Depends:** T2.3, T1.8.
**File:** `tests/ttnn/test_mtp.py`

---

## Phase 3 — TTNN mesh-sharded (BH GLX 8×4)

These tests require BH GLX. PCC requirements identical. Mesh tests must also verify no hang, no OOM.

### T3.1 — DeltaNet TP across rows (V-heads, K-heads sharded)
**What:** wrap T2.1's kernel with `ShardTensor2dMesh` sharding the 48 V-heads → 6/row, 16 K-heads → 2/row.
**Input:** real weights, `B=1, T=64`.
**RED:** the `out_proj` is row-parallel and needs ReduceScatter after the matmul; if missing, each row sees only its head slice → PCC < 0.5.
**GREEN:** PCC > 0.99 vs T1.2 reference.
**Depends:** T2.1.
**File:** `tests/ttnn/test_deltanet_mesh.py::test_recurrent_tp_rows`

### T3.2 — DeltaNet recurrent state sharded across rows
**What:** the SSM state `[B, 48, 128, 128]` lives sharded as `[B, 6, 128, 128]` per chip and persists across decode steps without drift.
**Input:** decode loop of 1024 steps from a single prefill.
**RED:** if state is replicated rather than sharded, OOM at higher batch; if sharded but not persistent across CCL calls, accumulator drift.
**GREEN:** step-1024 hidden state PCC > 0.99 vs reference.
**Depends:** T3.1.
**File:** `tests/ttnn/test_deltanet_mesh.py::test_state_persistence`

### T3.3 — DeltaNet chunked TP across rows
**What:** chunked prefill with sharded V/K-heads.
**Input:** `B=1, T ∈ {64, 4096, 32768}`.
**RED:** the Neumann series matmuls require the chunk to be local to one row — if cross-row CCL is needed inside the series, latency explodes.
**GREEN:** PCC > 0.99; latency within 2× of unsharded reference / 8.
**Depends:** T3.1.
**File:** `tests/ttnn/test_deltanet_mesh.py::test_chunked_tp`

### T3.4 — Gated attention TP across cols
**What:** 24 Q-heads sharded → 6/col; 4 KV-heads sharded → 1/col; H sharded → 1280/col.
**Input:** real layer-3 weights, `B=1, S=128`.
**RED:** the fused-AGM K-dim fix from `changh95` (commit fc90c3ac) targets exactly `n_heads*head_dim != dim` — without it, the all_gather_matmul dim mismatches and produces garbage. Expected RED before cherry-pick: PCC < 0.5.
**GREEN:** PCC > 0.99 with the fix applied.
**Depends:** T2.3.
**File:** `tests/ttnn/test_gated_attention_mesh.py`

### T3.5 — Gated attention paged_update_cache + sdpa_decode on mesh
**What:** KV cache replicated along rows (n_kv=1 per col, replicated 8×), decode step PCC.
**Input:** prefill S=512, 16 decode steps.
**RED:** if KV-head replication isn't wired (`changh95`'s n_kv-replication fix), all 8 rows write to the same cache slot → corruption after step 2.
**GREEN:** PCC > 0.99 at every step.
**Depends:** T3.4.
**File:** `tests/ttnn/test_gated_attention_mesh.py::test_decode_step`

### T3.6 — Distributed RMSNorm on mesh
**What:** existing `distributed_norm.py` path holds at our H=5120, sharded 1280/col.
**Input:** real layer-0 norm weights.
**RED:** already known-good; smoke test.
**GREEN:** PCC > 0.999.
**Depends:** none.

### T3.7 — MLP on mesh (intermediate sharded across rows)
**What:** I=17408 sharded → 2176/row, with AG-matmul (gate+up) and RS-matmul (down).
**Input:** real layer-0 MLP weights.
**RED:** `intermediate=17408` is not a multiple of 24×32 — without padding to 2304/row, sharded layout fails alignment.
**GREEN:** PCC > 0.99.
**Depends:** T2.6.
**File:** `tests/ttnn/test_mlp_mesh.py`

### T3.8 — Hybrid 4-layer mesh slice (3 DeltaNet + 1 Gated-Attn + MLPs + norms)
**What:** layer 0..3 wired with layer-type dispatch on mesh.
**Input:** real weights, `B=1, S=256`.
**RED:** without layer-type dispatch in `decoder.py`, layer 3 runs as DeltaNet → garbage.
**GREEN:** PCC > 0.99 vs T1.9.
**Depends:** T3.1, T3.4, T3.7.
**File:** `tests/ttnn/test_hybrid_4layer_mesh.py`

### T3.9 — Full 64-layer text decoder on mesh
**What:** full 64-layer forward with embedding + decoder + norm + LM head.
**Input:** real weights, prompt "The capital of France is" (no image).
**RED:** if LM head is replicated (5GB/chip), OOM at allocation; vocab-parallel LM head is required.
**GREEN:** PCC > 0.99 on logits; top-5 token match with HF for 16 tokens.
**Depends:** T3.8, T3.10.
**File:** `tests/ttnn/test_full_text_decoder.py`

### T3.10 — Vocab-parallel LM head on mesh
**What:** LM head sharded across 32 chips (vocab/32 ≈ 7760 per chip); reduce_scatter logits, AllGather final logits for argmax.
**Input:** hidden state from T3.9 layer 63, real LM head weights.
**RED:** `vocab=248320`, `32 chips`, `248320/32 = 7760` exactly — no padding needed; but if shard layout is wrong, argmax produces the wrong token.
**GREEN:** PCC > 0.999 vs replicated LM head reference.
**Depends:** T2.5 (norm).
**File:** `tests/ttnn/test_vocab_parallel_lm_head.py`

---

## Phase 4 — VLM integration

### T4.1 — ViT on mesh (replicated weights, data-parallel patches)
**What:** 27-block ViT forward; weights replicated, patch sequence data-parallel.
**Input:** real ViT weights + `pixel_values` from one image.
**RED:** the input pixel patches `[N_patches, 1536]` must shard along dim 0; if replicated, no parallelism benefit and CCL not exercised.
**GREEN:** PCC > 0.99 vs reference; latency proportional to N_patches / (8×4).
**Depends:** T2.8.
**File:** `tests/ttnn/test_vit_mesh.py`

### T4.2 — Patch merger on mesh (replicated, after AllGather)
**What:** AllGather ViT features along patches dim → run merger replicated → distribute to text path.
**Input:** ViT output from T4.1.
**RED:** if AllGather is skipped, merger sees only its shard → wrong output dim.
**GREEN:** PCC > 0.99 vs reference.
**Depends:** T4.1, T2.9.
**File:** `tests/ttnn/test_merger_mesh.py`

### T4.3 — Image-bidirectional mask construction
**What:** mask builder marks ALL of {248053, 248054, 248056, 248057} as vision; bidirectional inside image span, causal elsewhere.
**Input:** synthetic `input_ids` with a vision span at positions 10..200 including markers at 10, 200.
**RED:** if only `image_token_id` (248056) is marked, frame markers get causal-only attention → ~30pp accuracy drop in actual generation, but for unit-test it manifests as PCC < 0.97 on attention output.
**GREEN:** PCC > 0.99 against reference mask.
**Depends:** none.
**File:** `tests/ttnn/test_vlm_mask.py`

### T4.4 — Vision feature injection (replacement at image_token_id positions)
**What:** at `image_token_id` positions in `input_ids`, replace the embedded token with the corresponding row of merger output.
**Input:** synthetic sequence with 64 image tokens at known positions.
**RED:** off-by-one in position mapping → wrong image feature at each slot → PCC < 0.7 on next-layer hidden.
**GREEN:** PCC > 0.99 on first decoder layer output.
**Depends:** T4.2, T4.3.
**File:** `tests/ttnn/test_vision_injection.py`

### T4.5 — Full VLM forward with image (4-layer slice + ViT + merger)
**What:** end-to-end image input through ViT, merger, image injection, hybrid 4-layer decoder.
**Input:** real weights for ViT + layers 0..3, sample image.
**RED:** if MRoPE isn't passing per-position (T,H,W) coords to the gated-attention path, partial-RoPE indexes wrong.
**GREEN:** PCC > 0.99 on layer-3 hidden state vs reference.
**Depends:** T1.11, T3.8, T4.4.
**File:** `tests/ttnn/test_vlm_4layer.py`

### T4.6 — Full VLM forward end-to-end (image + full decoder + LM head)
**What:** image + prompt → logits.
**Input:** sample image, prompt "What is in this image?".
**RED:** common at this stage is the vision/text MRoPE position split being off by one for the text portion.
**GREEN:** top-5 token match with HF reference for first 8 tokens; PCC > 0.99.
**Depends:** T4.5, T3.9.
**File:** `tests/ttnn/test_vlm_e2e.py`

### T4.7 — Full VLM forward with video
**What:** same as T4.6 but with a 4-frame sampled video.
**Input:** sample video, prompt "Describe".
**RED:** frame timestamps used by MRoPE must match HF's `frames_indices` exactly; mismatch → wrong temporal positions.
**GREEN:** top-5 token match with HF for first 8 tokens.
**Depends:** T4.6, T1.12.
**File:** `tests/ttnn/test_vlm_video_e2e.py`

---

## Phase 5 — MTP integration

### T5.1 — MTP block on mesh (gated-attn shape)
**What:** the MTP block wired on mesh, identical layout to gated-attention.
**Input:** real `mtp.layers.0.*` weights.
**RED:** the MTP attention is inside `mtp.*` not `model.language_model.*`; weight loader must look up keys correctly. RED = KeyError or PCC ~0 (random weights).
**GREEN:** PCC > 0.99.
**Depends:** T3.4, T2.10.
**File:** `tests/ttnn/test_mtp_mesh.py`

### T5.2 — MTP fc + norms
**What:** the `[h‖e]→H` fc and the two pre-fc norms.
**Input:** outputs from T3.9 layer 63 and an embedded next token.
**RED:** missing the embedding norm gives a scale mismatch → MTP loss/agreement collapses.
**GREEN:** PCC > 0.99 on `fc` output.
**Depends:** T5.1.
**File:** `tests/ttnn/test_mtp_fc.py`

### T5.3 — MTP end-to-end agreement (speculative decode 2-token)
**What:** for 16 prompts, the MTP-proposed next-next token agrees with HF's MTP-proposed next-next token at acceptance rate ≥ HF baseline.
**Input:** 16 short prompts; greedy.
**RED:** if MTP shares lm_head with base (which it should), wrong sharing gives random token proposals.
**GREEN:** acceptance rate within 5pp of HF baseline.
**Depends:** T5.2, T3.9.
**File:** `tests/ttnn/test_mtp_speculative.py`

---

## Phase 6 — Performance + optimization

### T6.1 — Decode trace capture (no per-token Python overhead)
**What:** `ttnn.tracy` decode trace captures successfully; subsequent traced decodes use no host-side Python in the hot loop.
**Input:** prefill + 64 decode steps; capture trace at step 1, run steps 2..64 from trace.
**RED:** trace capture fails if any tensor is host-created inside the trace; `build_fused_chunked_delta_rule_constants` must run BEFORE trace capture.
**GREEN:** traced decode latency = untraced × (1 ± 0.05); no host Python in the per-step path (verified via Tracy).
**Depends:** T3.9.
**File:** `tests/perf/test_decode_trace.py`

### T6.2 — Tracy profile per block
**What:** generate Tracy profile and `run_block_profiles.sh` xlsx for every block.
**Input:** the per-block scripts in `tests/profile/`.
**RED:** missing scripts; output xlsx empty.
**GREEN:** xlsx contains rows for all 12 blocks; latency numbers logged in BRINGUP_LOG.md.
**Depends:** all of Phase 3.
**File:** `tests/perf/run_block_profiles.sh`

### T6.3 — Decode latency target
**What:** sustained tokens/sec for B=1 batch.
**Target:** **≥ 25 tok/s** (≈40 ms/tok). Initial budget: 64 layers × ~0.6 ms/layer.
**RED:** without trace, expect ~5-10 tok/s (Python overhead).
**GREEN:** ≥ 25 tok/s on BH GLX with prefill of 4K, B=1, sampling off.
**Depends:** T6.1.
**File:** `tests/perf/test_decode_perf.py`

### T6.4 — Prefill TTFT target
**What:** prefill time-to-first-token.
**Target:** **≤ 200 ms for S=512**, **≤ 2 s for S=8192**.
**RED:** chunked DeltaNet at cs=64 may bottleneck; expect 2-3× slower than target initially.
**GREEN:** within target bounds.
**Depends:** T3.3, T3.4.
**File:** `tests/perf/test_prefill_perf.py`

### T6.5 — Memory budget check
**What:** per-chip DRAM and L1 occupancy at decode steady-state.
**Target:** **< 85% DRAM**, **< 70% L1**.
**RED:** with replicated LM head, expect DRAM ~80% even before activations.
**GREEN:** numbers logged in BRINGUP_LOG.md; both under threshold.
**Depends:** T3.9, T3.10.
**File:** `tests/perf/test_memory_budget.py`

### T6.6 — Long-context (S=64K) prefill PCC
**What:** at max practical context, no numerical drift.
**Input:** long passage prompt, S=65536.
**RED:** at large T, both the `bfloat4_b` attention mask and the chunked DeltaNet Neumann series can accumulate error.
**GREEN:** PCC > 0.99 vs CPU reference run.
**Depends:** T3.9.
**File:** `tests/perf/test_long_context.py`

---

## Phase 7 — Server (tt-inference-server)

Per CLAUDE.md completion criteria: server PCC must be within 2-3 pp of demo PCC; 105-video suite or equivalent.

### T7.1 — Generator + KV cache lifecycle
**What:** `tt/generator.py` (lifted/adapted from `ssinghal/qwen3.5-27B`) supports prefill, decode loop, KV-cache reuse, and reset between requests.
**Input:** sequence of 4 requests with different prompts.
**RED:** if recurrent state isn't reset between requests, request 2 onward leaks state from request 1 → garbage tokens.
**GREEN:** each request matches its independent run.
**Depends:** T3.9.
**File:** `tests/server/test_generator.py`

### T7.2 — `generator_vllm.py` registration with tt-inference-server
**What:** `Qwen3.6-27B` registered in `tt-inference-server`; `vllm serve` starts up; `/v1/models` lists it.
**Input:** server start command.
**RED:** common: missing entry in `model_spec.py`, wrong dispatcher type.
**GREEN:** server runs, model listed.
**Depends:** T7.1.
**File:** `tests/server/test_vllm_registration.py`

### T7.3 — OpenAI API smoke test (chat completion)
**What:** chat completion request via OpenAI-compatible API returns sensible text.
**Input:** "What is the capital of France?" → expect "Paris" in response.
**RED:** generation works in demo but tokenizer mismatch in server gives gibberish.
**GREEN:** "Paris" in first 32 tokens of response.
**Depends:** T7.2.
**File:** `tests/server/test_openai_smoke.py`

### T7.4 — OpenAI API VLM (image)
**What:** image + question via vision API.
**Input:** image URL + "What is in this image?".
**RED:** vision backend not registered → image dropped, model sees text-only prompt.
**GREEN:** non-empty response that references image content.
**Depends:** T7.3, T4.6.
**File:** `tests/server/test_openai_image.py`

### T7.5 — OpenAI API VLM (video)
**What:** video URL + question.
**Input:** sample video URL.
**RED:** the model-specific vLLM video backend (see `/tt-inference-server` skill) must replicate HF frame sampling exactly. RED = wrong frame count or wrong timestamps.
**GREEN:** non-empty response that references video content.
**Depends:** T7.4, T4.7.
**File:** `tests/server/test_openai_video.py`

### T7.6 — Accuracy regression suite
**What:** run a 50-prompt text + 50-image + 5-video accuracy suite; compare server vs demo accuracy.
**Pass:** **server accuracy ≥ demo accuracy − 3 pp**.
**RED:** if server adds any sampling stochasticity that demo doesn't have, baseline shifts.
**GREEN:** both numbers logged in BRINGUP_LOG.md.
**Depends:** T7.3 through T7.5.
**File:** `tests/server/test_accuracy_suite.py`

### T7.7 — Server stability (no `tt-smi -r` needed between requests)
**What:** run 100 mixed requests sequentially; no device hang, no need to reset.
**Input:** 100-request mixed-mode workload.
**RED:** common at large scale: state leakage between requests, fabric link timeouts, KV cache slot fragmentation.
**GREEN:** 100/100 successful; server uptime > 24h in soak test.
**Depends:** T7.6.
**File:** `tests/server/test_stability.py`

---

## Test execution order (dependency-respecting)

```
Phase 1 (CPU, no device):
  T1.5 (MRoPE) ┐
  T1.6 (vis block) ┼──> T1.1 ──> T1.2 (DeltaNet HF parity)
  T1.7 (merger) ┘          T1.3 (gated attn HF parity) ──> T1.4 (KV consistency)
                           T1.8 (MTP)
                           ↓
                          T1.9 (hybrid 4-layer)
                           ↓
                          T1.10 (full text decoder vs HF)
                           ↓
                          T1.11 (VLM image) ──> T1.12 (VLM video)

Phase 2 (single BH chip):
  T2.5, T2.6 (smoke) parallel
  T2.1 (DeltaNet recurrent kernel) ──> T2.2 (chunked)
  T2.3 (gated attn) ──> T2.4 (decode step)
  T2.7 ──> T2.8 (vision block); T2.9 (merger); T2.10 (MTP)

Phase 3 (BH GLX mesh):
  T3.6 (smoke)
  T3.1 ──> T3.2 ──> T3.3
  T3.4 ──> T3.5
  T3.7
  T3.10 (vocab-parallel LM head)
  T3.8 (4-layer hybrid mesh) ──> T3.9 (full text decoder)

Phase 4 (VLM):
  T4.1 ──> T4.2
  T4.3
  T4.4 ──> T4.5 ──> T4.6 ──> T4.7

Phase 5 (MTP):
  T5.1 ──> T5.2 ──> T5.3

Phase 6 (perf — gated on Phase 3 + 4):
  T6.1 (trace) ──> T6.3 (decode perf), T6.4 (prefill perf)
  T6.2 (profile)
  T6.5 (memory)
  T6.6 (long context)

Phase 7 (server — gated on Phase 6):
  T7.1 ──> T7.2 ──> T7.3 ──> T7.4 ──> T7.5 ──> T7.6 ──> T7.7
```

---

## TDD discipline checklist (applies to every test above)

Before writing production code that the test exercises:
- [ ] Test file exists, test function exists, asserts the expected behavior
- [ ] Test was run; failure observed; failure message matches the **RED** entry above
- [ ] Failure is for the right reason (feature missing — **not** typo, import error, or shape mismatch unrelated to the behavior)
- [ ] If failure reason differs from this plan, update the plan before proceeding

After GREEN:
- [ ] Test passes
- [ ] All upstream-dependency tests still pass
- [ ] Output is pristine (no warnings, no traceback noise)
- [ ] Refactor (if any) keeps tests green and adds no new behavior

---

## What this plan does NOT include

- **No production code.** This is the relay-race contract. Phase by phase implementation lives in the corresponding phase skill (`/reference`, `/ttnn`, `/debug`, `/optimization`, `/tt-inference-server`).
- **No mock-based tests** for the model logic. Every test in Phase 1+ uses real weights and reference outputs.
- **No retrospective tests** added after the implementation. If a phase produces code without a logged failing test from this plan, that work is rejected per TDD discipline.

This document is the authoritative test contract. Modify only if a failure reveals an unforeseen requirement; record the modification with date and reason in BRINGUP_LOG.md.
