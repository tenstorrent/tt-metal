# TT multichip deepseek v3.2 MLA

## Goal
Implement deepseek v3.2 MLA layer for multichip TT setup. Start from existing v3 and introduce changes.
Identify missing ttnn operations.
Identify problems and what needs to be updated/changed.

## Scope
Prefill only.
MLA layer.

## Agreements
1. **Start from the existing v3 op and modify it**.
2. Keep track of issues and learning points.
3. K cache stays in the same format so that it doesn't affect decode downstream.
4. ~~TopK needs Row-major input.~~ CORRECTED 2026-06-10: ttnn.topk asserts TILE input (topk_device_operation.cpp:152), verified on Blackhole.
5. PCC truth is the CPU reference (reference_cpu), not v3 — v3.2 output is not assumed to match v3.
6. Weights: start from random initialization, move to pretrained later. Pretrained MLA loading already exists in reference_cpu; torch→ttnn conversion + sharding exists in v3 (_convert_and_cache_weights / build_ttnn_cache) and is reused.
7. Always follow tensor shapes — document the shape (and sharding) of every input/output/intermediate; CPU reference shapes are normative.
8. APIs for missing ops are derived from the fused-op reports in context/ (DeepGEMM fp8_mqa_logits for indexer, FlashMLA sparse attention) + composing existing ttnn ops, per "Approach to missing ops".
9. Every decision is documented here before/while implementing — keeps implementation unblocked and reviewable async.
10. Test-first whenever possible. For a missing op the first test checks shapes (op API runs end-to-end with the agreed input/output shapes); numerics vs CPU reference come after.
11. Target hardware: QuietBox, 4 Blackhole devices. Tests parametrized by mesh shape so Galaxy works later — no hardcoded 4-device assumptions.
12. Parallelism and sharding follow v3 decisions exactly. Any deviation forced by v3.2 is documented here. CPU reference is single-device truth only — it says nothing about distribution.
13. Mesh bring-up order: start 1x4 (pure TP, matches spec-multichip TP-only plan); add 2x2 (SP x TP, exercises chunked path) later. Both stay parametrized.
14. Scale bring-up order: small single-shot first (~4-8k tokens, cheap CPU reference); 50k cache + 5k chunk later as gating milestone with cached CPU outputs.
15. Chunk size is a configurable test parameter, 1k for the dev loop (CPU truth fast enough for iteration); 5k once proper kernels (fused sparse attention) land. Same code path either way.

## Status (2026-06-10)
1. **Done:** CPU reference (reference_cpu) — MLA + Indexer, matches DeepSeek reference.
2. **Done:** single-chip ttnn port (reference_tt_single_chip) — MLA + Indexer with CPU fallbacks; decisions/fallbacks documented in spec.md, multichip plan in spec-multichip.md.
3. **Done:** multichip scaffold (tt/ + tests/) reusing deepseek_v3_d_p as a library:
   - tt/mla/mla.py — ttMLA subclass of v3 ttMLA, passthrough for now; all DSA changes go here.
   - tt/tt_prefill_block.py, tt/tt_prefill_transformer.py — copies of v3, only the MLA/block import changed. Future work: add mla_class/block_class injection params to v3 so the copies can be deleted (see tt/README.md).
   - tests/test_mla.py — e2e MLA test reusing the v3 harness (monkeypatched MLA class); collects. Currently compares against the v3 reference — to be replaced, see Next.
4. **Done (step 1, test rewire to CPU reference):** Decisions taken:
   - Weight mapping MLACPU→v3 dict is a pure rename (wq_a→q_a_proj, q_norm→q_a_layernorm, wq_b→q_b_proj, wkv_a→kv_a_proj_with_mqa, kv_norm→kv_a_layernorm, wkv_b→kv_b_proj, wo→o_proj); same [out,in] layout, bf16.
   - Bring-up seq_len = 2048 = index_topk: there the DSA index mask is 0 over the causal region, so MLACPU is dense-equivalent and the passthrough must match (output PCC 0.98, KVPE 0.99 — v3 thresholds). seq>2048 diverges by design until indexer+sparse SDPA land.
   - MLACPU run with simulate_fp8=False (device KVPE stores bf16; same as single-chip port).
   - CPU outputs cached at $DEEPSEEK_V32_MLA_REF_CACHE (default /tmp/deepseek_v32_mla_ref_cache), keyed by tag+seq+seed.
   - Device side keeps the HF config (shape-asserted vs ModelArgs); RoPE equivalence interleaved↔reference proven in single-chip spec D5.
   - First hardware runs (1x4, passthrough): pipeline runs e2e clean on QuietBox; seq2k output PCC 0.785, seq256 0.889 — diagnosed, not an MLA bug: ModelArgs.max_seq_len was set to seq_len, which disables YaRN (max_seq_len ≤ original 4096) in the CPU reference while v3's device tables (HF DeepseekV3YarnRotaryEmbedding) always apply YaRN; freqs and softmax mscale² (1.87x) diverge. Host check confirmed. Fix: keep max_seq_len=16384, refs re-cached (cache tag includes max_seq_len).
   - **After fix, step 1 DONE — both pass on 1x4 QuietBox: seq256 output PCC 0.9991, seq2k 0.9986, KVPE ~0.9999** — weight mapping, RoPE convention, sp=1 harness all confirmed. seq2k cold run 431s (cached ref cuts CPU part on reruns).
5. **Done (step 2 shape contracts):** tt/ops.py + tests/test_ops_shapes.py — all 5 shape tests pass on 1x4 QuietBox (8.5s): indexer_logits composed from ttnn matmul/relu/permute (on device); topk via ttnn.topk — works on device at k=2048, TILE input, host fallback never triggered; sparse_mla CPU fallback.
6. **Done (step 2 numerics):** tests/test_ops_numerics.py vs reference_cpu functional path — indexer_logits PCC ≥0.99 (incl. ReLU + weighted head-sum), sparse_mla 0.999, ttnn.topk index-set overlap vs torch ≥1−max(2/k, 1%) per row (bf16 ties swap a boundary band that grows with k — ~0.5% on random logits, 11/2048 measured; not a bug; effect should shrink on real data with larger score spread). Next: step 3 — wire indexer ops into v32 ttMLA forward, integrate at seq>2048 vs CPU reference (sparse path now differs from v3 by design).
7. **MILESTONE (step 3 done, 2026-06-10): full v32 suite green on 1x4 QuietBox (12 tests, 8m17s)** — e2e vs functional-parity CPU reference: seq256 0.9991 / seq2k 0.9986 (dense passthrough), seq4k 0.9975 with DSA active (sparse rows 0.9948), KVPE 0.9999; ops shape+numerics all pass. Band PCC diagnostics kept permanently in test_mla.py (caught the head-sharding bug).
8. **Step 3 implementation log:** DSA wired into v32 ttMLA (tt/mla/mla.py): seq<=index_topk → super().forward unchanged (dense==sparse); seq>index_topk → _dsa_forward = copy of v3 single-shot forward with ring SDPA replaced by ops.sparse_mla over top-k latents, wkv_b2 applied after attention (chunked-path style). Indexer stems on host for bring-up (qr from MLA q_a stem host copy; F1 non-interleaved rope); logits+topk on device. Limits: sp=1 single-shot only (asserted). e2e seq4k case (sparse on both sides) added — indexer weights flow through WEIGHT_NAME_MAP, popped before v3 sees them; first run: device DSA path executed e2e clean; cold CPU ref at 4096 took 48 min (now disk-cached; reruns are fast).
   **Remaining for production (next):** chunked prefill (sparse_mla needs start_pos offset for causality; goal = 50k cache + 5k chunk with cached truth); 2x2 SP×TP (indexer needs seq AG or distributed topk); device-side indexer stems + non-interleaved rope (F1 still host); fused fp8 ops out of scope (C++ follow-ups per Approach §4).
   **Chunked plan (step 4, proposed):** (1) sparse_mla gains start_pos: causality drops index > start_pos + row; (2) v32 ttMLA keeps a host indexer K-cache (k_norm(wk(x)) per chunk, [max_seq, 128] per layer) since chunks only carry their own hidden; logits matmul against full cached prefix, mask [chunk, end_pos]; (3) attention reuses v3 _chunked_attn cache write but swaps ring_mla for sparse_mla over the populated prefix. Tests staged like step 3: shape/numerics for sparse_mla(start_pos>0) first; e2e at 4k cache + 1k chunk (~5k truth, cold ~1.5h, cache once) before 50k+5k (truth on big box overnight). CPU truth: chunk loop on MLACPU decode branch (kv from cache). Bring-up bugs found+fixed so far: (a) hidden is TP-sharded — indexer host stems concat shards; (b) epilogue is RS-only (RS+AG gave replicated 28672 output); (c) sparse_mla must re-impose causality for rows with <k causal keys (0.20→0.38; future indices from topk's -inf band) — now part of the op contract; (d) CPU truth ran the fp8/Hadamard indexer path — selection-divergent vs functional stems — set indexer.use_fp8_path=False per spec.md §104 parity; truth re-cached (46 min at 4096); (e) band diagnostics (dense rows also ~0.40) exposed the real bug: q is head-sharded across TP, but sparse_mla read shard 0 and replicated its 32 heads to all chips → 3/4 chips fed o_proj wrong heads. sparse_mla is now per-shard: each chip's heads computed separately, out re-sharded on heads — this is the op's distribution contract (q/out TP-sharded, kvpe+indices replicated); unit tests updated accordingly.
   - Gotcha: don't pipe pytest through tail — swallows exit code; log to file, check $?.
9. **In progress (step 4, chunked prefill):** slice 1 done — sparse_mla(start_pos) green (10/10 ops tests, single_shot + chunked cases). Remaining slices per Chunked plan (chunk size = test param, 1k dev default per agreement 15): (2) host indexer K-cache in ttMLA; (3) chunked DSA forward (v3 _chunked_attn cache write + sparse_mla(start_pos)); (4) chunked e2e harness — v3 run_mla_inference is single-shot, needs chunk loop + get_rope_tensors_indexed, CPU truth = chunk loop on MLACPU decode branch cached per chunk; first target 4k cache + 1k chunks (truth ~1h cold, cache once), then 50k+1k overnight; 5k chunks deferred to fused kernels.
10. **Backlog:** pretrained weights (V3.2 checkpoint with indexer weights into conftest; reuse reference_cpu loading + v3 conversion); 2x2 SP×TP; device indexer stems.

## References
1. models/demos/deepseek_v32/reference_cpu - deepseek's reference implementation running on CPU w/o fused ops and sparse attention
2. models/demos/deepseek_v32/reference_tt_single_chip - reference implementation using ttnn that runs on single chip and w/ CPU fallbacks
3. models/demos/deepseek_v3_d_p - tt multichip implementation for deepseek v3

## Issues
1. No fused indexing op in ttnn (fp8_index + causal mask) — composed-op workaround in tt/ops.py::indexer_logits (device, bf16, no fp8/Hadamard); fused C++ op is follow-up
2. ~~ttnn.topk k=2048 untested~~ RESOLVED 2026-06-10: works on Blackhole at k=2048 (TILE input, shape test green)
3. No sparse attention in ttnn — CPU fallback in tt/ops.py::sparse_mla carrying the agreed contract (TP-shard distribution, causality, start_pos); fused C++ op is follow-up
4. Missing non-interleaved RoPE op — host fallback (F1, single-chip spec), used in indexer host stems
5. v3 composition files hardcode ttMLA/TtPrefillBlock — forced copies in v32; fix by upstreaming injection params (tt/README.md)
6. V3.2 checkpoints (indexer weights) not wired into test conftest — tests run with v3 weights

### Missing op APIs (proposed 2026-06-10, step 2 — review async)
ttnn-shaped equivalents of the fused references (DeepGEMM fp8_mqa_logits, FlashMLA sparse fwd). All activations [1, B, S_local, ·] TILE bf16 like v3; indexer replicated across TP, S sharded on SP (spec-multichip §3.6). B=1 prefill.

1. `indexer_logits(q, k, w) -> logits` — q [1,B,Sq,H_idx*D_idx] (H=64, D=128), k [1,B,Skv,D_idx], w [1,B,Sq,H_idx] (fp32 weights_proj out). Out [1,B,Sq,Skv] bf16 (fp8 inputs later). Causal window per row (DeepGEMM ks/ke), no materialized mask. Workaround: per-head matmul + ReLU + weighted head-sum + causal mask add. CPU fallback for non-interleaved rope (F1).
2. `topk_indices(logits, k=2048) -> indices` — TILE in (corrected agreement 4), out [1,B,Sq,k] uint32, padded with last valid where Skv<k. Workaround: ttnn.topk; host fallback. K cache format untouched (agreement 3).
3. `sparse_mla(q, kvpe_cache, indices, scale) -> out` — q [1,H,Sq,576] absorbed; kvpe [1,1,Skv,576]; indices [1,B,Sq,2048]; out [1,H,Sq,512]; indices replace causal mask (FlashMLA contract). Workaround: gather (embedding-style) + chunked dense SDPA; stub: dense ring MLA (valid Sq≤2048). **API learning (e2e seq4k, PCC 0.20→fixed):** rows with <k causal keys receive arbitrary future indices from topk's -inf band and scores are recomputed from latents, so the op itself must drop index > row_pos (FlashMLA solves via per-row topk_length; fused op must too). Chunked prefill needs the start_pos offset here.

Shape tests are the first deliverable per op; numerics vs reference_cpu after.

### Approach to missing ops
When no op exist try to **0. define an API (inputs/outputs)** and
1. create a workaround by composing existing ops
2. fallback to CPU implementation
3. implement stub op that emits a warning and returns random/zeroes/ones tensor in the expected format.
4. Proper implementation of c++ ops is out of scope. That's follow-up that should be documented.

## Long-running tasks
Track every step that takes minutes — each is either a bug risk (silent hangs, stale state) or a caching/optimization opportunity. Add measured times as we collect them.

| Task | When | Time | Mitigation / caching |
|---|---|---|---|
| First e2e MLA test run (mesh init + fabric + weight upload, no output until end) | every fresh pytest | measured: 472s seq2k (cold CPU ref incl.), ~40s seq256 | pytest -s for live progress; track time per stage; weight cache reuses v3 build_ttnn_cache |
| CPU reference forward (uncached) | per (tag, seq, seed); 128 heads + 64-head indexer | measured: ~7 min seq2k, 48 min seq4k — quadratic; 50k+5k infeasible cold | disk cache /tmp/deepseek_v32_mla_ref_cache (env DEEPSEEK_V32_MLA_REF_CACHE) — keep tag/seed stable; 50k+5k truth must be cached once on a big box |
| ttnn incremental rebuild after .so staleness | after rebase/pull | ~3 min observed | rebuild ninja -C build ttnn (target is "ttnn", not "_ttnn"); fresh lib lands in build/ttnn/_ttnn.so |
| HF config-only download | first run / new variant | seconds-min, network | already cached by v3 conftest |
| Pre-commit hooks (isort/black, EOF fixer) | every commit | tens of sec | don't partial-stage — keep index clean or hooks loop on fix-rollback |

## Testing
- Primary goal: prefilled 50k cache, 5k chunk
- add determinism tests
- add accuracy tests that should match CPU reference (CPU reference outputs should be cached somewhere to speed-up testing)
