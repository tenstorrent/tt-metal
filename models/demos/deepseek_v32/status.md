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

## Status (2026-06-10)
1. **Done:** CPU reference (reference_cpu) — MLA + Indexer, matches DeepSeek reference.
2. **Done:** single-chip ttnn port (reference_tt_single_chip) — MLA + Indexer with CPU fallbacks; decisions/fallbacks documented in spec.md, multichip plan in spec-multichip.md.
3. **Done:** multichip scaffold (tt/ + tests/) reusing deepseek_v3_d_p as a library:
   - tt/mla/mla.py — ttMLA subclass of v3 ttMLA, passthrough for now; all DSA changes go here.
   - tt/tt_prefill_block.py, tt/tt_prefill_transformer.py — copies of v3, only the MLA/block import changed. Future work: add mla_class/block_class injection params to v3 so the copies can be deleted (see tt/README.md).
   - tests/test_mla.py — e2e MLA test reusing the v3 harness (monkeypatched MLA class); collects. Currently compares against the v3 reference — to be replaced, see Next.
4. **In progress:** Next step 1 (test rewire to CPU reference) implemented, first hardware run pending. Decisions taken:
   - Weight mapping MLACPU→v3 dict is a pure rename (wq_a→q_a_proj, q_norm→q_a_layernorm, wq_b→q_b_proj, wkv_a→kv_a_proj_with_mqa, kv_norm→kv_a_layernorm, wkv_b→kv_b_proj, wo→o_proj); same [out,in] layout, bf16.
   - Bring-up seq_len = 2048 = index_topk: there the DSA index mask is 0 over the causal region, so MLACPU is dense-equivalent and the passthrough must match (output PCC 0.98, KVPE 0.99 — v3 thresholds). seq>2048 diverges by design until indexer+sparse SDPA land.
   - MLACPU run with simulate_fp8=False (device KVPE stores bf16; same as single-chip port).
   - CPU outputs cached at $DEEPSEEK_V32_MLA_REF_CACHE (default /tmp/deepseek_v32_mla_ref_cache), keyed by tag+seq+seed.
   - Device side keeps the HF config (shape-asserted vs ModelArgs); RoPE equivalence interleaved↔reference proven in single-chip spec D5.
   - First hardware runs (1x4, passthrough): pipeline runs e2e clean on QuietBox; seq2k output PCC 0.785, seq256 0.889 — diagnosed, not an MLA bug: ModelArgs.max_seq_len was set to seq_len, which disables YaRN (max_seq_len ≤ original 4096) in the CPU reference while v3's device tables (HF DeepseekV3YarnRotaryEmbedding) always apply YaRN; freqs and softmax mscale² (1.87x) diverge. Host check confirmed. Fix: keep max_seq_len=16384, refs re-cached (cache tag includes max_seq_len).
   - **After fix, step 1 DONE — both pass on 1x4 QuietBox: seq256 output PCC 0.9991, seq2k 0.9986, KVPE ~0.9999** — weight mapping, RoPE convention, sp=1 harness all confirmed. seq2k cold run 431s (cached ref cuts CPU part on reruns).
5. **Done (step 2 shape contracts):** tt/ops.py + tests/test_ops_shapes.py — all 5 shape tests pass on 1x4 QuietBox (8.5s): indexer_logits composed from ttnn matmul/relu/permute (on device); topk via ttnn.topk — works on device at k=2048, TILE input, host fallback never triggered; sparse_mla CPU fallback. Numerics tests vs reference_cpu next, then wiring into ttMLA (step 3).
   - Gotcha: don't pipe pytest through tail — swallows exit code; log to file, check $?.
4. **Next (proposed, to agree on — no implementation until agreed):**
   1. Rewire tests/test_mla.py to use reference_cpu (MLACPU) as the PCC truth instead of the v3 reference — v3.2 and v3 outputs are not assumed to match, even for the passthrough. Reuse v3's run_mla_inference for the device side only; one set of random torch weights mapped into both the v3 ttMLA dict format and MLACPU; cache CPU reference outputs on disk (pattern exists in reference_tt_single_chip/test_model.py). Mesh shape parametrized; bring-up config = QuietBox 4x Blackhole.
   2. Define APIs (inputs/outputs, shapes, sharding) for the missing ops from the context/ reports + existing ttnn ops; record them here for review. For each missing op the first deliverable is a shape test: the agreed API runs e2e and produces the right shapes (stub/composed/CPU-fallback per "Approach to missing ops"); numerics vs CPU reference follow.
   3. Wire indexer into v32 ttMLA — sharding follows v3 exactly (TP plan: indexer replicated, only collective = RS+AG after wo; deviations documented here) — PCC vs the same CPU reference, random weights.
   4. Pretrained: add V3.2 checkpoint (indexer weights) to conftest; reuse reference_cpu loading + v3 ttnn conversion/sharding.

## References
1. models/demos/deepseek_v32/reference_cpu - deepseek's reference implementation running on CPU w/o fused ops and sparse attention
2. models/demos/deepseek_v32/reference_tt_single_chip - reference implementation using ttnn that runs on single chip and w/ CPU fallbacks
3. models/demos/deepseek_v3_d_p - tt multichip implementation for deepseek v3

## Issues
1. No fused indexing op in ttnn (fp8_index + causal mask)
2. ~~ttnn.topk k=2048 untested~~ RESOLVED 2026-06-10: works on Blackhole at k=2048 (TILE input, shape test green)
3. No sparse attention in ttnn
4. Missing non-interleaved RoPE op
5. v3 composition files hardcode ttMLA/TtPrefillBlock — forced copies in v32; fix by upstreaming injection params (tt/README.md)
6. V3.2 checkpoints (indexer weights) not wired into test conftest — tests run with v3 weights

### Missing op APIs (proposed 2026-06-10, step 2 — review async)
ttnn-shaped equivalents of the fused references (DeepGEMM fp8_mqa_logits, FlashMLA sparse fwd). All activations [1, B, S_local, ·] TILE bf16 like v3; indexer replicated across TP, S sharded on SP (spec-multichip §3.6). B=1 prefill.

1. `indexer_logits(q, k, w) -> logits` — q [1,B,Sq,H_idx*D_idx] (H=64, D=128), k [1,B,Skv,D_idx], w [1,B,Sq,H_idx] (fp32 weights_proj out). Out [1,B,Sq,Skv] bf16 (fp8 inputs later). Causal window per row (DeepGEMM ks/ke), no materialized mask. Workaround: per-head matmul + ReLU + weighted head-sum + causal mask add. CPU fallback for non-interleaved rope (F1).
2. `topk_indices(logits, k=2048) -> indices` — TILE in (corrected agreement 4), out [1,B,Sq,k] uint32, padded with last valid where Skv<k. Workaround: ttnn.topk; host fallback. K cache format untouched (agreement 3).
3. `sparse_mla(q, kvpe_cache, indices, scale) -> out` — q [1,H,Sq,576] absorbed; kvpe [1,1,Skv,576]; indices [1,B,Sq,2048]; out [1,H,Sq,512]; indices replace causal mask (FlashMLA contract). Workaround: gather (embedding-style) + chunked dense SDPA; stub: dense ring MLA (valid Sq≤2048).

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
| CPU reference forward (uncached) | per (tag, seq, seed); seq 2048, 128 heads | ~1-2 min, grows quadratically; 50k cache + 5k chunk infeasible cold | disk cache /tmp/deepseek_v32_mla_ref_cache (env DEEPSEEK_V32_MLA_REF_CACHE) — keep tag/seed stable; 50k+5k truth must be cached once on a big box |
| ttnn incremental rebuild after .so staleness | after rebase/pull | ~3 min observed | rebuild ninja -C build ttnn (target is "ttnn", not "_ttnn"); fresh lib lands in build/ttnn/_ttnn.so |
| HF config-only download | first run / new variant | seconds-min, network | already cached by v3 conftest |
| Pre-commit hooks (isort/black, EOF fixer) | every commit | tens of sec | don't partial-stage — keep index clean or hooks loop on fix-rollback |

## Testing
- Primary goal: prefilled 50k cache, 5k chunk
- add determinism tests
- add accuracy tests that should match CPU reference (CPU reference outputs should be cached somewhere to speed-up testing)
