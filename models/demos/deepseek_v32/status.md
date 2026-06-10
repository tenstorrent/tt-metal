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
4. TopK needs Row-major input.
5. PCC truth is the CPU reference (reference_cpu), not v3 — v3.2 output is not assumed to match v3.
6. Weights: start from random initialization, move to pretrained later. Pretrained MLA loading already exists in reference_cpu; torch→ttnn conversion + sharding exists in v3 (_convert_and_cache_weights / build_ttnn_cache) and is reused.
7. Always follow tensor shapes — document the shape (and sharding) of every input/output/intermediate; CPU reference shapes are normative.
8. APIs for missing ops are derived from the fused-op reports in context/ (DeepGEMM fp8_mqa_logits for indexer, FlashMLA sparse attention) + composing existing ttnn ops, per "Approach to missing ops".
9. Every decision is documented here before/while implementing — keeps implementation unblocked and reviewable async.
10. Test-first whenever possible. For a missing op the first test checks shapes (op API runs end-to-end with the agreed input/output shapes); numerics vs CPU reference come after.
11. Target hardware: QuietBox, 4 Blackhole devices. Tests parametrized by mesh shape so Galaxy works later — no hardcoded 4-device assumptions.
12. Parallelism and sharding follow v3 decisions exactly. Any deviation forced by v3.2 is documented here. CPU reference is single-device truth only — it says nothing about distribution.

## Status (2026-06-10)
1. **Done:** CPU reference (reference_cpu) — MLA + Indexer, matches DeepSeek reference.
2. **Done:** single-chip ttnn port (reference_tt_single_chip) — MLA + Indexer with CPU fallbacks; decisions/fallbacks documented in spec.md, multichip plan in spec-multichip.md.
3. **Done:** multichip scaffold (tt/ + tests/) reusing deepseek_v3_d_p as a library:
   - tt/mla/mla.py — ttMLA subclass of v3 ttMLA, passthrough for now; all DSA changes go here.
   - tt/tt_prefill_block.py, tt/tt_prefill_transformer.py — copies of v3, only the MLA/block import changed. Future work: add mla_class/block_class injection params to v3 so the copies can be deleted (see tt/README.md).
   - tests/test_mla.py — e2e MLA test reusing the v3 harness (monkeypatched MLA class); collects. Currently compares against the v3 reference — to be replaced, see Next.
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
2. ttnn.topk exists (last dim, bf16, needs row-major input) but k=2048 is untested
3. No sparse attention in ttnn
4. Missing non-interleaved RoPE op
5. v3 composition files hardcode ttMLA/TtPrefillBlock — forced copies in v32; fix by upstreaming injection params (tt/README.md)
6. V3.2 checkpoints (indexer weights) not wired into test conftest — tests run with v3 weights

### Approach to missing ops
When no op exist try to **0. define an API (inputs/outputs)** and
1. create a workaround by composing existing ops
2. fallback to CPU implementation
3. implement stub op that emits a warning and returns random/zeroes/ones tensor in the expected format

## Testing
- Primary goal: prefilled 50k cache, 5k chunk
- add determinism tests
- add accuracy tests that should match CPU reference (CPU reference outputs should be cached somewhere to speed-up testing)
