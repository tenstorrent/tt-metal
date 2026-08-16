# AutoFix report

AutoFix independently isolated and tested the remaining failures.

1. The CI burst crash was caused by interpreting vLLM's cumulative prefill end as a chunk length. A resumed request at start 48/end 100 executed 100 tokens from position 48 and narrowed its table to three blocks. The sliding cache then rejected width 3 because its 1024-token modulo needs sixteen 64-token blocks. Fixed by executing exactly the 52-token slice with absolute positions and retaining the scheduler's full table.
2. Degenerate qualitative output came from raw `/v1/completions` prompts that bypassed the instruction model's processor chat template. The chat endpoint produces the accepted full-model reference and the complete final suite passes.
3. An unsafe allocation warning came from allocating page-table staging tensors while a trace pinned allocator addresses. Trace release now precedes staging allocation; the final exact benchmark log is clean.
4. vLLM padded decode inputs to 32 rows and the adapter executed that as logical batch 32. This reproduced the historical 2.5 t/s path and caused duplicated subwords. Decode now validates and slices the packed active prefix. Primary serving is 21.78 t/s/u versus canonical full-model 23.76, and corrected qualitative output matches the full-model controls without corrupt joins.
5. Hybrid-cache capacity was proven by a served 262,143-input/1-output request. The fallback audit reran qualitative and benchmarks with `throw_exception_on_fallback=true`.
6. Logical-batch slicing initially exposed padded-token reshape and prime-batch shard-grid failures. The final contract executes true batch one for performance and the canonical 32 device lanes for every multi-user sampling batch, gating inactive rows. This restores shared seeded-stream reproducibility and the full sampling profile.
7. A focused staggered overlap test now uses Gemma's chat endpoint and quality-valid controls. The coherent 96-token essay crosses the 64-token page boundary while the two-word request reaches EOS after 3 tokens; both outputs exactly match isolated controls. The runner hard-fails on doubled tokens, adjacent repeated phrases, dominant-token collapse, failure to cross the page, or failure to finish early. This covers device token feedback, position advancement, mutable page tables, async read fencing, and visible output quality.

Validation: adapter contracts 15/15 pass; full sampling 72 passed/1 skipped; non-aligned 47 and 2051 pass; full-context 262,143+1 passes; final primary 1/1 and CI 32/32 benchmarks pass; qualitative degeneracy gate passes.
