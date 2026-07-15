# Qualitative verdict

Pass. The selected four-input-shard/block-2 LM head reproduces all six Stage 06 TT prompt continuations exactly. Two 64-token prompts match HF exactly; their repeated corpus phrases therefore are not TT-only degeneration. The Fibonacci control is coherent, has no repeated trigram loop, and restores the Stage 06 trajectory rejected by block 3. The separate 100-token story remains coherent English with zero adjacent repetitions and zero repeated trigrams.

The tokenizer exposes no chat template, so these are exact plain-tokenizer base-checkpoint completion controls. Instruction-like corpus autocomplete is a checkpoint/prompt-format limitation, not evidence of a runtime fallback. See `degenerate_output_check.json`, `lm_head_aligned_ab.json`, and `../autoregressive/`.
