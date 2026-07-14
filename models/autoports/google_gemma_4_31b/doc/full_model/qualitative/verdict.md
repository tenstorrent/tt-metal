# Full-model qualitative verdict

Prompt format: exact `GemmaTokenizer` plain completion. The cached exact model revision has `chat_template=None`; no synthetic instruction template was introduced. Sampling was greedy for both HF and TT.

## Common 100-token autoregressive prompt

HF continues with a coherent story about a shimmering object. TT continues with a coherent story about a peculiar shimmer and a mysterious discovery. The first generated token matches and the paths diverge at generated token two; only 8/100 token positions match thereafter. This is acceptable free-running sensitivity given the 91% top-1 and 100% top-5/top-100 teacher-forcing result. TT shows no adjacent repetition, wrong-language drift, or early incoherence.

## Six-prompt comparison

- Haiku: both produce coherent English haiku continuations and then autocomplete additional haiku prompts. TT is cut at the 64-token limit.
- Supervised versus unsupervised learning: HF and TT are token-identical and repeat the question as a corpus-style completion. This is a base-model control behavior, not TT divergence.
- Story: HF produces a conventional portal story. TT begins coherently, then shifts into a list-like story/poem-prompt autocomplete and repeats one full corpus-style prompt sentence that HF does not. Style is weaker but text remains English and intelligible; the separate 100-token TT story has no such repetition.
- Thermodynamics: HF and TT are token-identical question/equation-list completions rather than direct answers.
- French translation: neither path follows the instruction. HF autocompletes more English translation prompts; TT autocompletes English world-fact prompts and another translation prompt. With no chat template, this is recorded as base-checkpoint prompt-format behavior, not TT-only wrong-language drift.
- Fibonacci: both correctly define or enumerate the sequence. TT is coherent and truncated by the 64-token cap.

The mechanical checker found no degenerate output. Adjacent duplication is zero for prompts 0-4, 0.0263 for Fibonacci, and zero for the common autoregressive completion. High trigram-loop advisory scores on the supervised and thermodynamics cases mirror HF corpus-style repetition. Verdict: pass for Stage 06 correctness and implementation quality, with explicit base-model/no-chat-template limitations.
