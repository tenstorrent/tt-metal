# Recall request-mode hypothesis review

Independent read-only AutoFix review by serving_audit, 2026-11-17. Not a stage-review verdict.

The proposed change is justified as explicitly configured chat recall, but does not yet explain or resolve the original TT failure. Raw HF misses11/12 at24 tokens while reasoning; one default-thinking chat diagnostic completes after839 tokens. Explicit request-level thinking off plus existing --tt-recall-reasoning-effort=low adds no shared interface. Keep --tt-recall-reasoning-token-budget=0.

Required evidence:

- Preserve original full suite, traceback and exact failed responses; HF reasoning truncation cannot excuse cross-request contamination.
- Complete matched HF controls for both solo prompts and all12 concurrent-corpus prompts, including token IDs and effective settings.
- Run all3 canonical tests with unchanged prompts,24 cap, seeds, concurrency, rounds and assertions; preserve requests/responses.
- Verify actual ragged chunk/decode-interleaving coverage, because chat wrapping changes boundaries.
- Run same-code/same-chip generator control at the served token count. A serving-only failure requires serial-ragged versus interleaved discrimination.

Report successful configured results as explicit nonthinking chat recall, alongside original raw failure, never as an unchanged-default suite pass. Single greedy controls are not a model-card/ten-seed matched quality waiver.


Follow-up read-only control audit (serving_audit): served-geometry control constructs the selected adapter model, device sampler and all10distinct cache pools at capacity32. All14manifest fixtures and preserved HF source hashes match. Fixed before native execution: require14unique/complete/recalled outputs; protect mesh closure and result saving from generator-close failure. Summary now checks only JUnit failure/error/skipped tags, retaining harmless system-out/properties, and explicitly passes reasoning_effort=low to rendering. Exact answer equality is labeled a stricter diagnostic than the canonical matcher.

Native generator.generate resets state and prefills the whole prompt in one public call, then executes solo traced decode. This is a native model/sampler control; it does not reproduce scheduler chunk boundaries, pending prefills, remapping or async overlap. The served digest/partition/interleaving summary complements it; interleaving logs prove submissions, not direct device-byte preservation. Existing current tracked lifecycle/page-growth evidence supplies additional ownership/stale-input coverage. No stage acceptance verdict follows from this source audit.
