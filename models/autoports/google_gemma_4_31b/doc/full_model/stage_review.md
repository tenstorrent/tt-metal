# Stage 06 Independent Review

Historical review record. The 2026-07-17 capacity and non-greedy RNG
remediation is independently reviewed in
`stage_review_remediation_2026-07-17.md` with verdict `clean-pass`. That report,
the current README, work log, and `full_model_plan` contain the authoritative
post-remediation physical KV arithmetic and 23-test contract result; the
pre-remediation figures and 19-test result below are retained only as evidence
of the original Stage 06 checkpoint.

Final verdict: `clean-pass`

The first independent review returned `more-work-needed` for five evidence or implementation gaps:

1. mixed-prompt device-logit prefill was limited to batch one;
2. profiler evidence predated the final source repairs;
3. the decoder-stack performance lower bound was not recorded;
4. the full-context physical batch envelope was not recomputed;
5. trace-warning documentation was inconsistent, and short prompts exposed a tensor-ownership defect.

All findings were repaired and independently rereviewed. The rereviewer verified:

- mixed batch-two non-aligned prompts of lengths 33 and 17 on hardware and under watcher;
- source-current profiler evidence with raw CSV SHA-256 `cefa4861ae9713bc1d83c117dab38760939997a0a305ee71a03483f1c7d528a3`, sampler share 9.68%, and LM-head share 56.25%;
- the exact Stage 05 stack lower bound of 28.356925 ms/token and full-model overhead of 4.13%;
- the physical capacity arithmetic: batch three leaves 974,281,336 bytes/device while batch four is short 1,814,930,824 bytes/device;
- preallocated split-trace sampler buffers, persistent all-gather output, changed and unchanged page-table replay, reset, teardown, and watcher controls;
- the repaired short-prompt ownership path in normal and watcher tests;
- the source-current static contract suite at 19/19 passing.

The final review found no P1/P2 issue, no required work, and no blocking hard-check gap. Batch two at context 128 remains the largest hardware-tested batch; batch three at full context is documented only as a calculated physical upper bound. The raw 7 GB Tracy tree is deliberately excluded from version control; only compact, hash-linked performance artifacts are retained.

One TT qualitative story repeats a corpus-style prompt sentence absent from HF. This is recorded in the qualitative verdict; the separate 100-token TT generation is coherent and non-repetitive, and trace/token-feedback controls show no systematic decode-loop fault.
