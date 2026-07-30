# Stage review 1

Verdict: more-work-needed.

Required findings were: missing real-weight precision signoff, incomplete
precision-locked dominant-matmul geometry search, missing batch-32 prefill A/B,
and unclassified TTNN memory-config substitutions.

Remediation:

- recovered the official snapshot from `/huggingface` and added real-weight
  batch-1 and batch-32 prefill/cache-consuming traced decode PCC;
- measured BFP4/LoFi, BFP8/LoFi, and BFP8/HiFi2 with real weights;
- measured 8/16/32-core phase shards, larger blocks through 16, exact L1
  failure evidence, and repeated 16-vs-32 timing;
- measured functional and optimized batch-32 prefill in the same harness;
- tested the factory-computed non-rectangular grid directly and recorded the
  exact sharded-RMSNorm bounding-box blocker and profiler accounting;
- added exact-limit 131072 prefill execution.

The full independent review text is preserved in the parent Codex thread.
