# Prefill: deferred / skipped ideas (decisions as of 2026-10-06)

Skipped on purpose (the user decided to note and skip):
- Fused collective + matmul ops (all_gather_minimal_matmul_async, matmul_reduce_scatter_async, ...) via a private C++ build: estimated gain ~2-4% of prefill
  (only the attention output reduce-scatter ~42 ms/chunk and ~60 ms of all-gathers are matmul-adjacent; the big MoE collectives are not), weeks of C++ for a
  4x8 mesh (the ops hard-code the 1D fabric packet header; matmul_reduce_scatter_async only tested on a 1x4 box; the strided variant races on Blackhole).
  Decode attention output all-reduce (tt/attention.py:381, ~60-90 us/layer) has the same pattern: est. 2-4% of decode; no profile taken; our demo already uses
  FABRIC_1D_RING (demo/text_demo.py:568,620), so a one-layer feasibility test would be the first step if this is ever revisited.

Not done / rejected (evidence in the per-optimization notes):
- bfp4 routed experts: off by default (accuracy: full-chain PCC -0.05..-0.08); only helps decode (floor halves), ~1% for prefill.
- sparse_sdpa fp8 key table (DSV41_PFA_SP_FP8): fails the accuracy gate (layer PCC 0.9995 -> 0.997), only 2% TTFT.
- Sequence parallelism: no throughput gain at B>=4 (same work, extra attention CCL); only helps single-user TTFT.
- Phase-switch weight relayout (pf-sharedw): rejected, breaks interleaved prefill/decode serving; replaced by the one-copy ring weights.
- ENGRAM_OWN (DSV41_PF_ENGRAM_OWN): flips first tokens at isl8k_b4 / isl32k_b8; off.
