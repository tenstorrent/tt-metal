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

## Backlog (decision 2026-10-06: focus is (1) unified MoE + one-copy ring weights on by default and rebuilt, (2) adaptive-length spec decode that works for B up to 128; everything below waits)

Prefill (branches hold the code; numbers are the agents' measurements, 40 layers):
- MoE collective overlap (ssinghal/dsv4p1-pf-moeoverlap: DSV41_UNI_ROUTER=batched, DSV41_MO_OVERLAP=1 shared expert on a sub-device behind dispatch, DSV41_UNI_TOPO=ring):
  about -3..-6% prefill replay, bit-identical; segmented trace (81 segments) for the overlap. Not merged.
- Hot-expert skew in the unified MoE (real tokens: one expert gets ~63x the mean; dispatch/experts/combine 3.9/6.1/3.1 ms vs 1.5/2.8/1.0 ms with random tokens):
  replicate/split hot experts or a combine overlapped with the expert op. Biggest remaining MoE lever, not started.
- Chunk-size rule DSV41_PREFILL_ROW_TOKENS=auto2 (ssinghal/dsv4p1-chunkcal tip 277aec8a1a5, tt/chunk_rule.py): only the part needed by the unified default is being merged; short-prompt bucketing not done.
- sparse_sdpa is DRAM random-gather bound (~280 GB/s): sharing gathered keys across adjacent queries / block-sparse selection (changes selection semantics), fp8 table rejected (accuracy).
- Indexer fusions: one fused kernel for the ~20-op fp4 simulation (5.4 ms/index layer), build the latent masks once per chunk (~2 ms/index layer), layout glue; est. 1-2% total.
- mHC remaining small kernels / Engram per-layer ops: partly done (packed mHC, Engram batch); the rest ~10% of kernel time.
- Pipeline-parallel sub-meshes for prefill (MiniMax-M3 style), sequence parallelism for single-user TTFT: not scoped.
Serving:
- Interleaved prefill with decode (ssinghal/dsv4p1-interleave, DSV41_VLLM_INTERLEAVE=1, 40-layer exact at B=16) and the vLLM adapter (ssinghal/dsv4p1-vllm) + tt-inference-server entry (tt-inference-server branch ssinghal/dsv4p1): done on branches, not merged;
  vLLM not installed yet (needs the user's approval for the install into python_env; dry-run showed pydantic/pydantic-core/sse-starlette upgrades + 37 new packages).
Decode:
- Decode per-op profile at B=16/128, attention output all-reduce fusion (skipped), non-expert ~50 ms of 87.7 ms at B=128, bfp4 experts (off).
Spec decode (parts not in the focus above):
- k sweep beyond B=16 (ssinghal/dsv4p1-spec-k), workload sweep for acceptance by task type, confidence-scheduler variants.
Infra:
- Local copy of the weight cache per host to speed builds, higher build-slot cap, hangwatch default stall limit (8 min kills the ~9 min Engram load), host .31 faulty, host .42 hangs, run Claude Code inside tmux.
