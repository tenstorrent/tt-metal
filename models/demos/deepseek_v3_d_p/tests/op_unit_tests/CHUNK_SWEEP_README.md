# Attention-op chunk-size study (8x4 Blackhole Galaxy)

Goal: find the smallest prefill chunk (global chunk / SP=8) at which the attention-path ops of Kimi-K2.7 and
GLM-5.2 keep their relative perf, i.e. the same core count and roughly flat utilization.
Baseline: `main` @ `db596ae144d`. Each op has a traced op-level test (warm-up, trace capture, 10 replays,
realtime profiler, max over chips, median over replays), an env-gated compute-only switch, and a results
doc.

| op | model | test | compute-only switch | results |
|---|---|---|---|---|
| `ring_mla` (dense ring MLA, fused KV all-gather) | Kimi-K2.7 | `test_ring_mla_chunk_sweep.py` | `RING_SDPA_COMPUTE_ONLY=1` | [RING_MLA_CHUNK_SWEEP.md](RING_MLA_CHUNK_SWEEP.md) |
| `topk_large_indices` (indexer top-k) | GLM-5.2 | `test_glm_topk_chunk_sweep.py` | `TOPK_COMPUTE_ONLY=1` | [GLM_TOPK_CHUNK_SWEEP.md](GLM_TOPK_CHUNK_SWEEP.md) |
| `sparse_sdpa` (sparse MLA attention) | GLM-5.2 | `test_glm_sparse_sdpa_chunk_sweep.py` | `SPARSE_SDPA_COMPUTE_ONLY=1` | [GLM_SPARSE_SDPA_CHUNK_SWEEP.md](GLM_SPARSE_SDPA_CHUNK_SWEEP.md) |
| `ring_indexer_score_dsa` (indexer score, fused full-mesh gather) | GLM-5.2 | `test_glm_indexer_score_chunk_sweep.py` | `INDEXER_SCORE_COMPUTE_ONLY=1` | [GLM_INDEXER_SCORE_CHUNK_SWEEP.md](GLM_INDEXER_SCORE_CHUNK_SWEEP.md) |
| `high_bw_all_gather` (sparse-KV prefix gather) | GLM-5.2 | `test_glm_kvpe_gather_chunk_sweep.py` | n/a (pure CCL; % of fabric roofline instead) | [GLM_KVPE_GATHER_CHUNK_SWEEP.md](GLM_KVPE_GATHER_CHUNK_SWEEP.md) |

The compute-only switches are perf-experiment only: they stub every DRAM/NoC/fabric transfer while keeping
the CB handshakes, so outputs are garbage. They are read from the host environment when the program is built.

## Summary

| op | smallest chunk that keeps cores busy | behaviour below it | bound |
|---|---|---|---|
| ring_mla (Kimi) | 2k (256 / device): 60.5% util vs 67.0% at 5k (50k prefix) | 1k: 64 / 110 cores busy, ~31% util | compute (DM 2-10% at 50k prefix) |
| topk (GLM) | 2k: 1 row / core on the 80-core overlap grid, 115 us per 1k tokens vs 97 at 5k | 1k: 32 / 80 cores busy, 2.35x per token | compute (DM 1-2%) |
| sparse_sdpa (GLM) | no cliff down to 2k: 436 us at 2k vs 1058 at 5k (213 vs 207 us per 1k tokens) | 1k: 32 / 120 cores, compute bound, 243 us per 1k tokens | indexed KV gather (~300-375 GB/s per chip; DM +108-187%) |
| indexer score (GLM) | no per-call saving: about 320 us at 50k for 1k-5k (q_chunk 32), so per token 156 us/1k at 2k vs 64 at 5k | 1k: 318 us per 1k tokens (5x 5k) | walking the K prefix (compute-only flat in rows, steps with kv_len); production q_chunk 64 at 2k/4k costs 18-38% |
| sparse-KV gather (GLM) | no per-call saving: 739-809 us at 50k for 1k-5k, so per token 405 us/1k at 2k vs 157 at 5k | 1k: 790 us per 1k tokens | fabric bandwidth (80-92% of 100 GB/s roofline at >= 50k); O(prefix) per chunk |

### Bottom line

The attention compute ops (ring_mla, top-k, sparse_sdpa) keep their efficiency down to a **2k** chunk, and
break at 1k, where cores go idle. The two O(prefix)-per-call GLM ops do not: the indexer score and the sparse-KV
prefix gather cost about the same per call at any chunk size, so their per-token cost grows as 1/chunk, about
2.5x from 5k to 2k. For GLM-5.2, a 2k chunk is viable for attention only if the indexer walk and the prefix
gather are made incremental (score and gather only the new chunk against a cached result) or amortized. As
they are today, they dominate at small chunks.
