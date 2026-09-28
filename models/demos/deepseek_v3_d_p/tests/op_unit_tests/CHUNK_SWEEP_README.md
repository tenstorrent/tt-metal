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
| indexer score | GLM-5.2 | pending | pending | pending |
| `high_bw_all_gather` (sparse-KV prefix gather) | GLM-5.2 | pending | n/a (pure CCL) | pending |

The compute-only switches are perf-experiment only: they stub every DRAM/NoC/fabric transfer while keeping
the CB handshakes, so outputs are garbage. They are read from the host environment when the program is built.

## Summary so far

| op | smallest chunk that keeps cores busy | behaviour below it | bound |
|---|---|---|---|
| ring_mla (Kimi) | 2k (256 / device): 60.5% util vs 67.0% at 5k (50k prefix) | 1k: 64 / 110 cores busy, ~31% util | compute (DM 2-10% at 50k prefix) |
| topk (GLM) | 2k: 1 row / core on the 80-core overlap grid, 115 us per 1k tokens vs 97 at 5k | 1k: 32 / 80 cores busy, 2.35x per token | compute (DM 1-2%) |
| sparse_sdpa (GLM) | no cliff down to 2k: 436 us at 2k vs 1058 at 5k (213 vs 207 us per 1k tokens) | 1k: 32 / 120 cores, compute bound, 243 us per 1k tokens | indexed KV gather (~300-375 GB/s per chip; DM +108-187%) |
