# Gemma4 prefill: matmul core use and chunk size, starting point

Branch `kmabee/gemma4-matmul-dig-base`. It is `kmabee/gemma4-prefill-1005-combined` @ `72c7750e249` (main + Kyle's open Gemma4 prefill PRs) plus two model fixes:
- **RMSNorm for odd tile-row counts** (`tt/rms_norm.py`): chunks with an odd number of residual tile rows per device (1024 / 3072 / 5120) fell back to one core per tile row. First chunk: 1024 59.4 → 43.0 ms, 3072 88.5 → 72.3 ms, 5120 101.2 → 85.7 ms. No effect at 2048 / 4096 / 8192.
- **Global attention policy** (`tt/attention/ring_prefill.py`): q128 with a 2-way K split for chunks 5120 / 6144 instead of an unsplit q96. Chunk 6144: 2.183 → 2.003 s at 100k. No effect at 2048 / 4096 / 8192.

## Numbers to beat
Measured on this branch (`835cd919bcf` before this doc update), BH Galaxy 8x4 (CP8 / TP4), bh-glx-120-c03u02, 130 W TDP, 2026-10-05, with the one-process sweep command below. First chunk = TTFT of a prompt that fits one chunk; 100k = through the chunks covering 102,400 tokens; 256k = device time for 262,144 tokens.

| Chunk | First chunk | 100k wall | 100k device | 256k device |
|---|---|---|---|---|
| 2048 | 53.5 ms | 3.354 s | 3.188 s | 10.290 s |
| 4096 | 74.6 ms | 2.306 s | 2.216 s | 7.287 s |
| 8192 | 113.8 ms | 1.960 s | 1.919 s | 6.586 s |

- Wall minus device is host-side chunk staging in the test harness (~3 ms per chunk here). Device time is the cleaner comparison.
- Clock and power differ per box. **Run the sweep on your own box first** and compare against that.
- KV PCC gates (overall / RRMSE / min per-head): ≥ 0.97 / < 0.232 / ≥ 0.91. At 4096: 0.980254 / 0.1990 / 0.9354 (earlier build of this stack).

```bash
# HF config / tokenizer: HF_MODEL must stay the repo id (the KV PCC test checks it against the golden's model_id),
# so it needs a hub-format cache; Weka only has a plain copy. /mnt/models/huggingface is the shared CI hub cache.
export HF_MODEL=google/gemma-4-31B-it HF_HUB_OFFLINE=1 HF_HOME=/mnt/models/huggingface
# 8x4 weight cache (complete, so the safetensors are not read) and the KV PCC golden, both on Weka
export TT_CACHE_PATH=/mnt/weka/model-cache/scratch/google/gemma-4-31B-it-Cache
export PREFILL_TRACE_DIR=/mnt/weka/model-weights/llm/ref-data/gpu_traces/gemma4_d_p/gutenberg-135
# perf: 2048, 4096 and 8192 together in one process (~8 min warm; prints per-chunk device/wall and a 1k/10k/100k/256k wall table per size)
GEMMA4_SWEEP_CHUNK_SIZES=2048,4096,8192 pytest "models/demos/gemma4_d_p/demo/text_demo_prefill.py::test_prefill_chunk_sweep_traced[blackhole-ctx_256k-text-8x4]" -sv
# KV PCC (~12 min); the chunk size comes only from GEMMA4_TEST_CHUNK_SIZE
GEMMA4_TEST_CHUNK_SIZE=4096 pytest "models/demos/gemma4_d_p/tests/test_prefill_migration.py::test_prefill_migration[mock-256k]" -svv
# one layer under tracy (both layer types, chunk index 1)
python -m tracy -r -p -v -m pytest "models/demos/gemma4_d_p/demo/text_demo_prefill.py::test_prefill_layer_perf_chunk_n[blackhole-chunk1-both-sz4096-ctx_256k-8x4]" -sv
```

## Matmul core use: what we know
- **Why 96 cores at 4k / 8k:** the 2D multicast config splits the output width over the 12 grid columns and the tokens (M) over the 10 grid rows, with one block height for every row. Each device has chunk / 8 tokens: 16 tile rows at 4096, 32 at 8192. Over 10 rows that rounds to 2 / 4 tiles per row, so 8 rows have work: 8 × 12 = 96.
- **2k:** M = 8 tile rows takes the 1D config, which splits only the output width, 2 tiles per core: 84 cores for the 5376-wide projections, 64 for the 4096-wide QKV.
- **Chunk sizes whose tile rows divide over 10 grid rows:** 5120 (20) and 10240 (40). The sequence-parallel residual also needs chunk % 1024 == 0.

Measurements (BH Galaxy 8x4 unless noted):
- **Per-matmul time, one sliding layer** (09-30 tracy capture, older combined-main build; µs, cores):

  | Projection | 2048 | 4096 | 8192 |
  |---|---|---|---|
  | QKV (N 4096) | 108 (64) | 116 (96) | 158 (96) |
  | o_proj (K 2048) | 59 (84) | 71 (96) | 102 (96) |
  | MLP, 3 projections | 106, 126, 118 (84) | 166, 125, 140 (96) | 156, 175, 224 (96) |

- **96 vs 120 cores by padding M** (single chip, MLP 5376², 30 s sustained loop): 4k 158.6 µs (96 cores) vs 167.7 µs (padded to 640 rows, 120 cores); 8k 231.0 vs 249.9 µs (padded to 1280 rows).
- **Chunk 5120 in-model:** matmul time per token 0.84x of 4096's (one-layer tracy, 10-04).

Already tried (09-30 and later, not re-validated on this stack; raw data available from Kyle):
- Config families: 1D vs 2D, L1 width- and block-sharded, DRAM-sharded, transposed 2D, `minimal_matmul`, the matmul auto-tuner (#58143).
- In the model today: sharded L1 input at 2k, QKV output in L1, o_proj output sharded into the reduce-scatter.
- bf16 instead of bfp8 MLP weights at 2k: +0.6 ms per chunk.
