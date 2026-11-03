# Muse-Glimmer-30B

Text decoder of [meta-models/Muse-Glimmer-30B](https://huggingface.co/meta-models/Muse-Glimmer-30B) on Tenstorrent Blackhole.

## Setup

```bash
huggingface-cli download meta-models/Muse-Glimmer-30B --local-dir <ckpt_dir>
export HF_MODEL=<ckpt_dir>
export TT_CACHE_PATH=<writable_dir>   # optional; TT weight cache, default <ckpt_dir>/tt_cache
```

## Run

QB2 (4x Blackhole, TP4):

```bash
MESH_DEVICE=P300x2 pytest models/demos/muse_glimmer_30b/demo/text_demo.py -k "dp1-tp4-b1-greedy"
MESH_DEVICE=P300x2 pytest models/demos/muse_glimmer_30b/demo/text_demo.py -k "dp1-tp4-b32-greedy"
MESH_DEVICE=P300x2 pytest models/demos/muse_glimmer_30b/demo/text_demo.py -k "test_muse_glimmer_isl and tp4"
```

`test_muse_glimmer_isl` runs batch 1 on the Gemma4 demo inputs at ISL 128 to 128k; pick one ISL with e.g. `-k "isl32k-tp4"` (ids: `isl128`, `isl256`, `isl1k` … `isl128k`).

Blackhole Galaxy (32x Blackhole, one TP4 or TP8 replica):

```bash
MESH_DEVICE=BHGLX pytest models/demos/muse_glimmer_30b/demo/text_demo.py -k "dp1-tp4-b1-greedy"
MESH_DEVICE=BHGLX pytest models/demos/muse_glimmer_30b/demo/text_demo.py -k "dp1-tp8-b1-greedy"
MESH_DEVICE=BHGLX pytest models/demos/muse_glimmer_30b/demo/text_demo.py -k "dp1-tp8-b32-greedy"
```

## Supported

| | |
|---|---|
| ISL | up to 131072 tokens including generated tokens (batch 1); batch 32 runs with max sequence length 4096 |
| Batch size | 1 to 32 per replica |

## Precision

Default: BFP4 projection and LM-head weights, BFP8 KV cache, LoFi matmuls, HiFi2 SDPA. Set `MUSE_LOW_PRECISION=0` for BFP8 weights, BF16 KV cache, HiFi2 matmuls and HiFi4 SDPA (about 30% slower decode).

| QB2 TP4 | default | `MUSE_LOW_PRECISION=0` |
|---|---|---|
| GSM8K test set (1319, greedy) | 94.16% | 95.38% |
| Teacher-forced top-1 vs bf16 CPU reference (3200 tokens) | 88.8% | 97.6% |
| Reference top-1 in TT top-5 | 99.5% | 100% |

## Performance

QB2 (TP4), default precision, warm kernel cache. Batch 1, greedy, `test_muse_glimmer_isl`; ISL is the prompt length in Muse tokens. Prompts over 32768 tokens run global-layer attention in chunks against the paged KV cache.

| ISL | TTFT (s) | tok/s/user |
|---|---|---|
| 135 | 0.18 | 54.1 |
| 214 | 0.12 | 54.0 |
| 1015 | 0.21 | 53.3 |
| 1785 | 0.31 | 53.1 |
| 3897 | 0.68 | 52.8 |
| 7651 | 1.31 | 52.5 |
| 16337 | 2.68 | 52.1 |
| 30566 | 5.34 | 51.8 |
| 61362 | 12.20 | 50.1 |
| 130912 | 33.36 | 48.8 |

QB2 (TP4), `test_muse_glimmer_demo`, ISL ~64 tokens:

| | TTFT (s) | tok/s/user | tok/s |
|---|---|---|---|
| Batch 1, greedy | 0.16 | 53.9 | 54 |
| Batch 32, greedy | 2.37 | 52.7 | 1644 |
| Batch 32, top-p | 2.36 | 52.7 | 1640 |

### Blackhole Galaxy (earlier version)

These numbers were measured before the current decode and prefill optimizations and before the low-precision default. They are not re-measured.

Batch 1, greedy (TP4 on a 1x4 submesh, same code path as QB2).

| ISL | TP4 TTFT (s) | TP4 tok/s/user | TP8 TTFT (s) | TP8 tok/s/user |
|---|---|---|---|---|
| 128 | 0.20 | 23.8 | 0.21 | 29.9 |
| 256 | 0.21 | 23.5 | 0.20 | 29.7 |
| 1024 | 0.38 | 23.1 | 0.33 | 29.0 |
| 2048 | 0.63 | 22.5 | 0.53 | 28.6 |
| 4096 | 1.18 | 22.2 | 0.98 | 27.6 |
| 8192 | 2.42 | 21.7 | 1.81 | 26.8 |
| 16384 | 5.50 | 20.9 | 3.89 | 25.4 |
| 32768 | 14.11 | 18.8 | 9.27 | 22.7 |

Batch 32, ISL ~64 tokens:

| | tok/s/user | tok/s |
|---|---|---|
| TP4 | 23.4 | 728 |
| TP8 | 29.4 | 915 |
