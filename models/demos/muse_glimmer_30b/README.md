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

## Performance

Batch 1, greedy, measured on a Blackhole Galaxy (TP4 on a 1x4 submesh, same code path as QB2).

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

Batch 1, greedy, measured on QB2 (TP4) with `test_muse_glimmer_isl`; ISL is the prompt length in Muse tokens. Prompts over 32768 tokens run attention in chunks against the paged KV cache.

| ISL | TTFT (s) | tok/s/user |
|---|---|---|
| 135 | 0.16 | 27.6 |
| 214 | 0.13 | 27.6 |
| 1015 | 0.32 | 26.9 |
| 1785 | 0.60 | 26.3 |
| 3897 | 1.13 | 25.8 |
| 7651 | 2.24 | 25.1 |
| 16337 | 5.53 | 23.7 |
| 30566 | 13.32 | 21.7 |
| 61362 | 25.94 | 18.3 |
| 130912 | 71.69 | 13.1 |

Batch 32, ISL ~64 tokens:

| | tok/s/user | tok/s |
|---|---|---|
| TP4 (Galaxy) | 23.4 | 728 |
| TP8 (Galaxy) | 29.4 | 915 |
| TP4 (QB2) | 27.3 | 849 |
