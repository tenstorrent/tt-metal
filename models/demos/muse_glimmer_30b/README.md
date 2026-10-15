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
```

Blackhole Galaxy (32x Blackhole, one TP4 or TP8 replica):

```bash
MESH_DEVICE=BHGLX pytest models/demos/muse_glimmer_30b/demo/text_demo.py -k "dp1-tp4-b1-greedy"
MESH_DEVICE=BHGLX pytest models/demos/muse_glimmer_30b/demo/text_demo.py -k "dp1-tp8-b1-greedy"
MESH_DEVICE=BHGLX pytest models/demos/muse_glimmer_30b/demo/text_demo.py -k "dp1-tp8-b32-greedy"
```

## Supported

| | |
|---|---|
| ISL | up to 32768 tokens (batch 1); batch 32 runs with max sequence length 4096 |
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

Batch 32, ISL ~64 tokens:

| | tok/s/user | tok/s |
|---|---|---|
| TP4 | 23.4 | 728 |
| TP8 | 29.4 | 915 |
