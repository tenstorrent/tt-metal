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

## DFlash2 speculative decoding

The [z-lab/Muse-Glimmer-30B-DFlash2](https://huggingface.co/z-lab/Muse-Glimmer-30B-DFlash2) drafter proposes a block of tokens, and one 32-row target step verifies the blocks of all users. QB2 (TP4) only, 1 to 8 users, up to 131072 tokens per user. Drafts per user per round: 15 for 1-2 users, 7 for 3-4, 3 for 5-8. Greedy output follows plain greedy decode (it can differ at bf16 near-ties); sampled output uses lossless rejection sampling.

```bash
export MUSE_DFLASH_MODEL=z-lab/Muse-Glimmer-30B-DFlash2   # repo id or a local snapshot directory
MESH_DEVICE=P300x2 pytest models/demos/muse_glimmer_30b/demo/text_demo.py -k "test_muse_glimmer_dflash"
```

`test_muse_glimmer_dflash` runs batch 1 at ISL 128, 8k, 64k and 128k, greedy and sampled; pick one with e.g. `-k "dflash and isl8k and greedy"`.

### Batch 1 (demo)

`test_muse_glimmer_dflash`, 128 new tokens, warm kernel cache. Sampled: temperature 1.0, top-p 0.95, top-k 64 (the checkpoint's generation config). Accepted = tokens per verify round (at most 16). Plain decode is the batch 1 table above.

| ISL | TTFT (s) | Greedy tok/s/user | Greedy accepted | Sampled tok/s/user | Sampled accepted | Plain decode tok/s/user |
|---|---|---|---|---|---|---|
| 135 | 0.11 | 259.4 | 7.94 | 290.9 | 9.07 | 54.1 |
| 7649 | 1.57 | 271.7 | 9.07 | 211.0 | 7.47 | 52.5 |
| 61360 | 11.98 | 185.6 | 7.06 | 224.8 | 9.07 | 50.1 |
| 130896 | 31.12 | 190.7 | 9.07 | 202.8 | 9.77 | 48.8 |

The demo prompts make the model restate the prompt at the start of its reasoning, which the drafter predicts well. Longer and more varied outputs accept fewer tokens; see the serving numbers below.

### Serving (tt-inference-server, vLLM)

Model spec `impl: muse_glimmer_30b_dflash2` (dev), [vllm-tt-plugin](https://github.com/tenstorrent/vllm-tt-plugin) main with device-sampled speculative ids (`spec_ids_are_sampled`). `max_num_seqs` 8, context 131056 tokens per user, sampling on device in decode (`sample_on_device_mode: decode_only`). Default sampling is temperature 1.0, top-p 0.95, top-k 32 (the device sampler supports top-k up to 32). Server start takes about 330 s; about 220 s of it compiles all 33 prefill lengths before the first trace capture.

SPEED-Bench throughput split through `launchers/run_spec_decode.py --spec-decode-preset throughput`: 32 requests per point, up to 8192 output tokens, natural EOS, default sampling. The public dataset copy has about 42% placeholder prompts. Accepted = mean tokens per verify round from the vLLM spec-decode metrics. Output tok/s is over the whole run, so it includes prefill time and the tail with fewer active requests.

| Users | ISL | Accepted | Median TTFT (s) | Median TPOT (ms) | Output tok/s |
|---|---|---|---|---|---|
| 8 | 1k | 2.48 | 0.49 | 16.4 | 330 |
| 8 | 2k | 2.36 | 0.86 | 17.2 | 375 |
| 8 | 8k | 2.26 | 2.04 | 21.4 | 321 |
| 8 | 16k | 2.33 | 3.78 | 24.3 | 271 |
| 8 | 32k | 2.27 | 7.80 | 32.7 | 198 |
| 1 | 1k | 3.03 | 0.44 | 10.5 | 92 |

With sampling at temperature 1.0, acceptance on this dataset is low. At 1 user, TPOT is 10.5 ms against about 18.8 ms for plain decode at ISL 1k (1.8x). At 8 users (3 drafts per user), TPOT is close to plain decode, which stays near 19 ms per token up to batch 32; for throughput at many users, plain decode at batch 32 is faster. The TPOT under load also includes the stalls from other users' prefills.
