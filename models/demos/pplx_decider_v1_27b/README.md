# pplx-decider-v1-27b on Blackhole

- `perplexity-ai/pplx-decider-v1-27b`, a decision model fine-tuned from Qwen3.8-27B, on the qwen36 TP model.
- Batch 1, prefill only.

## Setup

```bash
hf download perplexity-ai/pplx-decider-v1-27b
export TT_CACHE_PATH=$HOME/tt_cache/pplx_decider_v1_27b
```

- Set `HF_MODEL` to a local checkpoint dir to skip the download.
- Keep `TT_CACHE_PATH` pplx-specific: the cache path does not include the model name.

## QuietBox 2 (1x4, TP4)

```bash
export MESH_DEVICE=P150x4
pytest models/demos/pplx_decider_v1_27b/demo/demo.py         # demo
pytest models/demos/pplx_decider_v1_27b/tests/test_model.py  # accuracy vs the CPU reference
pytest models/demos/pplx_decider_v1_27b/tests/test_perf.py   # latency, 256 to 4096 tokens
```

## Galaxy (4x8, SP4 x TP8)

```bash
unset TT_VISIBLE_DEVICES MESH_DEVICE
export PPLX_PARENT_MESH=4x8 PPLX_SUBMESH=4x8
pytest models/demos/pplx_decider_v1_27b/demo/demo.py
pytest models/demos/pplx_decider_v1_27b/tests/test_model.py
pytest models/demos/pplx_decider_v1_27b/tests/test_perf.py
```

- Prompts on 4x8 are limited to 4096 tokens, the largest prefill bucket: a 2D mesh has no chunked prefill, so `max_seq_len` is capped there and longer prompts raise `ValueError`.
- The tests open the full 4x8 mesh: a bare 1x8 or 1x4 open fails the fabric handshake on a Galaxy, and `TT_VISIBLE_DEVICES` breaks the fabric.
- For TP8 or TP4 on a Galaxy submesh, set one of these instead and run the same commands:

```bash
export MESH_DEVICE=P150x8 PPLX_PARENT_MESH=4x8 PPLX_SUBMESH=1x8  # 1x8, TP8
export MESH_DEVICE=P150x4 PPLX_PARENT_MESH=8x4 PPLX_SUBMESH=1x4  # 1x4, TP4
```

## Performance on Galaxy

Batch-1 decision latency (ms, p50, traced, `tests/test_perf.py`):

| Tokens | 4x8, SP4 x TP8 | 1x8 submesh, TP8 | 1x4 submesh, TP4 |
| ------ | -------------- | ---------------- | ---------------- |
| 256    | 35.9           | 41.1             | 61.2             |
| 512    | 41.2           | 52.3             | 71.4             |
| 1024   | 49.9           | 72.8             | 110.1            |
| 2048   | 62.9           | 130.0            | 195.2            |
| 3072   | 77.7           | 198.0            | 331.8            |
| 4096   | 92.7           | 270.6            | 440.8            |
