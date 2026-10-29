# strands-decider-2B (StrandsAgents/strands-decider-2B-qwen3.5-v1-2610) on one Blackhole chip

## Run the e2e test
```bash
source python_env/bin/activate
export TT_METAL_HOME=$PWD ARCH_NAME=blackhole
export TT_CACHE_PATH=$HOME/tt_cache/strands_agent

# QB2 (P300x2) only: open one of its 4 chips as a single P150
export TT_VISIBLE_DEVICES=0
export TT_MESH_GRAPH_DESC_PATH=$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto
export TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0

pytest -sv models/demos/strands_agent/demo/text_demo.py
```

## Perf
Prefill replayed from ttnn traces for prompts up to 2048 tokens (eager above), p50 of 5 runs after 2 warmups. One Blackhole chip of a QB2 (P300x2), 1x1 mesh.
B=32: one question repeated 32 times in one call, split into forwards of at most 8192 padded tokens. Throughput is 32 / B=32 latency.

| Prompt tokens | 128 | 256 | 512 | 1024 | 2048 | 4000 |
|---|---|---|---|---|---|---|
| B=1 latency (ms) | 10.6 | 13.0 | 19.6 | 27.5 | 47.9 | 91.3 |
| B=32 latency (ms) | 84.9 | 155.5 | 313.7 | 634.1 | 1272.9 | 2664.2 |
| Batch throughput (questions/s) | 377.0 | 205.8 | 102.0 | 50.5 | 25.1 | 12.0 |

```bash
pytest -sv models/demos/strands_agent/tests/test_perf.py
```
