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
Eager prefill, p50 of 5 runs after 2 warmups. One Blackhole chip of a QB2 (P300x2), 1x1 mesh.
Batch: one question repeated up to 32 times, at most 8192 padded tokens per forward.

| Prompt tokens | 128 | 256 | 512 | 1024 | 2048 | 4000 |
|---|---|---|---|---|---|---|
| B=1 latency (ms) | 46.9 | 49.1 | 90.4 | 161.1 | 283.8 | 537.2 |
| Batch size | 32 | 32 | 16 | 8 | 4 | 2 |
| Batch throughput (questions/s) | 46.6 | 26.9 | 13.9 | 7.0 | 3.7 | 1.9 |

```bash
pytest -sv models/demos/strands_agent/tests/test_perf.py
```
