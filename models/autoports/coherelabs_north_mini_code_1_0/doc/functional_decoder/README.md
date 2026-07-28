# North-Mini-Code-1.0 functional decoder

Status: complete. Fresh independent re-review verdict: `clean-pass`.

## Contract

The implementation is
`models/autoports/coherelabs_north_mini_code_1_0/tt/functional_decoder.py`.
It subclasses `LightweightModule` and loads canonical Hugging Face weights through:

```python
FunctionalDecoder.from_state_dict(
    state_dict,
    hf_config=config,
    layer_idx=layer_idx,
    mesh_device=mesh_device,
    batch=batch,
    max_cache_len=max_cache_len,
    page_size=32,
)
```

Prefill:

```python
prefill_forward(
    hidden_states,
    *,
    key_cache,
    value_cache,
    page_table,
    position_cos=None,
    position_sin=None,
)
```

Decode:

```python
decode_forward(
    hidden_states,
    *,
    key_cache,
    value_cache,
    page_table,
    current_positions,
    position_cos=None,
    position_sin=None,
)
```

Prefill uses `[1, batch, sequence, 2048]`; decode uses
`[1, batch, 1, 2048]`. The physical cache is
`[num_blocks, 4, page_size, 128]`. The page table is an INT32 device tensor.
Decode positions and selected cos/sin rows are stable caller-owned device tensors,
so the whole forward graph can be captured and replayed without a Python position.

The model has three meaningful decoder kinds:

| Representative | Attention | RoPE | MLP |
|---|---|---|---|
| layer 0 | full | forced | dense SwiGLU, intermediate 3072 |
| layer 1 | sliding window 4096 | yes | 128 experts, sigmoid top-8 |
| layer 4 | full | no | 128 experts, sigmoid top-8 |

Cohere RoPE pairs adjacent dimensions. Setup permutes Q/K projection rows to
split-half ordering, making TTNN's device RoPE algebraically equivalent while
preserving QK dot products.

## Correctness and capability evidence

| Claim | Evidence | Result | Remaining risk |
|---|---|---|---|
| dense/full/forced-RoPE prefill | synthetic layer 0, logical length 33 | PCC 0.999737 | none observed |
| dense paged decode | synthetic layer 0, position 33, trace replay output | PCC 0.999768 | none observed |
| sliding-RoPE-MoE prefill | real-statistics-derived nonzero router/experts, length 1025, selected tokens across chunk boundary | PCC 0.999827 | none observed |
| full/no-RoPE-MoE prefill | real-statistics-derived nonzero router/experts, length 33 | PCC 0.999763 | none observed |
| sliding/RoPE/MoE decode | populated permuted paged history, updated hidden/position/cos/sin buffers, replay at position 4097 | PCC 0.999849 | none observed |
| full/no-RoPE/MoE decode | nonzero router/experts, updated trace buffers | PCC 0.999823 | none observed |
| served MoE trace shape | layer 1, batch 32, nonzero router/experts, updated hidden buffer | PCC 0.998193 | none observed |
| real gate and experts | official revision `d11e61a...`, layer 1, all real tensors | decode PCC 0.999798; selected experts `[107,119,126,14,61,79,18,20]` | none observed |
| page-table placement | batch-2 prefill, reversed physical blocks, K and V checked block-by-block | PCC >= 0.995 | none observed |
| current positions | batch-4 positions `[5,17,31,63]`, reversed page table, physical K/V slots checked | PCC >= 0.995 | none observed |
| determinism | identical decode repeated against the same cache slot | bitwise-equal output | none observed |
| served trace shape | batch-32 complete forward compile/capture/replay | PCC >= 0.995 | none observed |
| sliding window | controlled nonzero-V layer-1 reference through logical length 4097 | PCC >= 0.995 | none observed |
| full advertised prefill | dense/full layer 0 at logical length 500000 | finite output; 159559.7 ms single pass | costly but supported |
| near-limit nonaligned prefill | all three kinds at logical length 499999 | finite output; 159650.6/22634.4/176343.7 ms | none observed |
| full advertised decode | all kinds at batch 1; dense at batch 32, position 499999 | traced finite output; see capability JSON | none observed |
| runtime fallback audit | source audit plus filtered profiler windows | no torch/from/to-torch/program config; zero host ops | none observed |
| watcher | all 22 tests under `TT_METAL_WATCHER=10` | pass; no suspicious watcher signature | none observed |

The functional path uses BF16, tile layout, DRAM defaults, and framework-default
compute kernels. It has no runtime matmul or SDPA program configuration. Decode
RoPE, paged update/SDPA, and head concat use only the required batch-derived L1
height-sharded layouts.

The MoE baseline intentionally evaluates all experts with ordinary batched TTNN
matmuls, then applies the exact sigmoid top-8 routing weights. This avoids
introducing the mandatory hand-authored program configuration required by the
current `ttnn.sparse_matmul` API; sparse expert topology belongs to optimization
unless the API gains a framework-derived default. It is fully device resident and
matches official real weights at PCC 0.999798.

Logical lengths 1/31/32/33/65 cover tile and page behavior. Layer-1 full
decoder probes cover MoE chunk lengths 1023/1024/1025 and sliding-window lengths
4095/4096/4097. Long non-divisible dense prefill passed at 8193 and 65537.
All three layer kinds then passed at the near-limit nonaligned length 499999,
and dense/full passed the advertised 500000-token length. `context_contract.json`
records the complete capability evidence and cache byte calculations. No context
reduction is applied.

Official layer-1 statistics for all 390 consumed checkpoint tensors (name,
shape, dtype, mean, and standard deviation) are in
`real_layer1_weight_stats.json`.

## Performance

All rows are warmed batch-1 runs at sequence 128 unless noted. Decode uses
complete-forward trace replay. Tracy rows are filtered between explicit start
and end signposts; `Device Time` is reported in microseconds by
`tt-perf-report` 1.2.8.

| Layer kind | Wall prefill | Tracy prefill | Wall decode | Tracy decode |
|---|---:|---:|---:|---:|
| dense/full/forced-RoPE | 0.636 ms | 586 us | 0.356 ms | 338 us |
| sliding/RoPE/MoE | 14.908 ms | 14644 us | 9.528 ms | 9452 us |
| full/no-RoPE/MoE | 14.655 ms | 14567 us | 9.524 ms | 9439 us |
| dense batch-32 decode | n/a | n/a | 6.652 ms | 6614 us |
| sliding/RoPE/MoE batch-32 decode | n/a | n/a | 11.122 ms | 11084 us |
| full/no-RoPE/MoE batch-32 decode | n/a | n/a | 11.129 ms | 11077 us |

Each Tracy directory contains the raw ops CSV, filtered CSV, human-readable
table, and command/provenance log. Every measured window reports zero host ops.

## Reproduction

```bash
pytest -q models/autoports/coherelabs_north_mini_code_1_0/tests/test_functional_decoder.py

python models/autoports/coherelabs_north_mini_code_1_0/tests/functional_decoder_capacity.py \
  --mode prefill --context 500000 --batch 1

python models/autoports/coherelabs_north_mini_code_1_0/tests/functional_decoder_capacity.py \
  --mode decode --context 500000 --batch 32

TT_METAL_WATCHER=10 TT_METAL_LOGS_PATH=/tmp/north-watcher \
  pytest -q models/autoports/coherelabs_north_mini_code_1_0/tests/test_functional_decoder.py
```

The real-weight test discovers the two official layer-1 shards through `HF_HOME`
or `/huggingface/hub`; alternatively set `NORTH_MINI_REAL_WEIGHT_DIR`.

## Anomaly and recovery record

The first 500000-token prefill command intentionally used a five-minute bound
but performed both a warmup and a measured pass; it timed out during the
duplicated workload and left device 3 active. A post-timeout mesh smoke caught
the stale cores. `/home/mvasiljevic/.ttsmi-venv/bin/tt-smi -r 3` reset only that
device, all four boards relisted, and the mesh smoke passed. The corrected
single-pass probe then completed in 159.560 seconds. This was infrastructure
recovery after externally terminating an active kernel, not a decoder failure.
The exact list/reset/smoke and failed-attempt logs are preserved here.
