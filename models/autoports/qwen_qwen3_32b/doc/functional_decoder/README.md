# Qwen3-32B functional decoder

This stage translates the pre-generated EmitPy graphs in
`/home/mvasiljevic/emit-qwen3` into one correctness-first Qwen3 decoder layer.
It implements both shipped paths (prefill and single-token decode) on a 1x1
mesh, preserves the emitted batch of 32, and uses full, unsharded Hugging Face
weights. No MLIR conversion or EmitPy regeneration was performed.

The stage is deliberately limited to `tt/functional_decoder.py`, its tests,
and functional-decoder documentation. It does not include fused-decoder,
optimized-decoder, multichip runtime, full-model, generation, or vLLM work.

## EmitPy provenance

The source is a flat, full-model, TP4 TTNN-to-EmitPy package with one
self-contained subpackage per graph:

| Package | Files read | Translated role | Shape/classification evidence |
| --- | --- | --- | --- |
| `/home/mvasiljevic/emit-qwen3/g0_prefill` | `main.py`, `consteval.py` | prefill | hidden `[1, 32, 17, 5120]`; `fill_cache`; no paged update or decode SDPA |
| `/home/mvasiljevic/emit-qwen3/g1_decode` | `main.py`, `consteval.py` | decode | hidden `[1, 32, 1, 5120]`; `paged_update_cache`; decode SDPA |

`utils.py` and `ttir_cpu.py` were inspected for package context, but no CPU
helper was copied into a runtime forward. The emits were not regenerated and
`ir_to_emit.sh` was never run.

The flat graph repeats 64 decoder layers. Layer 32 was selected as a
representative middle layer. In `g0_prefill/main.py`, its first input RMSNorm
starts at line 30450 and its final MLP residual add is at line 31311; layer 33
starts at line 31314. In `g1_decode/main.py`, the corresponding span is lines
5964 through 6069, with layer 33 starting at line 6073. Weight keys containing
`model.layers.32` and the four RMSNorm sites bound both segments.

The translated operation order is:

1. input RMSNorm (`eps=1e-6`);
2. fused Q/K/V projection and Q/K/V split;
3. per-head Q and K RMSNorm, RoPE, and K/V cache fill or paged update;
4. causal prefill SDPA or decode SDPA with scale `1/sqrt(128)`;
5. head concatenation, output projection, and attention residual;
6. post-attention RMSNorm;
7. SwiGLU (`silu(gate) * up`), down projection, and MLP residual.

This multiset and fusion order were cross-checked against
`transformers.Qwen3DecoderLayer`: hidden size 5120, 64 Q heads, 8 KV heads,
head dimension 128, intermediate size 25600, and two residual branches.

## TP4 collapse

The source mesh is `[1, 4]` with tensor parallelism on mesh dimension 1 and
`FABRIC_1D_RING`. Per rank it carries 16 Q heads, 2 KV heads, a Q/K/V fused
width of 2560, and an MLP width of 6400. Each representative layer has two
ring all-reduces: after `o_proj` and after `down_proj`. No all-gather,
reduce-scatter, mesh-shard, or collective-permute occurs inside either
representative layer segment.

The functional translation loads canonical full HF weights, fuses full Q/K/V
weights in Q-K-V order, and performs dense projections over 64 Q heads, 8 KV
heads, and the full 25600-wide MLP. This is mathematically equivalent to the
TP4 partial projections plus their all-reduces, so no source collective or
mesh-layout glue appears in a runtime forward. Exact tensor shapes, emitted
dtypes, shard axes, and collective placements are recorded in
`multichip_provenance.json` for a later multichip stage.

Both source constevals transpose projection weights and cast fused QKV, O,
gate, up, and down weights to `BFLOAT8_B`; normalization weights remain
`BFLOAT16`, and source caches are `BFLOAT8_B`. This correctness baseline uses
BF16 weights, activations, and caches with TILE/DRAM defaults. That intentional
normalization is not a datatype recommendation.

## Runtime contract

Construction:

```python
FunctionalDecoder.from_state_dict(
    state_dict,
    *,
    hf_config,
    layer_idx,
    mesh_device,
    batch=32,
    max_cache_len=128,
)
```

The state dictionary may use canonical `model.layers.<layer_idx>.*`,
language-model-prefixed, or layer-local keys. Construction validates the
Qwen3-32B architecture and 1x1 mesh, then performs host-side weight loading,
Q/K/V fusion, transposes, and RoPE-table preparation.

Runtime methods:

```python
decoder.prefill_forward(hidden_states, key_cache, value_cache)
decoder.decode_forward(hidden_states, key_cache, value_cache, *, current_pos)
```

- `hidden_states` is a device TTNN tensor `[1, 32, seq, 5120]`; prefill accepts
  `1 <= seq <= max_cache_len`, while decode requires `seq == 1`.
- `key_cache` and `value_cache` are device TTNN tensors
  `[32, 8, max_cache_len, 128]`.
- The output is a device TTNN tensor with the same shape as `hidden_states`.
- Prefill writes cache positions `[0:seq]`; decode updates `current_pos` in
  place and attends through that position.

Runtime forwards contain no Torch call, `from_torch`, `to_torch`, NumPy, or
host fallback. Decode uses only the L1 head sharding required by the TTNN
decode-head, paged-cache-update, and head-concatenation APIs; it contains no
source TP layout or collective.

## Correctness

Tests compare against `transformers.Qwen3DecoderLayer` using eager attention.
Cache checks compare rotated K and V with Hugging Face `DynamicCache` state.

| Weights | Path | Case | Output PCC | K-cache PCC | V-cache PCC |
| --- | --- | --- | ---: | ---: | ---: |
| synthetic | prefill | seq 4 | 0.999348 | 0.999904 | 0.999867 |
| synthetic | prefill | emitted seq 17 | 0.999480 | 0.999903 | 0.999869 |
| synthetic | prefill | larger seq 128 | 0.999640 | 0.999901 | 0.999867 |
| synthetic | decode | position 17 | 0.999578 | 0.999902 | 0.999850 |
| Qwen3-32B layer 32 | prefill | seq 17 | 0.998887 | 0.999900 | 0.999865 |
| Qwen3-32B layer 32 | decode | one step at position 17 | 0.998576 | 0.999902 | 0.999855 |

All output PCCs exceed the preferred 0.995 aim. The test suite also performs a
source-level runtime-fallback audit.

## Context and limitations

The HF-advertised context is 40,960. On one Blackhole p300c at the emitted
batch 32, a complete dense-layer prefill passes at sequence 4,096 and the
adjacent 4,097 probe reaches the 4,128 TILE allocation class and fails with a
verified DRAM OOM. One BF16 `[32, 40960, 25600]` MLP activation alone is
67,108,864,000 bytes, exceeding the device's 34,178,731,008 DRAM bytes.
`doc/context_contract.json` therefore records 4,096 as the current measured
single-device functional limit rather than claiming the advertised context.

This is a functional baseline, not a performance result. It translates one
representative layer only. The prefill implementation generalizes the emitted
17-token graph within the measured capacity; decode remains the emitted
single-token path. No warmed latency, throughput, profiler data, optimized
layout, embeddings, block stack, final norm, logits, generator, tracing,
multichip execution, or serving integration is claimed.
