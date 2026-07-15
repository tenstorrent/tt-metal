# Qwen3-32B functional decoder

This stage translates both forward paths shipped by the flat TTNN IR into one correctness-first Qwen3 decoder layer. It implements prefill and single-token decode on a 1x1 mesh, keeps the emitted batch of 32, and uses full unsharded Hugging Face weights. It does not contain optimized-decoder, multichip, full-model, or serving work.

## IR provenance

Source directory:

`/home/mvasiljevic/qb2-irs/ttnn-mlir-ttnn_qwen_3_32b_tp_qb2_bs32_isl128_1784008145090`

The graph classifier reported:

| Selected compiler graph | Role | `fill_cache` | `paged_update_cache` | decode SDPA | Logits | Runtime |
| --- | --- | ---: | ---: | ---: | --- | --- |
| `ttnn_qwen_3_32b_tp_qb2_bs32_isl128_runcf4a_g0_1784008145090.mlir` | prefill | 4096 | 0 | 0 | no | no |
| `ttnn_qwen_3_32b_tp_qb2_bs32_isl128_runcf4a_g1_1784008729649.mlir` | decode | 0 | 128 | 64 | no | no |

The counts are classifier signals over the flat 64-layer graph, not sequence lengths. Graphs g2/g3 repeat the same paths with logits, and the four `ttnn_runtime_*` files are runtime variants; none was used as the translation source.

Both selected compiler graphs were lowered with `scripts/ir_to_emit.sh`, and the lowered Python plus raw MLIR were read together. The raw shapes establish batch 32, hidden width 5120, a 17-token prefill graph, a one-token decode graph, and source KV caches shaped `[32, 2, 128, 128]` per TP shard. Layer 32 is the representative middle layer segmented from the repeated flat graph.

The source mesh is 1x4 (TP degree 4): each shard has 16 Q heads, 2 KV heads, and a local fused QKV width of 2560. The translation restores the dense 64-Q-head/8-KV-head Qwen3 math with a fused width of 10240. Source all-reduces after the attention output and MLP down projections are therefore represented by dense projections over full HF weights; no collective or source mesh/layout glue remains in the runtime path.

In both selected emits, the layer-32 fused QKV weight and the O, gate, up, and down projection weights are cast to `BFLOAT8_B` after load-time fusion or transposition; RMSNorm and Q/K norm weights remain BF16. This correctness-first functional stage intentionally normalizes all weights and activations to BF16. It does not claim or select the compiler-emitted BF8 precision policy; precision selection belongs to a later datatype stage.

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

The state dictionary may use canonical `model.layers.<layer_idx>.*` keys, language-model-prefixed keys, or layer-local keys. The constructor validates the Qwen3-32B architecture and a 1x1 mesh, performs Q/K/V fusion and weight transposes on the host, and prepares RoPE tables and position indices.

Runtime forwards:

```python
decoder.prefill_forward(hidden_states, key_cache, value_cache)
decoder.decode_forward(hidden_states, key_cache, value_cache, *, current_pos)
```

- `hidden_states`: TTNN tensor `[1, batch, seq, 5120]`; prefill accepts `1 <= seq <= max_cache_len`, decode requires `seq == 1`.
- `key_cache`, `value_cache`: TTNN tensors `[batch, 8, max_cache_len, 128]`.
- return: TTNN tensor with the same hidden-state shape.
- prefill fills cache slots `[0:seq]`; decode appends K/V in place at `current_pos` and runs decode SDPA over the persistent cache.

Runtime methods contain no `torch`, `ttnn.from_torch`, `ttnn.to_torch`, or host fallback. Functional defaults are BF16, TILE layout, and DRAM, distinct from the source weight policy recorded above. Decode retains only the minimal L1 head layouts required by `nlp_create_qkv_heads_decode`, paged cache update, and head concatenation; compiler-selected TP layouts and program configurations are intentionally excluded.

## Correctness

All tests use the emitted batch 32 and compare against `transformers.Qwen3DecoderLayer` with eager attention. Cache PCC checks compare rotated K and V against Hugging Face `DynamicCache` state.

| Weights | Path | Case | Output PCC | K-cache PCC | V-cache PCC |
| --- | --- | --- | ---: | ---: | ---: |
| synthetic | prefill | seq 4 | 0.999348 | 0.999904 | 0.999867 |
| synthetic | prefill | emitted seq 17 | 0.999480 | 0.999903 | 0.999869 |
| synthetic | prefill | larger seq 128 | 0.999640 | 0.999901 | 0.999867 |
| synthetic | decode | position 17 | 0.999578 | 0.999902 | 0.999850 |
| Qwen3-32B checkpoint, layer 32 | prefill | seq 17 | 0.998887 | 0.999900 | 0.999865 |
| Qwen3-32B checkpoint, layer 32 | decode | position 17 | 0.998576 | 0.999902 | 0.999855 |

The real-weight output checks exceed the preferred 0.995 PCC aim. The synthetic test also covers small, emitted, and larger prefill lengths. `tests/test_functional_decoder.py` contains the host-fallback audit.

## Context and limitations

The HF-advertised context is 40,960. An isolated Blackhole p300c capacity sweep at the emitted batch 32 passes 4,096 tokens and fails at 4,097 because TILE padding raises it to the 4,128 allocation class. The context contract therefore records 4,096 as the current single-device functional limit with device-DRAM evidence. At 40,960, one BF16 `[32, 40960, 25600]` MLP activation is 67,108,864,000 bytes, already larger than the device's 34,178,731,008 DRAM bytes. This stage does not claim the advertised context.

This is a functional baseline, not a performance result. The compiler IR originated with static prefill-17 and decode-1 shapes; the translated prefill implementation generalizes sequence length within the recorded device capacity, while decode remains a single-token path as emitted. It does not report warmed latency, throughput, profiler data, or optimized layouts. It translates one representative decoder layer only; embeddings, the block stack, final norm, logits, generation, multichip execution, tracing, and vLLM integration remain out of scope.
