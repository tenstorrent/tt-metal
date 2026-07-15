# Qwen2.5-Coder-32B-Instruct functional decoder

This stage translates both forward paths shipped by the flat TTNN IR into one correctness-first Qwen2 decoder layer. It implements prefill and single-token decode on a 1x1 mesh, keeps the emitted batch of 32, and uses full unsharded Hugging Face weights. It does not contain optimized-decoder, multichip, full-model, or serving work.

## IR provenance

Source directory:

`/home/mvasiljevic/qb2-irs/ttnn-mlir-ttnn_qwen_2_5_coder_32b_instruct_tp_qb2_bs32_isl128_1784014672995`

The graph classifier reported:

| Selected compiler graph | Role | `fill_cache` | `paged_update_cache` | decode SDPA | Logits | Runtime |
| --- | --- | ---: | ---: | ---: | --- | --- |
| `ttnn_qwen_2_5_coder_32b_instruct_tp_qb2_bs32_isl128_runc0ba_g0_1784014672995.mlir` | prefill | 4096 | 0 | 0 | no | no |
| `ttnn_qwen_2_5_coder_32b_instruct_tp_qb2_bs32_isl128_runc0ba_g1_1784015214915.mlir` | decode | 0 | 128 | 64 | no | no |

The counts are classifier signals over the flat 64-layer graph, not sequence lengths. Graphs g2/g3 repeat the same paths with logits, and the four `ttnn_runtime_*` files are runtime variants; none was used as the translation source.

Both selected compiler graphs were lowered with `scripts/ir_to_emit.sh`, and the lowered Python plus raw MLIR were read together. The raw shapes establish batch 32, hidden width 5120, a 17-token prefill graph, a one-token decode graph, and source KV caches shaped `[32, 2, 128, 128]` per TP shard. Layer 32 is the representative middle layer segmented from the repeated flat graph.

The source mesh is 1x4 (TP degree 4): each shard has 10 Q heads, 2 KV heads, a local fused QKV width of 1792, and a local MLP width of 6912. The translation restores the dense 40-Q-head/8-KV-head Qwen2 math with fused width 7168 and MLP width 27648. Source all-reduces after the attention output and MLP down projections are represented by dense projections over full HF weights; no collective or source mesh/layout glue remains in the runtime path.

The selected emits feed QKV constant evaluation in `[V, K, Q]` operand order, partition the tensors for TP, then reverse them into runtime Q-K-V order. The constructor performs the equivalent dense Q/K/V weight and bias concatenation over canonical HF tensors. In both emits, fused QKV, O, gate, up, and down projection weights are cast to `BFLOAT8_B` after load-time transforms; normalization weights and QKV bias remain BF16. This functional stage intentionally normalizes all parameters and activations to BF16. It records but does not claim the compiler-emitted BF8 precision policy.

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

The state dictionary may use canonical `model.layers.<layer_idx>.*` keys, language-model-prefixed keys, or layer-local keys. The constructor validates the Qwen2.5-Coder-32B architecture and a 1x1 mesh, performs Q/K/V fusion and weight transposes on the host, and prepares RoPE tables and position indices.

Runtime forwards:

```python
decoder.prefill_forward(hidden_states, key_cache, value_cache)
decoder.decode_forward(hidden_states, key_cache, value_cache, *, current_pos)
```

- `hidden_states`: TTNN tensor `[1, batch, seq, 5120]`; prefill accepts `1 <= seq <= max_cache_len`, decode requires `seq == 1`.
- `key_cache`, `value_cache`: TTNN tensors `[batch, 8, max_cache_len, 128]`.
- return: TTNN tensor with the same hidden-state shape.
- prefill fills cache slots `[0:seq]`; decode appends K/V in place at `current_pos` and runs decode SDPA over the persistent cache.

Runtime methods contain no `torch`, `ttnn.from_torch`, `ttnn.to_torch`, or host fallback. Functional defaults are BF16, TILE layout, and DRAM, distinct from the source weight policy recorded above. Decode retains only the minimal L1 head layouts required by `nlp_create_qkv_heads_decode`, paged cache update, and head concatenation. It also preserves the emitted post-RoPE logical-head slices: the TP-local 10-Q/2-KV slices become dense 40-Q/8-KV slices so SDPA sees the correct 5:1 grouped-query ratio rather than the 64-head padded extent.

## Correctness

All tests use the emitted batch 32 and compare against `transformers.Qwen2DecoderLayer` with eager attention. Cache PCC checks compare rotated K and V against Hugging Face `DynamicCache` state.

| Weights | Path | Case | Output PCC | K-cache PCC | V-cache PCC |
| --- | --- | --- | ---: | ---: | ---: |
| synthetic | prefill | seq 4 | 0.999447 | 0.999869 | 0.999870 |
| synthetic | prefill | emitted seq 17 | 0.999537 | 0.999869 | 0.999873 |
| synthetic | prefill | larger seq 128 | 0.999648 | 0.999865 | 0.999871 |
| synthetic | decode | position 17 | 0.999573 | 0.999850 | 0.999852 |
| Qwen2.5-Coder-32B-Instruct checkpoint, layer 32 | prefill | seq 17 | 0.998781 | 0.999877 | 0.999867 |
| Qwen2.5-Coder-32B-Instruct checkpoint, layer 32 | decode | position 17 | 0.998929 | 0.999883 | 0.999857 |

The real-weight output checks exceed the preferred 0.995 PCC aim. The synthetic test covers small, emitted, and larger prefill lengths, plus one decode step after an emitted-length prefill. `tests/test_functional_decoder.py` contains the host-fallback audit.

## Context and limitations

The target HF config advertises 32,768 positions. An isolated Blackhole p300c capacity sweep at emitted batch 32 passes 3,999 tokens and fails at the adjacent 4,000-token length. The failing final MLP projection requests a 1,310,720,000-byte output when the largest free DRAM block is 160,768,000 bytes. The context contract therefore records 3,999 as the current one-device functional limit with device-DRAM evidence. At 32,768, one BF16 `[32, 32768, 27648]` MLP activation is 57,982,058,496 bytes, already larger than the device's 34,178,731,008 DRAM bytes. This stage does not claim the advertised context.

This is a functional baseline, not a performance result. The compiler IR originated with static prefill-17 and decode-1 shapes; the translated prefill implementation generalizes sequence length within the recorded device capacity, while decode remains the emitted single-token path. It does not report warmed latency, throughput, profiler data, or optimized layouts. It translates one representative decoder layer only; embeddings, the block stack, final norm, logits, generation, multichip execution, tracing, and vLLM integration remain out of scope.
