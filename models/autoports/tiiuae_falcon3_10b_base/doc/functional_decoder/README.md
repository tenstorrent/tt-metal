# Falcon3-10B-Base functional decoder

This stage translates the supplied compiler-emitted TTNN IR into one correctness-first Falcon3 decoder layer. It is a single-device dense implementation: the source graph's 1×4 tensor-parallel shards and `all_reduce` operations are collapsed into full Hugging Face weights and dense matmuls. Compiler-selected shard grids, program configs, and collective layout glue are not carried into the runtime.

## IR provenance and classification

IR root:

`/home/mvasiljevic/qb2-irs/ttnn-mlir-ttnn_falcon3_10b_tp_qb2_bs32_isl128_1784012293355/ttnn_falcon3_10b_tp_qb2_bs32_isl128_1784012293355`

The selected non-runtime, non-logits graphs are:

| Path | Compiler graph | `fill_cache` | `paged_update_cache` | decode SDPA | Workload evidence |
|---|---|---:|---:|---:|---|
| Prefill | `ttnn_falcon3_10b_tp_qb2_bs32_isl128_runbd43_g0_1784012293355.mlir` | 2560 | 0 | 0 | Batch 32, sequence 17, full-cache fills into length-128 KV tensors |
| Decode | `ttnn_falcon3_10b_tp_qb2_bs32_isl128_runbd43_g1_1784012509714.mlir` | 0 | 80 | 40 | Batch 32, one token, persistent KV inputs, position-driven append and decode SDPA |

`g2` and `g3` classify as the same prefill/decode paths with full-vocabulary logits tails. The functional layer uses `g0`/`g1` because the logits tail is outside a decoder-layer boundary. Runtime variants were not translated.

Both selected graphs were converted with `.agents/skills/forge-functional-decoder-from-ir/scripts/ir_to_emit.sh`. The resulting flat emits contain 40 repeated layers, matching `AutoConfig`; the prefill emit has 38,619 lines and the decode emit has 11,159. Layer 20 was segmented as the representative middle layer. Raw MLIR records `meshShape = 1x4`, and all reductions use TP cluster axis 1.

The translated block is:

1. input RMSNorm (`epsilon=1e-6`);
2. fused Q,K,V matmul, with the const-eval fusion order Q then K then V;
3. 12 query heads and 4 KV heads at head dimension 256;
4. HF-style RoPE with theta 1000042 and scale `1/sqrt(256) = 0.0625`;
5. cache fill/update plus prefill or decode SDPA;
6. output projection and residual add;
7. post-attention RMSNorm;
8. `silu(gate_proj(x)) * up_proj(x)`, down projection, and residual add.

The IR's local projection widths (Q=768, K=256, V=256, MLP=5760) are one quarter of their dense widths. `from_state_dict` instead loads full unsharded HF tensors (Q=3072, K=1024, V=1024, MLP=23040), pre-transposes all matmul weights, fuses Q,K,V at load, and uses no collectives at runtime. The compiler capture's weight and cache policy is predominantly BF8; this correctness baseline deliberately normalizes weights and caches to bf16.

## Runtime contract

`FunctionalDecoder` subclasses `LightweightModule` and is built with:

```python
FunctionalDecoder.from_state_dict(
    state_dict,
    hf_config=hf_config,
    layer_idx=layer_idx,
    mesh_device=device,
    batch=32,
    max_cache_len=6528,
)
```

Canonical `model.layers.{layer_idx}.*`, layer-local, `layers.{layer_idx}.*`, and `model.language_model.layers.{layer_idx}.*` keys are accepted. Weights, activations, RoPE tables, and linear caches default to bf16, TILE layout, and DRAM. Decode uses only the minimal batch/head L1 sharding required by QKV head creation, cache update, decode SDPA, and head concatenation.

Forward signatures and logical shapes:

```python
prefill_forward(
    hidden_states,                 # [1, 32, seq_len, 3072]
    *,
    key_cache, value_cache,        # each [32, 4, max_cache_len, 256]
) -> [1, 32, seq_len, 3072]

decode_forward(
    hidden_states,                 # [1, 32, 1, 3072]
    *,
    key_cache, value_cache,        # persistent, mutated in place
    cache_position,                # device int32 [32]
    position_index,                # common scalar position for RoPE table slice
) -> [1, 32, 1, 3072]
```

Prefill writes the rotated K and V values for every batch slot, then evaluates causal attention over the in-memory sequence. Decode appends the single K/V token in place and reads the persistent cache through decode SDPA. As in `g1`, the decode path builds an inclusive device-side `position >= cache-index` mask, repeats it over query heads, and calls SDPA in explicit-mask/non-causal mode. The emitted Q path's height-sharded RoPE output is converted to interleaved L1, sliced to the exact head count, and moved to DRAM before SDPA. The source workload uses one common decode position repeated over all 32 users; the translated signature preserves that contract.

Runtime forward/helper methods contain no `torch`, `ttnn.from_torch`, `ttnn.to_torch`, or host fallback. Host preprocessing is restricted to `from_state_dict` and test boundaries.

## Validation

| Weights | Path | Batch | Sequence/position | PCC vs dense HF layer 20 | Status |
|---|---|---:|---:|---:|---|
| Synthetic bf16 | Prefill | 32 | 17 | 0.99893845 | pass |
| Synthetic bf16 | Prefill | 32 | 128 | 0.99897713 | pass |
| Synthetic bf16 | Decode Q before SDPA | 32 | position 17 | 0.99991428 | pass |
| Synthetic bf16 | Decode | 32 | position 17 | 0.99911663 | pass |
| Synthetic bf16 | Prefill K/V cache | 32 | prefix 17 | 0.99991646 / 0.99991996 | pass |
| Synthetic bf16 | Post-decode K/V cache | 32 | prefix 18 | 0.99991628 / 0.99991987 | pass |
| Real Falcon3-10B | Prefill | 32 | 17 | 0.99879820 | pass |
| Real Falcon3-10B | Decode | 32 | position 17 | 0.99851787 | pass |
| Static | Runtime fallback audit | — | both paths | forbidden-token audit passed | pass |

The real-weight test reads all nine layer-20 tensors from `model-00003-of-00005.safetensors` and compares against `transformers.models.llama.modeling_llama.LlamaDecoderLayer` at the emitted batch. The synthetic decode test additionally proves that prefill K/V caches match HF before decode, that the rotated decode Q matches HF, and that K/V remain correct after the in-place append through position 17. These TTNN-vs-dense-HF results confirm the TP4 collective collapse.

The compiler-emitted cache length remains 128 and is used for the numerical tests above. A separate serialized capacity search ran the complete translated prefill layer at batch 32 with bf16 TILE/DRAM tensors. Sequences 256, 512, 1024, 2048, 4096, 6144, 6400, and 6528 passed through output materialization. Sequence 6560, the next tile-aligned candidate, failed at the SwiGLU gate/up multiply with a TTNN DRAM out-of-memory error while requesting its 9,673,113,600-byte output. The default `max_cache_len=6528` and exact-cache-shape runtime invariant therefore match the measured single-device boundary. This is execution/capacity evidence; PCC evidence is deliberately reported separately in the table.

## Limitations

- This is the correctness-first functional stage. It intentionally has no optimized-decoder, multichip runtime, full-model, generator, or vLLM implementation.
- The current single-device contract is capped at the measured 6528-token boundary, below Falcon3's advertised 32768 positions. At preserved batch 32, one dense bf16 `[batch,32768,23040]` MLP buffer is 45 GiB and gate/up/product buffers alone are 135 GiB. Later stages need chunking or optimized layouts to extend context.
- The compiler capture used BF8 weights/caches in many optimized operations. This functional baseline normalizes weights and caches to bf16. The emitted policy is provenance for a later datatype sweep, not the functional default.
- Watcher, Tracy/device-profiler, long-context numerical PCC, paged page-table, warmed timing, and serving evidence are not claimed by this scoped IR-translation stage.
