# GPT-OSS 20B functional decoder

This is a dense, single-device translation of the already-generated TTNN EmitPy package at `/home/mvasiljevic/emit-gptoss`. No MLIR conversion or emit regeneration was run.

## Provenance and segmentation

Both shipped graphs were translated:

- `g0_prefill/main.py` and `consteval.py`: batch-one, sequence-17 prefill with `fill_cache`.
- `g1_decode/main.py` and `consteval.py`: batch-one, one-token decode with `paged_update_cache` and decode SDPA.

The sources are flat 24-layer full-model programs. Layer 12 was selected as a representative middle layer. Its two RMSNorm sites and `model.model.layers.12.*` weight keys bound lines 3879-4089 of the prefill forward and 3318-3488 of the decode forward. The segmented multiset matches the Hugging Face block: input RMSNorm, biased QKV attention with attention sinks, biased O projection and residual, post-attention RMSNorm, top-4 routing over 32 biased SwiGLU experts, and the second residual.

The emit was captured on a 1x4 ring (TP4). It uses 16 local Q heads, 2 local KV heads, and 8 local experts. `FunctionalDecoder.from_state_dict` instead loads the full canonical Hugging Face tensors (including load-boundary MXFP4 dequantization when raw blocks/scales are supplied), performs one dense matmul, and skips the corresponding attention all-reduce, routing `mesh_partition`, and expert all-reduce. Full sharding, dtype, and collective placement are recorded in `multichip_provenance.json`; none of them appear in runtime forwards.

## Runtime contract

Construction:

```python
FunctionalDecoder.from_state_dict(
    state_dict,
    *,
    hf_config,
    layer_idx,
    mesh_device,
    batch=1,  # emitted default; positive batches are supported
    max_cache_len=128,
    ...,
)
```

The mesh must contain one device. Weights, activations, and cache are BF16/TILE/DRAM by default. The emitted FP32 router promotion is retained, and the decode Q/K/V head kernels use one minimal height-sharded L1 buffer derived from batch one.

| Path | Signature | Shapes |
| --- | --- | --- |
| Prefill | `prefill_forward(hidden_states, key_cache, value_cache)` | input/output `[1, B, S, 2880]`; K/V `[B, 8, 128, 64]`; `2 <= S <= 128` |
| Decode | `decode_forward(hidden_states, key_cache, value_cache, *, current_pos)` | input/output `[1, B, 1, 2880]`; K/V `[B, 8, 128, 64]`; `0 <= current_pos < 128` |

The emitted default is `B=1`; batch 2 is also device-validated on both paths. Prefill selects each batch row on device for `fill_cache`, whose TTNN input contract is one row per call, and writes it to the corresponding cache row without host fallback.

Prefill implements the emitted manual attention sequence: QKV fusion in Q-K-V order, RoPE, cache fill, GQA repeat, QK scale `0.125`, causal mask, raw sink concat, softmax/drop-sink, V aggregation, O projection and residual. Decode retains the emitted sink pre-scaling, explicit non-causal mask, paged cache updates, and decode SDPA. MoE retains interleaved gate/up fusion, gate max clamp 7, up clamp [-7, 7], `(up + 1) * gate * sigmoid(1.703125 * gate)`, biased down projection, FP32 router, top-4 softmax/scatter, expert weighting, and sum.

Runtime forwards and their MoE helper contain no `torch`, `from_torch`, `to_torch`, host fallback, collective, or mesh partition call.

## Validation

Hardware validation used one Blackhole chip, device 0, with the single-chip P150 mesh descriptor needed when this host exposes a P300c endpoint one chip at a time.

| Evidence | Output PCC | K cache PCC | V cache PCC |
| --- | ---: | ---: | ---: |
| Synthetic prefill, S=4 | 0.999826 | 0.999947 | 0.999949 |
| Synthetic prefill, S=17 | 0.999800 | 0.999945 | 0.999951 |
| Real layer-12 prefill, S=17 | 0.999193 | 0.999948 | 0.999952 |
| Real layer-12 decode, position 17 | 0.999298 | 0.999946 | 0.999949 |
| Synthetic batch-2 prefill, S=4 | 0.999832 | 0.999944 | 0.999950 |
| Synthetic batch-2 decode, position 4 | 0.999800 | 0.999944 | 0.999949 |

The real test reads locally cached `openai/gpt-oss-20b` safetensors, dequantizes canonical MXFP4 expert tensors, constructs the Hugging Face layer on meta storage, and compares both translated paths directly to Hugging Face.

## Limitations

- Batch 1 is the emitted workload and default; batch 2 is additionally validated, while larger batches are not yet measured.
- The largest validated prefill is S=17. The functional cache/mask allocation extends to 128, but S=128 is not validated. The HF-advertised 131072 context requires a different long-context memory policy; see `../context_contract.json`.
- This artifact is one representative decoder layer, not a full model.
- Precision is normalized to the BF16 functional baseline. The emitted BF8 projection/cache policy is provenance only.
- This stage's checkpoint contains only the functional-decoder artifacts listed in the forge skill. Ignored optimized/multichip/full-model/pipeline-debug files already existed under the autoport root before this stage and were neither read as translation input nor modified or committed here.
