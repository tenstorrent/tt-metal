# GPT-OSS 20B functional decoder

> 🗄️ **OLD / SUPERSEDED EXPERIMENT (2026-07-25).** Single-device decoder translated directly from the
> EmitPy package. This is the earlier autoport lineage that fed the old `optimized_decoder`; the current
> prettify→optimize pipeline does **not** use it (it optimizes the *prettified multichip* model on
> ttnn-models branch `mvasiljevic/gpt-oss-optimize`). See the autoport top-level `../../README.md`.

This artifact is a correctness-first, dense, single-device translation of the
pre-generated TTNN EmitPy package at `/home/mvasiljevic/emit-gptoss`. Both
shipped paths are implemented. No MLIR conversion, `ir_to_emit.sh`, or emit
regeneration was run.

## Emit provenance and segmentation

The translation sources are:

- `g0_prefill/main.py` and `consteval.py`: batch-one, sequence-17 prefill,
  `fill_cache`, and manual scaled dot-product attention.
- `g1_decode/main.py` and `consteval.py`: batch-one, one-token decode,
  `paged_update_cache`, and decode SDPA.

The SHA-256 hashes of all four files are recorded in
`multichip_provenance.json`.

Each emit is a flat 24-layer full-model program. There are 49 RMSNorm sites:
two per decoder layer plus the final model norm. Layer 12 is the representative
middle layer. Its RMSNorm sites and `model.model.layers.12.*` keys bound:

- prefill `main.py` lines 3879-4089;
- decode `main.py` lines 3318-3488.

The segmented operation order matches the Hugging Face layer:

1. input RMSNorm with epsilon `1e-5`;
2. biased Q-K-V projections, RoPE, GQA attention, attention sinks, biased O
   projection, and residual;
3. post-attention RMSNorm;
4. FP32 router, top-4 routing over 32 biased experts, interleaved gate/up
   SwiGLU, biased down projection, and residual.

The exact emitted SwiGLU order is retained: clamp gate to a maximum of 7,
clamp up to `[-7, 7]`, then compute
`(up + 1) * gate * sigmoid(1.703125 * gate)`.

## TP4 collapse

The source graph targets a 1x4 ring:

- 16 Q heads and 2 KV heads per rank;
- 8 experts per rank;
- column-parallel Q/K/V tensors and a row-parallel O tensor;
- expert-parallel gate/up/down weights and biases: each rank owns 8 complete,
  full-feature-width experts out of 32;
- sum all-reduces after O projection and after the local expert reduction;
- an expert-axis routing `mesh_partition`;
- consteval bias/sink partitions and routing-base all-gathers.

`FunctionalDecoder.from_state_dict` loads canonical full Hugging Face weights
and computes the mathematically equivalent dense result on a 1x1 mesh. Raw
MXFP4 expert blocks/scales are dequantized at load time when supplied. No
collective or mesh partition remains in a runtime forward. Structured
per-tensor sharding, emitted dtypes, boundary activations, cache tensors, and
collective placement are in `multichip_provenance.json`.

The emit stores projection and expert weights in BF8_B and its KV cache in
BF8_B. This functional baseline intentionally uses BF16 weights, activations,
and cache with TILE layout and DRAM placement by default. RMSNorm parameters
use the TTNN-required row-major form. The emitted FP32 router-input/logit
boundary is retained. Decode Q/K/V heads use the emitted workload-derived
minimal L1 height-sharded form; this is not a mesh partition.

## Runtime contract

Construction:

```python
FunctionalDecoder.from_state_dict(
    state_dict,
    *,
    hf_config,
    layer_idx,
    mesh_device,
    batch=1,
    max_cache_len=128,
    ...,
)
```

`mesh_device` must be a 1x1 `ttnn.MeshDevice`. The emitted workload batch is
fixed at one. `max_cache_len` may be from 1 through 21,248 for this dense
functional topology.

| Path | Signature | Tensor contract |
| --- | --- | --- |
| Prefill | `prefill_forward(hidden_states, *, key_cache, value_cache)` | hidden input/output `[1, 1, S, 2880]`; BF16/TILE K/V cache `[1, 8, C, 64]`; `2 <= S <= C` |
| Decode | `decode_forward(hidden_states, *, key_cache, value_cache, cache_position, cache_position_tensor, attention_mask)` | hidden input/output `[1, 1, 1, 2880]`; same K/V cache; host integer and device INT32 position identify the same slot; additive BF16/TILE/DRAM mask `[1, 1, 64, C]` |

`create_kv_cache()` creates the required BF16/TILE/DRAM cache. Host/device
tensor transfer and weight preprocessing occur only at construction or test
boundaries. The runtime methods contain no `torch`, `from_torch`, `to_torch`,
host fallback, layout conversion, collective, or mesh-partition call.

The prefill emit spells out attention primitives. The translation uses the
dedicated causal TTNN SDPA with the same scale (`0.125`), sliding window, GQA
semantics, and sink contribution. The sink is divided by the scale at load
time because TTNN SDPA applies that scale internally. Decode preserves the
emitted `is_causal=False` decode-SDPA call and explicit additive mask. The
caller supplies that mask so construction stays outside the runtime forward.
The two RMSNorms retain the emitted HiFi4/FP32-accumulation configuration;
projections, attention, router, and expert matmuls retain emitted/default
compute-kernel behavior.

Decode selects the requested cosine/sine row on device, then invokes rotary
embedding with constant index zero. This avoids a TTNN program-cache defect in
which the rotary program hash omits the host `token_idx`; without this
workaround, a position-256 call after position 17 reused position 17's rotary
offset. `AUTODEBUG.md` records the source-level diagnosis. The boundary test
validates the new K/V cache row, attention residual, and full decoder result.

## Correctness evidence

Validation ran on one Blackhole chip from a P300c endpoint with
`TT_VISIBLE_DEVICES=2,3`. The final suite passed: 6 tests in 7.66 seconds.

| Test | Output PCC | Threshold |
| --- | ---: | ---: |
| Synthetic prefill, S=17 | 0.9999918034 | 0.99 |
| Synthetic prefill, S=128 | 0.9999962547 | 0.99 |
| Synthetic prefill, S=256 | 0.9999969718 | 0.99 |
| Official real layer-12 prefill, S=17 | 0.9933185739 | 0.99 |
| Official real layer-12 decode, position 17 | 0.9993172396 | 0.99 |
| Official real layer-12 prefill, S=256 boundary primer | 0.9913088292 | 0.99 |
| Official real position-256 K cache update | 0.9999342857 | 0.99 |
| Official real position-256 V cache update | 0.9999488041 | 0.99 |
| Official real position-256 attention residual | 0.9994680592 | 0.99 |
| Official real layer-12 decode, position 256 | 0.9994801582 | 0.99 |

The real test reads the locally cached official
`openai/gpt-oss-20b` safetensors, dequantizes its canonical MXFP4 expert
tensors, constructs the Hugging Face layer on meta storage, and compares both
translated paths directly to the dense Hugging Face reference. The JUnit
artifact is `test_results.xml`.

The dense functional topology also passed a prefill capacity probe at S=21,248
and hit a reproducible device-DRAM allocation failure at S=21,249. The context
contract records the allocation details and exact commands.

## Limitations

- This is one representative decoder layer, not a full model.
- Batch one is the only emitted and supported workload.
- The supported context of 21,248 is a measured DRAM limit of the dense
  all-expert functional topology, not a model-architecture limit. The Hugging
  Face YaRN target is 131,072.
- BF8_B emitted precision is provenance only in this stage; datatype selection
  belongs to a later stage.
- This stage reports correctness, not performance.
- The emitted/load-time dense QKV weight fusion is retained. No later-stage
  fused-decoder, optimized-decoder, multichip, full-model, or vLLM
  implementation is included.
