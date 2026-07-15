# Mistral Small 24B functional decoder

This directory records the correctness-first translation of the compiler-emitted TTNN IR for decoder layer 20 of `mistralai/Mistral-Small-24B-Instruct-2501`. Both forward paths present in the IR are implemented: cache-filling prefill and persistent-cache single-token decode. This stage does not include optimized-decoder, multichip, full-model, or vLLM work.

## IR provenance and classification

The graph bundle is:

`/home/mvasiljevic/qb2-irs/ttnn-mlir-ttnn_mistral_small_24b_instruct_2501_tp_qb2_bs32_isl128_1784013702712`

Running `scripts/classify_graphs.sh` found four compiler graphs and four runtime mirrors:

| Compiler graph | Role | `fill_cache` | `paged_update_cache` | decode SDPA | Logits |
| --- | --- | ---: | ---: | ---: | --- |
| `..._g0_1784013702712.mlir` | prefill | 2560 | 0 | 0 | no |
| `..._g1_1784014155192.mlir` | decode | 0 | 80 | 40 | no |
| `..._g2_1784014311458.mlir` | prefill | 2560 | 0 | 0 | yes |
| `..._g3_1784014392440.mlir` | decode | 0 | 80 | 40 | yes |

The runtime mirrors have the same role/signals. The no-logit compiler graphs selected for decoder translation are the full files:

- `ttnn_mistral_small_24b_instruct_2501_tp_qb2_bs32_isl128_runc399_g0_1784013702712.mlir`
- `ttnn_mistral_small_24b_instruct_2501_tp_qb2_bs32_isl128_runc399_g1_1784014155192.mlir`

Each selected graph was converted outside the repository with `scripts/ir_to_emit.sh`. The generated flat emits were 38,539 lines for prefill and 11,159 lines for decode. The raw MLIR remained the source of truth for operand types, shapes, cache semantics, and mesh metadata.

The graph is a flat, 40-layer model program. Layer 20 was segmented as a representative middle layer by its two RMS-norm boundaries and repeating attention/MLP structure. The source mesh is `1x4`, so its local shapes are TP4 shards: Q `[1024,5120]`, K/V `[256,5120]`, O `[5120,1024]`, gate/up `[8192,5120]`, and down `[5120,8192]`. The translation uses canonical full HF shapes and drops the now-identity all-reduces on a `1x1` mesh.

## Translated math

The translated order is:

1. input RMS norm (`epsilon=9.999999747e-6`);
2. fused Q/K/V projection in the emitted Q, K, V concatenation order;
3. grouped-query head split, Q/K rotary embedding, cache fill or in-place append, and causal prefill SDPA or decode SDPA with scale `1/sqrt(128)`;
4. output projection and residual add;
5. post-attention RMS norm;
6. `down(silu(gate) * up)` and the second residual add.

This model's residual width is 5120 while its attention width is `32 * 128 = 4096`; the dense Q and O shapes intentionally reflect that distinction. The source emits mark representative weights and caches as BF8 and contain compiler-selected layouts/program configurations. The functional baseline normalizes constants, activations, and caches to BF16, TILE layout, and DRAM. Only the minimal workload-derived L1 head layout required by decode head split/cache/head-concat operations is retained; compiler grid choices and collectives are not runtime dependencies.

## Runtime contract

Setup is performed by:

```python
FunctionalDecoder.from_state_dict(
    state_dict,
    *,
    hf_config,
    layer_idx,
    mesh_device,
    batch=32,
    max_cache_len=128,
    ...,
)
```

It accepts full, unsharded HF weights and prepares/transposes constants on the host once. Runtime methods are device-only and contain no `torch`, `ttnn.from_torch`, `ttnn.to_torch`, or other host fallback:

```python
output = decoder.prefill_forward(hidden_states, key_cache, value_cache)
output = decoder.decode_forward(hidden_states, key_cache, value_cache, current_pos=position)
```

Shapes and types:

- hidden input/output: TTNN BF16 TILE DRAM `[1, batch, sequence, 5120]`;
- prefill sequence: `1..max_cache_len`, PCC-validated at 4, emitted 18, and 128; capacity-validated at 3583 for batch 32;
- decode sequence: exactly 1, with integer `current_pos` in `[0, max_cache_len)`;
- key/value caches: TTNN BF16 TILE DRAM `[batch, 8, max_cache_len, 128]`, updated in place;
- batch: `1..32`, validated at 13 and the emitted workload batch 32;
- device: a one-device (`1x1`) mesh.

## Correctness evidence

All measurements compare device output with the official Hugging Face `MistralDecoderLayer` using BF16 weights and inputs. The real-weight case loads only layer 20's two required safetensor shards from the local HF snapshot.

| Weights | Path | Batch | Sequence/position | Output PCC | Cache PCC |
| --- | --- | ---: | --- | ---: | ---: |
| synthetic | prefill | 32 | sequence 4 | 0.999417 | K 0.999863, V 0.999866 |
| synthetic | prefill | 32 | sequence 18 | 0.999533 | K 0.999864, V 0.999867 |
| synthetic | prefill | 32 | sequence 128 | 0.999601 | K 0.999862, V 0.999867 |
| synthetic | prefill | 13 | sequence 4 | 0.999427 | covered by following decode append |
| synthetic | decode | 13 | position 4 | 0.999520 | K 0.999869, V 0.999869 |
| real layer 20 | prefill | 32 | sequence 18 | 0.999970 | populated cache used by decode |
| real layer 20 | decode | 32 | position 18 | 0.999970 | K 0.999879, V 0.999884 |

The test gate requires PCC at least 0.99; all measured output paths also exceed the 0.995 aim. The runtime source audit passes for the MLP, prefill, and decode methods.

## Context and limitations

HF advertises 32,768 tokens. At the emitted batch of 32, the current dense functional path has an exact measured context boundary of 3,583 tokens: sequence 3,583 completed and copied its full `[1,32,3583,5120]` output to the host, while the adjacent sequence 3,584 reproduced a TT DRAM allocator OOM twice. The retained expected-failure probe reported a 1,174,405,120-byte allocation request with 314,321,280 bytes free per bank but a largest contiguous block of only 114,688,000 bytes per bank. A post-failure `tt-smi -ls --local` health check found all four chips, and the device fixture had closed normally.

The boundary was narrowed in isolated processes through passing lengths 256, 512, 1024, 2048, 3072, 3328, 3456, 3520, 3552, 3553, 3568, 3576, 3580, 3582, and 3583. It measures capacity, not an additional long-context PCC claim; functional PCC remains validated through sequence 128. At HF context 32,768, one `[32,32768,32768]` BF16 MLP activation alone would require 68,719,476,736 bytes, exceeding the measured 34,178,731,008 bytes of device DRAM before weights, input, KV cache, or other live values. The opt-in reproduction is `tests/test_context_capacity.py`, and the machine-readable evidence is in `doc/context_contract.json`.

This artifact is a representative single layer, not a block stack or generator. It preserves the emitted batch as the default but does not retain the source TP mesh or performance tuning. No latency/profile claim was made, and no watcher or Tracy run was needed for the scoped correctness tests. Contexts above 3,583 at batch 32 require a lower-memory execution strategy in a later authorized stage.
