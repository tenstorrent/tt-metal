# Llama 3.1 8B Instruct functional decoder

This directory records the correctness-first, single-layer TTNN translation of the compiler-emitted Llama 3.1 8B graphs. Both forward paths present in the IR are implemented: full-sequence prefill and one-token decode with persistent KV-cache append.

## IR provenance

The input artifact is:

`/home/mvasiljevic/qb2-irs/ttnn-mlir-ttnn_llama_3_1_8b_instruct_tp_qb2_bs32_isl128_1784013027476`

The graphs are nested in its like-named child directory. Running the required classifier reported:

| Compiler graph | Role | `fill_cache` | `paged_update_cache` | decode SDPA | Logits |
| --- | --- | ---: | ---: | ---: | --- |
| `ttnn_llama_3_1_8b_instruct_tp_qb2_bs32_isl128_run8032_g0_1784013027476.mlir` | prefill | 2048 | 0 | 0 | no |
| `ttnn_llama_3_1_8b_instruct_tp_qb2_bs32_isl128_run8032_g1_1784013181786.mlir` | decode | 0 | 64 | 32 | no |

The g2/g3 compiler graphs add full-model logits and have the same decoder-layer roles; runtime mirrors were not selected because they wrap the same math in trace/execute plumbing. The selected raw MLIR was read for shapes and layouts, and each selected compiler graph was lowered with `scripts/ir_to_emit.sh` to `/tmp/meta_llama_llama_3_1_8b_instruct/{prefill,decode}.py`.

The emitted graph is flat and repeats 32 decoder blocks. Layer 16 was used as the representative middle layer. Its two RMSNorm sites and layer-16 weight arguments delimit the block in both flat emits.

The capture uses a `1x4` mesh and tensor-parallel degree 4. Column-parallel Q/K/V/gate/up projections and row-parallel O/down projections are collapsed to equivalent dense single-device matmuls over the canonical, unsharded HF weights. The emitted all-reduces after O and down therefore do not appear in the runtime. QKV load-time fusion preserves the emitted Q, K, V order.

## Runtime contract

`FunctionalDecoder.from_state_dict(state_dict, *, hf_config, layer_idx, mesh_device, batch=32, max_cache_len=128, ...)` accepts a canonical HF state dict, a one-device mesh, and full unsharded layer weights. All transposes, QKV fusion, weight conversion, and RoPE-table construction happen at this setup boundary.

The public hidden-state convention is `[1, batch, seq, 4096]` in and out. Caches are mutable tensors of shape `[batch, 8, max_cache_len, 128]`.

- `prefill_forward(hidden_states, key_cache, value_cache)` accepts `1 <= seq <= max_cache_len`, fills positions `[0, seq)`, and evaluates causal self-attention over the current sequence.
- `decode_forward(hidden_states, key_cache, value_cache, *, current_pos)` accepts `seq == 1`, appends K/V at `current_pos`, and applies decode SDPA over cache positions through `current_pos`.

The emitted workload batch of 32 is the default and is used by the correctness tests. Runtime forwards contain no `torch`, `ttnn.from_torch`, `ttnn.to_torch`, or other host fallback. The minimal L1 layouts required by decode head split, cache update, and head concatenation are derived from the emitted batch; compiler-selected grids and program configurations are not retained.

The translated block is pre-attention RMSNorm, fused QKV, GQA with Llama 3.1 RoPE and scale `1/sqrt(128)`, O projection and residual, followed by post-attention RMSNorm and the emitted SwiGLU MLP (`down(silu(gate) * up)`) plus residual.

## Validation

| Test | Weights | Batch | Sequence / position | Required PCC | Result |
| --- | --- | ---: | --- | ---: | --- |
| runtime fallback source audit | n/a | n/a | both forwards | no forbidden calls | pass |
| prefill small | synthetic BF16 | 32 | seq 4 | 0.99 | 0.998865 |
| prefill captured | synthetic BF16 | 32 | seq 18 | 0.99 | 0.998430 |
| prefill parameterized batch | synthetic BF16 | 13 | seq 4 | 0.99 | 0.998912 |
| decode parameterized batch | synthetic BF16 | 13 | position 4 | 0.99 | 0.998619 |
| prefill captured | real layer 16 | 32 | seq 18 | 0.99 | 0.999986 |
| decode after prefill | real layer 16 | 32 | position 18 | 0.99 | 0.999988 |

Every output and cache gate uses the official Hugging Face `LlamaDecoderLayer` with a `DynamicCache` as its reference. Synthetic cache-fill PCCs were 0.999898 or higher. Batch-13 decode K/V append PCCs were above 0.99990; the real batch-32 decode K/V append PCCs were 0.999889 and 0.999892. Tests ran with watcher and lightweight kernel assertions through the safe pytest wrapper on the free `TT_VISIBLE_DEVICES=2,3` endpoint pair; recovery evidence for the unavailable pair is in `work_log.md`.

## Limitations

- This is a functional decoder layer only. It does not start optimized-decoder, multichip, full-model, generation, or vLLM work.
- The source was specialized for batch 32, prefill sequence 18, decode sequence 1, and cache length 128. The translation accepts batch 1 through 32, derives one decode input shard per logical user, and supplies an explicit 32-head compute sub-core grid to decode head concatenation. Batch 13 is the non-factorable, non-emitted regression. Prefill is validated at sequence 4 and 18.
- HF advertises 131072 positions. The functional-stage context contract currently records sequence 18. At the emitted batch, the BF16 input plus KV cache alone would require 48 GiB, exceeding the device allocator's measured 34178731008 bytes before weights, residual/output tensors, RoPE tables, or temporaries.
- The compiler used lower-precision projection weights (mostly BF8, with BF4 gate/up). The correctness baseline intentionally recasts all weights to BF16; datatype recovery is deferred.
- Watcher correctness checks ran for the complete hardware suite. Tracy, warmed latency, long-context execution, and paged-KV performance were not measured and are not claimed.
