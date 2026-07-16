# Llama 3.1 70B Instruct functional decoder

This directory records the correctness-first, single-layer TTNN translation of the compiler-emitted Llama 3.1 70B graphs. Both forward paths present in the IR are implemented: full-sequence prefill and one-token decode with persistent KV-cache append.

## IR provenance

The input artifact is:

`/home/mvasiljevic/qb2-irs/ttnn-mlir-ttnn_llama_3_1_70b_tp_qb2_bs32_isl128_1784014705663`

The graphs are nested in its like-named child directory. The required classifier reports:

| Compiler graph | Role | `fill_cache` | `paged_update_cache` | decode SDPA | Logits |
| --- | --- | ---: | ---: | ---: | --- |
| `ttnn_llama_3_1_70b_tp_qb2_bs32_isl128_runbb45_g0_1784014705663.mlir` | prefill | 5120 | 0 | 0 | no |
| `ttnn_llama_3_1_70b_tp_qb2_bs32_isl128_runbb45_g1_1784015812679.mlir` | decode | 0 | 160 | 80 | no |

The g2/g3 compiler graphs add full-model logits and have the same decoder-layer roles. Runtime g0-g3 mirror the compiler graphs with trace/execute plumbing, so g0 and g1 are the selected non-runtime, non-logits sources. Each was lowered with `scripts/ir_to_emit.sh` to `/tmp/meta_llama_llama_3_1_70b_instruct_ir/{prefill,decode}.py`, and the raw MLIR was read for shapes, layouts, const-eval weight transforms, and the decode mask contract.

The flat graph repeats 80 decoder blocks. Layer 39 is the representative middle layer; all nine of its real weights reside in one locally available HF shard. Its two RMSNorm sites and layer-39 weight arguments delimit the translated block in both emits.

The capture uses a `2x2` mesh and two-dimensional tensor-parallel degree 4. Per-chip projection shapes are hidden 4096, intermediate 14336, and KV width 512. Column-parallel Q/K/V/gate/up projections and row-parallel O/down projections are collapsed to dense single-device matmuls over canonical full HF weights. Post-projection all-reduces on both cluster axes therefore do not appear at runtime. QKV load-time fusion preserves the emitted Q, K, V order.

## Runtime contract

`FunctionalDecoder.from_state_dict(state_dict, *, hf_config, layer_idx, mesh_device, batch=32, max_cache_len=128, ...)` accepts canonical full HF weights and a one-device mesh. All torch use, transposes, QKV fusion, weight conversion, RoPE-table construction, and mask-constant creation happen at this setup boundary.

The hidden-state convention is `[1, batch, seq, 8192]` in and out. Mutable caches have shape `[batch, 8, max_cache_len, 128]`.

- `prefill_forward(hidden_states, key_cache, value_cache)` accepts `1 <= seq <= max_cache_len`, fills cache positions `[0, seq)`, and evaluates causal attention over the sequence.
- `decode_forward(hidden_states, key_cache, value_cache, *, current_pos)` accepts `seq == 1`, appends K/V at `current_pos`, builds the emitted additive mask on device, and evaluates decode SDPA through that position.

The emitted workload batch 32 is the default and is preserved by every hardware PCC test. Runtime forwards contain no `torch`, `ttnn.from_torch`, `ttnn.to_torch`, or host fallback. The small L1 shard layouts used for cache update and head concatenation are derived from the emitted batch. Decode SDPA retains the emitted default program configuration; an explicit-grid ablation proved unnecessary after restoring the emitted mask signature.

The translated block is pre-attention RMSNorm, fused QKV, GQA with Llama 3.1 RoPE and scale `1/sqrt(128)`, O projection and residual, followed by post-attention RMSNorm and emitted SwiGLU (`down(silu(gate) * up)`) plus residual.

## Decode mask fidelity

The selected decode IR does not use causal SDPA with a cur-position tensor. It compares the scalar cache position with `[0, ..., 127]`, selects BF16 zero for admitted entries and BF16 `-inf` for future entries, repeats the mask across the local Q-head axis, and calls decode SDPA with `is_causal=False`.

After collapsing TP, the translation constructs the same `[1, 1, 64, 128]` mask entirely with TTNN operations. Cache update indices remain device tensors used only by the two in-place cache appends. This exact signature also avoids the two extra cur-position circular buffers whose causal variant exceeded the Blackhole kernel-config buffer.

## Validation

| Test | Weights | Batch | Sequence / position | Required PCC | Result |
| --- | --- | ---: | --- | ---: | ---: |
| runtime fallback source audit | n/a | n/a | both forwards | no forbidden calls | pass |
| prefill small | synthetic BF16 | 32 | seq 4 | 0.99 | 0.997370 |
| prefill captured | synthetic BF16 | 32 | seq 18 | 0.99 | 0.996538 |
| decode after prefill | synthetic BF16 | 32 | position 18 | 0.99 | 0.995945 |
| prefill captured | real layer 39 | 32 | seq 18 | 0.99 | 0.999993 |
| decode after prefill | real layer 39 | 32 | position 18 | 0.99 | 0.999994 |

All output and cache gates use the official Transformers `LlamaDecoderLayer` with a `DynamicCache`. Synthetic prefill cache PCCs are above 0.99985; synthetic decode K/V append PCCs are above 0.99986. Real decode K/V append PCCs are 0.999850 and 0.999858. Device tests ran with watcher, lightweight kernel assertions, and triage enabled through the safe pytest wrapper on `TT_VISIBLE_DEVICES=2,3`.

## Limitations

- This artifact is a functional decoder layer only. It does not begin optimized-decoder, multichip, full-model, generation, or vLLM work.
- The source specializes batch 32, prefill sequence 18, single-token decode, and cache length 128. The translation accepts batch 1 through 32, but this stage deliberately preserves and validates the emitted batch.
- HF advertises 131072 positions, while this functional-stage contract validates sequence 18. At batch 32, the BF16 input and KV cache alone require 80 GiB, exceeding the allocator-measured 34178731008 bytes before weights and temporaries.
- The compiler emits mostly BF8 projection weights and BF4 gate/up weights. The functional baseline intentionally uses BF16 throughout; emitted datatype recovery is deferred.
- The selected IR carries per-chip layout glue and collectives that are absent after dense collapse. Only operator-required cache/head sharding remains.
- Watcher correctness checks were run. Tracy, warmed latency, long-context execution, and paged-KV performance were not measured and are not claimed.
