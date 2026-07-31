# Phi-3.5 Mini Optimized Decoder

Optimized decoder pass for `microsoft/Phi-3.5-mini-instruct` under:

`models/autoports/microsoft_phi_3_5_mini_instruct`

## Scope

This stage starts from the completed functional decoder and implements the repo-local optimized single-layer decoder path:

- `tt/optimized_decoder.py`
- `tests/test_optimized_decoder.py`
- `doc/optimized_decoder/`

The model has one meaningful decoder layer kind for this stage: dense Phi-3.5-mini self-attention plus MLP. There is no MoE layer kind in this model, no CCL in the single-device decoder layer, and no LM-head or sampling path in this goal. Multichip decoder, full-model, and vLLM work are intentionally out of scope.

## Runtime Contract

`OptimizedDecoder.from_state_dict(state_dict, *, hf_config, layer_idx, mesh_device, block_size=32, max_position_embeddings=None, **_)` is the weight-loading boundary. It accepts the canonical HuggingFace Phi-3 decoder-layer keys:

- `self_attn.qkv_proj.weight`
- `self_attn.o_proj.weight`
- `mlp.gate_up_proj.weight`
- `mlp.down_proj.weight`
- `input_layernorm.weight`
- `post_attention_layernorm.weight`

The hot `prefill_forward`, `decode_forward`, RoPE, MLP, cache update, and SDPA paths are TTNN-only. Test setup and final PCC comparison use torch at explicit boundaries; the optimized runtime callables are statically audited for forbidden `torch`, `ttnn.from_torch`, and `ttnn.to_torch` fallback tokens.

Prefill preserves the functional decoder contract: TILE-layout TTNN hidden states `[1, 1, seq_len, 3072]`, block-size-aligned paged cache writes through `paged_fill_cache`, LongRoPE short/long table selection, and return shape `[1, 1, seq_len, 3072]`.

Decode preserves the functional decoder contract: TILE-layout TTNN hidden states `[1, 1, batch, 3072]`, batch size 1, tensor `current_pos`, optional tensor `position_ids` for trace-stable RoPE lookup, paged cache updates through `paged_update_cache`, paged decode SDPA, and return shape `[1, 1, batch, 3072]`.

## Chosen Policy

| Area | Final choice |
| --- | --- |
| Activations and norms | BF16 |
| Attention weights | BFP8_B for prefill and decode |
| MLP weights | BFP8_B for prefill, BFP4_B for decode |
| KV cache | BFP8_B |
| Decode activation layout | Width-sharded L1 residual stream |
| Decode Q/K/V layout | Height-sharded L1 for cache update and paged SDPA |
| Decode weights | DRAM width-sharded |
| Decode matmuls | `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig` |
| Prefill weights | Interleaved BFP8_B |
| Prefill long matmuls | Chunked 2D multicast configs |
| Math fidelity | HiFi2 matmuls, HiFi4 norms and full prefill SDPA |
| SDPA ops | `scaled_dot_product_attention` for prefill, `paged_scaled_dot_product_attention_decode` for decode |

Large prefill uses 2048-token QKV chunks and 1024-token output/MLP chunks, with a capped `in0_block_w` for large-N MLP matmuls to avoid L1 circular-buffer overflow. Short prefill uses default interleaved TTNN matmul for the 32-token acceptance workload because DRAM-sharded short-prefill trials were slower.

## Correctness Evidence

Acceptance threshold: PCC >= 0.995 for HF-vs-TTNN prefill and decode.

Final default command:

```bash
python -m py_compile models/autoports/microsoft_phi_3_5_mini_instruct/tt/optimized_decoder.py models/autoports/microsoft_phi_3_5_mini_instruct/tests/test_optimized_decoder.py
pytest --confcutdir=models/autoports/microsoft_phi_3_5_mini_instruct -q models/autoports/microsoft_phi_3_5_mini_instruct/tests/test_optimized_decoder.py -s
```

Result: `4 passed, 2 skipped in 26.67s`.

| Test path | Weights | Prefill PCC | Decode PCC | Notes |
| --- | --- | ---: | ---: | --- |
| `test_optimized_dense_layer_synthetic_prefill_decode_pcc_and_traced_decode` | deterministic synthetic from real layer-0 stats | 0.9999909427141687 | 0.9998422357977581 | paged prefill, paged decode, non-identity page table, traced decode replay |
| `test_optimized_dense_layer_real_weights_prefill_decode_pcc` | real HF layer-0 safetensors | 0.9999880147414637 | 0.999796077448235 | real checkpoint loading and real-weight PCC |
| `test_repeated_input_determinism` | deterministic synthetic from real layer-0 stats | 0.9999911722192633 vs HF | 0.9998535538636368 vs HF | repeated identical TTNN outputs asserted PCC >= 0.9999 |
| `test_runtime_forward_fallback_audit_static` | n/a | n/a | n/a | audits optimized hot paths and helpers for forbidden runtime fallback tokens |

Functional baseline real-weight PCC was prefill `0.9999957910376245` and decode `0.9999965913259444`. The optimized prefill delta is small and remains well above the acceptance bar. The optimized decode PCC delta is material and is explained by the accepted BFP4_B decode MLP weight policy; the BFP4_B MLP trial reduced traced decode time while preserving PCC well above the functional acceptance bar.

## Stress Evidence

Full advertised decode context:

```bash
PHI35_RUN_LONG_CONTEXT=1 pytest --confcutdir=models/autoports/microsoft_phi_3_5_mini_instruct -q models/autoports/microsoft_phi_3_5_mini_instruct/tests/test_optimized_decoder.py::test_full_context_decode_current_position_and_page_table -s
```

Result: `1 passed in 18.76s`. This exercises `current_pos=131071`, `max_seq_len=131072`, target config shapes, and a full page table.

Long prefill:

```bash
PHI35_RUN_LONG_PREFILL=1 PHI35_LONG_PREFILL_LEN=32768 pytest --confcutdir=models/autoports/microsoft_phi_3_5_mini_instruct -q models/autoports/microsoft_phi_3_5_mini_instruct/tests/test_optimized_decoder.py::test_long_prefill_page_table -s
```

Result: `1 passed in 13.60s`. This exercises the chunked long-prefill matmul path and paged cache fill with a non-identity page table.

Watcher-clean optimized correctness run:

```bash
TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_ETH=1 TT_METAL_WATCHER_NOINLINE=1 TT_METAL_WATCHER_APPEND=1 TT_METAL_LOGS_PATH=models/autoports/microsoft_phi_3_5_mini_instruct/doc/optimized_decoder/watcher/2026_06_15_1334_final_split pytest --confcutdir=models/autoports/microsoft_phi_3_5_mini_instruct -q models/autoports/microsoft_phi_3_5_mini_instruct/tests/test_optimized_decoder.py::test_optimized_dense_layer_real_weights_prefill_decode_pcc -s
```

Result: `1 passed in 12.00s`.

Watcher audit:

```bash
rg -n -i "TT_FATAL|TT_THROW|exception|assert|out.of.bounds|overflow|sanit|stack overflow|noc .*bad|bad noc|l1 .*overflow|watcher.*error" models/autoports/microsoft_phi_3_5_mini_instruct/doc/optimized_decoder/watcher/2026_06_15_1334_final_split/generated/watcher/watcher.log
```

Result: no matches.

## Performance Evidence

The profiler uses a warmed prefill window and a warmed traced decode replay window from the real layer-0 weight test. Times below are from `tt-perf-report`; unit is microseconds.

| Window | Functional device | Functional device + gap | Optimized device | Optimized device + gap | Speedup by device + gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| warmed prefill | 1807.085 | 2752.018 | 1328.898 | 2612.095 | 1.05x |
| warmed traced decode replay | 1752.534 | 1826.376 | 840.860 | 923.470 | 1.98x |

Final optimized decode host-side trace replay timing was `0.983737 ms/token` over the profiler timing loop. The device-only decode time was `0.840860 ms/token`; device plus op-to-op gap was `0.923470 ms/token`.

Final optimized perf artifacts:

- `tracy/dense/ops_perf_results_2026_06_15_13_30_42.csv`
- `tracy/dense/prefill_perf_report.csv`
- `tracy/dense/prefill_perf_report.txt`
- `tracy/dense/prefill_perf_report.console.log`
- `tracy/dense/decode_perf_report.csv`
- `tracy/dense/decode_perf_report.txt`
- `tracy/dense/decode_perf_report.console.log`
- `tracy/dense/prefill_perf_report_stacked.csv`
- `tracy/dense/decode_perf_report_stacked.csv`
- `tracy/dense/perf_summary.json`
- `tracy/raw_real/profile_split_mlp_policy_final.log`

The source profiler CSV for the final policy is:

`tracy/raw_real/reports/2026_06_15_13_30_42/ops_perf_results_2026_06_15_13_30_42.csv`

Functional baseline artifacts are under `doc/functional_decoder/tracy/dense/`, with summary values copied into `tracy/dense/perf_summary.json`.

## tt-perf-report Conclusions

Final prefill report:

- 46 device ops, 0 host ops.
- QKV prefill matmul: 152 us, BFP8 weights, DRAM-bound at 191 GB/s.
- O prefill matmul: 63 us, BFP8 weights.
- MLP gate/up prefill matmul: 274 us, BFP8 weights, DRAM-bound at 188 GB/s.
- MLP down prefill matmul: 161 us, BFP8 weights.
- High op-to-op gap advice remains because this module test does not trace prefill. Prefill is a variable-length, one-shot path in this stage; traced decode was the required traced latency target.
- DRAM-sharded short prefill matmul advice was tried and rejected because it slowed the measured short-prefill acceptance workload.
- L1 input placement advice for short O/down prefill matmuls was tried and accepted.

Final decode report:

- 56 device ops, 0 host ops.
- QKV decode matmul: 122 us, DRAM-sharded BFP8 weights, marked optimized.
- O decode matmul: 50 us, DRAM-sharded BFP8 weights, marked optimized.
- MLP gate/up decode matmul: 178 us, DRAM-sharded BFP4 weights, FLOP-bound.
- MLP down decode matmul: 92 us, DRAM-sharded BFP4 weights, FLOP-bound.
- Residual high op-to-op gap advice applies only to the first layer norm, with about 4 us possible saving.
- The remaining decode matmul advice says to increase grid size from 12 for the BFP4 MLP matmuls. This was rejected for this module because the accepted decode path uses DRAM-sharded weights and `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig`, whose grid is tied to the DRAM-bank sharded layout in this TTNN path and does not expose an independent compute-grid override. Moving away from the 12-bank DRAM-sharded layout would discard the accepted DRAM-sharded decode-weight optimization and was slower or less applicable in the tested alternatives.

Data-movement audit:

- Measured prefill/decode have 0 host ops and no torch/from_torch/to_torch in the optimized hot path.
- The remaining TTNN layout ops in decode are tied to op contracts: sharded decode matmuls require width-sharded L1 activations, `nlp_create_qkv_heads_decode`/paged cache update/paged SDPA require the head tensors in their supported layouts, and Phi LongRoPE currently needs an on-device decomposed HF split-half rotation.
- The experimental `rotary_embedding_hf` composite op was inspected and rejected for Phi-3.5 mini because it requires padded `head_dim` to be 32 or divisible by 64; Phi uses `head_dim=96`, whose HF split midpoint is not tile-aligned. The Llama and fused-QK RoPE variants use tile-local transformation-matrix semantics and do not implement the required HF split-half LongRoPE mapping for `head_dim=96`.
- No remaining layout conversion in the final report was identified as a removable host fallback or removable sharding mismatch within the current TTNN op contracts.

## Tried Options

| Option | Result | Evidence |
| --- | --- | --- |
| DRAM-sharded decode weights and DRAM-sharded decode matmuls | Accepted | Decode device + gap improved from 1826.376 us to 923.470 us |
| Width-sharded decode residual stream | Accepted | Final decode report shows sharded decode setup and no host ops |
| Height-sharded decode Q/K/V for cache update and paged SDPA | Accepted | Final decode uses paged update/cache and paged SDPA with sharded layouts |
| TTNN prefill SDPA and paged decode SDPA | Accepted | Runtime uses composite SDPA ops, no host fallback |
| BFP8 attention weights | Accepted | PCC remains above bar; decode QKV/O marked optimized |
| BFP4 MLP for both prefill and decode | Rejected as all-path policy | Prefill PCC dropped to 0.9997876497432253 and untraced prefill total worsened to 2688.945 us |
| BFP8 MLP prefill plus BFP4 MLP decode | Accepted | Prefill PCC recovered to 0.9999880147414637; decode kept 923.470 us device + gap |
| DRAM-sharded short-prefill matmuls | Rejected | Trial: prefill 1918.781 us device, 3053.170 us device + gap, slower than final |
| L1 placement for short O/down prefill matmul inputs | Accepted | Trial carried into final short prefill path |
| Large-prefill chunked matmul configs | Accepted | 32768-token long-prefill stress passed |
| Composite RoPE replacement for decomposed Phi LongRoPE | Rejected | `rotary_embedding_hf` does not support head_dim 96; Llama/fused-QK variants do not match HF split-half LongRoPE semantics |
| Larger BFP4 MLP decode grid | Rejected | Not exposed independently by the accepted DRAM-sharded TTNN matmul program path |
| MoE active-expert execution | Not applicable | Phi-3.5-mini decoder layer is dense |
| Fused CCL | Not applicable | Single-device decoder layer, no CCL in scope |
| LM head and sampling optimizations | Not applicable | Decoder layer goal only; full-model and vLLM stages are out of scope |

## Optimization Checklist

| Checklist item | Status | Evidence |
| --- | --- | --- |
| Optimized decoder file exists | Complete | `tt/optimized_decoder.py` |
| Tests exercise optimized path, not functional fallback | Complete | `tests/test_optimized_decoder.py` imports `OptimizedDecoder` only |
| Prefill/decode semantics preserved | Complete | HF-vs-TTNN PCC tests, page-table tests, long-context decode, long prefill |
| Paged KV-cache behavior preserved | Complete | `paged_fill_cache`, `paged_update_cache`, non-identity page table, full page table |
| Determinism covered | Complete | repeated input test asserts TTNN output PCC >= 0.9999 |
| Representative layer-kind coverage | Complete | dense Phi-3.5-mini layer kind covered with synthetic and real layer-0 weights |
| Warmed prefill and traced warmed decode measured before/after | Complete | functional and optimized perf tables above |
| tt-perf-report human-readable tables and CSVs exist | Complete | `tracy/dense/*_perf_report.txt` and `*.csv` |
| Advice-backed actionable options tried | Complete | DRAM-sharded decode, BFP4 MLP, L1 prefill placement, prefill program configs |
| Rejected advice documented with evidence or concrete reason | Complete | tried options table and tt-perf-report conclusions |
| Canonical precision/fidelity policy chosen | Complete | BF16/BFP8/BFP4 policy table above |
| Sharded layouts used where model-applicable | Complete | decode residual/Q/K/V and decode weights are sharded |
| DRAM-sharded decode matmuls addressed | Complete | QKV/O/MLP decode use DRAM-sharded matmul configs |
| Large prefill program configs addressed | Complete | chunked 2D multicast configs, 32768-token stress |
| SDPA/composite ops addressed | Complete | prefill SDPA and paged decode SDPA |
| Memory/program/compute-kernel configs addressed | Complete | explicit memory configs, program configs, HiFi2/HiFi4 kernels |
| Runtime data movement audited | Complete | final measured windows have 0 host ops; static fallback audit passes |
| Unnecessary torch/from_torch/to_torch absent in hot path | Complete | static audit and perf reports |
| Unnecessary measured host fallback absent | Complete | final prefill/decode reports show 0 host ops |
| Stress or repeated-run coverage exists | Complete | repeated determinism, full-context decode, 32768 prefill |
| Watcher-clean correctness run exists | Complete | watcher run and no-error grep above |
| MoE active-expert execution addressed | Complete, n/a | dense Phi model has no MoE |

Within the current single-device TTNN decoder-layer capabilities, no remaining model-applicable decoder optimization item is deferred from this stage.

## Limitations

- Decode remains batch size 1, matching the functional decoder contract for this autoport stage.
- Prefill remains untraced in this module test. The optimized path reduces prefill device time and preserves quality, while traced replay is used for decode latency.
- Full 131072-token prefill is still not a practical module test because full causal attention materialization is extremely large; the optimized chunked prefill path was stress-tested at 32768 tokens.
- This stage does not include multichip, full-model generator, LM head, sampling, or vLLM serving integration.
- Use `--confcutdir=models/autoports/microsoft_phi_3_5_mini_instruct` for these tests in this checkout because the repo-root `conftest.py` imports a missing local demo module.
