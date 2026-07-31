# Phi-3.5 Mini Functional Decoder

Functional decoder bringup for `microsoft/Phi-3.5-mini-instruct` under:

`models/autoports/microsoft_phi_3_5_mini_instruct`

## Scope

This stage implements the single dense Phi-3.5-mini decoder layer kind only. It does not include optimized decoder, multichip decoder, full-model, or vLLM work.

Target HF config:

| Field | Value |
| --- | --- |
| hidden size | 3072 |
| intermediate size | 8192 |
| layers | 32 |
| attention heads | 32 |
| KV heads | 32 |
| head dim | 96 |
| max position embeddings | 131072 |
| original max position embeddings | 4096 |
| norm | RMSNorm, eps 1e-5 |
| RoPE | Phi LongRoPE short/long factors |
| attention bias | false |

## Runtime Contract

`FunctionalDecoder.from_state_dict(state_dict, *, hf_config, layer_idx, mesh_device, block_size=32, max_position_embeddings=None, **_)` is the weight-loading boundary. It accepts canonical HF decoder-layer keys:

- `self_attn.qkv_proj.weight`
- `self_attn.o_proj.weight`
- `mlp.gate_up_proj.weight`
- `mlp.down_proj.weight`
- `input_layernorm.weight`
- `post_attention_layernorm.weight`

The hot forward paths are TTNN-only. Test setup and final PCC comparison use torch at explicit boundaries; `prefill_forward`, `decode_forward`, and their helper paths do not call torch, `ttnn.from_torch`, or `ttnn.to_torch`.

Prefill:

```python
prefill_forward(
    hidden_states,
    *,
    page_table,
    kv_cache,
    user_id=0,
    start_pos=0,
    rope_sequence_length=None,
    cache_position_modulo=None,
)
```

`hidden_states` is TILE-layout TTNN `[1, 1, seq_len, 3072]`, with `seq_len` a multiple of the paged-cache block size. `page_table` is int32 `[max_batch, num_blocks]` or wider. `kv_cache` is `(k_cache, v_cache)` with paged cache shape `[num_blocks, 32, block_size, 96]`. The path fills the paged cache and returns `[1, 1, seq_len, 3072]`.

Decode:

```python
decode_forward(
    hidden_states,
    *,
    current_pos,
    page_table,
    kv_cache,
    position_ids=None,
    rope_sequence_length=None,
    cache_position_modulo=None,
)
```

`hidden_states` is TILE-layout TTNN `[1, 1, batch, 3072]`. `current_pos` is int32 TTNN `[batch]` and is used by paged cache update and paged SDPA. `position_ids` is optional uint32 TTNN `[batch]` for trace-stable on-device RoPE table lookup. This functional stage supports decode batch size 1. The path updates the paged cache and returns `[1, 1, batch, 3072]`.

## Correctness Evidence

Acceptance threshold: PCC >= 0.995 for HF-vs-TTNN prefill and decode.

Final default command:

```bash
python -m py_compile models/autoports/microsoft_phi_3_5_mini_instruct/tests/test_functional_decoder.py models/autoports/microsoft_phi_3_5_mini_instruct/tt/functional_decoder.py
pytest --confcutdir=models/autoports/microsoft_phi_3_5_mini_instruct -q models/autoports/microsoft_phi_3_5_mini_instruct/tests/test_functional_decoder.py -s
```

Result: `4 passed, 2 skipped in 12.99s`.

| Test path | Weights | Prefill PCC | Decode PCC | Notes |
| --- | --- | ---: | ---: | --- |
| `test_dense_layer_synthetic_prefill_decode_pcc_and_traced_decode` | deterministic synthetic from real layer-0 stats | 0.9999970054274875 | 0.9999976050540407 | paged prefill, paged decode, non-identity page table, traced decode replay |
| `test_dense_layer_real_weights_prefill_decode_pcc` | real HF layer-0 safetensors | 0.9999957910376245 | 0.9999965913259444 | proves real checkpoint loading and real-weight PCC |
| `test_repeated_input_determinism` | deterministic synthetic from real layer-0 stats | 0.9999970458001098 vs HF | 0.9999975174967267 vs HF | repeated identical TTNN outputs asserted PCC >= 0.9999 |
| `test_runtime_forward_fallback_audit_static` | n/a | n/a | n/a | audits hot paths and helpers for forbidden torch/from_torch/to_torch tokens |

Real weight stats are recorded in `weight_stats_layer0.json` with name, checkpoint key, shape, dtype, mean, and std for every tensor used by the TTNN layer. Synthetic tests use that artifact deterministically.

## Long-Shape Evidence

Full advertised decode context:

```bash
PHI35_RUN_LONG_CONTEXT=1 pytest --confcutdir=models/autoports/microsoft_phi_3_5_mini_instruct -q models/autoports/microsoft_phi_3_5_mini_instruct/tests/test_functional_decoder.py::test_full_context_decode_current_position_and_page_table -s
```

Result: `1 passed in 8.40s`. This exercises `current_pos=131071`, `max_seq_len=131072`, real target config shapes, and a full page table.

Longest feasible nonchunked prefill tested:

```bash
PHI35_RUN_LONG_PREFILL=1 PHI35_LONG_PREFILL_LEN=32768 pytest --confcutdir=models/autoports/microsoft_phi_3_5_mini_instruct -q models/autoports/microsoft_phi_3_5_mini_instruct/tests/test_functional_decoder.py::test_long_prefill_page_table -s
```

Result: `1 passed in 8.84s`. Full 131072 prefill is not feasible for this unoptimized nonchunked functional path because a materialized attention score tensor of `[1, 32, 131072, 131072]` at BF16 is about 1.0 TiB before other tensors.

## Performance Evidence

Profiler command:

```bash
PHI35_READ_DEVICE_PROFILER=1 PHI35_SKIP_MESH_CLOSE=1 python -m tracy -r -p -v --dump-device-data-mid-run -o models/autoports/microsoft_phi_3_5_mini_instruct/doc/functional_decoder/tracy/raw_real -m pytest --confcutdir=models/autoports/microsoft_phi_3_5_mini_instruct -q models/autoports/microsoft_phi_3_5_mini_instruct/tests/test_functional_decoder.py::test_dense_layer_real_weights_prefill_decode_pcc -s
```

The signposted windows use a warmed prefill pass and warmed traced decode replay. `tt-perf-report` reports `Device Time` in microseconds.

| Window | Device time | Op-to-op gap | Device + gap | Host ops |
| --- | ---: | ---: | ---: | ---: |
| warmed prefill | 1807.085 us | 944.933 us | 2752.018 us | 0 |
| warmed traced decode replay | 1752.534 us | 73.842 us | 1826.376 us | 0 |

Artifacts:

- `tracy/dense/prefill_ops.csv`
- `tracy/dense/prefill_perf_report.txt`
- `tracy/dense/prefill_perf_report.csv`
- `tracy/dense/decode_ops.csv`
- `tracy/dense/decode_perf_report.txt`
- `tracy/dense/decode_perf_report.csv`
- `tracy/dense/perf_summary.json`
- raw provenance under `tracy/raw_real/`

## Runtime Fallback Audit

The static fallback audit covers `prefill_forward`, `decode_forward`, `_mlp_forward`, RoPE helpers, and dtype helper code. It passed in the final default suite.

The filtered `tt-perf-report` CSVs show `num_host_ops: 0` in `tracy/dense/perf_summary.json` for both measured windows.

## Watcher Evidence

Clean watcher command:

```bash
TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_ETH=1 TT_METAL_WATCHER_NOINLINE=1 TT_METAL_WATCHER_APPEND=1 TT_METAL_LOGS_PATH=models/autoports/microsoft_phi_3_5_mini_instruct/doc/functional_decoder/watcher/2026_06_15_1253 pytest --confcutdir=models/autoports/microsoft_phi_3_5_mini_instruct -q models/autoports/microsoft_phi_3_5_mini_instruct/tests/test_functional_decoder.py::test_dense_layer_real_weights_prefill_decode_pcc -s
```

Result: `1 passed in 31.96s`. The watcher log is `watcher/2026_06_15_1253/generated/watcher/watcher.log`.

Audit command:

```bash
rg -n -i "TT_FATAL|TT_THROW|exception|assert|out.of.bounds|overflow|sanit|stack overflow|noc .*bad|bad noc|l1 .*overflow|watcher.*error" models/autoports/microsoft_phi_3_5_mini_instruct/doc/functional_decoder/watcher/2026_06_15_1253/generated/watcher/watcher.log
```

Result: no matches. The log contains normal stack-usage summaries, zero Ethernet retraining events, and detach lines for devices 0 through 7.

`TT_METAL_WATCHER_DISABLE_ETH=1` was used because the first watcher attempt with ETH checking enabled hit a watcher dispatch idle-ERISC code-size overflow before running the decoder. That failed attempt and the reset are recorded in `work_log.md`.

## Notes

Use `--confcutdir=models/autoports/microsoft_phi_3_5_mini_instruct` for these tests in this checkout. The repo root `conftest.py` imports `models.tt_transformers.demo.trace_region_config`, which is not present in this local tree.
