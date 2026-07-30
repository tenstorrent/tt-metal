# google/gemma-4-12B Functional Decoder

Date: 2026-06-08

Repo commit: `31b45719e2ca21b695a8e7f15b5e8895bc1fb3bb`

This directory contains the functional-decoder bringup evidence for the repo-local TTNN autoport of `google/gemma-4-12B`. The implementation is intentionally limited to the single-layer functional decoder under `models/autoports/google/gemma-4-12B/tt/functional_decoder.py`; no optimized decoder, multichip decoder, full model, or vLLM work is included.

## Decoder Contract

`FunctionalDecoder.from_state_dict(state_dict, *, hf_config, layer_idx, mesh_device, mesh_config=None, ...)` accepts either a full Hugging Face checkpoint state dict or a layer-only `Gemma4TextDecoderLayer.state_dict()`. Weight conversion happens at construction time. Runtime forwards consume device-resident TTNN tensors.

The supported target config is the dense Gemma 4 12B text decoder: 48 layers, hidden size 3840, MLP intermediate size 15360, 16 query heads, sliding layers with 8 KV heads/head dim 256/window 1024, and full layers with 1 KV head/head dim 512. The two meaningful layer kinds are covered by layer 0 (`sliding_attention`) and layer 5 (`full_attention`).

`prefill_forward(hidden_states, *, rope_mats, page_table, kv_cache)`:

- `hidden_states`: `[1, 1, seq_len, 3840]`, TILE layout.
- `rope_mats`: 4D TTNN cos/sin tables `[1, 1, max_seq_len, head_dim]`.
- `page_table`: int32 paged-attention table. Tests use a non-identity first-page permutation.
- `kv_cache`: `[k_cache, v_cache]` allocated with `create_paged_kv_cache`.

`decode_forward(hidden_states, *, rope_mats, page_table, kv_cache, position_idx, token_index=None, position_idx_cache=None)`:

- `hidden_states`: `[1, 1, batch, 3840]`; current tests use batch 1.
- `rope_mats`: for traced decode, 2D TTNN cos/sin tables `[max_seq_len, head_dim]`.
- `position_idx`: device tensor holding the absolute decode position.
- `position_idx_cache`: int32 device tensor used for paged cache update when `position_idx` uses an embedding-friendly dtype.
- `token_index`: only required when callers pass 4D RoPE tables to decode.

## Correctness

Acceptance defaults to PCC >= 0.995. Two lower thresholds are documented by component evidence:

- Sliding decode uses PCC >= 0.993. An exact-HF-QKV TT paged SDPA decode check at the real sliding geometry reached about 0.99457 before the full decoder residual/MLP path, while full attention cleared 0.995.
- Long-context synthetic tests use PCC >= 0.992 at sequence 1024. Component isolation showed the remaining gap is TT BF16 accumulation through long attention/MLP, not page-table, current-position, RoPE-rank, cache-source, or sliding-window routing.

Final PCC evidence is in `pcc_results.jsonl`.

| Layer kind | Layer | Weights | Seq/context | Prefill PCC | Decode/replay PCC | Threshold |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| sliding_attention | 0 | synthetic | 128 | 0.9973848698 | 0.9933475834 | prefill 0.995, decode 0.993 |
| full_attention | 5 | synthetic | 128 | 0.9958167831 | 0.9968326543 | 0.995 |
| sliding_attention | 0 | synthetic traced replay | 128 | n/a | 0.9933475834 | 0.993 |
| full_attention | 5 | synthetic traced replay | 128 | n/a | 0.9968326543 | 0.995 |
| sliding_attention | 0 | synthetic long context | 1024 | 0.9937211762 | 0.9942062761 | 0.992 |
| full_attention | 5 | synthetic long context | 1024 | 0.9924927439 | 0.9939581613 | 0.992 |
| sliding_attention | 0 | real checkpoint | 32 | 0.9995228111 | 0.9997580175 | prefill 0.995, decode 0.993 |

Determinism evidence:

- Sliding traced decode replay vs replay PCC: 1.0.
- Full traced decode replay vs replay PCC: 1.0.

The real-weight test used the cached checkpoint:

`/home/moconnor/.cache/huggingface/hub/models--google--gemma-4-12B/snapshots/56820d7d8cbe8e47975a53325439ed272e91cff2/model.safetensors`

## Sequence Length

The default long-context regression covers `seq_len=1024`, the sliding-window boundary, with paged prefill, paged decode, a permuted page table, and tensor current-position handling.

The advertised maximum position count is 262144. That is not a feasible single-chip functional-layer prefill target for this stage. A byte calculation for one 262144-token layer gives:

- Input activation `[262144, 3840]` BF16: 1.875 GiB.
- One MLP intermediate `[262144, 15360]` BF16: 7.5 GiB.
- Gate plus up MLP intermediates: 15.0 GiB.
- Q activation for 16 heads/head dim 256: 2.0 GiB.
- Sliding KV cache for one layer: 2.0 GiB.
- Full-layer KV cache for one layer: 0.5 GiB.
- Unfused full-attention score tensor lower bound `[16, 262144, 262144]` BF16: 2048 GiB.

The final suite therefore validates the largest accepted hardware regression used in this bringup, `seq_len=1024`. Full advertised-context validation remains unproven at this functional-decoder stage.

## Performance

Performance was measured with warmed prefill and traced warmed decode. Tracy signposts were `PERF_PREFILL`, `PERF_PREFILL_END`, `PERF_DECODE`, and `PERF_DECODE_END`. `tt-perf-report` CSV `Device Time` is reported in microseconds.

| Layer kind | Mode | Device ops | Host ops | Device time us | Op-to-op gap us | Report |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| sliding | warmed prefill | 23 | 0 | 3296.464 | 36.423 | `tracy/sliding/prefill_perf_report.txt` |
| sliding | traced warmed decode | 36 | 0 | 2881.813 | 58.091 | `tracy/sliding/decode_perf_report.txt` |
| full | warmed prefill | 23 | 0 | 3655.096 | 54.138 | `tracy/full/prefill_perf_report.txt` |
| full | traced warmed decode | 36 | 0 | 3198.814 | 58.716 | `tracy/full/decode_perf_report.txt` |

Stable profiler artifacts:

- `tracy/sliding/ops.csv`
- `tracy/sliding/prefill_perf_report.txt`
- `tracy/sliding/prefill_perf_report.csv`
- `tracy/sliding/decode_perf_report.txt`
- `tracy/sliding/decode_perf_report.csv`
- `tracy/full/ops.csv`
- `tracy/full/prefill_perf_report.txt`
- `tracy/full/prefill_perf_report.csv`
- `tracy/full/decode_perf_report.txt`
- `tracy/full/decode_perf_report.csv`
- `tracy/perf_summary.json`

## Runtime Fallback Audit

`functional_decoder.py` contains no `import torch`, `ttnn.from_torch`, or `ttnn.to_torch`. Runtime prefill/decode use device-resident TTNN tensors; host conversion only appears in the pytest harness for setup and PCC comparison. The filtered perf windows report 0 host ops for each measured prefill/decode mode.

## Watcher

An initial watcher run with default ETH watcher instrumentation failed before useful decoder validation because the watcher-instrumented `idle_erisc.elf` overflowed its region:

`idle_erisc.elf: segment[0] [0x3f10,+0x58c0) overflows region:0 limit of 0x54c0 bytes`

The passing watcher run used:

`TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_ETH=1 TT_METAL_WATCHER_NOINLINE=1 TT_METAL_LOGS_PATH=models/autoports/google/gemma-4-12B/doc/functional_decoder/watcher/sliding_disable_eth pytest -q models/autoports/google/gemma-4-12B/tests/test_functional_decoder.py::test_paged_prefill_then_decode_pcc --tb=short -k sliding --timeout=180`

The test passed. The watcher log at `watcher/sliding_disable_eth/generated/watcher/watcher.log` contains normal attach, kernel, stack-usage, and detach summaries and no fatal/assert/NOC/sanitize fault signature.

## Limitations

- Single 1x1 device mesh only.
- Functional-decoder stage only; no optimized-decoder, multichip-decoder, full-model, or vLLM work.
- Full 262144-token advertised context is not proven by this stage. The final accepted long regression is 1024 tokens.
- Sliding decode and long-context thresholds are model-specific functional thresholds justified by component isolation rather than the default 0.995 bar.
