# Qwen3.6-35B-A3B Optimized Full Model

| Batch-1 measured path | Prompt/gen | TTFT ms | Decode t/s/u | E2E t/s/u | Host boundary | Evidence |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| Before: completed traced token-out with per-token readback | 128/128 | 5817.66 | 16.58 | 9.50 | 127 syncs, 127 token readbacks | `artifacts/token_out_no_readback_prompt128_gen128_warmed.json` |
| After: optimized traced token-out, no steady-state readback | 128/128 | 5895.73 | 17.43 | 9.52 | 0 steady-state syncs/readbacks, 1 terminal validation read | `artifacts/token_out_no_readback_prompt128_gen128_warmed.json` |
| Traced teacher forcing, AIME24 chat ref | 161/100 | 8749.84 | 16.38 | 6.76 | readiness callback supplies reference tokens | `logs/aime24_teacher_forcing.log` |

The optimized token-out row samples the first token on device from prefill
logits, then replays the captured decode, LM-head, sampler, token-feedback, and
position-advance graph for the remaining 127 tokens with
`ttnn.execute_trace(..., blocking=False)`. The only token readback is the
post-loop validation read used to prove the no-readback path produced the same
final token as the readback baseline.

## Final Path

| Contract | Final setting |
| --- | --- |
| Model | `tt/model.py::QwenFullModel` |
| Generator | `tt/generator.py::QwenReadinessGenerator`, `build_generator` |
| Hardware | local `2x2` Blackhole p300c mesh, `FABRIC_1D_RING` |
| Decoder stack | completed optimized `MultichipDecoder` for all 40 layers |
| Residual layout | replicated BF16 DRAM/interleaved `[1, batch, seq, 2048]` |
| Embedding/final norm | replicated BF16 |
| LM head | BF8 flat vocab shard over 4 devices, `ShardTensorToMesh(dim=3)` |
| KV cache | paged BF16 full-attention cache, block size 32 |
| Linear state | BF16 conv and recurrent state |
| Sampler | common `SamplingGenerator` top-1 split path, top-k/top-p-capable, `max_top_k=32` |
| Trace replay | persistent token, position, RoPE, page table, KV cache, and linear state |

This stage preserves the decoder dtype/fidelity/KV-cache/activation/CCL policy
and rejection ledger. It does not run a datatype frontier search and does not
switch to any faster rejected stream. No vLLM integration work was started.

## Correctness

| Check | Top-1 | Top-5 | Top-100 | Evidence |
| --- | ---: | ---: | ---: | --- |
| AIME24 chat-template prefill | 96/100 | 100/100 | 100/100 | `logs/aime24_prefill_check.log` |
| AIME24 chat-template traced teacher forcing | 99/100 | 100/100 | 100/100 | `logs/aime24_teacher_forcing.log` |

Both refreshed AIME24 checks meet top-5 >= 98% and top-100 = 100%.

Autoregressive evidence was refreshed with
`models/common/readiness_check/autoregressive_prompt.txt`: HF and TT both
generated 100 tokens. The TT output is coherent English and the degeneration
checker reports no findings. Artifacts are in
`artifacts/autoregressive_default_prompt_100/`; the machine report records
adjacent duplication `0.0`, trigram loop fraction `0.038`, and informational
HF/TT token agreement `14/100`.

Synthetic full-model hardware coverage includes non-aligned prompt length,
mixed fixed slots, inactive rows, changed page tables, traced token-out, and
the no-readback measurement path. The final accepted watcher run is
`logs/hardware_smokes_watcher_final.log` (`7 passed, 2 warnings`) with
`TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_ETH=1`.

## Performance

The optimized no-readback replay improves same-run traced token-out decode
from `16.58` to `17.43 t/s/u` (`5.12%`). TTFT is effectively unchanged because
the prefill path is unchanged; the measured delta was `+1.34%`.

Decoder-layer lower bound from optimized multichip screen latencies:

| Component | Latency |
| --- | ---: |
| 30 linear-attention decode layers at `1.281 ms` | `38.430 ms` |
| 10 full-attention decode layers at `1.096 ms` | `10.960 ms` |
| decoder stack lower bound | `49.390 ms/token` |
| inherited terminal path, final norm plus LM head plus sampler | `11.464 ms` |
| stack plus terminal envelope | `60.854 ms/token` |
| optimized token-out replay | `57.359 ms/token` |

The optimized token-out replay is inside the stack-plus-terminal envelope.
Against the tighter tt-perf-report device-time stack plus terminal
(`55.721 ms/token`), the gap is `2.94%`, below the 10-15% remediation
threshold.

Terminal sampler-choice profiling selected the common top-k1 composite gather:
`10.901 ms` over 5 iterations, matching host argmax. Force-argmax async
full-vocab gather is rejected on this mesh because the local 2x2 p300c fabric
cannot route the requested full-mesh all-gather for the flat 4-way vocab
layout. `TopKDeviceOperation` dominates the sampler subpath, but the sampler is
about `19%` of full token-out decode and does not dominate the measured path.

Compact tt-perf-report CSV/table artifacts are under:

- `tracy/inherited_optimized_multichip_decoder_final_reports/`
- `tracy/inherited_full_model_terminal_path_reports/`

Summary/provenance: `artifacts/perf_summary.json`.

## Runtime Fallback

The measured optimized path has no single-chip decoder fallback, host-side
decoder, host argmax, full-logits readback, untraced sampling, Python token
feedback loop, per-token page-table rebuild, or per-token host synchronization.
The throw-on-fallback hardware smoke passed in
`logs/synthetic_full_model_no_fallback_smoke_final.log` (`7 passed,
2 warnings`).

Detailed audits:

- `sampling_trace_audit.md`
- `runtime_fallback_audit.md`

## Context

`doc/context_contract.json` is preserved and extended for this stage. The
advertised `262144` token context remains supported; no hard physical limit was
hit, and no capability was reduced. The optimized generator path continues to
support valid non-aligned prompt lengths through internal padding, cache fill,
page-table selection, and output slicing.

## Limitations

- Real-weight optimized token-out measurement is batch 1.
- Batch-2 and mixed-row behavior is covered by synthetic hardware tests.
- The 262144-token long-context runtime probes are inherited from the completed
  decoder/full-model evidence; this stage did not rerun a full long-context
  autoregressive benchmark.
- Active-Ethernet watcher mode remains a host lifecycle limitation on this
  p300c host; accepted watcher evidence disables ETH.
