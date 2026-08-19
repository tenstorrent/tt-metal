# Qwen3.6-35B-A3B Full Model

| Measured path | TTFT ms | Decode t/s/u | End-to-end t/s/u | Evidence |
| --- | ---: | ---: | ---: | --- |
| Optimized token-out greedy, traced batch 1 | 2930.03 | 16.42 | 11.16 | `logs/token_out_trace_perf_default_prompt_100.log` |
| Readiness teacher forcing, traced decode with host-token callback | 8747.52 | 16.35 | 6.76 | `logs/run_teacher_forcing_aime24_chat_100.log` |

The optimized token-out row samples the first generated token on device from
prefill logits, then captures and replays decode plus greedy split sampling for
the remaining 99 tokens. The teacher-forcing row captures low-level decode in
the same trace-safe path, but the readiness callback supplies reference tokens
between replays, so it is reported separately from free-running token-out.

## Final Path

| Contract | Final setting |
| --- | --- |
| Full model | `tt/model.py::QwenFullModel` |
| Generator | `tt/generator.py::QwenReadinessGenerator`, exported through `build_generator` |
| Hardware | local `2x2` Blackhole p300c mesh, `FABRIC_1D_RING` |
| Decoder stack | completed optimized `MultichipDecoder` for all 40 layers |
| Residual layout | inherited replicated BF16 DRAM/interleaved layer boundary `[1, batch, seq, 2048]` |
| Embedding/final norm | replicated BF16 |
| LM head | BF8 flat vocab shard over 4 devices, `ShardTensorToMesh(dim=3)` |
| KV cache | paged BF16 full-attention cache, block size 32, caller-owned page table |
| Linear state | BF16 conv and recurrent state, fixed-slot cache owned by caller/generator |
| Sampler | common `SamplingGenerator` greedy top-1, `max_top_k=32`, flat 4-way vocab gather |
| Trace path | decode logits, common sampler, 1-wide decode input buffer, tile-width sampler output buffer, device token-copy, and position increment captured together |
| Host compatibility | explicit `host_sampling_compat` mode for teacher-forcing/tests requiring host sampling |

The public generator accepts valid prompt lengths up to the advertised 262144
token context. Prefill chunks internally in 64-token windows; the public path
owns padding, page-table row selection, cache fill, absolute positions, masks
through paged SDPA, and output slicing. Non-aligned prompt lengths are covered
by the synthetic smoke test.

Serving-ready low-level calls are exposed through `allocate_kv_cache`,
`prefill_forward`, and `decode_forward`. These calls take explicit cache,
page-table, prompt-length, position, and batch state. The full-model wrapper
supports mixed prompt rows and inactive rows by processing active users against
their fixed cache slot and returning zero logits for inactive rows.

## Correctness

Fresh AIME24 HF-tokenizer chat-template reference:

```bash
timeout 14400 ./python_env/bin/python -m models.common.readiness_check.generate \
  --hf-model Qwen/Qwen3.6-35B-A3B \
  --prompt-source aime24 --chat-template --gen-len 100 --top-k 100 \
  --output models/autoports/qwen_qwen3_6_35b_a3b/doc/full_model/artifacts/aime24_chat_100.refpt
```

Result: `161` prompt tokens and `100` generated reference tokens.

| Check | Top-1 | Top-5 | Top-100 | Evidence |
| --- | ---: | ---: | ---: | --- |
| Prefill, AIME24 chat ref | 96/100, 96.0% | 100/100, 100.0% | 100/100, 100.0% | `logs/run_prefill_check_aime24_chat_100.log` |
| Decode teacher forcing, AIME24 chat ref | 99/100, 99.0% | 100/100, 100.0% | 100/100, 100.0% | `logs/run_teacher_forcing_aime24_chat_100.log` |

Both checks meet the full-model expected bar: top-5 >= 98% and top-100 = 100%.

The autoregressive qualitative run generated 100 HF tokens and 100 TT tokens
from `models/common/readiness_check/autoregressive_prompt.txt`. HF and TT share
the opening continuation, then diverge naturally under greedy token-out. The TT
completion is coherent English, has no repetition loop, no wrong-language drift,
and no early duplicate-token failure. Artifacts are under
`artifacts/autoregressive_default_prompt_100/`.

The shared qualitative prompt suite generated 64 traced TT tokens for each of
the 6 prompts from `models/common/readiness_check/vllm_prompts.txt`, rendered
through the HF tokenizer chat template with `add_generation_prompt=True`.
Prompts 0-4 matched HF exactly for all 64 generated tokens. Prompt 5 diverged
at generated token 44 but stayed coherent on the requested Python/Fibonacci
task. `logs/check_degenerate_output_qualitative_chat_suite_64.log` reports no
degenerate output. Artifacts are under
`artifacts/qualitative_chat_suite_64/`.

Synthetic full-model smokes cover:

- readiness generator exports;
- runtime fallback audit strings;
- non-aligned prompt length 5;
- mixed batch with an inactive row and explicit nontrivial page table;
- changed page-table decode override, including traced decode coverage;
- traced token-out path compared against semantically greedy host compatibility.

Final synthetic fallback run:
`logs/runtime_fallback_audit_synthetic.log`, `6 passed, 2 warnings in 46.56s`.
Final watcher run:
`logs/watcher_synthetic_composite_gather.log`, `6 passed, 2 warnings in 50.05s`.

## Context And Capacity

`doc/context_contract.json` is updated for the full-model stage and keeps the
advertised 262144-token context. No hard physical capacity limit was hit.

Per-device modeled DRAM at context 262144, batch 1:

| Category | Bytes | GiB |
| --- | ---: | ---: |
| Transformed TT weights under selected policy | 15,985,073,536 | 14.8873 |
| Full-attention KV cache, 10 layers | 2,684,354,560 | 2.5000 |
| Linear conv/recurrent state, page table, decode RoPE tables | 83,853,312 | 0.0781 |
| Total weights plus runtime state | 18,753,281,408 | 17.4654 |

Real model load and full context cache allocation passed in
`logs/real_full_model_load_context_alloc.log`: `REAL_FULL_MODEL_LOAD_OK`,
`max_seq_len=262144`, `full_layers=10`, `linear_layers=30`,
`page_table_shape=(1, 8192)`.

## Sampling And Trace

Two common samplers were compared before token-out measurement:

| Candidate | Result |
| --- | --- |
| `models.common.modules.sampling.sampling_1d.Sampling1D` | Rejected. The implementation is 1D-sharded and rejects the flat 4-way vocab-sharded logits shape used by this full model. |
| `models.common.sampling.tt_sampling.SamplingGenerator` | Selected. It supports the flat 4-way vocab-sharded LM head with `cluster_shape=(1,4)`, top-k gather, and device output token buffer. |

No custom sampler was written. The selected path is canonical split sampling:
LM-head logits stay sharded, common greedy sampler writes the next token to a
persistent tile-width TT sampler output buffer, the trace copies output slot 0
into a separate 1-wide TT decode input buffer, and the traced decode body
increments the persistent TT current-position tensor. The page table is
allocated once and reused during trace replay; it is not rebuilt per token.

The native all-gather path in the common sampler exposed a watcher-only BRISC
assert on this small top-k gather tensor. `$autofix` root-caused it and added an
opt-in composite row-major all-gather path in `models/common/sampling/tt_sampling.py`.
Only this Qwen full model enables `use_composite_topk_all_gather=True`.

Terminal-path profiling is recorded in
`artifacts/terminal_path_profile_summary.json` and
`tracy/terminal_path_reports/`. The final norm plus LM head takes `0.510 ms`,
the sampler takes `10.938 ms`, and the terminal subpath takes `11.464 ms`.
`tt-perf-report` shows `TopKDeviceOperation` accounts for `10,608 us`, so the
sampler dominates only the terminal subpath. Against the full traced token-out
decode step (`60.891 ms/token`), sampler cost is `18.0%`, so it does not
dominate token-out decode.

Detailed sampling and trace evidence: `sampling_trace_audit.md`.

## Runtime Fallback

The measured optimized token-out path does not use single-chip, replicated
decoder, host-side decoder, host argmax, full-logits readback, untraced sampling,
or Python token feedback. Host logit materialization remains available only for
readiness checks and explicit host-sampling compatibility mode.

Detailed runtime audit: `runtime_fallback_audit.md`.

## Limitations

- No vLLM integration work was started in this stage.
- Long-context runtime evidence for 262144 tokens is inherited from the
  functional/optimized decoder probes plus this stage's real full-model load and
  full-context cache allocation. A full 262144-token autoregressive run was not
  rerun.
- The optimized token-out measurement is batch 1. Fixed-slot mixed prompt and
  inactive-row behavior is covered by synthetic smoke, not by a full real-weight
  batch-2 generation benchmark.
- Active-Ethernet watcher mode remains a host lifecycle limitation from the
  decoder stages. The accepted watcher evidence uses
  `TT_METAL_WATCHER_DISABLE_ETH=1`.

## Artifacts

- Implementation: `tt/model.py`, `tt/generator.py`
- Tests: `tests/test_full_model.py`
- Reference: `artifacts/aime24_chat_100.refpt`
- AIME checks: `logs/run_prefill_check_aime24_chat_100.log`,
  `logs/run_teacher_forcing_aime24_chat_100.log`
- Autoregressive qualitative: `logs/run_autoregressive_default_prompt_100.log`,
  `artifacts/autoregressive_default_prompt_100/`
- Chat qualitative suite: `logs/run_qualitative_chat_suite_64.log`,
  `artifacts/qualitative_chat_suite_64/`
- Token-out perf: `logs/token_out_trace_perf_default_prompt_100.log`
- Terminal-path profile: `artifacts/terminal_path_profile_summary.json`,
  `tracy/terminal_path_reports/`
- Fallback and watcher: `logs/runtime_fallback_audit_synthetic.log`,
  `logs/watcher_synthetic_composite_gather.log`
- Device health: `logs/tt_smi_post_watcher.log`
- Autofix reports: `AUTOTRIAGE.md`, `AUTOFIX.md`
