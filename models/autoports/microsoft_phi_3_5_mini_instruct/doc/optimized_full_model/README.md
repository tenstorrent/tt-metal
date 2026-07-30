# Optimized Full Model

Model: `microsoft/Phi-3.5-mini-instruct`

Scope: completed full-model/generator path on the existing optimized 1x8 multichip decoder stack. No vLLM integration work was started.

Status: optimized-full-model state is complete for the current target mesh. The final token-out path is traced, uses canonical split sampling, feeds sampled tokens back through `tt_out_tok`, advances position on device, keeps page tables persistent/changed-only, and has no per-token host readback or synchronization in the measured decode loop.

## Final Defaults

- Mesh: T3K 1x8 ring, same decoder dtype/fidelity/KV-cache/activation/CCL/inter-layer residual policy as `doc/optimized_multichip_decoder`.
- Decoder stack: unchanged `MultichipDecoder` stack, replicated BF16 residual between layers.
- Embedding: hidden-dim sharded lookup followed by all-gather to replicated residual.
- Final norm: TTNN RMSNorm, no host boundary.
- LM head: vocab-sharded matmul with `per_device_vocab_size=8192`, total padded vocab 65536.
- Logits mask: padded vocab rows remain masked and `logits_to_torch` slices back to the canonical 32064 vocab at explicit readiness boundaries.
- Sampling: force-argmax disabled. Greedy uses the same top-k/top-p-capable split sampler with `top_k=1`, `top_p=0.0`, `temperature=1.0`.
- Token-out benchmark: persistent token/position/page-table inputs, device-side position advance, `tt_out_tok` feedback, nonblocking model and sampling trace replay, no sampled-token readback.

## Before/After Summary

| Path | Before | Final |
| --- | ---: | ---: |
| AIME24 traced teacher forcing TTFT | 221.54 ms | 226.93 ms |
| AIME24 traced teacher forcing decode | 36.88 t/s/u | 40.15 t/s/u |
| Token-out no-readback TTFT, prompt128/gen128 | 227.15 ms | 254.47 ms |
| Token-out no-readback decode, prompt128/gen128 | 50.52 t/s/u | 56.43 t/s/u |
| Token-out no-readback E2E, prompt128/gen128 | 35.72 t/s/u | 38.82 t/s/u |

The selected path trades a larger padded LM head for a much faster steady-state sampler. TTFT for the prompt128 token-out harness regressed by about 27 ms, but warmed no-readback decode improved from 19.79 ms/token to 17.72 ms/token.

## Correctness

Final AIME24 chat-template evidence:

| Check | top-1 | top-5 | top-100 | Perf |
| --- | ---: | ---: | ---: | --- |
| Prefill | 96/100 | 100/100 | 100/100 | n/a |
| Traced teacher forcing | 91/100 | 100/100 | 100/100 | TTFT 226.93 ms, decode 40.15 t/s/u |

Autoregressive evidence:

- `logs/final_run_autoregressive_2026_06_15.log`
- `../../readiness_autoregressive/tt_completion.txt`
- `logs/final_check_degenerate_output_2026_06_15.log`

The TT completion produced 128 tokens and the degeneracy scan reported no degenerate output. The current CPU HF greedy run produced repetitive text, so HF/TT token agreement is treated as informational only.

## Sampling Optimization

The bottleneck was the generic single-core `TopKDeviceOperation` in split sampling. The first completed full-model path used per-device vocab width 4032; reduced full-path profiling showed TopK at 2255.32 us, 70.25% of reduced token-out device time.

Trials:

| Strategy | Result | Decision |
| --- | --- | --- |
| 4032 per-device vocab | Full token-out no-readback 50.52 t/s/u; reduced TopK 2255.32 us on 1 core. | Rejected. Generic TopK dominated. |
| Explicit sampler pad to 4096 | One-layer decode was slower than unpadded; separate pad op did not remove generic TopK. | Rejected. |
| LM-head contract padded to 4096 | TopK still used 1 core because this TopK requires width >=8192 for multicore. | Rejected. |
| LM-head contract padded to 8192 | Full token-out no-readback 56.43 t/s/u; reduced TopK 99.11 us on 17 cores. | Accepted. |
| Force-argmax/full-vocab all-gather | Inspected only. Not required, not canonical split sampling, and not used in final path. | Not selected. |

The final path still uses `TopKDeviceOperation`, but not the generic dominating single-core form. In the selected reduced profile TopK is 99.11 us and 8.48% of device time; LM head matmul is now the largest individual terminal op at 207.99 us.

## Lower Bound

Decoder-layer lower bound from `doc/optimized_multichip_decoder`:

- Optimized layer device time: 543.090 us/layer.
- Optimized layer host traced time: 559.258 us/layer.
- 32-layer device stack: 17.37888 ms/token.
- 32-layer host traced stack: 17.896256 ms/token.

Selected full-path accounting:

| Item | Value |
| --- | ---: |
| Full token-out measured decode | 17.72088 ms/token |
| Reduced one-layer selected full-path device time | 1168.96 us |
| Estimated terminal device work | 625.87 us |
| Decoder-stack device lower bound + terminal estimate | 18.00475 ms/token |
| Decoder-stack host traced lower bound | 17.89626 ms/token |

The measured token-out path is within the lower-bound plus measured terminal-work envelope, so no >10-15% avoidable gap remains to split.

## Perf Artifacts

Primary final artifacts:

- `perf/token_out_no_readback_prompt128_gen128_lmhead8192.json`
- `tracy/reduced_1layer_lmhead8192/reports/2026_06_15_18_18_41/ops_perf_results_2026_06_15_18_18_41.csv`
- `perf/reduced_1layer_lmhead8192_token_out_perf_report.csv`
- `perf/reduced_1layer_lmhead8192_token_out_perf_summary.csv.csv`
- `perf/reduced_1layer_lmhead8192_token_out_perf_table.txt`
- `perf/reduced_1layer_lmhead8192_prefill_perf_report.csv`
- `perf/reduced_1layer_lmhead8192_prefill_perf_summary.csv.csv`
- `perf/reduced_1layer_lmhead8192_prefill_perf_table.txt`
- `perf_summary.json`

For rejection provenance, earlier 4032 and 4096 profile/report artifacts remain in `logs/`, `perf/`, and `tracy/`.

## Runtime Audit

Passed:

- Static fallback audit:
  `logs/final_static_fallback_audit.log`
- Full 32-layer token-out watcher run:
  `watcher/2026_06_15_full_token_out_watcher10_lmhead8192/pytest.log`
- Watcher log:
  `watcher/2026_06_15_full_token_out_watcher10_lmhead8192/generated/watcher/watcher.log`
- Watcher scan:
  `watcher/2026_06_15_full_token_out_watcher10_lmhead8192/watcher_scan.log`

Watcher attached and detached all eight devices. The scan found no fatal/error/overflow/bad-NOC/retraining signatures. The watcher stack summary reported minimum free stack of 416 bytes on TRISC0 in `sdpa.cpp`, with Ethernet retraining events at 0.

## Checklist

| Requirement | Status |
| --- | --- |
| Full path optimized, not just decoder | Done. Embedding/norm/LM-head/logits/sampling/cache/trace/generator path audited; LM-head/sampling contract changed. |
| Canonical split sampling preserved | Done. Force-argmax disabled; top-k/top-p-capable `TTSampling` path retained. |
| Greedy semantically greedy | Done. Greedy uses `top_k=1`, `top_p=0.0`, `temperature=1.0` through split sampler. |
| Token feedback/no host boundary | Done. `tt_out_tok` feedback, device-side position advance, changed-only page tables, no per-token readback in benchmark counters. |
| Decoder policy preserved | Done. No decoder dtype/fidelity/KV-cache/activation/CCL/residual policy change. |
| AIME24 evidence refreshed | Done. Prefill and teacher forcing meet top-5/top-100 requirements. |
| Lower-bound accounting | Done. Full token-out is within stack lower bound plus measured terminal work. |
| Generic TopK no longer dominates | Done. TopK reduced from 2255.32 us to 99.11 us. |
| tt-perf-report provenance | Done. Selected and rejected profile CSVs/tables are preserved. |
| Runtime fallback audit clean | Done. Static audit and watcher smoke passed. |
| Broad datatype sweep avoided | Done. No full-model dtype frontier search was run. |
| vLLM not started | Done. |

