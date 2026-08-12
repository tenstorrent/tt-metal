# Optimized full model: Mistral Small 24B Instruct 2501

The final target is batch-1 inference on a `1x4` Blackhole p300c mesh with TP4.
The optimized path keeps the completed full model's accuracy-selected decoder,
cache, collective, residual, and endpoint policies. It adds a public steady-state
token-out window in which model replay, split Sampling1D replay, sampled-token
feedback, cache/position/RoPE advance, and unchanged page-table state remain on
device. vLLM integration is intentionally out of scope.

## Headline performance and accuracy

| final-source measurement | before / compatibility path | optimized path | result |
|---|---:|---:|---|
| warmed batch-1 TTFT, 128-token prompt | 57.314 ms | **57.939 ms** | +1.09%, within run noise; prefill unchanged |
| traced token-out, exact 128-token window | 54.353 t/s/u with caller token observation | **54.452 t/s/u** with no per-token host work | 18.364851 ms/token |
| AIME24 chat-template prefill | inherited 99% top-1, 100% top-5/top-100 | **99% / 100% / 100%** | pass |
| AIME24 teacher-forcing decode | inherited 97% top-1, 100% top-5/top-100 | **97% / 100% / 100%** | pass; 52.562 traced t/s/u |
| autoregressive trace | inherited 55.936 t/s/u | **54.210 t/s/u** | coherent EOS output; device feedback |

“Before” and optimized decode in the exact 128/128 row come from the same final
40-layer process and differ only in caller observation versus the new host-free
window. TTFT is measured after an explicit prefill compile and from two adjacent
cache-reset requests; the decode-only optimization does not change prefill.
Teacher forcing is deliberately separate: forced tokens cross the host
boundary and its 31.38 t/s/u aggregate decode number must not be presented as
token-out throughput. The trace-only forced-token interval is 52.562 t/s/u.

## Selected full-path policy

- Mesh: `1x4`, TP degree 4, `FABRIC_1D`, 200,000,000-byte trace region per
  DRAM bank.
- Decoder: 40 layers, BFP4 LoFi dense weights/kernels, BF16 activations and
  norms, BFP8 KV cache, BFP8 persistent async decode all-reduces, BF16 general
  prefill all-reduces, and the selected 11-core L1 block-sharded inter-layer
  residual. There is no replicated/public restore between layers.
- Endpoints: TP4 BF16 embedding, BF16 final norm and LM-head weights with HiFi2,
  rank-local sharded logits, split Sampling1D, and `tt_out_tok` feedback.
- State: explicit cache, page table, signed cache/SDPA position, unsigned RoPE
  position, prompt length, sampling parameters, active batch, and fixed slots.
  Page tables copy only when changed. Mixed prompt lengths 7/11 and inactive
  rows are covered; arbitrary non-tile prompt lengths remain supported.
- Trace: the model and sampling traces are separate but chained on device.
  `replay_token_out_window(128)` performs 128 nonblocking model/sampler replays,
  zero token/position/page/sampling-param host copies, zero token or full-logit
  readbacks, and one end-of-window synchronization for measurement.
  A read only after that window verifies the final token, signed position, RoPE
  position, page table, and first/last-layer K/V caches exactly against a
  caller-observed control with the same 130 generated tokens.

No datatype frontier was reopened. The optimized-multichip rejection ledger is
binding: BFP8 dense weights, HiFi2 decoder kernels, BFP8 activations, replicated
stream carries, general BF16 decode CCL, fused MRS boundaries, and alternate
residual families remain rejected. No measured default switches to any of them.

## Lower bound and terminal gap

The retained warmed multichip layer latency is 0.414822 ms, so the 40-layer
stack is 16.592880 ms. The current one-layer complete terminal control is
1.747149 ms; subtracting one layer leaves 1.332327 ms for final norm, full
BF16 head, Sampling1D, feedback, and orchestration. The composed expectation is
17.925207 ms versus 18.364851 ms measured. The 0.439644 ms remainder is only
2.394%, below the 10–15% gap trigger.

The retained device-profiler layer number is 348.538 us versus a 153.954 us
byte-only DRAM floor. The refreshed one-layer full terminal is 1,856.835 us of
device time; composing its incremental terminal with 40 layers gives 15.449817
ms device time. The equivalent composed byte floor is 6.795218 ms. These
device-kernel and roofline values are reported separately from warmed wall time
in `perf_summary.json` and the profiler tables.

## LM-head and sampler decisions

The selected LM-head geometry remains 8,192 rank-local columns per split,
10 input cores, 64 output cores, and `max_in0_block_w=4`. A direct 16,384-column
load first exceeded L1 (2,208,512 versus 1,572,864 bytes), but this was not
treated as a final rejection: staging tilization through interleaved DRAM and
using the maximum legal 80-input-core/block-1 runtime geometry made it pass.
Over 256 exact replays it measured 1.733358 ms versus 1.747072 ms control, a
0.785% full-terminal gain and below the 1% materiality threshold. Likewise,
adapting the block-8 failure with 80 input cores made the legal effective
block-2 family pass at 1.743618 ms, only 0.198% faster. The 32-output-core
candidate is 0.065% faster. Defaults therefore remain unchanged. The complete
sweep and adaptations are in `lm_head_candidates.csv` and raw logs.

Canonical split greedy remains selected. The refreshed non-watcher comparison
is 0.339 ms versus 1.261 ms for force-argmax. The reduced terminal report puts
TopK plus sampling/seed at 169.475 us/replay, 9.127% of the one-layer full
terminal, so it is not dominant; the final full path does not all-gather the
full vocabulary. Top-k/top-p capable sampling uses
the same split trace and mutable request-boundary parameter buffers.

## Context and serving contract

`doc/context_contract.json` is preserved and extended. Advertised context stays
at the HF limit of 32,768 because the established full-stack physical ceiling
with reserve is 34,464. The generator keeps mixed prompts, fixed slots,
inactive rows, nonuniform positions, changed-only page tables, reset-safe trace
reuse, and non-aligned prompt lengths. No capability was reduced.

## Evidence map

- primary full-stack benchmark and terminal sweep: `logs/full_40layer_128x128_correctness_warmed.log`,
  `logs/lm_head_*.log`, and `lm_head_candidates.csv`;
- AIME24 checks: `logs/run_prefill_check.log` and
  `logs/run_teacher_forcing.log`;
- autoregressive and shared qualitative outputs: `autoregressive/`,
  `qualitative_suite/`, `logs/run_autoregressive.log`, and
  `logs/qualitative_suite.log`;
- runtime audit and reduced functional gates: `logs/runtime_boundary_source_audit.log`
  and `logs/reduced_real_*.log`;
- profiler: `profiler/reduced_terminal_trace/`, `perf_summary.json`, and
  `lower_bound_accounting.md`;
- watcher and final device health: `logs/watcher_full_terminal.log`,
  `logs/watcher_split_trace.log`, and `logs/final_hardware_health.log`;
- commands, decisions, limitations, review iterations, and commit record:
  `work_log.md`.

## Limitations

- The public compatibility `generate()` API still returns Python token IDs and
  therefore observes tokens for streaming/EOS. The serving-ready fast contract
  is the device-resident token returned by `replay_token_out_window`; a serving
  layer may inspect it only at an explicit request boundary.
- Full-context batch-32 prompt ingestion retains the decoder-owned streaming
  policy documented in `context_contract.json`; this is not a context reduction.
- The nanobind shutdown reference-leak warning remains a framework teardown
  diagnostic after successful device close; it does not alter outputs or leave
  devices attached.
- `logs/full_40layer_128x128.log` is a superseded harness failure: pytest's
  original 300-second timeout expired during the normal layer-40 weight load.
  It was not a device hang; the 1,800-second retry completed and was later
  superseded by the correctness/warmed benchmark above.
