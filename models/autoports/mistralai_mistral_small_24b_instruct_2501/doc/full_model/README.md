# Full model: Mistral-Small-24B-Instruct-2501

Status: complete. Implementation and all hardware gates pass; the independent
`$stage-review` verdict is `clean-pass`. Scope is the repo-local TTNN full model
and standard Metal readiness generator only. No vLLM integration was started.

## Result

`tt/model.py` assembles the sharded BF16 embedding, all 40 optimized TP4
decoder layers, final BF16 RMSNorm, untied BF16 TP vocabulary head, and common
Sampling1D sampler. `tt/generator.py` supplies the standard readiness factory,
explicit low-level cache/page/position state, public prefill, traced decode,
high-level generation, reset, and a named host-sampling compatibility mode.

| Fresh AIME24 chat-template reference, 100 tokens | Top-1 | Top-5 | Top-100 |
| --- | ---: | ---: | ---: |
| full prefill | 99/100 | 100/100 | 100/100 |
| traced teacher-forcing decode | 97/100 | 100/100 | 100/100 |

| Batch-1 performance | Result | Boundary |
| --- | ---: | --- |
| teacher-forcing TTFT, prompt 327 | 2808.99 ms | prefill through first sampled token |
| teacher-forcing decode | 34.78 t/s/u | includes callback observation and forced-token copies |
| teacher-forcing trace interval | 54.77 t/s/u | 99 model+sampler replays; excludes callback and forced-token copies |
| shared-suite TTFT, prompt 128 | 5812.15 ms | exact 128-token synthetic aggregate prompt |
| shared-suite token-out decode | 55.00 t/s/u | 127 post-capture model+sampler replays for 128 requested tokens |
| free-running TTFT, prompt 238 | 6057.47 ms | chat-template prefill through first token |
| split-trace setup | 197.22 ms | warmup plus capture, excluded below |
| free-running token-out decode | 55.94 t/s/u | 53 post-capture model+sampler replays, caller-visible tokens |

The teacher and token-out numbers are deliberately separate. Teacher forcing
must observe every prediction and replace the next input with the HF reference;
free running keeps feedback entirely on device.

## Preserved optimized policy

The full model calls `prepare_*_residual` once, every layer's
`*_forward_stacked`, and the final finish/norm boundary once. It does not
restore a public layout, gather, reshard, or reduce between layers.

- topology: logical 1x4 Blackhole p300c, TP4, Linear two-link collectives;
- dense decoder weights/kernels: BFP4 LoFi, separate gate/up projections;
- residuals and attention/MLP activations: BF16;
- decode CCL: persistent asynchronous all-reduce, BFP8 output, shared BF16 L1
  workspace/semaphore;
- prefill CCL: selected general BF16 all-reduce;
- inter-layer decode state: `[1,1,B,5120]`, 11-core L1 block-sharded;
- inter-layer prefill state: `[1,1,B*S,5120]`, BF16 DRAM with explicit logical
  sequence length;
- cache: direct per-rank two-local-head BFP8 paged K/V, block size 32;
- endpoints: BF16 hidden-sharded embedding and untied BF16 vocab-sharded head.

All accepted/rejected decoder candidates remain in
`doc/optimized_multichip_decoder`. The full-model wrapper does not introduce a
single-chip, replicated, host-compute, or lower-precision substitute.

## Context and request contract

The public maximum remains 32,768. Arbitrary positive logical lengths,
including non-aligned lengths, are accepted. The generator owns padding to 32,
causal masking through the decoder, logical output slicing, cache fill,
position selection, and page-table normalization. An optimized 576-token
prefill window handles the prefix; longer prompts fill the remaining cache via
device decode steps. This is functionally complete but intentionally listed as
a long-prompt performance limitation below.

Low-level prefill/decode expose tokens, caller- or generator-owned K/V cache,
page table, prompt lengths, signed current positions, unsigned rotary
positions, active batch, and per-user sampling parameters. Fixed slots receive
unique physical pages. Inactive decode rows keep signed position `-1` while
RoPE uses a separate clamped row. A real one-layer TP4 gate passes mixed prompt
lengths 7 and 11 in two active slots plus one inactive slot, and verifies
unchanged/changed persistent page-table copy behavior.

The strengthened physical capacity gate opens the same mesh with a 200,000,000
byte trace region per DRAM bank (1,600,000,000 bytes/device) and keeps both
decode and prefill matrix representations for all 40 layers, batch-32 full 32K
caches, BF16 endpoints, and a physically allocated 1.5 GiB/rank reserve alive.
With that complete live set it executes a 32-token-per-user batch-32 prefill
chunk and then paged decode at position 32,767 without releasing prefill
weights. Recalculated steady state is 31,378,609,024 bytes/rank, leaving
1,189,509,248 after reserve. The reserve-adjusted page-aligned physical ceiling
is 34,464, above HF's limit. The exact arithmetic and human-readable claim are
machine-checked by `test_full_model.py` against `doc/context_contract.json`.

## Split tracing and sampling

The optimized greedy path is `k=1`, `p=0`, `temperature=1` through Sampling1D
local-top-k split sampling. Model and sampler are captured as separate traces.
Sampling writes directly into the persistent token tensor that the next model
trace consumes. The model trace advances signed cache/SDPA positions with
negative entries skipped and advances the unsigned RoPE position tensor.

The full 40-layer free-running proof reports:

- 53 model replays and 53 sampling replays;
- final position 291, exactly `238 + 53`;
- sampler output and next-token input share the same device buffer address;
- one request-bound page-table copy, three token copies and six position copies
  during warmup/capture only, and zero per-token copies;
- zero full-logit readbacks; 54 single-token caller observations only;
- sampling parameters remain unchanged with zero host copies.

The reduced real gate separately proves reset retains traces, unchanged page
tables add zero copies, a changed mapping adds exactly one copy, and a replay
then writes the newly mapped physical cache block without modifying the old
block. Host teacher forcing intentionally overwrites the feedback token after
each prediction; it retains both traces. The explicit
`sampling_mode="host"` compatibility path is the only path that gathers full
logits and calls host argmax, and releases active traces before its
shape-dependent prefill.

The two distinct common sampler implementations were compared before
token-out. SamplingGenerator's parameter formatting, unseeded state, disabled
penalty/log-prob state, internal trace capture/replay, and trace release were
exercised; Sampling1D's trace is owned externally by this generator.

| Common greedy path | Token | Mean latency |
| --- | ---: | ---: |
| Sampling1D local-top-k split (`k=1,p=0,temp=1`) | 29317 | 0.314182 ms |
| SamplingGenerator force-argmax (`k=1,p=0,temp=1`) | 29317 | 1.266483 ms |

Split greedy is semantically greedy and 4.03x faster. No custom sampler was
needed. Its isolated 0.314 ms is about 1.76% of the measured 17.88 ms
full model+sampler token interval, so sampling does not dominate token-out.

The reduced full-terminal profiler covers one exact optimized decoder layer,
final norm, all four rank-local LM-head pieces, Sampling1D, device token
feedback, and ten trace replays. It measures 1.378589 ms/replay: matmul is
69.21%, TopK 9.06%, Sampling 1.97%, and ManualSeed 1.25%; the sampler ops do
not dominate. Independent lower-bound accounting scales the inherited
0.414822 ms/layer decoder result to 16.592880 ms for 40 layers versus the
measured 17.877457 ms full token interval, a 1.284577 ms (7.18%) gap.

## Qualitative result

The canonical autoregressive runner used the corrected Mistral regex and chat
template, an explicit all-ones HF attention mask, greedy decoding, and EOS
stopping. HF produced 58 tokens; TT produced 54. Token IDs are identical
through index 23 and first diverge at index 24. Both outputs coherently ask for
the missing story detail, remain in English, preserve the topic, contain no
token or phrase loop, and end with `</s>`.

The shared six-prompt readiness suite then generated 128-token HF and TT
completions for a haiku, ML explanation, fiction, thermodynamics explanation,
requested French translation, and Fibonacci code. All TT outputs were read:
they are coherent and on-topic, show no unintended-language drift, token or
phrase loop, gibberish, or control-token leakage, and the repeated first prompt
is byte-identical. The exact degeneration checker reports no finding. The
texts, prompt-format proof, per-prompt metadata, aggregate 128/128 benchmark,
checker JSON, and verdict are under `qualitative_suite/`.

## Runtime fallback and reset audit

`logs/runtime_boundary_source_audit.log` covers model/generator execution,
cache ownership, host-logit boundaries, sampling, and reset. The representative
final-source reduced endpoint and profiler gates additionally execute with
`ttnn.CONFIG.throw_exception_on_fallback=true` and pass. Summary:

- model runtime contains no PyTorch compute or host conversion;
- decoder weights/caches remain rank-local TP4; constructor rejects other mesh
  shapes;
- optimized generation has no host argmax, full-logit readback, untraced
  sampling, or host token feedback;
- public returned-token observation is not used for feedback;
- page and sampling state copy only when changed;
- reset zeros device cache in place and retains weights, buffers, and traces;
- new prefill or host compatibility releases prior traces at the request
  boundary before shape-dependent allocations.

## Limitations and rejected alternatives

- Full-context prompt ingestion after the first 576 tokens uses sequential
  device decode cache fill. It preserves the 32K contract and non-aligned
  semantics but is not a high-throughput long-prefill implementation.
- `release_prefill_weights_for_decode()` remains an explicit irreversible
  compatibility transition for decode-only deployments. The supported 32K
  context and ordinary reset do not require or use it.
- The language head uses four 8192-column/rank DRAM-sharded pieces. An
  eight-K-tile input block exceeded Blackhole L1 circular-buffer capacity
  (1,782,528 > 1,572,864 bytes); block four is the passing adapted config.
- Single-chip, replicated caches/weights, host decoder/head, and silent eager
  trace fallbacks are rejected by construction.
- Force-argmax common sampling is rejected by the semantic/latency comparison;
  the explicit host mode remains compatibility-only.
- The compiler shard advisor could not start because the preinstalled
  `libTTMLIRRuntime.so` and current tt-metal operator ABI disagree on
  `ttnn::experimental::moe_compute`. Per advisor policy the toolchain was not
  rebuilt. The exact terminal capture and failure are retained under
  `shard_advise/`; accepted decoder advisor artifacts still cover every layer.
- The first full-model worker-watcher run exposed the shared linear all-gather
  endpoint bug: an outward connection was fetched even when that edge worker
  had no valid target. The exact established fix from commit `ff8ced34251` was
  applied to `minimal_default_writer.cpp`; canonical Sampling1D alone and the
  complete reduced gate now pass with watcher and cleanly detach all devices.
  Active-Ethernet inspection remains disabled because firmware 19.8.0 cannot
  fit its 27,776-byte watcher config in the 25,600-byte region; fabric and CCL
  execute while worker BRISC/NCRISC are checked.

## Evidence map

- reference: `artifacts/aime24_chat_100.refpt` and manifest;
- final-source prefill/decode accuracy and runner performance:
  `logs/run_prefill_check_final_source.log`,
  `logs/run_teacher_forcing_final_source.log`;
- free-running trace/performance: `logs/run_autoregressive.log` and
  `autoregressive/autoregressive_meta.json`;
- qualitative texts/verdict: `autoregressive/`, `qualitative_suite/`,
  `logs/qualitative_suite.log`, and `logs/check_degenerate_output.log`;
- sampler, feedback, page-table, host mode, reset, and mixed slots:
  `logs/reduced_real_common_samplers_final_source.log`,
  `logs/reduced_real_batch1_final_source.log`,
  `logs/reduced_real_split_trace.log`,
  `logs/reduced_real_serving_state_gate.log`,
  `logs/reduced_real_mixed_slots_retry.log`,
  `logs/reduced_real_mixed_slots_final.log`, and
  `logs/reduced_real_mixed_slots_final_source.log`;
- 32K capacity with retained prefill weights, 200 MB/bank trace reservation,
  and batch-32 prefill plus terminal decode:
  `logs/full_context_capacity_prefill_resident_final.log`;
- fallback audit: `logs/runtime_boundary_source_audit.log`;
- reduced one-real-layer full-terminal profiler: `profiler/reduced_terminal_trace/`
  `profiler/lower_bound_accounting.md`, and
  `logs/reduced_terminal_trace_profiler.log`;
- watcher failure, isolation, and fixed passes:
  `logs/watcher_full_model_failure.log`,
  `logs/watcher_sampling1d_isolation.log`,
  `logs/watcher_sampling1d_fixed.log`, and
  `logs/watcher_full_model_fixed.log`;
- final post-gate device health: `logs/final_hardware_health.log`;
- advisor capture/failure: `shard_advise/`;
- exact commands, adaptations, review, and commit record: `work_log.md`.
