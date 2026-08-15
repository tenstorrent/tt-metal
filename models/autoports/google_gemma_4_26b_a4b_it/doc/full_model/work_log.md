# Full-model work log

## 2026-08-15 implementation start

- Baseline optimized multichip implementation commits and rejection ledger:
  `doc/optimized_multichip_decoder/README.md` and `work_log.md`.
- Unrelated pre-existing worktree state excluded:
  `models/autoports/qwen_qwen3_6_27b/` and `third_party/tt-metal/`.
- Read `$full-model`, `$tt-device-usage`, `$multichip`, `$optimize`, and
  `$tt-enable-tracing`, plus LLM performance section 4.5.
- Initial bounded `tt-smi -ls --local` showed four P300C devices. A configured
  `FABRIC_1D_RING` 1x4 open/close printed `MESH_SMOKE_OK`.
- Added `tt/model.py`, `tt/generator.py`, and fast full-model contract tests.
- Selected `Sampling1D`; rejected the stateful TTTv1 `SamplingGenerator` for
  the explicit-state reasons recorded in README. Force-argmax comparison is
  still pending and is not completion evidence.

### Reduced probe repair loop

1. Real embedding + layers `[0,5]` + final norm/LM head failed before decoder
   execution because RoPE embedding lookup received INT32 indices. Generator
   staging was already UINT32; fixed the probe and documented the contract.
2. Per `$tt-device-usage`, reset/relist/mesh-smoke passed. The second run
   reached the first decoder norm and proved the TP embedding all-gather left a
   ROW_MAJOR tensor. Added the one model-entry `to_layout(TILE)` conversion.
3. Reset/relist/mesh-smoke passed again. The third fallback-raising run passed
   end to end in 6.88 seconds wall time (4.31 seconds test call including cache
   loads/compilation).

No decoder math, dtype, fidelity, CCL placement, KV dtype, expert routing, or
inter-layer residual policy changed during these repairs.

### Full-stack capacity recomputation

The real reduced-probe TTNN cache files provide exact selected tensor shapes
and serialized sizes. TP-sharded files were divided by four and replicated
files counted once per device, then extrapolated as 25 sliding + 5 full layers:

- decoder stack: 7,216,213,640 bytes/device;
- sharded embedding + tied sharded LM head + final norm: 738,378,402 bytes;
- replicated full-context RoPE tables: 805,306,368 bytes;
- BF16 full-model KV cache: 2,736,783,360 bytes;
- persistent subtotal: 10.707119265571237 GiB/device;
- with trace, persistent CCL, 6 GiB workspace, and 4 GiB allocator reserves:
  20.779690066352487 GiB/device, leaving 11.220309933647513 GiB/device.

No context reduction is justified. The all-layer physical probe remains open.

### AutoFix: split-sampling trace blocker

The reduced real-weight test was extended into a minimal canonical split-trace
repro. Model-entry RoPE decode shapes were fixed to the established decoder
contracts: sliding `[1,B,1,D]`, full `[1,1,B,D]`. Model-only capture then
passed under fallback-raising runtime.

`$autofix` isolated two independent optional-output limitations:

- row-major `plus_one` and in-place elementwise position updates cannot be
  captured; device-only position increments between the two trace replays are
  retained, with no host rebuild;
- both common semantically greedy sampler paths reject
  `tt_out_tok=token_input` during sampling trace capture: k=1 top-k fails in
  `ttnn.sampling`, and force-argmax fails in `ttnn.argmax`.

Artifacts: `/tmp/gemma4_embed_trace.log`, `/tmp/gemma4_model_trace.log`,
`/tmp/gemma4_model_plus_trace.log`, `/tmp/gemma4_model_add_trace.log`,
`/tmp/gemma4_split_trace_device_positions.log`, and
`/tmp/gemma4_split_argmax_trace.log`. See repo-root `AUTODEBUG.md` and
`AUTOFIX.md`.

Follow-up proved the failures were program-warmup mismatches rather than a
hardware limitation. The sampler was initially warmed without an output tensor,
then capture invoked the distinct optional-output program and attempted a
forbidden binary upload. Warming the exact `tt_out_tok=token_input` graph and
using the standard `[1,1,32]` fixed-slot decode tensor fixed the failure.
`/tmp/gemma4_split_exact_warm.log` passes the fallback-raising real-weight
two-layer repro, including two replays, exact +1 position progression, and zero
host token refreshes. Force-argmax remains selected; k=1 top-k is rejected as
semantically equivalent but slower and unnecessary for greedy readiness.

The full all-layer accuracy, qualitative, performance, stage-review, and
commit gates remain open.

## 2026-08-15 full-stack completion evidence

- The first 30-layer run exposed cumulative L1 residency from five independent
  persistent async-all-reduce pools. `MultichipDecoder.from_state_dict` now
  accepts a shared resource dictionary, and `Gemma4FullModel` shares one
  three-buffer/semaphore pool across sequential full-attention layers. The
  fallback-raising all-layer probe then passed in 40.84 seconds:
  `/tmp/gemma4_full_stack_shared.log`.
- Generated `readiness_aime24_chat.refpt` fresh with 100 tokens/top-100. Gemma
  4 publishes its canonical template through `AutoProcessor` and
  `chat_template.jinja`, not `AutoTokenizer.chat_template`; the readiness
  generator now uses that exact processor template as a fallback. The earlier
  plain reference is retained only as rejected diagnostic evidence because its
  HF control repeated punctuation.
- Prefill command from README passed top-1 96/100, top-5 100/100, top-100
  100/100 (`/tmp/gemma4_prefill_chat.log`).
- Initial teacher forcing allocated all-gather buffers while a trace was live
  and fell to 87/100 top-100. Teacher forcing now replaces only the input token
  while retaining traced device argmax. One remaining top-100 miss was isolated
  to step 0 with an impossible uninitialized token. Prefill logits had one row
  while Sampling1D requires the 32-slot decode tile. Padding to that fixed-slot
  contract and replaying a short-lived first-token sampling trace fixed it.
  Final teacher forcing: top-1 95/100, top-5 100/100, top-100 100/100, TTFT
  385.43 ms, decode 2.59 t/s/u (`/tmp/gemma4_teacher_chat_final.log`).
- Mixed serving probe passed with logical prompt lengths 33/47, batch 2, one
  inactive row, positions `[34,-1]` then `[35,-1]`, unchanged page-table
  addresses, and no host token/page-table refreshes (`/tmp/gemma4_mixed_probe.log`).
- Free-running 100-token HF/TT comparisons were produced for AIME24 and a sky
  explanation. Both TT outputs were fluent, coherent English and followed the
  HF topic/structure without doubled-token collapse, wrong-language drift, or
  early semantic divergence. Machine degeneracy check: clean. Artifacts are
  `autoregressive_aime24/`, `autoregressive_explanation/`, and
  `degeneracy_report.json`.
- Optimized 64-token token-out measurement: TTFT 215.445 ms, 63-token traced
  decode 25.1519 s, 2.5048 t/s/u (`/tmp/gemma4_autoreg_perf.log`).
- Recomputed full-context envelope after sharing the persistent CCL pool:
  20.771633425727487 GiB/device, margin 11.228366574272513 GiB/device. Public
  context remains 262,144.

## 2026-08-15 stage-review remediation

- The all-30-layer `max_seq_len=262144,max_batch_size=32` real-weight probe
  passed in 23.93 seconds after state allocation was changed from full context
  per row to one explicit global full-attention budget. Equal batch-32 slots
  receive 8,192 global tokens each and retain independent 1,024-token sliding
  rings. Command: `GEMMA4_FULL_MODEL_PROBE=1 GEMMA4_FULL_STACK_PROBE=1
  GEMMA4_BATCH32_PROBE=1 pytest -q ...::test_reduced_real_weight_full_model_probe`;
  log: `/tmp/gemma4_full_stack_batch32_rerun.log`. Recomputed worst profile:
  22.285305300727487 GiB/device, 9.714694699272513 GiB margin.
- Changed page-table identity now invalidates and recaptures the matching
  model/sampling trace exactly once. The 33/47 mixed prompt test proved two
  unchanged replays, a swapped physical mapping, one recapture, and stable
  reuse afterward (`/tmp/gemma4_mixed_changed_table3.log`).
- Faithful TP4 `[1,1,32,262144]` sampler comparison selected native
  force-argmax at 2.3434 ms over semantically greedy top-k=1 at 10.7359 ms;
  both returned identical tokens. Artifact: `sampler_comparison.json`.
- Primary batch-1 performance used exactly 128 chat prompt tokens and 128
  generated tokens: TTFT 9,170.188 ms, traced decode 2.6345 t/s/u, 127 trace
  replays, zero token/position/RoPE/page-table refreshes, one setup sync, and
  scalar token readback only. Artifact: `autoregressive_perf_128/` and log
  `/tmp/gemma4_autoreg_perf128.log`.
- Decoder-stack-only lower bound is 32.329 ms/token from prior warmed layer
  evidence. A fresh signpost-scoped reduced profile (sliding + full + terminal
  + sampler) reports 24.830 ms merged device-op sum, 9.7% modeled DRAM
  roofline, and argmax at 5.85%; canonical sampling is not dominant. The
  profiler's strict new-log merge found one absent host op; processing the same
  valid capture while omitting that unmatched row produced
  `profile_reduced_decode.csv`, `profile_reduced_summary.csv.csv`, and
  `profile_reduced_report.txt`. The temporary multi-gigabyte raw capture was
  removed after extracting these compact reports.
- Six shared canonical-chat prompts were read. The initial haiku artifact
  exposed that the text subconfig carries only EOS 1 while the outer HF config
  carries `[1,106]`; the model now preserves those generation EOS IDs. The
  corrected haiku stops cleanly at `<turn|>` with 14/18 prefix-token agreement.
  All six prompts are coherent and on-topic with no wrong-language drift or
  repetition collapse. Artifacts: `shared_qualitative_suite.json`,
  `autoregressive_haiku_fixed/`, `qualitative_verdict.json`, and clean
  `degeneracy_report_final.json`.
- Local host/static contract suite at that point passed; hardware probes were
  environment-gated and run separately.

Fresh stage rereview and local commit are the remaining gates.

## 2026-08-15 autofix after rereview

- Root cause of the 12x decode gap: the persistent fixed-32 sampling feedback
  tensor was sent through embeddings and all decoder layers at logical batch 1.
  Inactive rows gated cache writes but not norms/MoE/MLP. The model now slices
  hidden states, positions, and page tables to the logical batch before decoder
  compute and pads only logits at the sampling boundary. Exact 128/128 batch-1
  performance improved from 2.6345 to **23.7629 t/s/u**; TTFT is 320.841 ms.
  At 42.08 ms/token this is within 1.30x of the 32.329 ms decoder-only bound,
  leaving 9.75 ms for terminal, split sampling, trace orchestration, and scalar
  output. Artifact: `autoregressive_perf_128_fixed/`; log:
  `/tmp/gemma4_autoreg_perf128_fixed.log`.
- Added sampled split-trace plumbing for explicit top-k/top-p/temperature/seed
  parameters. Semantic params are part of the trace key and their device
  tensors live with the trace. Reduced real-weight hardware alternated greedy,
  k=8/p=.95/t=.8 sampled, then greedy. The first implementation retained both
  traces and triggered TT-Metal's active-trace allocator warning. Autofix now
  safely releases the active trace before a semantic mode transition allocates
  and recaptures; sampled `tt_out_tok` still aliases the persistent device
  token input. The warning-free final log is
  `/tmp/gemma4_sampled_trace_probe_safe.log`.
- Public prefill no longer writes padded future tokens into the sliding ring;
  logical nonaligned inputs are passed to the decoder's exact tile-padding and
  slicing path. The fallback-raising public generator accepted **262,111**
  tokens through all 30 layers and first traced decode in 341.29 seconds
  (`/tmp/gemma4_long_context_public_full_stack.log`). The first attempt was
  interrupted only by pytest's default 300-second timeout; rerun used
  `--timeout=1800`.
- Revalidated after fixes: prefill 96/100 top-1 and 100/100 top-5/top-100
  (`/tmp/gemma4_prefill_final_batchslice.log`); teacher forcing 98/100 top-1,
  100/100 top-5/top-100, 25.40 t/s/u
  (`/tmp/gemma4_teacher_final_batchslice.log`); mixed changed-page-table probe
  passed (`/tmp/gemma4_mixed_final_batchslice.log`); full-stack batch-32 probe
  passed (`/tmp/gemma4_full_stack_batch32_final.log`).
- Replaced the stale shared-suite haiku and French entries with corrected
  single-EOS TT outputs. `shared_qualitative_degeneracy.json` audits every
  suite entry and is clean.
- Final current host/static invocation: 6 passed and 2 environment-gated
  hardware probes skipped in 4.26 seconds
  (`/tmp/gemma4_full_model_contract_final_current.log`).
  Every skipped hardware body was also run explicitly above with its gate set.

Fresh stage rereview and local commit remain.
