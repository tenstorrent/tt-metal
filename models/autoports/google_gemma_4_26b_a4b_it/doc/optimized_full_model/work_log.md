# Optimized full-model work log

## 2026-08-15 baseline audit

- Loaded the requested `$multichip`, `$optimize`, and `$tt-device-usage`
  contracts plus `tech_reports/LLMs/llms.md` sections 3.3 and 4.
- Baseline commit: `c2fc33243b8`. Unrelated untracked
  `models/autoports/qwen_qwen3_6_27b/` and `third_party/tt-metal/` trees are
  excluded from this stage.
- `timeout 60 tt-smi -ls --local` reported four P300C Blackhole devices.
- A configured `FABRIC_1D_RING` 1x4 open/close smoke printed
  `MESH_SMOKE_OK`.
- Current host/static full-model contract: six passed, two serialized hardware
  probes skipped by their explicit environment gates.
- Inherited like-for-like batch-1 128/128 baseline is 320.841 ms TTFT and
  23.7629 traced token-out t/s/u. The old 64-token 2.5048 t/s/u row predates
  the logical-batch slicing repair and is not a valid competing baseline.

## Position-advance trace-fusion candidate

The inherited steady-state path replays the model trace nonblocking, dispatches
device-side `plus_one` for current and RoPE positions, then replays the sampling
trace nonblocking. A candidate moved both increments into the sampling trace to
remove two eager host dispatches.

- Reduced real-weight TP4 probe with fallback exceptions enabled: pass. It
  preserved two replays, `tt_out_tok` aliasing, and exact +1 position progress.
- Full 30-layer 128/128 run: 759.520 ms TTFT and **20.4766 t/s/u** over 127
  decode tokens (`6.202206 s`). This is 13.83% slower than the inherited
  23.7629 t/s/u baseline.
- Decision: rejected and reverted. Trace fusion serializes the position ops
  behind sampling and loses the useful overlap of the inherited nonblocking
  schedule. The device-only persistent position contract remains unchanged.
- Exact full-stack command used a 1x4 ring mesh, the canonical 128-token prompt
  from `doc/full_model/autoregressive_perf_128_fixed/autoregressive_meta.json`,
  `max_seq_len=512`, batch 1, 128 generated tokens, and
  `/tmp/gemma4_full_model_cache`.

Artifact: `candidates/position_advance_in_sampling_trace.json`.

## No-host-boundary token-feedback baseline

The complete 30-layer default was warmed for five replays, then executed for
128 token-feedback iterations with persistent token, current-position, RoPE,
cache, and page-table tensors. Each iteration issued nonblocking model and
sampling trace replays plus device-side position advances. There was no token
readback or per-token synchronization; one device synchronization closed the
timed window.

- 128 iterations: 4.570362 seconds, **35.7060 ms/token**, **28.0065 t/s/u**.
- Final current position 262 exactly matched the expected 128 prompt + one
  capture replay + five warmups + 128 measured iterations.
- Refresh/readback counters: token 0, position 0, RoPE 0, page table 0,
  token readback 0.
- Decoder-stack lower bound: 32.3291 ms/token. Complete token-out overhead is
  3.3769 ms/token, or 10.45% over the decoder-only bound, including final norm,
  vocabulary-sharded LM head, force-argmax sampler, position advance, and trace
  scheduling.
- The host-visible autoregressive path remains a separate metric because it
  intentionally reads each scalar token to return text and stop on EOS. Its
  inherited 23.7629 t/s/u is not used as the no-host serving-loop number.

Artifact: `no_host_boundary_token_out.json`.

## DRAM-sharded LM-head candidate

The inherited tied BF16 LM head is vocabulary-sharded across TP4 but uses an
interleaved weight and auto-selected matmul. A BF16 DRAM-width-sharded trial
used the eight P300C DRAM-adjacent cores, a width-sharded L1 hidden input, and
the largest legal K block (`in0_block_w=11`).

1. A single 65,536-column local matmul failed with 19,030,784 bytes of static
   circular buffers versus 1,572,864 bytes of L1. This was an initial shape
   mismatch, not a family rejection.
2. Adapting the common `LMHead1D` split family to sixteen 4,096-column local
   chunks reduced `per_core_N` from 256 to 16. The reduced real-weight TP4
   fallback-raising trace probe passed.
3. Full 30-layer no-host-boundary result: 36.0360 ms/token, 27.7500 t/s/u.
   The inherited interleaved single-matmul path is 35.7060 ms/token,
   28.0065 t/s/u. The adapted DRAM-sharded family is 0.9167% slower because
   sixteen matmuls, sixteen sharded-to-interleaved conversions, and concat
   outweigh the individual matmul improvement.

Decision: rejected and code reverted. The terminal remains vocab-sharded and
never performs a full-vocab gather in the measured no-host path.

Artifact: `candidates/dram_sharded_lm_head.json`; failing logs:
`/tmp/gemma4_dram_lm_head_candidate.log` and
`/tmp/gemma4_dram_lm_head_candidate2.log`; passing probe and perf logs:
`/tmp/gemma4_dram_lm_head_candidate3.log` and
`/tmp/gemma4_dram_lm_head_nohost_perf.log`.

## Accuracy, autoregressive, and serving-contract refresh

- Prefill command: `python -m models.common.readiness_check.run_prefill_check
  --model-dir models/autoports/google_gemma_4_26b_a4b_it --reference
  models/autoports/google_gemma_4_26b_a4b_it/doc/full_model/readiness_aime24_chat.refpt
  --mesh-device P300X2 --fabric-config FABRIC_1D_RING`. Result: top-1 96/100,
  top-5 100/100, top-100 100/100; `aime24_prefill.log`.
- Teacher-forcing command used the matching `run_teacher_forcing` arguments.
  Result: top-1 98/100, top-5 100/100, top-100 100/100, TTFT 417.80 ms,
  traced decode 25.43 t/s/u; `aime24_teacher_forcing.log`.
- AIME24 free-running greedy generation used the checkpoint chat template and
  100 tokens. TT and HF produced coherent equation setups; the scoped
  degeneracy checker passed. Artifacts: `autoregressive_aime24/`,
  `autoregressive_aime24_degeneracy.json`, and `qualitative_verdict.json`.
- Exact 128/128 host-visible rerun: TTFT 280.598 ms, 127-token traced decode
  4.844966 s, 26.2128 t/s/u, one setup synchronization, zero token/position/
  RoPE/page-table refreshes, and scalar token readback only. Artifact:
  `autoregressive_perf_128/autoregressive_meta.json`.
- Fallback-raising mixed prompt probe passed logical lengths 33/47, fixed slots,
  inactive row handling, unchanged-table reuse, changed-only table recapture,
  and stable reuse afterward; `mixed_prompt_probe.log`.
- Fallback-raising all-30-layer batch-32 allocation/prefill/two-replay probe
  passed in 132.06 seconds; `full_stack_batch32_probe.log`.
- Static contract result: six passed, two explicitly hardware-gated tests
  skipped; every gated body used for this stage was run separately.

## Full-path profiling and sampler closure

- Tracy command and signpost scope are recorded in `profile/provenance.json`.
  `tt-perf-report` merged four devices over `FULL_MODEL_REDUCED_DECODE` through
  `_END`; compact CSV/table/PNG artifacts are retained. The temporary 1.1 GiB
  raw capture was deleted after extraction and is not recoverable from the
  worktree.
- Reduced full-path device-op sum: 5.585 ms; modeled DRAM roofline 15.1%.
  Argmax is 1.393 ms, async vocabulary all-gather 0.790 ms, and generic top-k
  0.049 ms. Against the 32.329 ms full decoder-stack lower bound, none is a
  token-out dominant operation. No LM-head/sampler contract rewrite is needed.
- The shape-faithful TP4 sampler evidence remains the correct current-mesh
  choice: force-argmax 2.3434 ms versus semantic split top-k=1 10.7359 ms,
  identical tokens. The top-k path uses correctly shaped local-vocabulary
  logits and remains available for sampled serving; force-argmax is not used as
  a workaround for a malformed generic sampled path.

## AutoFix: watcher-only shared CCL assertions

- Original greedy-plus-sampled fallback-raising watcher probe aborted on device
  0 BRISC line 119 in `minimal_default_writer.cpp`. AutoTriage and isolated
  experiments refuted persistent-resource reuse: persistent-off reproduced,
  while corrected one- and two-full-layer model-only probes passed, including
  first modulo-three reuse.
- Verified root cause 1: the direct-fabric writer unconditionally acquired the
  selected direction connection on linear endpoints, although the outward
  direction has zero targets and no connection. Guarding acquisition with
  `detail::valid_targets(direction)` removed line 119.
- The next rerun exposed a distinct line-260 assert. Verified root cause 2: the
  header setup unconditionally initialized scatter state with
  `num_tiles_to_write_per_packet=1`, below the scatter API minimum of two. The
  data loop already uses unicast for one tile; guarding only scatter setup with
  a compile-time `>1` condition preserves behavior.
- Exact original watcher command after both fixes, with sampled trace transition:
  `TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_ETH=1
  TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback":true}'
  GEMMA4_FULL_MODEL_PROBE=1 GEMMA4_SAMPLED_TRACE_PROBE=1 pytest -q
  models/autoports/google_gemma_4_26b_a4b_it/tests/test_full_model_contract.py::test_reduced_real_weight_full_model_probe`.
  Result: one passed in 11.56 seconds; `watcher_full_path_fixed_v2.log`.
- No host synchronization, persistent CCL disablement, packet-pool reset, dtype
  change, topology change, or sampling change was retained. See
  `AUTOTRIAGE.md` and `AUTOFIX.md`.

## Final selected state before review

- Only the two verified shared CCL watcher fixes changed implementation code.
  All model optimization candidates that regressed performance were reverted.
- Decoder dtype/fidelity/KV-cache/activation/CCL policy and replicated
  inter-layer residual layout are unchanged. No datatype frontier sweep was
  performed.
- `doc/context_contract.json` retains 262,144 tokens and now records this stage's
  1x4 validation. The largest public all-layer non-aligned evidence remains
  262,111 tokens; mixed 33/47 prompts were freshly revalidated.
- Runtime audit is clean for measured token-out decode; see
  `runtime_fallback_audit.json` and `performance.json`.

## Stage-review remediation

- The first independent review returned `more-work-needed` for shared-suite,
  rank-contract, and benchmark-provenance gaps. Its report is retained as the
  review history until rereview replaces the verdict.
- Refreshed all six canonical chat prompts at 64 greedy tokens each. Every
  per-prompt directory stores rendered token IDs, HF control, TT output, and
  timing metadata. Manual inspection found coherent, on-topic output; the
  combined machine degeneracy audit is clean. See
  `shared_qualitative_suite.json` and `shared_qualitative_degeneracy.json`.
- Clarified the accuracy contract in `autoregressive_accuracy_contract.json`:
  `run_teacher_forcing` is the shifted-left autoregressive rank runner. It
  scores every traced greedy TT prediction, then feeds the reference token for
  the next step; its refreshed result is 100/100 top-5 and top-100. Separate
  free-running AIME24 evidence validates autonomous `tt_out_tok` feedback and
  text quality, where divergence from HF is expected after alternate coherent
  greedy choices.
- Added a durable gated all-layer benchmark branch to
  `tests/test_full_model_contract.py`. An initial invalid run inherited
  `max_seq_len=128` and exceeded capacity at position 128; it was terminated
  after a live triage capture whose UMD read helpers were incompatible. The
  harness now allocates 512 positions for 32 prompt + capture + five warmups +
  128 measured tokens. After reset/relist, the exact fallback-raising rerun
  passed with the required 128-token prompt: 4.568965 s, 35.695039 ms/token,
  28.015097 t/s/u, final position 262, zero token/readback/
  position/RoPE/page-table refreshes, zero timed synchronizations, and 134 total
  trace replays. Raw log: `no_host_boundary_token_out.log`.
- The historical/current host-visible difference is now explicitly labeled
  observational rather than attributed to an optimization: no performance
  candidate remains in the model/generator diff. The current reproduced
  host-visible and no-readback values are the final results.

## Local commits

- Stage implementation and evidence: `d351ed07056` (`Optimize Gemma4 full-model
  TP4 path`). This commit is local and was not pushed.
