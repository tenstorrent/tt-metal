# Stage Review

Verdict: more-work-needed

## Required Work

- P1: The public prompt/context contract is internally contradictory and does not meet the advertised 262,144-token capability.
  Evidence: `doc/context_contract.json` advertises `current_supported_context: 262144` and `full_model.batch_1_supported_context: 262144`, while `tt/generator.py:107-109` rejects every public prefill physical extent above `MAX_PREFILL_TOKENS = 192511`. `README.md:98-101` explicitly says longer public prompt prefill is not claimed. The capacity artifacts prove B1 cache residency at C262,144, but they do not prove a public autoregressive prompt of that length can run; the recorded earlier-stage contiguous-allocation failure instead establishes that the present single-pass implementation cannot.
  Why this matters: the goal and `$full-model` require valid logical prompts through the supported context, with internal padding/chunking owned by the generator. A decode-only cache extent cannot be represented as the model's overall supported context while the public prompt API rejects part of it.
  Required next step: implement semantics-preserving full-stack internal chunking through 262,144 and provide a non-aligned near-limit public-generator run, or change the externally advertised prompt capability only if the contract can accurately distinguish prompt capacity from total decode context and the goal accepts that hard-limit reduction. In either case, make JSON, code, README, and tests agree.

- P1: Required `$qualitative-check` shared-suite evidence is absent.
  Evidence: `doc/full_model/qualitative_prompt_format.json` says `"TT shared-suite outputs pending"`. The evidence tree contains only the single AIME24 HF/TT autoregressive comparison; no shared qualitative prompt-suite outputs are present. `$full-model` requires the shared suite through both HF and TT as soon as text generation works, and `$stage-review` treats this omission as required work.
  Why this matters: one math continuation, even when coherent, does not cover prompt-format failures, language drift, repetition, instruction behavior, or other generation-path anomalies across representative prompts.
  Required next step: run the shared qualitative suite with the tokenizer chat template through HF and the optimized TT token-feedback path, preserve rendered prompt/token metadata and both outputs, read and classify every output, and update the README/work log.

- P1: The delivered generator does not expose a serving-ready sampled low-level decode contract and the optimized path is fixed to greedy.
  Evidence: `tt/generator.py:149-161` exposes `decode_forward`, but it always gathers and reads the full vocabulary to host. The device token-out trace is only available through private methods/state (`_capture_token_out_trace`, `_seed_token_out_trace`, `_decode_trace_id`, `_trace_logits`). `generate` accepts no sampling parameters and construction permanently installs `top_k=1`; consequently the delivered public optimized path cannot exercise top-k/top-p despite `$full-model` requiring the same split-sampling path to support them. The only explicit public alternatives are host-logit compatibility decode and batch-1 high-level greedy generation.
  Why this matters: a later serving adapter cannot drive the required explicit cache/page-table/position/fixed-slot sampled state without depending on private internals or falling across a full-logit host boundary. This leaves serving sampling to be invented during vLLM integration, which this stage is supposed to prevent.
  Required next step: expose a stable public low-level trace setup/replay/token-out API with explicit state and sampling parameters, retain greedy force-argmax as the optimized measured mode, and demonstrate at least one non-greedy-capable top-k/top-p configuration through the common sampler without host logits or token feedback.

- P1: The required reduced full-model profiler and sampler-dominance gate are missing.
  Evidence: there is no `tt-perf-report`, device-profiler CSV, or Tracy artifact under `doc/full_model/`. The README reports 17.1085 t/s/u from `artifacts/full_model_perf_b1.json`, but provides no terminal/sampler operation breakdown, no layer-stack lower-bound calculation, and no evidence that `ArgMaxDeviceOperation`, full-vocabulary all-gather, or another sampler operation does not dominate. The claim at `README.md:45-47` that common local top-k was slower is not backed by comparable timing rows in the cited logs.
  Why this matters: the user explicitly requires canonical split sampling and requires fixing the LM-head/sampling contract if sampler operations dominate. `$full-model` also mandates a reduced one-real-layer-per-kind `tt-perf-report` and comparison against the decoder-stack lower bound.
  Required next step: profile the reduced TP4 model with one linear and one full-attention layer plus the real terminal and split sampler, run `tt-perf-report`, record operation-level terminal/sampler costs, compare semantically greedy common paths in the same regime, calculate the 48/16-layer stack lower bound, and optimize any dominant sampler/full-model-only cost before rereview.

- P2: Batch>1 evidence does not validate the delivered full model/generator contract claimed by the documentation.
  Evidence: `tests/mixed_prompt_state.py` constructs only `MultichipDecoder` layer 0 and feeds random hidden states; it does not instantiate `Qwen36Model`/`Qwen36Generator`, embedding, the 64-layer mixed linear/full-attention stack, LM head, common sampler, token feedback, or output formatting. Nevertheless `README.md:27-33` presents this as evidence for the generator's fixed-slot serving contract and calls it a real TP4 "layer stack" test. Other reduced B2 AutoFix logs do not replace the `$full-model` requirement for batch>1 full-model prefill, decode, cache/page-table indexing, token feedback, and output formatting.
  Why this matters: batch-sensitive defects have already occurred at the full-attention gate and sampler's physical 32-row contract. Single-layer linear-state evidence cannot close those boundaries in the delivered wrapper.
  Required next step: add a batch>1 reduced/full-model generator test that includes at least one real layer of each kind and the real terminal/sampler, covering mixed non-aligned prompts, fixed slots, active/inactive rows, page-table/cache indexing, traced feedback, and returned token formatting. Correct the README's description of the existing layer-0 test.

## Other Concerns

- `reset()` only zeroes the entire model cache and there is no public per-slot reset/reuse operation. The docs narrow inactive rows to empty/reset slots, but a serving-ready fixed-slot contract should explicitly prove how one completed slot is cleared and reused without destroying active slots.
- The fallback audit is largely an `rg` transcript plus prose. It identifies intended host boundaries, but it does not exercise a slot-reuse reset path or a public low-level sampled serving path because those paths are not presently exposed.
- The worktree includes unrelated untracked Tracy, Falcon, and third-party paths. A later stage checkpoint must isolate only stage-owned files, as required by `$stage-review`.

## Hard-Check Gaps

- No full-model Watcher/runtime-integrity run is recorded. Earlier decoder-stage Watcher artifacts cover inherited kernels, but not the new embedding, terminal LM head, common sampler, or split-trace lifecycle.
- No repeated full-model determinism artifact compares logits across clean runs and batch positions. Reduced trace token equality and one free-running run are useful but narrower.
- The B32 capacity bracket is a physical allocation probe, not execution evidence at the claimed largest feasible B32 context. The JSON should label that distinction unambiguously.
- The README records teacher-forcing 6.96 t/s/u and token-out 17.11 t/s/u but does not report traced latency in ms/token explicitly or the requested full-model-only/layer-stack cost accounting.

## Anomaly Ledger

- Observed anomaly: the S128 performance artifact's decoded continuation contains repeated chat markers (`walking<|im_end|>...`) and malformed leading text.
  Evidence: `doc/full_model/artifacts/full_model_perf_b1.json`, field `text`.
  Affected path: canonical split-trace greedy token feedback used for the reported 17.1085 t/s/u.
  Control or comparison: `autoregressive_final/tt_completion.txt` uses the same optimized feedback path with a valid complete chat prompt and is coherent/non-repetitive; `artifacts/degenerate_output.json` passes that control.
  Likely subsystem: performance harness prompt construction, which plain-encodes and truncates the already chat-rendered prompt at 128 tokens (`tests/full_model_perf.py:30-31`), rather than an established token-feedback defect.
  Investigation performed: inspected both generated texts, the perf token IDs, the performance harness prompt encoding, and the standard HF/TT autoregressive outputs.
  Resolution: controlled for correctness by the valid-prompt autoregressive run, but the performance artifact should be labeled as a malformed/truncated prompt workload rather than silently presented as qualitative output.

- Observed anomaly: HF and TT free-running AIME24 completions agree for only 3/100 tokens.
  Evidence: `artifacts/degenerate_output.json` and `autoregressive_final/{hf,tt}_completion.txt`.
  Affected path: optimized device-owned greedy feedback.
  Control or comparison: prefill top-5 is 100%, teacher-forced decode top-5 is 100%, both completions remain coherent English and follow the same algebraic setup, and the degeneracy checker reports no mechanical loop.
  Likely subsystem: normal free-running sensitivity to top-1 differences under the selected low-precision policy.
  Investigation performed: directly read both completions and checked the top-k and degeneracy artifacts.
  Resolution: controlled for this single prompt; broader shared-suite evidence remains required above.

## Scope Inspected

- Goal/skill paths: user full-model goal; `.agents/skills/stage-review/SKILL.md`; `.agents/skills/full-model/SKILL.md`.
- Artifact paths: `doc/full_model/README.md`, `work_log.md`, accuracy/trace/AutoFix logs, reference metadata, HF/TT completions, degeneracy JSON, capacity JSON, performance JSON, fallback audit, and `doc/context_contract.json`.
- Code paths: `tt/model.py`, `tt/generator.py`, modified decoder files, `tests/mixed_prompt_state.py`, `tests/full_model_perf.py`, capacity probe, shared reference generator, and repository diff/status.
- Commands run: read-only `sed`, `find`, `rg`, `git status`, `git diff`, `git branch`, `git rev-parse`, and artifact text/JSON inspection. No hardware was opened and no tests or servers were launched.

## Residual Risk

- Accuracy, B1 split-trace feedback, page-table refresh counters, position coherence, full terminal mapping, and physical capacity evidence are strong for the tested shapes. The principal residual risks are the unsupported portion of the advertised prompt range, absence of public sampled serving APIs, unprofiled terminal/sampler dominance, and missing cross-prompt/batch full-wrapper evidence.
