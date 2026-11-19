# Stage Review

Verdict: clean-pass

Independent round 1 of 3, Stage 9 vLLM integration for
`Qwen/Qwen3.8-Flash-Next`. Reviewed the live worktree at tt-metal
`fe932db222e763b2cf7789114309df7ad3a4a604`, including staged closure evidence,
and plugin `629f07cb385494b14d0ee3d138de039d71d09546` (runtime change
`edb95ca695d719a5340d919ae3d6fc2f6dd219a8`). This verdict accepts the reviewed
Stage 9 implementation and evidence; the owner's post-review local commit and
closure bookkeeping remain the normal next steps. The run plan says never push.

## Required Work

- None. No correctness, required-contract, or missing required-evidence finding
  remains in the inspected scope.

## Other Concerns

- Resident capacity is two, although both burst clients submit 32 requests.
  The README correctly distinguishes queued concurrency from resident batch
  size. B32's non-KV state exceeds the measured physical capacity; B2 is the
  largest tested serving configuration, not a proof of the largest feasible
  intermediate batch. Preserve that distinction in later reports.
- Optional dense batched prefill remains unimplemented. The measured
  `prefill_burst` is a valid serial reference, with 32/32 completions and 1024
  output tokens. An enabled-arm ratio and its batched-equals-solo acceptance
  are not missing requirements for this serial implementation. The historical
  serving-contract `batched_prefill` call row proves multi-row API semantics,
  not shared dense execution.
- The B1 top-k40 host compatibility ratio remains 1.179 advisory. Compact host
  readback, position-upload changes, dense batched prefill, and the Stage 10
  chunk-size/load work remain performance candidates, not rejected optimizations
  or Stage 9 blockers.
- Native constrained reasoning remains an eight-of-nine control, not a passing
  global budget proof. Its early-EOS sample and awkward constrained greetings
  are retained. The existing optional allowance produced a passing served
  profile with unchanged assertions; it does not establish that every future
  trajectory finishes within that allowance or that the initial unrecorded
  failures had exactly one cause.

## Hard-Check Gaps

- The aggregate gate exits 1 for advisories, not a critical failure. I checked
  the underlying P2/P3/P5 results and the raw serving outputs rather than
  treating that exit code or the README as the verdict.
- `mixed.json` has a null non-overlapped comparator. Its passing flag proves
  the listed lifecycle/routing checks, not synchronous token equivalence.
  Separate recorded-history immediate/queued runner controls and direct adapter
  state comparisons supply the required overlap evidence.
- Served determinism compares the top 20 normalized logprobs; the standalone
  control checks full logits. The configured all-vocabulary logprob request
  receives the expected 400 and the existing suite skips it. No full-vocabulary
  served comparison is claimed or inferred.
- The full sampling log contains 67 passes and 75 skips: 71 speculative tests,
  three separately opt-in chunked-prefill tests, and the all-vocabulary logprob
  test. Model-specific mixed, maximum-context and P5 evidence independently
  exercises chunked prefill. The skips are not 75 passing hardware tests.
- P3's deprecated `reasoning_content` `/tokenize` mismatch remains advisory.
  Actual c07 chat returns `Ada` with 88 prompt tokens; source establishes the
  endpoint-normalization difference. Equal token count is not exact actual
  c07 input-token evidence. The six shared qualitative prompts do have actual
  input-token equality, which I independently recomputed.

## Anomaly Ledger

- Observed anomaly: intermediate chunks used the host/full-logit boundary.
  Evidence: `reduced_control_v3/mixed.json`, `reduced_async_v4/mixed.json`,
  `plugin_changes.md`, and the approved plugin diff.
  Affected path: mixed chunked prefill while another request decodes.
  Control or comparison: final tracked mixed run has device/none routing,
  zero intermediate outputs/readbacks/sampling, and starts 0/256/512.
  Likely subsystem: previously unexpressible scheduler finality metadata.
  Investigation performed: interface ownership audit, fail-before/pass-after
  plugin tests, canonical mask token/RNG checks, and served replay.
  Resolution: fixed by the exactly authorized opt-in mask interface; other
  models retain the default-false behavior.

- Observed anomaly: historical sync/async comparisons differ on six of eight
  greedy responses, both before and after the mask change.
  Evidence: `reduced_async_v4/mixed.json`, `masked_async_v1/mixed.json`,
  `AUTODEBUG_async.md`, and recorded-history controls in both arrival orders.
  Affected path: different physical-width/slot histories during overlapping
  prefill and decode.
  Control or comparison: immediate and queued replays match all 307 and 310
  recorded output boundaries exactly; queued overlap is 91, immediate overlap
  zero, unsafe overlap zero. Direct adapter comparisons match tokens, history,
  final cache/state and RNG. Full48 concurrent repeats preserve all six outputs.
  Likely subsystem: execution-history dependence; the numerical cause of the
  original differing-history outputs is not established.
  Investigation performed: source diagnosis, fixed scheduler histories,
  reverse-arrival/intermediate-prefill coverage, and state-level controls.
  Resolution: controlled for async finalization correctness. The historical
  failures remain failures; arbitrary-history bitwise invariance is not claimed.

- Observed anomaly: P2 two-user host compatibility ratio initially exceeded 1.15.
  Evidence: `P2_initial*.json`, `AUTOFIX_P2.md`, and
  `P2_table_snapshot_v2*.json`.
  Affected path: authoritative host-token reload with unchanged full-width
  scheduler page tables.
  Control or comparison: final ratio 219.0/198.9 = 1.101; changed tables,
  page boundaries, remaps and prefill still take full validation.
  Likely subsystem: repeated Python table normalization.
  Investigation performed: source audit, adversarial CPU checks, reduced
  physical-page/state checks, exact-source full48 logit control and P2 rerun.
  Resolution: fixed by decode-only reuse of cloned validated snapshots.

- Observed anomaly: sampling collection and seven initial sampling tests failed.
  Evidence: retained collection/initial-failure logs, plugin test diffs,
  `AUTOFIX_reasoning_budget.md`, native control, and full-profile observer files.
  Affected path: test bootstrap, small-capacity client workloads, statistical
  diversity, and reasoning-chat answer allowance.
  Control or comparison: generic workload repairs preserve assertions; the
  final log records 67 passed, 75 skipped, zero failures. Eleven recorded chat
  responses stop normally; the twelfth recorded call is the expected oversized
  logprobs rejection. Seed0 uses 112 tokens, beyond its original 100-token cap.
  Likely subsystem: multiple test/workload constraints; original missing chat
  termination details and native early EOS remain distinct uncertainties.
  Investigation performed: host regression/2343-pass suites, focused served
  checks, native generation, source diagnosis and unchanged full-suite rerun.
  Resolution: fixed for served acceptance; native budget proof remains failed
  and is not used as a waiver.

- Observed anomaly: EngineDeadError appears in the v2 server log.
  Evidence: `acceptance_tracking_v2/server.log:11094` onward.
  Affected path: shutdown output-handler notification.
  Control or comparison: it follows the explicit idle-server SIGTERM/resource
  teardown, after completed acceptance requests; final performance has no
  execution error and cleanup records no device holders or owned processes.
  Likely subsystem: upstream shutdown ordering.
  Investigation performed: inspected surrounding raw log ordering and cleanup.
  Resolution: controlled teardown event, not an aborted acceptance workload.

## Scope Inspected

- Goal/skills: `/home/ttuser/qwen38-flash/multigoal_logs/09-09-vllm.prompt.txt`;
  installed 0.1.24 `stage-review`, `vllm-integration` and prefill/plugin/load
  references, `qualitative-check`, `tt-device-usage`, and tracing rules;
  plugin `AGENTS.md` and relevant interface/source contracts.
- Artifacts: README/work log, acceptance manifest, plugin-change and commit
  records, context/serving contracts, capacity and allocation-tracker positive
  control; direct async probes, both recorded scheduler-history controls,
  historical failed comparators, full48 mixed/history/logit artifacts; full
  sampling and host logs; HF/TT/served qualitative manifests and outputs;
  P2/P3 and all nine P5 requests/responses; full raw maximum-context response;
  raw/normalized benchmarks, process provenance, route audit, cleanup,
  aggregate gate and final precommit log.
- Code: full adapter; serving changes to `generator.py`, `model.py`,
  `cache_budget.py`, `factored_decode.py`; generator allocation, preparation,
  trace, mask, reload and output paths; mixed/async/replay/P5 observation
  drivers; approved plugin runtime and test diffs. No new trace exemptions
  were introduced in the model paths inspected.
- Commands: read-only `rg`, `sed`, `cat`, `git diff/status`, and stdlib-only
  Python artifact analysis. No imports of TTNN/vLLM, tests, HTTP requests,
  servers, device operations or implementation edits. Only this report was
  written.
- Independently rederived: 202 tracked mixed capture signatures, each once,
  with no request-time retirement; 198 performance signatures, each once;
  2207 device decode and 35 device prefill submissions with zero host/OOV
  events on performance. All source metadata entries in the final standalone
  manifest still match the worktree, and the served manifest matches it.
- Independently checked outputs: all 12 served qualitative texts exactly
  equal their same-length TT controls; all six prompt-token arrays and all 12
  actual served input arrays match HF. I read HF and served text, P3 answers,
  recorded sampling chat answers, and every P5 response. P5 recalls all nine
  unique passphrases with normal stop and no request error.
- Independently checked numbers: maximum-context raw input contains exactly
  262136 copies of token785 and eight valid output IDs, total usage262144.
  Benchmarks complete 1/1, 32/32 and 32/32 with 128, 3200 and 1024 output
  tokens. Primary raw TTFT is 6472.729225ms and mean TPOT103.350699ms;
  `1000 / TPOT` is 9.675793 tokens/s/user, matching the report. CI and prefill
  burst rates similarly rederive to 4.958759 and 4.961431.

## Residual Risk

- This is Stage 9 acceptance, not broad accuracy, release readiness or a long
  load soak. Full context was exercised once as required; the repeated-token
  maximum-context request is capacity evidence only.
- Serving proof is full48/C262144/B2/TP4 on the pinned stack and selected
  precision. Exact cache/RNG immediate-versus-queued state comparison uses
  reduced representative layers; full48 supplies served repeats, mixed
  lifetime, logits, quality, sampling and long-context acceptance.
- Long qualitative answers truncate at256; constrained bad-word outputs and
  historical execution-history sensitivity retain the limits above. No new
  universal determinism or model-quality exemption is inferred.
- Large raw artifacts remain in this persistent workspace. Compact staged
  evidence is not a substitute for preserving those raw files. Final closure
  should record the local checkpoint SHA and this review without pushing.
