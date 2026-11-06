# Coordinator dry observation handoff

Operational checklist for the accepted GPQA-only schedule; no dry has been launched or passed by this document. Use the already frozen GPQA run, exact command/config/image record and local benchmark arm. This is not a new harness, audit or permission request. All raw console files remain private under bringup/artifacts. Substitute known IDs from the coordinator's dispatched-run record; do not choose a job by guess or act on another owner's run.

## Read-only capture commands

```bash
dry_run_id='<confirmed dry run ID>'
dry_job_id='<confirmed GPQA device job ID>'
dry_private_dir='bringup/artifacts/gpqa-release-dry'
mkdir -p "$dry_private_dir"
date -u '+%Y-%m-%dT%H:%M:%SZ' > "$dry_private_dir/observation-start.txt"
gh api "repos/tenstorrent/tt-agentic-bringup-qb2/actions/runs/$dry_run_id" \
  --jq '{id,status,conclusion,created_at,updated_at,run_started_at,head_sha}' \
  > "$dry_private_dir/run-before.json"
gh api "repos/tenstorrent/tt-agentic-bringup-qb2/actions/runs/$dry_run_id/jobs?per_page=100" \
  --jq '{jobs:[.jobs[]|{id,name,status,conclusion,runner_name,started_at,completed_at}]}' \
  > "$dry_private_dir/jobs-before.json"
gh api "repos/tenstorrent/tt-agentic-bringup-qb2/actions/jobs/$dry_job_id/logs" \
  > "$dry_private_dir/console-snapshot.log"
date -u '+%Y-%m-%dT%H:%M:%SZ' > "$dry_private_dir/observation-end.txt"
```

Use a new timestamped filename for each necessary observation; never overwrite the snapshot that established the stop condition. Check each command's exit status and ensure the expected job ID maps to this run before reading or canceling. `gh api` follows the log-download redirect internally; omit `--include`, debug/verbose output and request URL diagnostics so signed redirects/credentials are not printed or stored. Do not fetch auth tokens. Downloads can take tens of seconds; record start and end. Do not replace these with `gh run view --log`, which rejects incomplete runs in the installed CLI. No-cache/query busting did not improve measured staleness; do not loop on them.

Inspect the snapshot privately. Match and record compact counts/line references, never raw prompts/reasoning. Useful safe index commands output only matched markers and line numbers:

```bash
rg -n -o 'autoport_registration|generator_trace_prepare|serving_pool_budget|impl-of-model|inference-server-git-ref|tt-metal-git-ref|vllm-git-ref|sha256:[0-9a-f]{64}|RUNTIME_MODEL_SPEC_JSON_PATH|Application startup complete|Uvicorn running' "$dry_private_dir/console-snapshot.log"
rg -n -o 'Requesting API:[^\r\n]*[0-9]+/[0-9]+|"active_rows": [0-9]+|"sampling_mode": "[^"]+"' "$dry_private_dir/console-snapshot.log"
rg -n -o 'Could not parse generations|API request failed|Retry attempt|Exception:|request batch\(es\) failed|Replacing with error sentinels|padding with error sentinels|API returned [0-9]+ response|API returned null content|__INFERENCE_ERROR__|__PARTIAL_OUTPUT__|Traceback|TimeoutError' "$dry_private_dir/console-snapshot.log"
rg -n -o 'Running postprocessing|Postprocessing|Processing results|exact_match|results_[0-9]|samples_r1_gpqa|Total evaluation time|Evaluation completed|Writing results|Saving results' "$dry_private_dir/console-snapshot.log"
```

These are indexes, not an automatic pass/fail parser. Startup command text may itself contain metric names or filenames, so inspect matched context privately before calling it a completed score. Progress regexes can include carriage-return updates; use the literal counter numerator/denominator and its **event timestamp**, not grep line count. Record the newest relevant event timestamp and fetch-end age. Match full source SHA/image digest against the frozen manifest, full60 layer list, adapter module path, B8/maxseq8/context262144, configured C4, stream=false/batch_size1, actual frozen IDs and unchanged request/output/sampling settings. Job wrapper SHA alone is not model source provenance. Image override TT/plugin flags alone are not proof of image contents; use already preserved build/push provenance and actual successful digest pull/startup.

## What successful progress proves, and what it does not

Pinned af401ef `api_models.py:684–727` checks HTTP status, awaits JSON for stream=false, calls `parse_generations`, and returns. Exceptions re-raise and enter retry. `_track` at867–870 increments only after this return; exhausted exception tasks do **not** increment and later become sentinels at875–894. Therefore k>=4 proves at least four non-throwing API-call completions, not merely four requests launched or four HTTP200 headers. Active_rows4 separately proves concurrent model work. It does not identify each initial request, which slots completed, or which replies contained answers.

Critical parser nuance: `openai_completions.py:212–251` catches malformed generation parsing and returns an empty string while logging **Could not parse generations**; these calls can increment progress. `choices=[]` can return an empty list with no immediate warning, and api_models.py895–910 detects its wrong cardinality and adds error sentinels **only after all tasks gather**. Null visible content may also be legitimate hidden reasoning and is not automatically a parse failure. Hence counters alone, even with no currently visible sentinel line, cannot categorically exclude every malformed/empty-response path. Scan the entire available invocation window for the listed errors/retries/warnings; do not describe k as independently verified nonempty answer bodies. For a strict first-agent-reply criterion, record this residual stock-log visibility limitation honestly. Existing same-route successful rungs support normal behavior but are not native per-response proof for a fresh dry. No extra instrumentation or response-producing request is authorized by this handoff.

## Stop decision, preservation and stock cancellation

Before cancellation, coordinator must have the frozen startup/config/image binding, observed activeC, partial successful-call progress C<=k<N, no compromising errors in the available invocation, local benchmark reachability, and evidence that this invocation has not reached score aggregation. Save the complete snapshot, run/job metadata and observation timestamps first. **Do not cancel merely because a stale snapshot shows partial progress.** A job can remain in progress during post-eval upload/cleanup. A stale log's missing final score is not proof of an absent score. Current metadata plus recent ongoing model events strengthen observation but do not make the download channel synchronous. Observed GPQA20 partial-to-final window was80min21.8s; latest REST lag was435s, with no proven upper bound. Actual dry timing may differ. If the no-score condition cannot be established, record the ambiguity/failure; do not invent a clean dry.

Only after the coordinator chooses the verified intended dry run and records the stop condition, use the ordinary authorized cancellation:

```bash
gh run cancel "$dry_run_id" -R tenstorrent/tt-agentic-bringup-qb2
date -u '+%Y-%m-%dT%H:%M:%SZ' > "$dry_private_dir/cancel-requested-at.txt"
gh api "repos/tenstorrent/tt-agentic-bringup-qb2/actions/runs/$dry_run_id" \
  --jq '{id,status,conclusion,created_at,updated_at,run_started_at}' \
  > "$dry_private_dir/run-after-cancel-request.json"
```

Cancellation is asynchronous. Through normal coordinator monitoring, preserve final run/job state and any stock cleanup logs using fresh filenames, without force-cancel, runner access, reset or a new dispatch. A cancel request timestamp is not the stop timestamp. If subsequent final logs reveal all requests completed or score aggregation before cancellation took effect, the dry did not satisfy no scores. Do not rely on post-cancel artifact upload, which may be absent. Keep the initial pre-cancel snapshot and final state even on a failed attempt. Only the coordinator's once-only dry record may declare pass; this checklist does not resolve residual observation races by wording.

For the local benchmark, use its frozen queue/commands and preserve successful response reachability plus clean idle shutdown, not spec tests or new accuracy scoring. Following a clean combined dry, release uses exactly its source/config/cohort/image identities. Any change requires another dry. The stage's actual release-start deadline and no-fixes release rule remain unchanged.

## Pinned successful-response cardinality invariant

Additional source verification narrows the silent-empty-choices caveat for the actual stock route. Local tracked-clean `bringup/artifacts/vllm-server-source-v0.31.0` is exact tag **v0.31.0**, commit **db9527a46873454610df6dbedf79a36d6bf1a7f6**. Stock TT pluginca7a18246e4111629ed6cb4a7d1192002749fccf explicitly installs `vllm==0.31.0` from source in `docs/install-vllm-tt.sh:42`. Confirm the actual frozen server startup still reports this version and normal V1 engine; source reasoning does not excuse mismatched installed runtime.

For one successful normal text generation with `n=1`, the source constructs **exactly one choices element**, even when the parsed visible content is empty or reasoning-only:

- `vllm/entrypoints/openai/chat_completion/protocol.py:228` defaults n=1;:700 forwards n;:720 chooses FINAL_ONLY for stream=false. Confirm no per-request override changes either value.
- `vllm/v1/engine/async_llm.py:500–502` takes the n=1 branch, adds the request with parent_req=None and request index0.
- `vllm/v1/engine/output_processor.py:308–312` suppresses unfinished FINAL_ONLY output. At:345–365 it creates a completion and sets `outputs=[output]` for parent_req=None; `_new_request_output` stores this list at:407 and `_new_completion_output` preserves request index at:453. An empty list is not the ordinary output path.
- `vllm/entrypoints/openai/chat_completion/serving.py:946–960` drains generation and returns an error if no final result exists. At:968 it iterates each final output;:970–972 rejects error finish reasons. At:1109–1128 it builds one ChatCompletionResponseChoice and unconditionally appends it, without filtering on content length. At:1189–1193 that choices list becomes the response. The `choices=[]` occurrence elsewhere in this file is the **streaming usage-only chunk**, not this nonstreaming route.
- `vllm/entrypoints/openai/chat_completion/api_router.py:64–73` returns ErrorResponse with its error HTTP code, or the normal ChatCompletionResponse as JSON. Ordinary empty visible content does not redirect into an error response or drop the choice.

Accordingly, once actual source/version, n=1, nonstreaming route, generated completion and successful HTTP response are bound to this dedicated evaluator traffic, the model-output `choices=[]` hypothetical is excluded by the normal stock implementation's cardinality invariant. Combine that source guarantee with successful lm-eval completion progress and **absence of caught parse warnings/errors** in the complete observed invocation window. An ordinary reasoning-only/empty model answer still has one choice and is a received model reply for dry reachability; it is not evidence of a correct answer, and its later scored cap outcome requires the normal trajectory audit. A `Could not parse generations` warning is different: the harness caught a malformed response and returned a fallback empty string, so it invalidates using that counter as clean parse evidence.

This is a conditional source argument, not a fresh native-body observation or a claim that a dry passed. If logs cannot bind the completed generation/HTTP traffic to the isolated evaluator, if runtime differs, if warnings are truncated, or if a response violates the stock invariant, retain uncertainty and investigate. Source guarantees cannot establish request identity, response chronology absent from logs, or remove the asynchronous log-publication/cancellation/no-score race. The earlier blanket silent-empty-choices concern applies to arbitrary/mock/malformed API producers, not the proved ordinary successful n=1 nonstreaming stock-vLLM path.

## Initial agents remain a separate required identity check

The exact skill clause is **“each suite's first agents get model replies.”** The coordinator requires the actual initial C request identities, not C later completions while initial requests remain unanswered. Therefore the aggregate k>=C rule in the earlier preparation is **necessary but insufficient for this stricter initial-agent confirmation**. Do not declare a dry clean from that counter alone.

Existing source/logs provide partial association only. Pinned lm-eval api_models.py985–1028 collates generation inputs, then820–862 creates asynchronous request batches and starts them through a shared semaphore/connector. The progress counter has no request ID. Adapter sampling rows carry no native request identity, and slot order must not be treated as task order. Stock Uvicorn access records expose client TCP peer ports and response status, but not model request IDs or question identity; a port can serve multiple sequential requests and may reconnect. Counting distinct ports cannot replace identified initial requests without additional already-present connection/lifecycle evidence.

Actual completed GPQA20 C4 console shows the first18 HTTP200 responses on only two client peer ports; the other two peer ports first appear at18:29:25.723UTC, immediately before all20 completion/scoring. The first four successful completions therefore were demonstrably repeated traffic on two connections, not four independently identified initial agents. This is consistent with two long initial requests holding slots while the other two refill, but it does not alone label their question IDs. It warns that the measured80min partial-progress window is **not a measured window after all initialC replies**. The schedule's early partial-progress duration cannot be represented as proof that the exact initial-agent dry criterion fits that window.

For the future actual frozen dry, search only existing stock logs for initial request creation/IDs and matched response completion. If that binding is absent, record `initial_agent_reply_identity=unknown` and do not silently weaken the criterion to aggregate completions. Native vLLM cardinality invariants solve the empty-choices question, not this identity/timing problem. This handoff adds no logging, request markers, alternate harness, runner checks or route changes. Coordinator must retain this unresolved confirmation before declaring any clean dry or relying on the revised schedule as an achieved operational route.
