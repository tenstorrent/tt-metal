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
