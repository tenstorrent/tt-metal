# Task-boundary dry route: authorized candidate, not yet demonstrated

Operator decision8 explicitly permits splitting release work on task boundaries and requires each release job to use the SHAs its dry run proved. A partitioned dry/release plan therefore does not inherently need another exception. This supersedes the skill's literal single-CI-job reading to that extent. It does **not** waive the complete frozen cohort, 12-hour per-suite fit, no-score dry condition, measured configuration acceptance, or evidence requirements.

The strongest unchanged-route candidate is **one selected task per job, planned_clients1**, frozen identically for that partition's dry and release. This removes both the lazy multi-wave environment problem and ambiguity about which task issued a model call. It is a concrete design to validate, not proof that the current release route already passes. The existing C2 five-task configuration is not silently replaced by this report.

## Authorization and accounting

Sources: `AGENTS.md:108-116`, especially “split release work on task boundaries so each job fits the CI job limit”; `bringup/agentic_evals/inputs.json` extended48h/run12h/max_parallel6; `/home/mvasiljevic/gemma4-31b-pipeline-v3/device-instructions.md` explicitly permits source-only model entries, immutable per-job SHAs and task-boundary splitting while forbidding workflow changes.

Freeze a manifest whose disjoint partitions have union exactly the already selected full cohort, including failures and difficult tasks. Do not remove tasks or assign seeds/settings based on outcomes. Each partition can use the ordinary model-specific task-list setting in its immutable TTI commit; it is not a new selector or harness. Its release repeats its own dry's exact TT/TTI/plugin/image/settings. The shared branch may move, but dispatch references remain immutable.

Keep the suite's total serialized projected duration, including repeated setup and appropriate queue/tail allowance, within12h on one device-equivalent. Five independent12h allowances or a five-device wall-clock speedup would not meet the extended mode's one-machine suite fit. Parallel dry jobs may fit the granted6-run pool if the coordinator records them; parallel release partition scheduling must not be represented as a one-device throughput measurement. A conservative sequential partition plan avoids that ambiguity. Pool labels cannot guarantee physical runner affinity; decision8 grants this pool route, not an undisclosed host selector. Record actual hosts and same device type rather than claiming an affinity control exists.

## Why singleton partitions can prove more than aggregate C2 progress

Pinned Harbor `e76c762fca5267dfb36e1f900f0a7724ed72e735`:

- `src/harbor/trial/trial.py:601-658`: a trial prepares/starts its environment, runs healthcheck and installs the agent before `_run` can query the model. A bare START still does not prove this, but an attributable agent model request does.
- `src/harbor/trial/single_step.py:35-52`: `_run_agent()` finishes before verification starts. While that sole agent is executing another model call, no earlier task in that partition exists to have been graded.
- `src/harbor/agents/installed/mini_swe_agent.py:1011-1014`: mini executes as the task environment command; its console is written inside `/logs/agent`, not streamed as native task-tagged progress into the CI log.
- mini2.2.8 `src/minisweagent/agents/default.py:77-125`: `step()` synchronously queries, receives the response, adds it and executes actions/observations before the next query. Recovered malformed tool responses append feedback and also require a received response. Ordinary uncaught exceptions terminate; they do not manufacture a growing successful conversation.

Thus a singleton's **second distinct, extended conversation request**, after its first request completes, can serve as indirect proof that its sole agent received a reply and its sole task environment started successfully. This is stronger than an HTTP200, a running count or continuing decode of the first request. A same-prompt transport retry or the second chunk of one long prefill is insufficient.

Task identity is then a proof by exclusion: exact immutable resolved singleton list, actual job total1, one model agent client, peer suites disabled, no other traffic, and model activity only after this suite starts. No task-ID log is magically added. If those premises cannot be verified for a particular dry job, this inference is unavailable and that job must not pass.

## Observable evidence and its limits

The old completed multi-task console `bringup/artifacts/stage12-swe-repeat-complete.log` demonstrates the signal channel, not a singleton dry pass:

- Lines1599–1621,13:53:35–36UTC show two initial chunked prefills:0→2048→2323 and0→2048→2206.
- Lines1824 and1950,13:53:41–45UTC show later cached-prefix prefills2176→2340 and2304→2694, after intervening decode.
- Line1624 shows2active rows; line3101 later shows0/5done,2running.

These observable starts/ends and decode transitions can distinguish chunk continuation from a new growing conversation when there is exactly one agent. In the old C2 run they do not, by themselves, label which task replied. For singleton proof require the initial prompt's final prefill, nonzero decode, termination of that response and a new extended request with cached prior prefix. Merely seeing two prefill events or a higher end position is not sufficient because chunked prefill itself does that. Record the exact event sequence and timestamps; compare it against the known single-client scheduler state and complete initial prompt progression.

The log snapshot must show the next request still actively generating, before the only agent ends; a stale0/1progress line alone cannot exclude verification already starting. Save that snapshot and run/commit metadata before normal stock cancellation. The cancellation still must occur early enough; if the task finishes/grades before the observation/stop, that attempt is not a no-score dry. No reliance on cancelled artifact uploads, no scoring the dry, no completed rehearsal masquerading as a dry, and no new wrapper or readiness mode.

A practical live proof can fail because the task submits on its first call, context shrinks, prefix is fully reused without an observable prefill, snapshots lag, or cancellation arrives too late. Those are limitations to record, not grounds to invent a reply identity or weaken “No scores.” The preserved logs do not establish that all five singleton tasks have a usable observation window. Actual future dry evidence remains necessary.

## Batch, capacity and score acceptance

The normal smallest supported batch at or above planned_clients1 is B1/max_num_seqs1. The current tool-suite evidence is C2/B8. Moving to singleton/B1 changes scheduling and may change outputs, so the skill requires a scored comparison on the same full frozen cohort against the current best before accepting that release configuration. No available GPQA C4 pair establishes SWE/Terminal equivalence. Existing B1 fixed-prompt performance likewise does not prove long real-agent capacity. Replay a faithful long history at B1 and confirm pool/limits; use actual task-defined CPU/memory/storage. An explicit measured reason could retain B8 for singleton jobs, but it must not be invented merely to avoid testing.

A2+2+1 partition retains C2/B8 for the pairs but still lacks task-identifiable first-reply proof in those pairs; total2active rows plus several replies could belong to refilling one agent. Singleton partitions are the simpler evidence route. Their repeated setup and lower concurrency may materially worsen suite time, so calculate before spending devices. If aggregate fit fails, this candidate does not rescue the release.

Terminal's agent is not mini; extending this singleton proof to Terminal requires checking its exact synchronous call/retry/control flow and an actual observable second-request window. This report does not claim that separate check is done.

## Coordinator decision boundary

There is no new authorization blocker merely to **prepare/evaluate** the frozen singleton-partition candidate: explicit task-boundary permission already exists. There is also no clean dry pass yet. The coordinator can rank the candidate's measured cost against other validity work, validate B1/score parity and total time, then freeze and attempt the exact per-partition dry using the proof above. Existing unresolved SWE output-limit validity remains a separate prerequisite; partitioning does not repair it.

If the strict proof cannot be obtained from a live singleton attempt without changing route/settings, retain the prior unresolved route question. Do not resolve it now based on this source argument alone, and do not ask the operator the same permission again merely because the task list is being partitioned under decision8.

Device use0; no runtime source, workflow, cohort, central evidence or published branch changed by this investigation.
