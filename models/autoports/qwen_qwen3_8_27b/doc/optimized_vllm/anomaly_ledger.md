# Anomaly ledger

## Allocations while traces are live

Observed: allocator.cpp emits a generic potential-corruption warning in the
tracking-disabled performance server (`after_b32_server.log`, line182).
Affected path: warmup/capture and later eager operations alongside live traces.
Control: `after_b32_tracked_command.json` and `after_b32_lifecycle.json` exercise
identical model/adapter source with allocation tracking enabled before import,
program-cache allocations included, all four trace signatures and mixed prefill,
decode, sampling, row orders, offsets and tails. All112 requests pass. The reduced
exact-state control also passes with tracker enabled. No new buffer acknowledgment
or allocation scope was introduced. Runtime source hashes match the performance
server. Resolution: controlled; the warning explains why tracked acceptance and
untracked performance are separate. It is not evidence of detected corruption.

## Small observed performance deltas

Observed: mixed-traffic255.429→255.249s and CI237.215→237.238tokens/s.
Control: same workload/argv/environment; terminal-call elimination independently
established by exact reduced controls and full-server chunk events.
Resolution: controlled; no meaningful serving speedup is claimed from these tiny
single-run differences. The change removes demonstrably unused work while
preserving the already-fast decode path. Long-prefill pauses remain disclosed.

## Probe environment and mesh recovery

Observed: first probe lacked cache environment and failed with permission denied
at //.cache/tt-metal-cache; corrected probe then failed opening the mesh on an
unchanged Ethernet heartbeat, before model execution.
Evidence: probe_cache_environment_failure.log, probe_mesh_open_failure.log.
Investigation: corrected cache paths; stopped only failed probe1655652; bounded
list/reset/list all exited0, all4chips visible; ring1x4 open/close passed.
Resolution: recovered; repeated device control and full B32 serving passed.
No operator reboot, lock removal or profiler collection was needed.

## Watcher Ethernet kernel capacity

Observed: ACTIVE_ETH program29264B exceeded26624B by2640B during mesh open.
Evidence: terminal_suppression_watcher.log.
Investigation: prescribed TT_METAL_WATCHER=10 with
TT_METAL_WATCHER_DISABLE_ETH=1 retry passed exact state/logit/decode controls and
allocation tracking; terminal_suppression_watcher_noeth.log.
Resolution: scoped tooling limitation; compute watcher assertions enabled,
Ethernet watcher excluded. Not represented as full Ethernet watcher coverage.

## Synthetic low-token-ID traffic text

Observed: first_137/138/139 SSE initial chunks contain decoded prompt bytes or
replacement characters; synthetic low-ID fixtures generate nonlinguistic text.
Affected path: run_serving_lifecycle.py deliberately synthetic token-ID requests,
not chat-template quality evaluation. Independent review compared first137/138/139,
both long chunks and prefix cold/hit against Stage09
readiness_vllm/full_b32_budget131072_lifecycle.requests.json; all seven raw output
texts match exactly. Resolution: controlled against prior serving. These requests
prove lifecycle/state behavior, not language quality. Final prompt-correct shared
and packaged suites remain separate acceptance evidence.

## Sampling-test HTTP client teardown

Observed: two `AsyncClient.aclose()` tasks log `Event loop is closed` inside the
captured log of passing test_repetition_penalty_mixed_batch.
Evidence: after_b8_sampling_tests.log, lines186-237; same client-teardown signature
exists in Stage09 full_b8_sampling_tests.log. Stack is in httpx2/asyncio teardown,
not model/device execution. All76 assertions pass, subsequent presence/frequency
penalty tests pass, and server shutdown leaves no live API/EngineCore process.
Resolution: controlled harness teardown warning; not a sampling correctness pass
waiver and not represented as graceful closure of those HTTP client tasks.

## Excluded in-flight B1 run and rejected snapshot candidate

Observed: a terminal-only B1 process overlapped an experimental kernel ABI edit. Its source stability could not be proven. Resolution: exclude `superseded_b1_*` from acceptance, stop the process, isolate the new ABI filename, then rerun B1 on stable selected source. Snapshot batching passed tracked/watcher copy correctness and full mixed traffic but did not improve measured B16 or mixed-traffic latency; reject it and restore the simpler runtime. `selected_source_restoration.json` verifies all runtime hashes against accepted terminal-only B32 evidence. Candidate artifacts named `final_b16_*`/`final_b32_*` are historical rejected experiments.

## Final shutdown and process scan

Observed: explicit runner SIGTERM uses vLLM abort timeout0 and reports force-killing the remaining EngineCore before its adapter shutdown audit prints. All requests had completed and API shutdown completed. Evidence: selected_b1_server.log, cleanup.json. No final B1 shutdown-counter claim is made; no-host fallback evidence is the stable source/manifest and tracked same-source B32 plus B16 audit. Final scan finds no live server/engine or accessible device holder. The first comm-only scan counted an inherited zombie PID1635300 (started10:51:37, PPID1); ps stateZ/empty cmdline/noFD proves it is not a live leftover from this stage. Resolution: clean resource ownership, explicitly not a graceful adapter-destructor claim.

## Stage checker advisories

Exit1 is advisory-only: historical isolated plugin-test16384 log lines, inherited measured weight copies, and configured289-ID hybrid-pool shortfall for planning32x32768. Current final manifests serve262144 and no capacity was lowered. The memory ledger includes physical pool/scratch allocation checkpoints and does not claim a true transient peak. See stage_check_verdict.json. No critical gate failed.

## Capped qualitative outputs

All nine greedy outputs exactly match all-layer autoregressive controls, including three length-capped256-token outputs. Shared sampled code also reaches explicit finish_reason=length at256, with a complete function and incomplete example. Raw reasoning remains in content under the checkpoint template with no parser configured. Resolution: controlled model/prompt/budget behavior; report capped smoke scope and do not claim JSON-only whole responses or complete long answers.
