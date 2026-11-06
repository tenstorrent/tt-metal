# AutoDebug: late Docker launch survives failure and collides with its retry

## Verdict

**Confirmed generic lifecycle defect:** `run_docker_command()` declares launch failure after 30 seconds without cancelling or reconciling its pending Docker launch. It raises before registering cleanup and before returning the container identity. `ServerCommand.execute()` then retries with no usable ownership handle. The original container can become running after the failure decision and compete with its retry for the same published port.

For Terminal smoke run **42182632490**, the first own container, **tt-inference-server-e9ac5761**, demonstrably continued into the pinned model download after being declared failed. The second own container, **tt-inference-server-2b55ff39**, failed to allocate **0.0.0.0:8000**. Self-contention from the abandoned first launch is the leading, strongly supported explanation; an external owner is not required. The artifact does not contain a contemporaneous port-to-container inspection or Docker event history, so the precise port owner at the second bind and the source of the initial startup delay remain unproven.

This is a container launch/ownership failure before model readiness or Terminal task execution, not evidence of a Gemma numerical, device-kernel, or Terminal evaluation defect.

## Scope and revision integrity

Inspected the exact TTI revision **a087be086b6e2cd6edf215dddb4c47c48463f13a** using `git show`, in `/home/mvasiljevic/gemma4-31b-pipeline-v3/tt-inference-server-stage12`. The prepared checkout HEAD was `8a3c949317fdf43ba11086ad90d11a6555046d54`; comparison found no differences in the three causal files below. All source line references in this report refer to the pinned revision, not an assumed current implementation.

Evidence log: `/home/mvasiljevic/tt-metal-gemma4-31b-pipeline/bringup/artifacts/stage12-terminal-explicit-job.log` (line numbers below). Startup contracts were read. No target execution, Docker commands, hardware operations, source changes, workflow changes, credential access, or old Gemma implementation inspection were performed. Host tests below are proposed, not executed.

## Direct observations

| UTC time | Observation | Job-log evidence |
|---|---|---|
| 08:35:14 | Checkout resolves to the requested `a087be0…` SHA. | 768–789 |
| 08:37:06.459 | First command names `tt-inference-server-e9ac5761`, publishes `0.0.0.0:8000:8000`, and launches with `--rm`; its stdout/stderr goes to the first Docker log. Image pull has completed. | 1270–1298 |
| 08:37:36.857 | First launch reports the 30-second timeout and raises from `run_docker_command`. | 1299–1316 |
| 08:37:36.858 | Caller immediately begins attempt 2/2, including another image pull. | 1317–1321 |
| 08:37:45.128 | Second command names `tt-inference-server-2b55ff39`, publishes the same port, and launches into a distinct log. | 1330–1351 |
| 08:37:45.425 | First container's own log reports loading this run's mounted runtime spec and setting P300x2. Subsequent requests access the pinned HF revision `842da3794eaa0b77d5f08bae87a17459d91ff475`. | 1921–1960 |
| 08:37:45.526 | Job streamer discovers the first named container. | 1352 |
| 08:38:15.571 | Second launch also reports the generic 30-second failure; server command fails after two attempts. | 1353–1377 |
| 08:38:25 | Collected Docker stderr identifies the second endpoint and says port 8000 is already allocated. Separate `docker logs` output is explicitly headed with the first container's name. | 1916–1923 |
| 08:38:26 onward | End-of-job workflow invokes broad stop/remove commands, after both launch failures. | 2004–2018 |

The outer 08:38:25 timestamps are log collection times. The embedded 08:37:45 timestamps belong to the first container's application log. The port-error line has no original event timestamp; do not claim the bind failed exactly at 08:37:45.128. Likewise, first application output is not the Docker daemon's exact `StartedAt` time.

## Complete causal chain

1. `workflows/run_docker_server.py:599` generates a new name on each invocation. Lines 856–858 call `Popen(docker run …)` asynchronously; this returns a CLI process handle, not proof that Docker has completed creation/network setup/start.
2. Lines 860–872 wait only for a nonempty `docker ps` result filtered by that name, for 30 seconds. This is running-container discovery, not an HTTP health wait. The loop does not inspect all container states and does not check `docker_process.poll()`.
3. Lines 874–882 convert non-observation into a generic exception. There is no stop/remove, process wait, or launch reconciliation in this branch. Closing a parent file handle would not cancel a child or daemon operation, either.
4. Cleanup registration at lines 884–894 happens only after discovery succeeds. The identity/process payload at lines 910–916 is also unreachable on timeout. `--rm` concerns container exit; it does not cancel a pending launch when Python raises.
5. `workflows/server_lifecycle_provider.py:41–49` simply forwards the launch. `workflow_module/commands.py:441–447` catches its exception, calls `_teardown_server(spec, None)`, and continues. `_payload_get(None, …)` returns None (211–214); Docker teardown only runs when the payload supplies a name or ID (400–407). Thus this attempted teardown does nothing.
6. The next attempt obtains a new name but requests the same host port. The original launch remains capable of completing. The first named container's subsequent application output directly proves that timeout did not establish termination. The second endpoint's bind error matches competition with that surviving launch.
7. Missing CLI exit polling also masks the second attempt's daemon error behind another full generic timeout. Process liveness checks in the later readiness stage cannot help: `_wait_until_ready` is called only after `_launch_once` returns (449), which neither attempt does.

`TT_SERVER_READY_TIMEOUT_SECONDS` controls a later HTTP readiness budget (`commands.py:139–160`); raising it cannot alter this hard-coded 30-second discovery deadline. Increasing only the discovery timeout might avoid this particular timing window, but does not repair failure ownership. Reducing attempts to one avoids self-collision while still allowing the first launch to escape supervision; it is not a lifecycle fix.

## Competing explanations and limits

| Candidate | Evidence and prerequisites | Assessment / discriminating check |
|---|---|---|
| Own late launch survives timeout and blocks retry | Direct missing-cleanup path; first named container logs after timeout; same published port; second endpoint bind failure. | Leading explanation. Exact container inspect/network state and daemon events would establish port ownership and creation/start timing. |
| External service owns port 8000 | Would require an unrelated owner at the relevant instant; none is identified in this artifact. | Not proven or needed. Snapshot port bindings and host listener ownership at failure to distinguish. Prior broad cleanup alone cannot rule out a later external owner. |
| HF download consumes the 30-second startup budget | Download appears in first container logs after timeout; Docker discovery does not wait for download completion. | Does not explain the discovery failure as an application-readiness timeout. A slow daemon/start path before visible running state remains possible. |
| Image pull itself exceeded discovery timeout | Explicit image pull completed before each `Popen`. | The observed explicit pull is outside the discovery timer. Storage, mount/volume initialization, runtime/network setup, or daemon load could delay launch, but the log does not discriminate among them. |
| First container was permanently dead | Contradicted by named-container application logs after failure. | Rejected for this run. |

The final cleanup prints `afb6b1b7bc45`, but this artifact does not explicitly map that ID to a name. It is not independent proof of which container owned the port. No claim is made that a later server would have become healthy or that downloads completed.

## Smallest discriminating host tests

Use isolated mocks with a fake monotonic clock and fake subprocess/Docker state; no device, network, real Docker daemon, or model imports are necessary. These are regression tests for a separate generic fix, not changes to the frozen workflow.

1. **Late startup, then retry:** model the first `docker run` remaining pending beyond 30 seconds and becoming running at t=39. Return empty running-only discovery before then. Drive `ServerCommand` through two attempts. Track names, pending launches, port reservation, stop/remove calls, and process reaping. Existing code should expose a live first attempt and second bind conflict. A fix must either continue supervising that first launch or establish termination before permitting the second launch.
2. **Termination/create race:** at timeout, let an initial inspect say absent, then let the pending daemon create/start complete. A fix must not treat one “not found” or merely killing the Docker CLI as proof of cancellation. Assert that uncertain cleanup prevents retry; an unrelated container remains untouched.
3. **Immediate CLI failure:** return a nonzero Docker CLI exit and stderr containing the bind error, while discovery stays empty. Assert prompt failure with the actual error and cleanup of only owned resources, rather than a 30-second generic timeout. Include a successful discovery control preserving the process payload and normal teardown.

The pinned `tests/test_run_docker_server.py:10–31` covers immediate successful discovery and retention of the process handle only. `tests/workflow_module/test_commands.py:437–458` tests retries after a synthetic exception with no live resource; lines 494 onward test a dead process only after a payload is returned. Neither covers this pre-payload ownership gap.

If daemon-level confirmation is later needed, use a disposable host-only container and isolated port under an authorized host test. Capture exact-name identity, Created/StartedAt/state, network port mapping, CLI return code, and Docker events around the transition. Do not use a TT/model reproduction merely to test this lifecycle defect.

## Smallest intervention and safe cleanup boundary

Keep the main harness and workflow stock. The confirmed defect warrants a focused generic fix on a separate branch: make `run_docker_command` own the entire launch transaction, including exceptional exits before payload return. Preserve identity before starting, observe CLI failure and container states separately, and reconcile/cancel the exact owned attempt before returning failure eligible for retry. Use bounded subprocess calls and a monotonic deadline. Cleanup failure/uncertain pending creation must fail closed against another launch. Registering only an `atexit` callback earlier is insufficient because retries occur before process exit.

A fix may return structured launch-failure identity to the caller or perform cleanup in the launcher; launcher-local ownership is the smaller boundary because it already has the name and `Popen` handle. Any timeout adjustment is secondary and needs its own rationale. No model settings, serving precision, task selection, or CI workflow change is implied.

For follow-up cleanup, first preserve the two attempt logs and narrowly selected state metadata (avoid dumping secret-bearing environment fields). Resolve the exact names `tt-inference-server-e9ac5761` and `tt-inference-server-2b55ff39` against this job's creation interval, image and runtime-spec mount; retain immutable IDs where available. Stop/remove only verified owned instances, and reconcile pending creation plus the CLI process before retry. Do not target all containers, a name prefix, or “whatever owns 8000.” Do not delete the shared model volume/cache or reset TT devices. If an unrelated port owner exists, leave it alone and report the collision. Broad cleanup commands in the historical workflow are evidence, not a safe operator cleanup recipe. No cleanup was performed here.

## Draft review / other potential issues

Rechecked the headline against the exact timeout branch, atexit registration order, exception handler, payload accessor, teardown guard, and explicitly named container log. The conclusion survives that review. Startup delay duration and exact port-owner attribution remain qualified rather than inferred as facts.

Additional robustness issues, not needed for this causal chain: `docker ps` name filtering is not an exact identity check; its subprocess has no timeout; wall-clock timing can jump; and query exceptions can also bypass pre-payload cleanup. No artifact evidence establishes these as triggers here. The retry-test class docstring describes opt-in retries while its test asserts retry-by-default; this documentation discrepancy does not cause the failure.
