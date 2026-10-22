# Stop helper interruption — October 22, 05:15 UTC

## Facts and observations

Controller 4862 completed the five p2880/c8192 cases and saved the 29-case audit at 05:14:58.446376: zero errors, failed cases or pending measurements. The 116 requests remain saved. The prefill worker stop completed at 05:15:33.731529 with no forced PID. The next decoder-worker stop failed at 05:15:45.616220 while evaluating `forced = [p for p in selected if live(p)]`. Its `/proc/<pid>/stat` read raised `ProcessLookupError: [Errno 3] No such process`.

The helper handles `FileNotFoundError` but not `ProcessLookupError` in that read. A host-only reproduction executes the original `live()` function with the recorded exception injected at the filesystem boundary; the exception escapes exactly as recorded. This is a process-disappearance handling defect in cleanup. The exact descendant PID and kernel scheduling interleaving are unknown. There is no evidence here of a model failure or transport failure.

Both allocations remained RUNNING. Controller 4862 was absent before publisher 4863 was identity-checked and stopped at 05:17:06.265243. Their stopped sources, receipts and logs are preserved under `evidence/failed/controller-stop-process-race`. A shared snapshot hashes 2813 accepted evidence files, records exactly 29 DONE cases, and confirms no shutdown sentinel was sent for current prefill 856294.

## Concurrency model and hypothesis

The stop helper and its selected service processes execute concurrently. TERM initiates process exit but does not lock the `/proc` entries. Reading a selected process after exit can return either a missing file or ESRCH. The observed ESRCH escaped the helper's narrower exception handler, aborting the controller before manager stops and prefill closure. Confidence in this local failure mechanism exceeds 80% because the traceback identifies the boundary and the original function reproduces it. Model state and remaining service descendants must still be inspected independently.

## Change and verification plan

A new versioned helper treats only `FileNotFoundError` and `ProcessLookupError` during process inspection as disappearance. ESRCH while sending a signal also means the process has already gone. Other errors, including permissions, remain fatal. The helper retains start-tick and zombie checks and narrows accepted records to group manager/worker services. It never accepts model-owner records. No stop, launch or model request is retried.

Host-only tests cover the recorded ESRCH, disappearance at initial capture and signal delivery, ordinary TERM, PID reuse, zombies, permission errors, host/identity mismatch and owner-record rejection. A read-only inspector will establish which original service process groups remain. A new continuation may finish only the outstanding cleanup, close the retained prefill using the supported sentinel protocol, then run exactly the remaining 58 cases in eight new groups. It must preserve all accepted hashes and the loaded decoder, pass real preflight, and acquire the original controller lock. No existing entrypoint is restarted.

## Remaining questions

Which service processes remain after the interrupted stop? Is the retained prefill still ready? Are the decoder driver and all four manifest owners live? Record the inspection and regression results before dispatching any further mutation. The earlier network reset and login timeout remain separate incidents with unresolved transport causes.

## Verification results

Eleven helper regressions and fourteen continuation/transport guards passed (25 total). No test fixture or model request was published. Read-only process inspection found both worker roots, descendants and process groups absent; both original managers remain live, no old stop helper is active, and no case/request process remains. The prefill owner remains present. The continuation therefore omits both already-performed worker stops, stops only the two still-live managers, verifies their absence, and then sends the first supported shutdown sentinel. It retains the existing decoder and base services. The exact service inspection and new source hashes are saved in `evidence/stop-race-service-inspection.json` and `evidence/stop-continuation-guards.json`.

## Live preflight and launch

Real preflight passed with exactly 29 accepted cases, all 2,813 saved file hashes unchanged, retained ready prefill 856294, ready decoder 3773856 and four exact manifest owners, both base services live, workers absent and both managers live. No future group existed and no sentinel had been sent. Sole new controller 10762 and publisher 10763 launched once at 05:24:08 UTC. Receipts record the exact source hashes. Controller repeats these guards before any mutation and will finish only the outstanding cleanup before the eight remaining groups. Actual closure and resumed measurements are not yet claimed.

## Successful interrupted closure

Controller10762 repeated preflight and began cleanup05:25:10.239765. It stopped only the outstanding managers at05:25:18.612513 and05:25:23.867004 with no forced PIDs, verified group service absence, and sent the first sentinel. Prefill856294 logged sentinel received05:26:03.015 and shutdown complete05:26:03.355, exited0, and the controller verified idle devices and closure05:26:12.013120. Decoder3773856 remained ready. The controller launched the next p4352/c8192 prefill925695 at05:26:32.284055; exact identity and effective4352/default2880/exactb3b5/coreSHA passed05:27, with compilation in progress and no exit/error. Recovery evidence is saved in `evidence/stop-continuation-recovery.json`. The accepted29 cases remain unchanged; no request or completed worker-stop mutation was retried.

## First full group after recovery

The p4352/c8192 group completed its five scheduled cases at05:34:04.798197. Saved audit05:34:08.180609 confirms34 accepted cases,136 requests and no errors, failed cases or pending measurements. The revised helper stopped both workers and both managers with no forced PIDs; the evidence retains every stop receipt. Controller verified idle devices and supported prefill closure at 2026-10-22T05:35:50.155457+00:00. Owner925695 is closed; never restart it. Read fresh current-pd-run.json and subsequent launch records for p8704/c8192. This verifies ordinary group cleanup with the revised helper; it does not identify the exact PID/interleaving of the earlier ESRCH.

## All 8K groups complete; 4K transition

All five p8704/c8192 cases completed by05:43:56.649466. The audit at05:44:00.030700 records39 accepted cases,156 requests and zero errors, failed cases or pending measurements. Every valid64K/32K/16K/8K case is now complete for all payloads. Both workers and managers stopped with no forced PIDs, and the supported prefill961949 shutdown finished with sentinel received, shutdown complete and exit0. Controller verified idle devices and closure05:45:39.103456 before launching p2880/c4096 owner990835 at05:45:59.250592. Its request loop began05:48:38.408; exact identity and effective/default2880, exactb3b5/coreSHA passed05:48. Decoder3773856 stayed loaded. See evidence/checkpoint-0544.json and c8k-complete-c4k-transition.json. There are48 cases in six4K/1K groups left; no request, launch or completed stop was repeated.
