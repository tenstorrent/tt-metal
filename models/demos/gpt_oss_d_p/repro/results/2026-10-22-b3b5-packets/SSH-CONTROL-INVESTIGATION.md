# SSH control interruption during decoder capture

## Observed failure

At 2026-10-22 01:36:44.915 UTC, local controller93538 exited while verifying decoder owner174718 with a read-only SSH `ps` command. SSH reported `Connection to UNKNOWN port 65535 timed out`. No inference case had started. The independent decoder continued writing capture logs; 12 unique image timings and no rank Traceback/fatal errors were recorded at01:38:20. Both Slurm142547/142578 remained divanovic RUNNING, with no preemption. The retained prefill103710 remained ready and reachable.

The controller's exit and exact source/logs were preserved in `evidence/failed/controller-ssh-auth-timeout`. Publisher93539 was verified and stopped01:42:33 before its shared module was copied into a new version. Neither model was signalled, reset or relaunched.

## Diagnostics and limits

- Slurm reports b04 ALLOCATED, job RUNNING, approximately393GB free host memory, and recent node telemetry. This does not diagnose SSH authentication latency.
- Name resolution matches the recorded decoder IP10.32.16.129. A bounded TCP check reached port22 and received the OpenSSH banner.
- Verbose SSH completed key exchange, matched the known host key, and reported that the server accepted the offered user key; it then timed out before session authentication completed.
- An otherwise equivalent prefill SSH check succeeded. This was not a general loss of the laptop/jump-host path.
- A decoder check using ConnectTimeout30, ServerAliveInterval15 and ServerAliveCountMax3 succeeded in33.747s, confirming driver174718 and ranks175157–175160 with their exact original commands.

The underlying authentication delay remains unresolved. The successful longer check does not prove whether the timeout settings alone or a transient server-side recovery changed the outcome. No PAM denial, host-key rejection, ownership loss, decoder error or device failure was observed. Do not claim a repaired hardware or authentication root cause.

## Bounded continuation

New local files: `resume-after-ssh-timeout.py`, `ops_b04_control.py`, `case_ops_b04_control.py`, `publish-b04-control.py`. The existing controller was already gone and publisher was stopped before copying the shared modules. Old files remain unchanged.

The new SSH settings allow more authentication time. Only allocation queries and owner identity reads get up to three transport retries with10s spacing. Every attempted compute-host identity read rechecks allocations. Ownership loss, permission denial, host-key errors, stale identity and model errors fail immediately. Launches, services, model requests and shutdown operations are never retried by this mechanism. Existing allocation, benchmark, service and owner deadlines remain unchanged.

The continuation requires exact retained decoder174718 and prefill103710 records and zero case records. It has no initial build, health probe or model-launch path. It waits for the existing decoder, checks the actual flat MPI rank-error files, requires all41 image timings and readiness files, then runs the original87-case matrix and native prefill closure protocol. Readiness cannot override an error.

Twelve host-only regression/guard tests passed; `evidence/ssh-continuation-guards.json` records source hashes. Tests cover bounded read retries, no mutation retry, ownership checks before compute, rejection of stale identities and security errors, actual rank-file errors overriding readiness, exact87-case matrix and unchanged deadlines. No fixture values were published and no extra model request was sent. A real read-only preflight is required before launch; its receipt is `evidence/ssh-continuation-preflight.json`.

## Continuation launch

Read-only preflight passed01:45:30: exact original owners,20/41 decoder image timings, no errors/exit and zero case records. New local controller95673 and publisher95674 launched01:45:56, with fresh launch records and original locks. Initial process identities were verified; inspect resume-after-ssh-timeout.log for completion of startup guards. This is a control continuation, with no model relaunch or changed measurement settings.

At01:46:58.244741UTC the new controller completed its initial guards and entered the retained-decoder capture wait. The original model owners were reused; the87-case sweep remains pending decoder readiness. This confirms resumed control, not completed capture or a fixed authentication root cause.

## Repeated control failure and new diagnostic (01:50–01:57 UTC)

FACT: continuation95673 exhausted all three owner-identity transport attempts and exited01:49:58.834118. Publisher95674 remains live; no copying/editing of its active module has occurred. Shared logs at01:49:54 show24/41 captures without rank errors/exit, no readiness and no cases. No model was signalled.

OBSERVATION: one bounded read-only Slurm step reached b04 but `/bin/sh` reported `Cannot fork` before its hostname/process/config checks; the step exited2. This adds direct evidence of an inability to create a child process in that session.

HYPOTHESES: (H1, medium) a process/thread or cgroup limit affects SSH authentication and the shell; (H2, low) memory pressure prevents fork; (H3, medium) SSH authentication delay and the shell failure have distinct causes. No specific limit or causal link is established yet.

EXPERIMENTS: one verbose SSH identity check with120s connection allowance and180s overall bound tests whether authentication can complete without repeated sessions. A separate direct Python Slurm read (no shell child forks) reads own-user process/thread counts, limits, cgroup PID counters/events and memory. A reached PID limit would supportH1; low memory would supportH2; healthy limits/memory leaveH3 or a transient condition open. Both are read-only, allocation-guarded, and do not alter models/devices.

### Diagnostic results01:54–01:56UTC

FACT: b04 had759 own-user threads, including188 in each of four original decoder ranks; only9 user processes. Slurm's diagnostic step inherited softRLIMIT_NPROC512 while the live model limit is2318132. Available host memory was471162140kB, cgroup pids.max=max and pids.events max0. A controlled single-child fork failedEAGAIN atsoft512 and succeeded when only that diagnostic process raised its soft limit4096 (below unchangedhard2318132), then restored it. This identifies the separate Slurm shell fork failure. No model or host-global limit was changed.

The120s verbose SSH attempt still failed after121.520s. Crucially, the client emitted macOS `Unexpected keychain error ... User interaction is not allowed` and then `storing passphrase in keychain` immediately before the stall/remoteclose. This is new evidence for a local Keychain integration stall; it was not visible at the earlier debug level. The key was accepted by the server. A direct-key check with UseKeychain=no, preserving IdentityAgent=none, known-host checking and allocation ownership, is testing this hypothesis. The Slurm soft-limit failure must not be conflated with SSH's root cause.

FACT: UseKeychain=no did not recover decoder SSH: it still timed out after47.206s after key acceptance. Thus Keychain-only causality is rejected. The server's public PAM configuration includes Slurm ownership adoption; the drained-node helper's query is bounded3s. Which authentication/account stage blocks remains unknown. No PAM/SSH configuration, group, trust or access control was modified.

NEXT: validate the standard `srun --overlap --jobid142578` control path under the existing allocation. This path already reads exact model identities successfully. Any continuation must keep the laptop as sole coordinator, verify allocations before every command, preserve native service commands and deadlines, hold a Slurm step for each service's actual lifetime (avoiding detached-child cleanup at step end), and preserve full CPU affinity. Read-only steps can use a task-local softNPROC4096 within the unchanged hard limit; live models are untouched. This is a transport alternative while SSH remains unresolved, not an SSH repair.

## Slurm decoder-control continuation (prepared02:03UTC)

The second controller and its read-only publisher were both verified inactive before preserving their sources/logs under `evidence/failed/controller-ssh-timeout-repeated` and creating new versions. New local controller `resume-with-slurm-control.py` imports `ops_b04_slurm.py` and `case_ops_b04_slurm.py`; publisher `publish-b04-slurm.py` uses the same read-only collector/baselines. `slurm_control.py` routes only exact decoder owner-identity reads, recorded decoder service stops, and guarded manager/worker starts through the current Slurm142578 allocation. Unsupported operations fail closed. Prefill remains on the existing laptop ProxyJump path. No initial owner/build/health launch path exists.

Every Slurm dispatch checks both exact divanovic RUNNING allocations before running a step and checks expected host/job inside it. Short identity/stop steps terminate after the command. Each native decoder service instead keeps a separate bounded Slurm step for its lifetime using `runtime/slurm-service-supervisor.py`; the supervisor calls the original guarded `start-service-b04.py`, records the step/affinity, acknowledges the exact native launch, and waits for that recorded PID/start-time to exit. The laptop remains the sole sweep coordinator. Mutating commands are not retried, acknowledgement timeouts require inspection, and exclusive launch/step/local transport records prevent duplicates. Native service/owner/benchmark deadlines remain unchanged.

Control steps raise only their own inherited softNPROC from512 to4096 within unchangedhard2318132; existing decoder/pre-fill limits and threads are unchanged. **Measured transport difference:** Slurm service/control steps have60 allowed CPUs (exclude1,2,33,34), while retained decoder ranks keep64. The initial full-affinity assertion failed and is preserved in `evidence/b04-slurm-affinity-identity.json`; no affinity or cgroup restriction was bypassed. These service resources must be recorded as an additional setup/cohort difference against previous experiments. All new packet cases use the same service-launch policy.

Twenty host-only guard checks passed. They include the prior retained-owner/error/87-case/no-relaunch checks plus exact Slurm allocation/host guards, unsupported-operation rejection, persistent-service routing, unchanged deadlines and no mutation retries. Real child-lifetime and retained-owner preflight receipts are required before activating this continuation. SSH authentication remains unresolved; this changes the supported control transport without modifying SSH/PAM configuration, identity or trust.

## Verified activation and readiness02:07UTC

The real Slurm child-lifetime test passed5.941s including a4s child. Preflight02:03:57 verified exact original model owners,39images/no errors and zero cases. Controller96693/publisher96694 launched02:04:45; initial guards passed02:05:06. Decoder reached41images/full36/16KVconfigs at02:04:55.821, with original175157–175160 ranks, capacity131072/twoallocatedslots and2406.082720s launch-to-ready. No model restart/reset.

The controller automatically started native decoder manager584522 at02:05:58 and worker585301 at02:06:15 in persistent Slurm steps, plus the existing-policy prefill services/etcd/frontend onb09. Services reached the controller's health/discovery barrier. Firstp2880/64K/64K benchmark driver296589 launched02:06:54. This verifies the new transport can launch and retain the native services and reach measurement; it does not establish a repaired SSH cause or a completed87-case result.

### First complete group confirms service lifetime and stop protocol

First2880/64K/64K case completed02:07:59; audit02:08:02 has1DONE andzeroerrors/failed/pending. Allfour native services stopped without forcedPIDs. The two decoder service supervisors recorded their service exits and released their Slurm steps normally. Nativeprefill sentinel/shutdown-complete/exit0/idle succeeded, and controller recordedclosure02:09:42. It then launched next4352/64Kprefill307109 at02:10:01 while retaining the originaldecoder. See evidence/first-group-closed.json. The successful first transition validates this control path in the real run; the87-case experiment remains incomplete.
