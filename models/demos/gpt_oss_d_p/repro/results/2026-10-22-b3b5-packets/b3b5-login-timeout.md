# Slurm login connection timeout — October 22

## Facts and observations

- Controller 1338 exited at 03:56:24.157729 UTC while reading `squeue` during the p8704/c65536 prefill readiness wait. SSH reported `connect to host ... port 22: Operation timed out`.
- The failed operation was an allocation read. No third case or p8704 group service had started.
- At 03:59–04:00, the same login path worked and both exact allocations remained divanovic RUNNING. Decoder3773856 and all four manifest owners were live and ready. Prefill604717 became ready at03:56:01.493 and remained live, without error or exit.
- Two accepted cases remain DONE. Their saved file hashes are recorded in `evidence/b08-login-timeout-state.json`.
- The existing retry classifier returned false for the exact observed SSH message. It recognized `Connection timed out` but omitted `Operation timed out`.
- Controller1338 was verified exited. Publisher1194 was then verified and stopped at04:00:16 before shared modules were copied. Both locks were free. Old sources/logs are preserved under `evidence/failed/controller-login-connect-timeout`.

## Unknowns and hypotheses

The underlying network/login timeout cause is unknown. The evidence does not distinguish a temporary network outage, login-host availability, or another transport problem. This is separate from the earlier compute-host authentication delay.

The control-path diagnosis has high confidence: the exact transport-error wording escaped the bounded read-only retry classifier. This explains immediate controller termination; it does not explain the network event itself.

## Experiments and intended change

1. Rechecked allocations through the unchanged login path. Both RUNNING; ownership loss was not observed.
2. Verified exact retained model identities and readiness, no third case, and no new group services. No model restart is needed.
3. Reproduced the old classifier returning false on the recorded error. Add only this spelling to a new immutable module's transport-read classifier. Test bounded retries and immediate rejection of ownership/security/identity/model failures; mutations remain unretried.
4. Prepare a new continuation requiring exactly the two accepted cases, their unchanged hashes, current ready p8704 prefill604717, ready decoder3773856/four exact ranks, and existing frontend/etcd. It has no initial owner/build/base-service launch path and runs only the remaining85 cases.

No running script, timeout wrapper, model source, cache, authentication setting, CPU policy, or allocation deadline is changed. Successful continuation will not establish the network timeout's root cause.

## Verification and continuation launch

Fifteen host-only guard tests passed at04:03:55; the recorded error fails the old classifier and passes the new one. Retry count remains at most3 with10s spacing; mutation paths remain single-attempt. The loop covers exactly85 unmeasured cases and retains both initial owners. Syntax checks passed. An initial local code-generation quoting error was corrected before any continuation was launched.

The real preflight passed: two accepted cases and all saved hashes unchanged, current prefill604717 ready, decoder3773856 and ranks3774197–3774200 ready/live, existing frontend/etcd live, no third-case or p8704 service/request records. New local controller4862 and publisher4863 launched once at04:05:13; exact receipts and source hashes are preserved. Controller repeats its preflight before attaching services. No new model owner, build, reset or request was manually started.

Controller4862 completed its repeated live preflight and reported both retained owners ready at04:06:02.678046. It automatically attached managers/workers at04:06:20–04:06:40. Both controller and read-only publisher identities were verified live. The original models were retained; the network root cause remains unresolved.

## First completed case after continuation

The retained p8704/c65536 model served its scheduled warmup and three measurements; the case completed04:08:04.785352. Saved audit04:08:27.178144 shows three accepted cases and no errors, failed cases or pending measurements. The controller then drained the group, closed the prefill by the supported sentinel protocol with exit0 and verified idle devices, and launched p2880/c32768 at04:10:21.747891. The decoder remained loaded. This demonstrates resumed execution; it does not establish the root cause of the earlier login timeout.
