# AutoFix: aligned mapped input upload

Failing contract: stage04 comparison must execute4096-token inputs on the single-chip baseline and TP4, without changing model math.
Diagnosis: AUTODEBUG.md identified explicit mesh mapper selecting device-side tilization as the strongest unverified path difference. Existing baseline harness omitted that mapper. No evidence implicated decoder attention/page capacity.

Discriminating tests:
1. Original mapper on count1 stalled before prefill (sliding_rep4096.log, instrumented sliding_rep4096_retry.log).
2. Omit mapper only on count1: count1 completed, TP4 stalled at upload [1,1,4096,5376] (sliding_rep4096_uploadfix.log).
3. Construct tiled mapped tensors on host, then to_device: both completed, PCC .99995167/.99993317;129 trace replays with tracking enabled (sliding_rep4096_hosttilize.json and manifest).

Repair: tests/run_multichip_decoder.py and tests/profile_multichip_decoder.py now separate host layout conversion from device transfer. This is setup-only; runtime forward, precision, positions and cache ownership are unchanged. Underlying device tilize kernel root cause is not established; this avoids its failing setup path and does not claim a kernel repair.

Recovery: three run-owned processes266295,267599,268126 were terminated only after attempted live triage. Triage was blocked in KMD power/ARC routines; first attempt returned dependency failures after workload termination; later bounded attempts returned124. No valid kernel-state report was obtained. After each, /proc FD scan found no device owners; bounded all-chip reset returned0, subsequent list showed all4, and FABRIC_1D mesh open/close passed. recovery_reset[2,3].log, recovery_list_after[2,3].log, recovery_mesh[2,3].log retain evidence. No other user's process was stopped.

Status: focused setup repair verified. Long-context, watcher, profiler and remaining multichip acceptance work continue.
