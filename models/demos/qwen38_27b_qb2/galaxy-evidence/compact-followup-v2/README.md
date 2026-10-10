# Compact follow-up queue, October 10, 2026

The latest confirmed B16/32K/TP4 decode result is **16.55 TSU**, with a new
control repeat at 16.5473 TSU (60.433 ms). The target is 30 TSU / 33.333 ms.
The compact GDN block's isolated saving projects to about 20 TSU; its whole
model timing remains pending. These are native decode rates, excluding
prefill, HTTP and router time. Precision stays BFP8 weights/KV, BF16
activations and FP32 recurrent state; no speculative decoding.

After the active compact control/candidate/control comparison, a persistent
follow-up requires a >=1% B16/32K gain, stable controls, identical generated
tokens and matching source, precision, runtime settings and prompts. It then
runs eight-replica G0, all 198 GPQA questions with the existing 65536-token
output budget, and a B16/32K unprofiled/profiled full-model trace pair. A low
GPQA score is retained and can still be profiled, but never becomes a
qualification or serving promotion. The profiler pair must use identical
inputs and outputs; host/device timing reconciliation is recorded separately.

The follow-up model and config hashes exactly match the compact source.
Only controller/profiler/test files differ. Initial preflight found profiling
helpers absent in that older source snapshot; that attempt never launched
hardware. A new snapshot includes the helpers and passed **541 CPU tests,
69 subtests, one skip**, plus physical-test collection. Those CPU results
validate queue and accounting behavior, not the unrun device profile.

At 18:14 UTC, the queue was:

1. `qwen38-compact-gdn-v2-20261010.service`: active before-control sweep;
   B16/32K complete, B16/16K started. Invocation
   `43f1990817724573a9b130ed61f3ff0e`.
2. `qwen38-compact-followup-v2-20261010.service`: waiting, hardware not
   started. Invocation `1a140fb3cc4e4aea92234ad4d38de9d6`.
3. `qwen38-projection-sweep-v4-20261010.service`: waiting behind the
   follow-up, frozen projection source unchanged. Invocation
   `dd0ad89df8004f78aa4270a047e35f91`.

Only the exact task-owned projection v3 waiting controller was stopped to
reorder it behind profiling. The active compact benchmark was untouched.
All hardware work retains `/tmp/tt-device.lock`. Services survive SSH/session
disconnects, not reboot; predecessor invocation and clean terminal receipts
are mandatory. Follow-up is bounded at 36 hours/256 GiB host RAM. Its profile
artifacts go under `/dev/shm/qwen38-compact-profile-v1-20261010`, with per-stage
24 GiB total/8 GiB per-file caps and 32 GiB minimum free RAM-disk space. The
host had 275 GiB free there at staging; no NFS or native install changes.

Expected follow-up hardware duration is roughly 2–3.5 hours after the current
comparison, mostly G0 and full GPQA. This is an estimate, not a completion
promise. The follow-up is skipped if the candidate has no measured gain.

Receipts are snapshots taken while the queues were waiting/running. They
must not be read as completed compact full-model or GPQA results.
