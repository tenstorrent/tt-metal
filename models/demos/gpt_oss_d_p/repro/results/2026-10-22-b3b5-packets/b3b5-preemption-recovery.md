# Decoder preemption and replacement — October 22

## Accepted evidence and retained prefill

The first case, p2880 / ISL65536 / chunk65536, completed at02:07:59.232515UTC. Saved audit02:08:02.649138 verified one DONE case with zero errors, failed cases or pending measurements. All warmup and three measured requests remain unchanged. It used b09u02 prefill and b04u02 decode with native decoder services inside Slurm steps. No second case had started.

The first prefill103710 closed through the drained H2D sentinel protocol with shutdown-complete, exit0 and verified idle devices. New p4352/c65536 prefill307109 launched02:10:01.247006 and reached readiness02:12:46.678 (165.431s). Its PID, generated owner script and H2D/ACK identities are retained. No reset or relaunch occurred on the healthy prefill.

## b04u02 ownership loss

Slurm preempted allocation142578 at02:11:43UTC. Controller96693 detected COMPLETING and exited at02:11:44.135932 before further compute access. The scontrol query subsequently confirmed PREEMPTED. That job later aged out of scontrol. No b04u02 compute access occurred after ownership loss. Survival of driver174718 and ranks175157–175160 is unknown; old ready files do not establish liveness.

The b04 decoder had completed all41 captures/full36 layers at02:04:55.821 after2406.083s and served the accepted first case. The preemption is distinct from the earlier b10 seven-bank incompatibility and SSH authentication failures. Preserve those attempts separately.

The controller's stopped source/logs and available shared evidence were saved under evidence/failed/decoder-preempted-b04. The old decoder output and launch script remain at their original paths; no old model reset, restart or script modification was attempted. Publisher96694 was verified stopped before copying its shared local modules.

## b05u02 rejected replacement

Idle individual bh-glx-120-b05u02 was allocated as142653 at02:15:42 in a fragmented partition. Initial device check found PCIe24 reading0xffffffff. No model/build ran there. Bounded site recover-hosts driver323498 started02:19:03.918712 with max-attempts1. Recovery exited1 at02:20:03; devices24–31 remained unreachable after cleanup reset, with secondary bus reset and POST_RESET failures. Full health, reset, cluster-debug and exit evidence is under evidence/recovery-decode-b05.

After recovery exit, process and fuser checks confirmed no recovery processes or device holders. Allocation142653 was cancelled02:21:48 and subsequently verified CANCELLED. Never access the released host. Local b05 continuation drafts were prepared while recovery ran, but were never launched or installed as a sweep controller.

## b04u08 selected replacement

Idle individual bh-glx-120-b04u08 was allocated as142657 at02:22:00, expiry08:37:00UTC. The b04/b05 quad was already fragmented (b04u02 drained); no intact free quad/SC16 was taken. Initial health02:22:20.764149 passed32devices/32DRAM and all32 GDDR masks0xff/eightbanks, with no device holders. No reset was needed. DecodeIP10.32.16.161; retained b09IP10.32.17.193, allocation142547 expires08:36:05UTC.

Native build304494 launched02:23:48.843110 under runtime/build-decode-b04u08.sh, exclusive record runtime/build-decode-b04u08-launch.json and evidence/build-decode-b04u08.log/exit. New model output will be runtime/repro-b04u08-full36/decode, separate from prior attempts, with socket prefix gptoss_b3b5_20261022_b04u08_decode. Do not launch another build/model.

The continuation preserves one accepted case, checks the existing frontend/etcd and current ready prefill, waits the recorded build, checks idle eight-bank hardware, and launches one new decoder. It runs only the remaining86 cases. Services use guarded native launchers and bounded persistent Slurm steps; model startup uses guarded laptop ProxyJump. Read-only owner checks have bounded transport retries; mutations and requests are never retried.

All deadlines remain benchmark07:50, services07:55, owners08:00, work08:05UTC; allocation guard stops new work07:35. The existing prefill wrapper is unchanged. Exact Metal b3b5 and dependency pins remain unchanged; do not commit during measurements.

## Interpretation

The accepted first case used b09/b04u02; subsequent cases use b09/b04u08 if startup succeeds. Keep per-group host metadata and service affinity records. Host, cache, packet order and resource-policy differences prevent packet-only or commit-only causal attribution. No numerical accuracy claim is made. Do not exclude or rerun slower samples.

## Build-completion control race

Native build and TTNN import succeeded, exit0 at02:29:14.983867 (326.141s). Controller98045 read the exit marker before it was written, then queried the build PID after normal process exit. The ps Slurm step returned1, and the controller exited02:29:17.143386. Both allocations remained RUNNING; no decoder launch occurred. This is a controller race at successful build completion, not a native build failure. Evidence is preserved under evidence/failed/controller-build-exit-race.

New controller98291, resume-after-b04u08-build.py, launched02:31:29.163764 after verifying98045 had exited. It uses the unchanged ops/case/Slurm modules and verifies completed build exit0, TTNN import, exact launch command/PID/job/host before post-build idle health and the single decoder launch. It has no build launch or wait loop. Twenty-six host-only guards passed, including successful completion and failed/missing/stale receipt rejection. Existing publisher98046 remains running unchanged.

Replacement decoder346592 launched02:32:13.849339 after post-build health passed at02:32:07.519476. Its exact current launch identity was verified at02:32:41; no exit and not ready at that checkpoint. The controller continues capture monitoring. All18 future group configurations passed host-only validation of host/job, decoder output symlink, packet/chunk, worker endpoints and socket identities.

## Second decoder preemption and b08u02 replacement

Slurm confirmed142657 PREEMPTED at02:59:08UTC. Controller98291 stopped02:59:22.411306 before further compute access. Shared logs contain28/41 image timing records, no ready manifest and no second case. Old process survival is unknown after ownership loss; no further b04u08 compute access occurred. The accepted first case and ready p4352/c65536 prefill307109 remain unchanged. Publisher98046 was verified stopped before adaptation. Available shared logs and stopped local controller/publisher sources are preserved under evidence/failed/decoder-preempted-b04u08.

Idle bh-glx-120-b08u02 was reserved as142679 for5h30m, preserving intact free quads/SC16. Its idle health03:02:32.257006 passed32devices/32DRAM/all32 eight-bank masksff, without reset. IP10.32.17.129. Native build3738742 launched03:04:09.581332 with runtime/build-decode-b08.sh. Prepared continuation requires its completed exit0/import/exact receipt before one new decoder launch; it has no build or initial prefill launch path. All26 continuation guards and18 group configurations passed. Remaining86 cases will use b09/b08 if startup succeeds; all prior attempts remain separate.

B08 native build and TTNN import PASSED exit0 at03:09:13.644438UTC,304.063s after launch. Real completed-build preflight verified the exact build receipt, unchanged one-case audit, retained prefill, frontend/etcd and no duplicate model/services. Sole local controller1338 resume-on-b08.py launched03:09:53.127867; receipt b08-continuation-launch.json. Sole read-only publisher1194 publish-b08.py launched03:06:41.114339. TCP roundtrip passed; all accepted-case file hashes remained unchanged.

Post-build health passed32/32/eightbanks at03:10:30.769054. Sole controller launched replacement decoder3773856 at03:10:37.040328 under runtime/start-decode-owner-b08.sh, with new output/runtime socket. Capture startup is in progress and readiness must be verified separately; no manual traffic or duplicate launch occurred.

## B08 readiness and resumed measurements

- Decoder b08u02 is **READY since 03:48:34.688 UTC**: all 41 images, 36 layers, 16 KV configurations, capacity 131072, two allocated slots. Launch-to-ready was **2277.647672 s (37m58s)**, separate from prior attempts. Driver 3773856 and manifest owners 3774197, 3774198, 3774199, 3774200 were verified live; no rank error or owner exit. See `evidence/b08-ready-summary.json` and `b08-ready-detail.json`.
- Sole controller 1338 automatically attached both native managers and workers at 03:49–03:50 UTC. It started p4352/64K/64K at 03:50:36; the case runner launched at 03:50:41. At the service-attachment snapshot, the first accepted case remained DONE and the new case was IN PROGRESS. Read fresh case records before reporting totals.
- Both b08 decoder service steps have CPU affinity **0–63 (64 CPUs)**. Manager step 142679.92 and worker step 142679.93 are persistent until their recorded native service exits. This differs from the first accepted b04 case's 60-CPU service affinity; retain the cohort caveat. See `evidence/b08-service-attachment-checkpoint.json`.
- Prefill 307109 stayed ready throughout startup. No model relaunch, manual request or duplicate service was used. Keep the decoder loaded across groups; use `runtime/current-pd-run.json` as prefill changes.


The first b08 case, p4352/64K/64K, completed at 03:51:27.715678 UTC. Saved audit 03:51:31.118717 confirms 2/87 DONE with zero errors, failures or pending measurements. The controller began draining the p4352 group at 03:51:39 before the next payload; decoder owners stay loaded. All four request samples are retained.
