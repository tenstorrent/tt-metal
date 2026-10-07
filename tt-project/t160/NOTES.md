# t160: exabox for LTX-2.5 1080p 6s 4x8 jobs

## Dit nodes (how identified)
Slurm has no partition, feature or reservation named "dit". The nodes come from the Markham Allocation
Weekly Schedule (Google Sheet 1HGi57KTOnNwoIeFeDyMEpElrfHdY7AGXK_pwZbRZs98): rows 120-B23 and 120-B45 are
allocated to Wan (a DiT model), and B45 is noted "active DIT dev". That gives 8 nodes,
bh-glx-120-b0{2,3,4,5}u{02,08}, all in partition bh_sc5_B2B9_D12. Each one is a 32-chip BH Galaxy.

## Inventory (UTC)
20:15:57 and again at 20:23: no node qualifies (each needs State=IDLE and LastBusyTime >= 2 h ago).
- b02u02 ALLOCATED, LBT 12:36 (127094 nwoodall, 8 h)
- b02u08 IDLE, LBT 20:15 (just freed)
- b03u02 DOWN (unexpected reboot 19:44), LBT 18:52
- b03u08 IDLE at 20:23, LBT 20:19 (127373 ctr-ifabijanic ended)
- b04u02 ALLOCATED, LBT 17:22, no job visible in squeue (not ours: squeue -u smarton is empty)
- b04u08 ALLOCATED, LBT 19:55 (losullivan, do not touch)
- b05u02 ALLOCATED, LBT 19:51 (pshah, UNLIMITED)
- b05u08 ALLOCATED, LBT 15:21 (kmabee, 12 h)

## Staging on exabox (/data/smarton/fasth3; the login-node home is not visible to compute nodes)
- t48/: blob:none clone + t160.bundle (ttp/t158-g15blx02-ltx25-4x8), detached at bf7db12a149 plus
  submodules tracy, umd, tt-cluster-descriptors. Script stage_src.sh, log stage_src.log (STAGE_RC=).
- models/ltx-2.5/: the 4 LTX-2.5 files rsynced (-L) from the g15blx02 /mnt/MLPerf snapshot (~70 GB).
  The first copy only sent HF symlinks; it was redone. Log: runs/648/t160-copy.log (COPY_OK).
- LTX-2.3 checkpoint (video VAE) and Gemma-3-12B: shared /mnt/models/huggingface/hub, not copied.
- prep.sh (started after staging, log prep.log, PREP_RC=): managed python + venv in python_env/,
  requirements-dev, `build_metal.sh --configure-only` (CPM fetch). No compile on the login node.
- job.sbatch: builds if _ttnn.so is missing, then runs the e2e (run_e2e.sh env) and writes to t160/out-<job>.
  submit.sh: re-checks qualify.sh, submits with -w <node> --exclusive --time 03:00:00, cancels the job
  if it is still PENDING after 60 s, and logs to t160/alloc.log.

## Run 724 (2026-10-07 02:28Z)
- No HF token on g15blx02 or exabox, and LTX-2.5 is gated: weights can only come over the Mac tunnel.
  The run-648 rsync died at 21:41 with gemma at 5.8 GB (1.5 MB/s). pcopy.sh now sends 1 GiB chunks over 8 ssh
  streams (~3.4 MB/s, ~5 h for 62 GB), resumable via pcopy.done, verifies sha256 against the HF blob names.
  Detached: state/runs/724/t160-pcopy.{log,rc}; done when the log ends PCOPY_OK. If it dies, rerun the same
  command (`ttp detach t160-pcopy -- ttp lock t160-copy -- bash tt-project/t160/pcopy.sh`); done chunks are skipped.
- 600 s cap: job.sbatch split into MODE=build (no device, 45 min) and MODE=e2e (--exclusive, 10 min Slurm limit,
  pytest --timeout 540). A cold box (no JIT/DiT cache) will time out the first e2e job(s); caches persist,
  so rerun e2e jobs (one at a time) until one completes. Never raise the limit.
- Slurm lists these nodes as CPUTot=1 RealMemory=1: -c/--mem requests never schedule; use --exclusive.
- Our own job resets the node's LastBusyTime, so each next job lands on another qualifying node.
- /data (shared 80 TB) has ~240 GB free. job.sbatch refuses e2e on a cold cache below 230 GB free
  (caches ~80 GB, keep >= 150 GB for others). After the weights land (~62 GB) free will be ~175 GB: the first
  cold e2e will refuse unless others free space. Then exabox stays unusable; report, do not lower the guard.
- Bugs fixed: the EXIT trap's `pkill -P $$` returned 1 under set -e and aborted the trap (no .rc file);
  `du` of the missing cache dir failed under pipefail and ended the job silently.

## Allocation log (UTC)
- 127512 b04u02 build, submitted 02:36:26, PENDING (asked -c 64 --mem), cancelled by us 02:37:26.
- 127513 b04u02 build, 02:38:30 start, FAILED at 0 s (du/pipefail bug). Node reset to LastBusyTime 02:38:30.
- 127515 b03u08 build, 02:42:01 submitted, RUNNING at 02:43; limit 45 min; rc in t160/job-127515.rc.

## Next step
1. Build: `ssh exabox-login cat /data/smarton/fasth3/t160/job-127515.rc` shows JOB_RC=0 (else read slurm-127515.out).
2. Copy: state/runs/724/t160-pcopy.log ends PCOPY_OK.
3. `ttp lock exabox -- ssh exabox-login 'MODE=e2e bash /data/smarton/fasth3/submit.sh'`; ttp note the job.
   Hand off waiting on `ssh exabox-login test -e /data/smarton/fasth3/t160/job-<J>.rc`. Repeat while cold.
4. On completion: copy the mp4 to tt-project/t160/, take PCC/PSNR against baselines/ltx25_1080p_6s/ref_dv145
   (seed 0, _0.mp4 = DEFAULT prompt), save a still, check `squeue -u smarton` is empty, write state/ready/exabox.READY.

## 08:47 UTC check (light wake)
- Tunnel up. Build 127515: JOB_RC=0 at 02:55:49Z (13 min). squeue -u smarton empty.
- Copy alive (5 procs): gemma 25/25 chunks, transformer 39/40 chunks done; last chunk + sha256 check left, ETA <30 min.
- /data now 105 GB free (was ~240): others filled it. Below job.sbatch's 230 GB cold-cache guard, so the first
  e2e will refuse until space frees. Do not lower the guard. Next wake: if PCOPY_OK, check df; if <230 GB, hand off blocked/waiting on disk.

## Run 817 (2026-10-07 ~08:56 UTC)
- pcopy ended rc=1: gemma sha256 OK; transformer sha256 MISMATCH. Cause: chunk() runs in a child `bash -c`
  without pipefail, so chunks whose ssh hit "Timeout, server 127.0.0.1 not responding" (8 times) were marked
  done anyway. Fixed: per-try pipefail check, 5 retries, ssh timeout 1800 s + keepalive.
- pverify.sh (detached, state/runs/817/t160-pverify.{log,rc}): per-1GiB-chunk sha256 on both sides,
  drops mismatched chunks from pcopy.done, reruns pcopy.sh (recopies only those, checks full sha256, PCOPY_OK).
  Local hashing over NFS + 40 GB remote hash read: ~10-20 min, plus ~10 min per bad chunk at tunnel speed.
- /data at 08:55: 102 GB free (100%). Below the 230 GB cold-cache guard: no e2e until others free space.
- Next wake: log ends PCOPY_OK -> check `df -h /data`; >= 230 GB free -> step 3 (submit e2e); else hand off
  waiting on disk (probe: df avail >= 230 GB). Not PCOPY_OK -> read the log, rerun pverify.sh the same way.
