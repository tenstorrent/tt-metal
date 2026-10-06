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

## Next step
1. `ssh exabox-login bash /data/smarton/fasth3/qualify.sh` exits 0, and prep.log shows PREP_RC=0 PREP_OK.
2. `ssh exabox-login 'TLIM=03:00:00 bash /data/smarton/fasth3/submit.sh'`, then `ttp note` the allocation.
3. Hand off waiting on `ssh exabox-login test -e /data/smarton/fasth3/t160/job-<J>.rc`.
4. On completion: copy the mp4 to tt-project/t160/, take PCC/PSNR against baselines/ltx25_1080p_6s/ref_dv145/seed0.mp4,
   save a still, check `squeue -u smarton` is empty, and write state/ready/exabox.READY.
