# K3.1 P→D on exabox: plan

Host: `bh-glx-120-c04u08` (job 191886), tmux session `k31`. Second galaxy: `bh-glx-120-c05u08` (job 192244, started about 21:41; the original c05u02 hold 191888 was cancelled by the 2 h idle limit).
Partition `bh_saad_k3`, 9 h limit, started about 18:35 on 2026-11-10.
Source tree: Ryan's `/data/rmiller/k31-migration` at `7370b8c2685` (read-only for us).
Our tree: `/data/bklockiewicz/k31-migration`.

## Steps

1. [x] Wait for `/data/bklockiewicz/k31run/copy.log` to print `git done: 7370b8c2685...`.
2. [x] Clone the venv:
   `/data/rmiller/jobs/k31/venv_clone.sh /data/rmiller/k31-migration /data/bklockiewicz/k31-migration`
   Needs CPython 3.10.19 in the node's uv store. The 3.7 GB venv has absolute paths, so it cannot be copied as is.
3. [x] Copy `/data/rmiller/jobs/k31/*.sh` to `/data/bklockiewicz/jobs/k31/`.
   The job script hard-codes `/data/$USER/jobs/k31` and also needs `/data/bklockiewicz/jobs/.built.k31-migration` containing the commit.
4. [x] Make our own weight-cache overlay. The script defaults to `/data/$USER/k31cache_overlay`, and Ryan's is read-only for us.
   Build symlinks to the Weka cache in our own directory.
5. [x] Check the build. `build_Release` was built at Ryan's path, so rpaths or cached paths may point into `/data/rmiller`.
   If anything fails to load, rebuild with `k31_build.sh`.
6. [ ] Run the 71 prefill host tests (five files under `models/demos/common/prefill/tests/` and `models/demos/deepseek_v3_d_p/tests/`).
7. [ ] Run the single-Galaxy 24-layer smoke on c04u08 inside the held allocation, with `ulimit -St unlimited`
   and `LOGURU_LEVEL=INFO` in RUNNER_ENV. If it hangs after fabric init, run `tt-triage`.

## Do not

- Write anything under `/data/rmiller`, or touch the `tt-metal-wt*` trees and branch `rmiller/k31-sparse-mla-trace`.
- Submit to `bh_saad_k3` without checking `squeue` (it can preempt tier-1 jobs). Never cancel or requeue other people's jobs.
  Use `PARTITIONS=bh_sc36_2,bh_sc36_6` for new jobs.
- Put large files in `~` on the galaxies (`/` is 96% full, 143 GB free). Use `/data/bklockiewicz`.

## Open

- Holds now: c04u08 (191886) and c05u08 (192244). c05u02 was lost to the idle cancel and is taken by jyuan 192178.
- Home dir is not shared between nodes: ~/tt-blaze exists on c04u08 only. /data is shared.
- Two-SC4 P→D run needs two free pods in `bh_saad_k3`. Unconfirmed whether c04u08 + c05u08 count.
- GitHub push works from the galaxies via the forwarded ssh agent. After reconnecting ssh, re-export `SSH_AUTH_SOCK` in the tmux shell.
- Keepalive: c04u08 loop PID 86371 (`kill <pid>` to stop); c05u08 loop runs inside job 192244 (ends with the job).
