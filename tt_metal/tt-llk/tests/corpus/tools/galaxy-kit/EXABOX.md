# EXABOX.md — running CRAQ-SFPI experiments on the Exabox galaxies

How this project uses the Exabox Blackhole galaxies (32 chips per node), and
why. Companion to README.md (the three-command runbook). The upstream
operations bible is ~/tt-matmul-codegen/EXABOX.md; this file records what is
specific to us.

## The route (quietbox cannot reach exabox directly)

quietbox has no DNS route to the cluster. The owner's MacBook relays:

    hop 1:  ssh mac-relay                     # alias in ~/.ssh/fleet-mac-relay.conf
    hop 2:  export SSH_AUTH_SOCK=$HOME/.ssh/qz-exabox-agent.sock
            ssh nkapre@slurm-login.exabox.tenstorrent.com

Everything through both hops runs BatchMode with a ConnectTimeout. Files move
two-stage (quietbox -> mac:/tmp -> node or /data); the Mac's disk is small, so
stage small archives and clean up. macOS has no `timeout` command.

KEEP THE MAC AWAKE: a sleeping laptop severs the only route mid-campaign
(node-side srun work survives; staging/collection stalls). `caffeinate -s` on
the Mac, or Energy Saver "prevent sleep when plugged in".

## Slurm hold vs direct ssh — use both, for different jobs

The compute nodes are ordinary Slurm scheduler nodes (not the k8s pool) and
ARE directly ssh-able via the login jump host:

    ssh -A -J slurm-login.exabox.tenstorrent.com nkapre@<node>

But ssh alone RESERVES NOTHING. An unheld node shows `idle` and the scheduler
(or the daily ci-provisioner, or an infra commandeering sweep) can take it
mid-run. The Slurm allocation is the reservation:

    salloc --no-shell --immediate=10 --job-name=craq_<purpose> -w <node> -p <partition>

Then run workers INSIDE the hold:

    srun --overlap --jobid=<jobid> ...

Why srun for the workers rather than raw ssh, given both work:
- every process is tracked under the job — one `scancel <jobid>` reaps the
  whole campaign cleanly on a shared machine;
- the footprint is visible: `squeue` shows who owns the node, instead of
  mystery processes on an "idle" host.

Why ssh still matters: srun steps inherit Slurm's soft nproc limit (512) —
the kit raises `ulimit -u` first — while ssh logins land in user.slice with
the full limit. The kit uses ssh for diagnostics/recovery and srun for the
tracked workers.

## Etiquette (hard rules)

- Never scancel or disturb a job you did not create. Label holds
  `craq_<purpose>` so they are identifiable in announcements.
- Grab only the nodes the task needs; release (`scancel <our jobid>`) at
  campaign end.
- Blackhole reset is `tt-smi -r` once per node up front — NEVER per-worker
  (shared reset domain; a per-worker reset cascades across all 32 chips).
- Watch for infra commandeering announcements; our holds are the answer to
  "are we on host X" — check `squeue -u nkapre` against the list.

## Measurement honesty on galaxies

Galaxy cycle counts are NOT comparable to the p150 canon board. Same-chip
sem-vs-expert ratios are the valid statistic; every A/B pair runs both arms
on the same chip in the same session, corr/golden gates first, multiple reps,
chip id recorded per cell. Replication data informs the paper and dashboard;
the p150 board remains the booking authority.

## The kit

stage.sh (compile on quietbox at a pinned toolchain, ship ELFs — the nodes
run execute-only), run_bench.sh (per-chip work-stealing workers under
srun --overlap; --pilot gate first), collect.sh (pull + emit
REPLICATION-LEDGER.tsv). See README.md for the exact commands and
prerequisites.

run_bench.sh --batch turns on OPT-IN pytest-session batching (default off):
one corr session per claimed batch of ops and one session per rep index over
all gated perf nodes, with per-node demux by the kit's pytest plugin and a
per-batch solo audit. Verified on the quietbox p150: verdict ledgers
byte-identical to the solo grain at ~6.5x less wall time per op — a full
146-row x 8-chip campaign's session count drops from ~14k to ~2k. Honesty
rules are unchanged (same-chip pairs, corr-gate-first, 5 reps, one-shot
upfront reset only); anything a batch cannot prove falls back to solo
sessions automatically.
