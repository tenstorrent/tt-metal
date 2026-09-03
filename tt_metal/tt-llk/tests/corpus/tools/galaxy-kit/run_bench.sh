#!/usr/bin/env bash
# galaxy-kit run_bench.sh — launch the parallel benchmark on the held galaxy.
#
#   run_bench.sh -j <jobid> -c <chips> [-n <node>] [-r <reps>] [-k <copies>] \
#                [-o <ops>] [-d <dest>] [--batch [N]] \
#                [--pilot <op[__leg]> [--chip N]] \
#                [--seed-only | --no-seed] [--status]
#
#   -j  Slurm job id of the OWNER'S HOLD (srun --overlap only; this kit
#       never scancels or releases anything)
#   -c  chips: "all" (0-31) | a count N (0..N-1) | a comma list
#   -n  node name (adds -w <node> to srun; usually unneeded — the hold
#       pins the node)
#   -r  perf reps per cell (default 5)
#   -k  chip-copies per row (default 4): each row is measured on K
#       DISTINCT chips (work-stealing; same-chip sem/hand pairs always)
#   -o  comma list of ops to seed (default: every staged row)
#   --pilot  run ONE row on ONE chip first (red/green: reproduce a known
#            board cell before trusting anything)
#   --batch  OPT-IN session batching (default OFF = the proven solo grain):
#            each worker claims up to N ops (default 8) and amortizes pytest
#            startup — one corr session per batch, one session per rep index
#            over all gated perf nodes, per-node demux via the kit's pytest
#            plugin, per-batch solo audit, solo fallback for anything the
#            batch cannot prove.  ~6.5x wall-time per op measured (see
#            README "Batched sessions").
set -uo pipefail
KIT=$(cd "$(dirname "$0")" && pwd)
source "$KIT/lib/remote.sh"

JOBID=""; CHIPS=""; NODE=""; REPS=5; COPIES=4; OPS=""; PILOT=""; PCHIP=0
SEED=1; SEED_ONLY=0; STATUS=0; BATCH=""
while [ $# -gt 0 ]; do
  case "$1" in
    --batch)
      if [ $# -gt 1 ] && [[ "$2" =~ ^[0-9]+$ ]]; then BATCH=$2; shift 2
      else BATCH=8; shift; fi;;
    -j) JOBID=$2; shift 2;;
    -c) CHIPS=$2; shift 2;;
    -n) NODE=$2; shift 2;;
    -r) REPS=$2; shift 2;;
    -k) COPIES=$2; shift 2;;
    -o) OPS=$2; shift 2;;
    -d) LK_DEST=$2; shift 2;;
    --pilot) PILOT=$2; shift 2;;
    --chip) PCHIP=$2; shift 2;;
    --seed-only) SEED_ONLY=1; shift;;
    --no-seed) SEED=0; shift;;
    --status) STATUS=1; shift;;
    *) echo "unknown arg $1"; exit 2;;
  esac
done
route_check || exit 2
SRUN="srun --overlap --jobid ${JOBID:?-j <jobid> required}${NODE:+ -w $NODE}"

if [ "$STATUS" = 1 ]; then
  exa "echo queue: \$(ls $LK_DEST/queue 2>/dev/null | wc -l) items, claimed \$(ls $LK_DEST/claims 2>/dev/null | wc -l), done \$(ls -d $LK_DEST/claims/*/done.txt 2>/dev/null | wc -l); for f in $LK_DEST/wlogs/chip*.log; do [ -f \"\$f\" ] && echo \"\$(basename \$f): \$(tail -1 \$f)\"; done 2>/dev/null | head -40"
  exit 0
fi

if [ -n "$PILOT" ]; then
  case "$PILOT" in *__*) :;; *) PILOT="${PILOT}__plain";; esac
  echo "PILOT: $PILOT on chip $PCHIP (jobid $JOBID)"
  exa "cd $LK_DEST && mkdir -p queue claims && $SRUN bash -c 'ulimit -u \$(ulimit -Hu); hostname; LK_BASE=$LK_DEST LK_REPS=$REPS $LK_DEST/venv/bin/python $LK_DEST/worker.py --chip $PCHIP --item $PILOT'"
  exit $?
fi

if [ "$SEED" = 1 ]; then
  exa "cd $LK_DEST && LK_BASE=$LK_DEST ./venv/bin/python seed.py --copies $COPIES --special-copies $COPIES ${OPS:+--ops \"$OPS\"}"
fi
[ "$SEED_ONLY" = 1 ] && exit 0

: "${CHIPS:?-c <chips> required}"
echo "launching workers: chips=$CHIPS reps=$REPS copies=$COPIES jobid=$JOBID${BATCH:+ batch=$BATCH}"
exa "LK_BASE=$LK_DEST LK_REPS=$REPS ${BATCH:+LK_BATCH=1 LK_BATCH_OPS=$BATCH }$SRUN bash $LK_DEST/galaxy_launch.sh $CHIPS"
