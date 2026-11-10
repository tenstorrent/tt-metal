#!/bin/bash
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -t 3:00:00
# k31_sc1_job.sh <commit>: Kimi-K3.1 prefill smoke, run_multirank_pcc.sh kimi_k3_1 sc1 on one galaxy, from the
# /data/$USER/k31-migration tree (built at <commit>, marker jobs/.built.k31-migration). Copy of sc4/sc1_job.sh.
# PREFILL_TTNN_CACHE: the symlink overlay of Momcilo's Weka K3.1 cache, so a missing tensor is written under /data.
set -u
ulimit -Su "$(ulimit -Hu)"; ulimit -St unlimited  # a forked rank inherits the job soft CPU cap (86400 s): SIGXCPU, rc 152, in weight load (job 189440)
COMMIT=$1 MODEL=kimi_k3_1
JOBS=/data/$USER/jobs S=/data/$USER/jobs/k31 REPO=/data/$USER/k31-migration
JOB=${SLURM_JOB_ID:?}
HOSTS=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | paste -sd,)
export HOSTS_SP=${HOSTS//,/ } REPO
echo "[k31-sc1] job $JOB partition ${SLURM_JOB_PARTITION:-?} hosts=$HOSTS commit=$COMMIT $(date -u +%FT%TZ)"
cd $REPO || exit 1
head=$(git rev-parse HEAD); want=$(git rev-parse "$COMMIT^{commit}" 2>/dev/null)
[ "$head" = "$want" ] || { echo "[k31-sc1] ABORT: $REPO is at $head, not $COMMIT"; exit 5; }
[ "$(cat $JOBS/.built.k31-migration 2>/dev/null)" = "$want" ] || [ "$(cat $JOBS/.built.k31-migration 2>/dev/null)" = "$COMMIT" ] || { echo "[k31-sc1] ABORT: build marker is $(cat $JOBS/.built.k31-migration 2>/dev/null)"; exit 5; }
every_host() { srun --overlap -N1 --ntasks-per-node=1 --label "$@"; }
every_host bash $S/hostprep.sh prep || { echo "[k31-sc1] ABORT: host prep failed"; exit 3; }
every_host bash $S/hostprep.sh trust || { echo "[k31-sc1] ABORT: ssh trust failed"; exit 6; }
for v in $(compgen -e | grep -E "^(SLURM|SLURMD|PMI|PMIX)_"); do unset "$v"; done
export PRTE_MCA_plm=^slurm
export TT_METAL_HOME=$REPO PYTHONPATH=$REPO
source $REPO/python_env/bin/activate
RUN=$S/run/$JOB; mkdir -p $RUN/ttop $RUN/prefill_summaries
printf '%s\n' $HOSTS_SP > $RUN/ttop/hostfile
export TTRUN_DIR=$RUN/ttop PREFILL_SUMMARIES=$RUN/prefill_summaries PREFILL_KEEP_RUN_DIR=1 PREFILL_TCP_INTERFACE=ens5f0np0
export PREFILL_TTNN_CACHE=${PREFILL_TTNN_CACHE:-/data/$USER/k31cache_overlay}
t0=$(date +%s)
timeout -k 120 165m bash models/demos/common/prefill/runners/ci/run_multirank_pcc.sh "$MODEL" sc1
rc=$?
echo "[k31-sc1] run_multirank_pcc rc=$rc wall=$(( $(date +%s) - t0 ))s"
for h in $HOSTS_SP; do
  ssh -n -o BatchMode=yes -o ConnectTimeout=10 $h "pkill -u $(id -u) -f '[p]refill_runner|[p]refill_producer|[t]trun.py|[p]rted'; sleep 3; \
    pkill -9 -u $(id -u) -f '[p]refill_runner|[p]refill_producer|[t]trun.py|[p]rted'; \
    find /dev/shm -maxdepth 1 -user $(id -u) \( -name '*prefill*' -o -name 'TT_UMD_LOCK*' \) -delete 2>/dev/null; true"
done
echo "[k31-sc1] done rc=$rc $(date -u +%FT%TZ)"
exit $rc
