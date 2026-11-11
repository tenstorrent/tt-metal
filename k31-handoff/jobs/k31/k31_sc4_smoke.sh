#!/bin/bash
# k31_sc4_smoke.sh <host0,host1,host2,host3>: K3.1 prefill-only smoke on one quad (sc4, 93 layers, migration off),
# with a single-rank feed-only producer (PREFILL_K31_SC4_SMOKE=1 in run_multirank_pcc.sh). Run on host0, inside an
# allocation that holds it; the other hosts are reached over ssh by prterun (PRTE_MCA_plm=^slurm), so each must be
# held by us and trusted (hostprep.sh prep + trust). Tree must be at a commit whose build marker matches.
set -u
ulimit -Su "$(ulimit -Hu)"; ulimit -St unlimited  # forked ranks inherit the soft CPU cap otherwise (SIGXCPU, rc 152)
HOSTS=${1:?usage: k31_sc4_smoke.sh host0,host1,host2,host3}
REPO=/data/$USER/k31-migration S=/data/$USER/jobs/k31
TAG=sc4smoke-$(date -u +%m%d-%H%M)
RUN=$S/run/$TAG; mkdir -p $RUN/ttop $RUN/prefill_summaries
tr ',' '\n' <<<"$HOSTS" > $RUN/ttop/hostfile
echo "[k31-sc4] $TAG hosts=$HOSTS commit=$(git -c safe.directory='*' -C $REPO rev-parse --short HEAD) $(date -u +%FT%TZ)"
for v in $(compgen -e | grep -E "^(SLURM|SLURMD|PMI|PMIX)_"); do unset "$v"; done
export PRTE_MCA_plm=^slurm
cd $REPO || exit 1
export TT_METAL_HOME=$REPO PYTHONPATH=$REPO
source $REPO/python_env/bin/activate
export TTRUN_DIR=$RUN/ttop PREFILL_SUMMARIES=$RUN/prefill_summaries PREFILL_KEEP_RUN_DIR=1 PREFILL_TCP_INTERFACE=ens5f0np0
export PREFILL_TTNN_CACHE=${PREFILL_TTNN_CACHE:-/data/$USER/k31cache_overlay}
export PREFILL_K31_SC4_SMOKE=1
export PREFILL_NUM_USERS=${PREFILL_NUM_USERS:-4}
t0=$(date +%s)
timeout -k 120 165m bash models/demos/common/prefill/runners/ci/run_multirank_pcc.sh kimi_k3_1 sc4
rc=$?
echo "[k31-sc4] run_multirank_pcc rc=$rc wall=$(( $(date +%s) - t0 ))s"
for h in ${HOSTS//,/ }; do
  ssh -n -o BatchMode=yes -o ConnectTimeout=10 $h "pkill -u $(id -u) -f '[p]refill_runner|[p]refill_producer|[t]trun.py|[p]rted'; sleep 3; \
    pkill -9 -u $(id -u) -f '[p]refill_runner|[p]refill_producer|[t]trun.py|[p]rted'; \
    find /dev/shm -maxdepth 1 -user $(id -u) \( -name '*prefill*' -o -name 'TT_UMD_LOCK*' \) -delete 2>/dev/null; true"
done
echo "[k31-sc4] done rc=$rc $(date -u +%FT%TZ)"
exit $rc
