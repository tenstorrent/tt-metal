#!/bin/bash
# One command on one cabled BH Galaxy quad (four Galaxies, one MPI rank each), or on one cabled pair (two Galaxies,
# glxq_pairs.tsv), through tt-run in its rank-binding mode. Submit it with glxq_run.sh, which stages the inputs and
# checks the rules; see MULTI_GALAXY_HARNESS.md.
#
# Inputs (environment, set by glxq_run.sh): STAGE (staged helpers + rank_binding.yaml, on /data), TREE (tree directory),
# FSD (Factory System Descriptor for the link preflight, or "none"), FSD_RUN (1: also pass it to tt-run), ROUTER (main | new | port), RUN_TIMEOUT (s), SMOKE (1: run the smoke row first),
# MPI_IF (default ens5f0np0, the Exabox MPI NIC), CMD_FILE (the command, one shell line, run in each rank's tree).
# Pair jobs also: PAIR (the pair's name in glxq_pairs.tsv), PAIR_GROUP and PAIR_LOCK (the sibling group: glxq_run.sh
# --pair submits one job per candidate pair; the first to start wins), SMOKE_YAML (the staged smoke row for the form).
#
# Steps (any failure stops the job; nothing here cancels a running job):
#   0. group     pair jobs: the first sibling to start takes the group's lock (mkdir) and cancels the siblings still
#                PENDING (scancel --state=PENDING: never a running job); a sibling that starts after it exits at once
#   1. guard     every node bh-glx-110-a*, the allocation is one quad of glxq_quads.tsv (hosts in ring order) or the
#                job's pair of glxq_pairs.tsv (hosts in rank order)
#   2. busy      other users' device processes on any node stop the job
#   3. tools     mpirun-ulfm, tt-smi, the tree, its python_env on every node
#   4. reset     tt-smi -glx_reset on every node in parallel; 32 Blackhole chips listed on each
#   5. links     read_eth_speed.py on every node; Ethernet channels UP and trained >= the FSD's count inside the quad
#   6. smoke     glxq_smoke.yaml (data-checked, all-to-all on the quad mesh) unless SMOKE=0
#   7. run       the command, RUN_TIMEOUT
#   On a timeout: this job's processes are killed on every node, the routers captured read-only (hang_capture.py per
#   node), every node reset, the job stops.
# Outputs under $G/jobs/<id>/: results/ (step logs, eth dumps, summary), ranks/ (per-rank logs and info), cache/r<k>
# (kernel caches, kept), the staged inputs (inputs/).
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --job-name=glxq
#SBATCH --output=/data/aagarwal/frs-glx/logs/%x-%j.log
#SBATCH --chdir=/data/aagarwal/frs-glx

set -uo pipefail
G=${G:-/data/aagarwal/frs-glx}
J=$G/jobs/${SLURM_JOB_ID:?not a Slurm job}
STAGE=${STAGE:?}; TREE=${TREE:?}; FSD=${FSD:-none}; FSD_RUN=${FSD_RUN:-0}; ROUTER=${ROUTER:-new}
PAIR=${PAIR:-}; SMOKE_YAML=${SMOKE_YAML:-glxq_smoke.yaml}
RUN_TIMEOUT=${RUN_TIMEOUT:-1800}; SMOKE=${SMOKE:-1}; MPI_IF=${MPI_IF:-ens5f0np0}; CMD_FILE=${CMD_FILE:-$STAGE/cmd.sh}
TSMI=${TSMI:-tt-smi}
say() { echo "[glxq $(date -u +%T)] $*"; }
die() { say "!!! $*"; exit 1; }

# 0. group (pair jobs) ------------------------------------------------------------------------------------------------
# Before anything else, so a sibling that lost exits within seconds and writes nothing but its Slurm log.
if [ -n "$PAIR" ]; then
    PAIR_GROUP=${PAIR_GROUP:?}; PAIR_LOCK=${PAIR_LOCK:?}
    if ! mkdir "$PAIR_LOCK/won" 2>/dev/null; then
        say "pair group $PAIR_GROUP already started (job $(cat "$PAIR_LOCK/won/job" 2>/dev/null || echo ?)): sibling $PAIR exits"
        exit 0
    fi
    echo "$SLURM_JOB_ID $PAIR $(date -u +%FT%TZ)" > "$PAIR_LOCK/won/job"
    if [ "${GLXQ_STUB:-0}" != 1 ]; then
        pending=$(squeue -u "$USER" -h -t PENDING -o "%i %j" | awk -v p="glxp-$PAIR_GROUP-" 'index($2, p) == 1 {print $1}')
        if [ -n "$pending" ]; then
            scancel --state=PENDING $pending
            say "pair group $PAIR_GROUP: job $SLURM_JOB_ID ($PAIR) won; cancelled pending siblings: $(echo $pending)"
        else
            say "pair group $PAIR_GROUP: job $SLURM_JOB_ID ($PAIR) won; no pending sibling"
        fi
        echo "$pending" > "$PAIR_LOCK/won/cancelled"
    fi
fi
mkdir -p $J/results $J/ranks $J/cache
cp -r $STAGE $J/inputs
on_all() { srun --overlap --nodes=$SLURM_JOB_NUM_NODES --ntasks-per-node=1 --label "$@"; }  # --overlap: usable after a hang
# Stub mode (GLXQ_STUB=1, local dry run only: no Slurm, no device): STUB_NODES stands for the allocation, each node
# step runs locally once per node (GLXQ_HOST names it), TTRUN_PY / READ_ETH replace tt-run and the ETH reader.
if [ "${GLXQ_STUB:-0}" = 1 ]; then
    on_all() { local k=0 n; for n in $STUB_NODES; do GLXQ_HOST=$n "$@" | sed "s/^/$k: /"; k=$((k + 1)); done; }
fi
export J TREE STAGE MPI_IF TSMI
node_step() { on_all bash $STAGE/glxq_node.sh "$@"; }  # glxq_node.sh on every node

# 1. guard ----------------------------------------------------------------------------------------------------------
if [ "${GLXQ_STUB:-0}" = 1 ]; then mapfile -t NODES < <(tr ' ' '\n' <<< "$STUB_NODES"); else
    mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST"); fi
say "job $SLURM_JOB_ID nodes ${NODES[*]} tree $TREE router $ROUTER fsd $FSD timeout ${RUN_TIMEOUT}s"
for n in "${NODES[@]}"; do case $n in
    bh-glx-110-a*) ;;
    bh-glx-120-c*) [ "${GLXQ_C120_EXCEPTION:-0}" = 1 ] || die "$n: bh-glx-120-c needs the user's one-time exception (GLXQ_C120_EXCEPTION=1)" ;;
    *) die "$n is not a bh-glx-110-a* node (study rule)" ;; esac; done
QUAD=""; RING=""
b=$(printf '%s\n' "${NODES[@]}" | sort | paste -sd,)
if [ -n "$PAIR" ]; then
    [ ${#NODES[@]} = 2 ] || die "${#NODES[@]} nodes, a pair has 2"
    while IFS=$'\t' read -r name slurm hosts rest; do
        [ "$name" = "$PAIR" ] || continue
        a=$(tr ',' '\n' <<< "$hosts" | sort | paste -sd,)
        [ "$a" = "$b" ] && { QUAD=$name; RING=$hosts; }
    done < $STAGE/glxq_pairs.tsv
    [ -n "$QUAD" ] || die "the allocation ${NODES[*]} is not pair $PAIR of glxq_pairs.tsv"
    say "pair $QUAD, rank order $RING"
    echo "pair=$QUAD hosts=$RING group=$PAIR_GROUP" > $J/results/quad.txt
else
    [ ${#NODES[@]} = 4 ] || die "${#NODES[@]} nodes, a quad has 4"
    while IFS=$'\t' read -r name slurm ring leaf; do
        [[ $name == \#* || -z $name ]] && continue
        a=$(tr ',' '\n' <<< "$ring" | sort | paste -sd,)
        [ "$a" = "$b" ] && { QUAD=$name; RING=$ring; }
    done < $STAGE/glxq_quads.tsv
    [ -n "$QUAD" ] || die "the allocation ${NODES[*]} is not one cabled quad of glxq_quads.tsv"
    say "quad $QUAD, ring order $RING"
    echo "quad=$QUAD ring=$RING" > $J/results/quad.txt
fi

# 2. busy -----------------------------------------------------------------------------------------------------------
busy=$(node_step busy)
if [ -n "$busy" ] && [ -z "${IGNORE_BUSY:-}" ]; then echo "$busy"; die "other users' processes on the quad (may hold the devices)"; fi

# 3. tools ----------------------------------------------------------------------------------------------------------
tools=$(node_step tools)
[ -z "$tools" ] && say "tools: mpirun-ulfm, tt-smi, tree, python_env, $MPI_IF, /data writable on every node" || { echo "$tools"; die "tools missing"; }

# 4. reset ----------------------------------------------------------------------------------------------------------
reset_all() {  # reason
    say "tt-smi -glx_reset on every node ($1)"
    node_step reset
}
r=$(reset_all "before the steps"); echo "$r"
[ $(grep -c "rc=0 chips=\(3[2-9]\|[4-9][0-9]\)" <<< "$r") = ${#NODES[@]} ] || die "reset failed or chips missing on a node"

# 5. links ----------------------------------------------------------------------------------------------------------
node_step eth
if [ "$FSD" != none ]; then
    $TREE/python_env/bin/python -I $STAGE/glxq_links.py check $FSD $RING $J/results || die "Ethernet links short of the FSD"
else
    say "links: no FSD, counts only: $(for n in "${NODES[@]}"; do echo -n "$n $(grep -c 'port_status=1 train_status=2' $J/results/eth_$n.txt) "; done)"
fi

# environment for the ranks (tt-run passes TT_* to every rank) -------------------------------------------------------
env | grep -E '^TT_' | sort > $J/results/submit_env.txt
for v in $(env | grep -oE '^TT_[A-Za-z0-9_]+'); do unset $v; done
unset PYTHONPATH VIRTUAL_ENV
lib=$TREE/build_Release
export TT_METAL_HOME=$TREE TT_METAL_RUNTIME_ROOT=$TREE TT_GLXQ_JOB=$J TT_GLXQ_TREE=$TREE
export LD_LIBRARY_PATH=$lib/tt_metal:$lib/tt_stl:$lib/lib:$lib/tt_metal/third_party/umd/lib
export TT_FABRIC_NO_BW_TELEMETRY=1 TT_METAL_FABRIC_OPT_LEVEL=O3 TT_METAL_CLEAR_L1=1 TT_METAL_DEVICE_PROFILER=0
export TT_METAL_FABRIC_ROUTER_SYNC_TIMEOUT_MS=1000000 TT_METAL_LOG_KERNELS_COMPILE_COMMANDS=1
case $ROUTER in
    main) export TT_FABRIC_KERNEL_VERSION=1 TT_FABRIC_EXP_FLAGS=0 ;;
    new) export TT_FABRIC_KERNEL_VERSION=2 TT_FABRIC_EXP_FLAGS=13129088 TT_FABRIC_EXP_SEQ_UPDATE_TIMEOUT=1000 ;;
    port) ;;  # a mainline-port tree: its router is the only one, no selection
    *) die "ROUTER $ROUTER: main | new | port" ;;
esac
# FSD_RUN=1 also hands the FSD to tt-run. With an FSD the control plane requires RELAXED system health for a RELAXED
# mesh graph (the quad MGDs); RELIABILITY_MODE (runtime option, passed by tt-run) asks for it, but test_tt_fabric's first
# control plane is built STRICT before its yaml is read (MULTI_GALAXY_HARNESS.md §5.1): off by default.
FSD_ARGS=()
if [ "$FSD_RUN" = 1 ] && [ "$FSD" != none ]; then FSD_ARGS=(--factory-system-descriptor $FSD); export RELIABILITY_MODE=${RELIABILITY_MODE:-relaxed}; fi
tar czf $J/results/src_$ROUTER.tgz -C $TREE tt_metal/fabric/hw/inc tt_metal/fabric/impl/kernels/edm_fabric tt_metal/hw/inc \
    tt_metal/hostdevcommon 2>/dev/null

kill_mine() { node_step kill > /dev/null 2>&1; sleep 3; }
capture() {  # label
    say "hang capture, every node (read-only)"
    node_step capture $1
}
ttrun_rc() {  # label timeout CMD...: one tt-run launch, ranks in ring (rank) order; returns tt-run's exit code.
    # RB: the rank binding (default the job's); a timeout (124, a hang) is captured and every node reset here.
    local label=$1 to=$2; shift 2
    export TT_GLXQ_LABEL=$label  # tt-run passes TT_* to the ranks: glxq_rank.sh names each rank's log by step
    say "=== $label: $* (timeout ${to}s)"
    (cd $TREE && timeout $to python_env/bin/python ${TTRUN_PY:-$TREE/ttnn/ttnn/distributed/ttrun.py} \
        --rank-binding ${RB:-$STAGE/rank_binding.yaml} --tcp-interface $MPI_IF "${FSD_ARGS[@]}" \
        --mpi-args "--host $RING" \
        $STAGE/glxq_rank.sh "$@") > $J/results/$label.log 2>&1
    local rc=$?
    grep -a "\[glxq\]" $J/results/$label.log | sed 's/^/   /'
    say "$label: exit $rc"
    if [ $rc = 124 ]; then
        say "!!! $label timed out (hang?): killing this job's processes, capturing, resetting"
        kill_mine; capture $label; reset_all "after a timeout in $label"
    fi
    return $rc
}
ttrun() {  # as ttrun_rc, but any failure stops the job
    ttrun_rc "$@"
    local rc=$?
    [ $rc = 124 ] && exit 1
    [ $rc = 0 ] || { reset_all "after a failure in $1"; exit 1; }
}

# 6. smoke + 7. run -------------------------------------------------------------------------------------------------
[ "$SMOKE" = 1 ] && ttrun smoke 900 ./build/test/tt_metal/tt_fabric/test_infra/test_tt_fabric --test_config $STAGE/$SMOKE_YAML
if [ -n "${SUITE:-}" ]; then
    # A suite (glxq_run.sh --suite): one tt-run per line of $STAGE/suite.tsv, in order: config, label, form (its rank
    # binding $STAGE/rank_binding_<form>.yaml), timeout (s), env (K=V,K=V or -), yaml (staged). A hang (timeout) is
    # captured, every node reset, and that config's remaining lines are skipped; a failure resets every node and the
    # suite goes on. Results: results/suite_summary.tsv (one line per step). GLXQ_TREE=<name> in a line's env runs that
    # line on the tree <name> next to $TREE (an A/B in one allocation), with its rank logs and kernel caches under
    # $J/tree_<name>/ (ranks/, cache/r<k>); its hang capture uses that tree's python_env.
    hung=" "
    printf 'config\tlabel\tform\tenv\tresult\tseconds\ttests\n' > $J/results/suite_summary.tsv
    while IFS=$'\t' read -r -u 3 cfg label form to envs yaml; do  # fd 3: tt-run and srun read stdin
        [[ -z $cfg || $cfg == \#* || $cfg == config ]] && continue
        if [[ $hung == *" $cfg "* ]]; then
            say "--- $label skipped: config $cfg hung earlier"
            printf '%s\t%s\t%s\t%s\tSKIPPED_AFTER_HANG\t0\t-\n' "$cfg" "$label" "$form" "$envs" >> $J/results/suite_summary.tsv
            continue
        fi
        t0=$(date +%s)
        (
            [ "$envs" = - ] || for kv in ${envs//,/ }; do export "$kv"; done
            if [ -n "${GLXQ_TREE:-}" ]; then
                TREE=$(dirname $TREE)/$GLXQ_TREE; lib=$TREE/build_Release
                export TREE TT_METAL_HOME=$TREE TT_METAL_RUNTIME_ROOT=$TREE TT_GLXQ_TREE=$TREE TT_GLXQ_JOB=$J/tree_$GLXQ_TREE
                export LD_LIBRARY_PATH=$lib/tt_metal:$lib/tt_stl:$lib/lib:$lib/tt_metal/third_party/umd/lib
                mkdir -p $TT_GLXQ_JOB/ranks $TT_GLXQ_JOB/cache
                say "--- $label on tree $TREE (rank logs and caches in $TT_GLXQ_JOB)"
            fi
            RB=$STAGE/rank_binding_$form.yaml ttrun_rc $label $to ./build/test/tt_metal/tt_fabric/test_infra/test_tt_fabric \
                --test_config $yaml --show-progress
        ) < /dev/null
        rc=$?
        tests=$(grep -aoE "Tests: [0-9]+ executed, [0-9]+ failed" $J/results/$label.log | sort -u | tr '\n' ';')
        case $rc in
            0) res=PASS ;;
            124) res=HANG; hung+="$cfg " ;;
            *) res="FAIL(rc=$rc)"; reset_all "after a failure in $label" > /dev/null ;;
        esac
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$cfg" "$label" "$form" "$envs" "$res" $(($(date +%s) - t0)) "${tests:--}" \
            >> $J/results/suite_summary.tsv
    done 3< $STAGE/suite.tsv
    say "suite:"; column -t -s $'\t' $J/results/suite_summary.tsv | sed 's/^/   /'
else
    ttrun run $RUN_TIMEOUT bash $CMD_FILE
fi
for k in $(seq 0 $((${#NODES[@]} - 1))); do
    for f in $J/ranks/r$k/tree/generated/fabric/*.csv; do [ -f "$f" ] && cp "$f" $J/results/r${k}_$(basename $f); done
done
say "done; results in $J/results, rank logs in $J/ranks"
