#!/bin/bash
# Fabric router study on one Blackhole Galaxy node (RESULTS §98, Galaxy comparison): main's router against the lead
# (v3q), on the same node, telemetry off, one link per test.
#
#   sbatch fabric_exp/galaxy/frs_glx.sbatch          (from the study tree; uses the trees under $G/trees)
#   SUITES="sweep_mesh sweep_linear" CONFIGS="v3q main v3qfw v3qr" VALIDATE=0 sbatch fabric_exp/galaxy/frs_glx.sbatch
#
# SUITES (default "mesh linear ring") name fabric_exp/galaxy/glx_<suite>.yaml; CONFIGS (default "main v3q mainka") run
# in that order within each suite:
#   main    tree "base", router v1 at flags 0 (main's router), firmware keepalive (8000)
#   mainka  tree "base", router v1 at flags 0 with the study's keepalive override (1000): the keepalive's share alone
#   v3q     tree "v3q", the lead in v2's place, keepalive from the host setting (1000 on Blackhole)
#   v3qfw   v3q with the firmware keepalive (override 0: the router leaves the firmware's value)
#   v3qr    v3q again (same kernel cache): a repeat, for run-to-run spread
# With VALIDATE=1 (default), first a data-checked validation of both routers (mesh sanity). Per-job copies of the trees
# (hard links, no extra space) keep this job's generated/ CSVs its own; each config has its own kernel cache.
#SBATCH --partition=bh_sc36_6
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=02:30:00
#SBATCH --job-name=frs-glx
#SBATCH --output=/data/aagarwal/frs-glx/logs/%x-%j.log
#SBATCH --chdir=/data/aagarwal/frs-glx

set -uo pipefail
G=${G:-/data/aagarwal/frs-glx}
J=$G/jobs/${SLURM_JOB_ID:-manual}
mkdir -p $J/results
echo "node=$(hostname) job=${SLURM_JOB_ID:-} start=$(date -u +%FT%TZ)"
for t in base v3q; do cp -al $G/trees/$t $J/$t || { echo "!!! copy of $t failed"; exit 1; }; done
echo "trees copied $(date -u +%T)"

run() {  # label cache tree kernel_version flags yaml [ENV=VAL...]
    local label=$1 cache=$J/cache_$2 tree=$J/$3 kv=$4 flags=$5 yaml=$6; shift 6
    local lib=$tree/build_Release
    rm -f $tree/generated/fabric/bandwidth_summary_results_blackhole.csv $tree/generated/fabric/bandwidth_results_blackhole.csv
    echo "=== $label ($(date -u +%T)): tree ${tree##*/}, kernel v$kv, flags $flags, $(basename $yaml) $*"
    (cd $tree && env -u TT_METAL_SLOW_DISPATCH_MODE TT_METAL_HOME=$tree \
        LD_LIBRARY_PATH=$lib/tt_metal:$lib/tt_stl:$lib/lib:$lib/tt_metal/third_party/umd/lib \
        TT_METAL_CACHE=$cache TT_METAL_CLEAR_L1=1 TT_METAL_DEVICE_PROFILER=0 \
        TT_FABRIC_NO_BW_TELEMETRY=1 TT_FABRIC_KERNEL_VERSION=$kv TT_FABRIC_EXP_FLAGS=$flags "$@" \
        timeout 1200 ./build/test/tt_metal/tt_fabric/test_infra/test_tt_fabric --test_config $yaml \
        > $J/results/$label.log 2>&1)
    local rc=$?
    for f in bandwidth_summary_results_blackhole:"" bandwidth_results_blackhole:_tests; do
        [ -f $tree/generated/fabric/${f%%:*}.csv ] && cp $tree/generated/fabric/${f%%:*}.csv $J/results/$label${f##*:}.csv
    done
    echo "$label: exit $rc; $(sed 's/\x1b\[[0-9;]*m//g' $J/results/$label.log | grep -aoE 'Tests: [0-9]+ executed, [0-9]+ failed' | tail -1)"
}

config() {  # config suite yaml
    local c=$1 suite=$2 yaml=$3
    case $c in
        main) run ${c}_$suite main base 1 0 $yaml ;;
        mainka) run ${c}_$suite mainka base 1 0 $yaml TT_FABRIC_EXP_SEQ_UPDATE_TIMEOUT=1000 ;;
        v3q | v3qr) run ${c}_$suite v3q v3q 2 13129088 $yaml ;;
        v3qfw) run ${c}_$suite v3qfw v3q 2 13129088 $yaml TT_FABRIC_EXP_SEQ_UPDATE_TIMEOUT=0 ;;
        *) echo "!!! unknown config $c" ;;
    esac
}

S=$J/base/tests/tt_metal/tt_fabric/test_infra/test_yamls
Y=$J/base/fabric_exp/galaxy
if [ "${VALIDATE:-1}" = 1 ]; then
    run main_val main base 1 0 $S/test_fabric_mesh_sanity_common.yaml
    run v3q_val v3q v3q 2 13129088 $S/test_fabric_mesh_sanity_common.yaml
fi
for suite in ${SUITES:-mesh linear ring}; do
    for c in ${CONFIGS:-main v3q mainka}; do
        config $c $suite $Y/glx_$suite.yaml
    done
done
echo "done $(date -u +%T)"
