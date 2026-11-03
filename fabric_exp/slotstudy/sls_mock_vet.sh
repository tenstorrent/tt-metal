#!/bin/bash
# Slot study: vet a yaml on the mock Galaxy (CPU only, no device) with a /data Galaxy tree's host and router (the VC2 line
# agent's mock_vet.sh, adapted). A mock run ends at the first program launch, so each test group gets its own run.
#   sls_mock_vet.sh single|split|split4x4|split4x2 YAML [ENV=VAL ...]     (TREE=/data/aagarwal/frs-glx/trees/sls_lead)
# single: the whole Galaxy as one mesh; split*: ttrun.py with every rank mocked on the Galaxy's cluster descriptor and
# the rank bindings of fabric_exp/slotstudy (glx_2x2_split_*, glx_4x4_split_*, glx_4x2_split_*). Output and caches in
# MOCK_DESC=<cluster descriptor> mocks another system (e.g. fabric_exp/a09u32_cluster_desc.yaml, single mode).
# $SCR (default the session scratchpad's sls_vet/<yaml>); P parallel runs (default 3), nice 10. Tests run with
# benchmark_mode false (the mock's 0 MHz clocks stop benchmark mode before the traffic setup); same traffic.
set -uo pipefail
MODE=$1; Y=$(realpath $2); shift 2
TREE=${TREE:-/data/aagarwal/frs-glx/trees/sls_lead}
SCR=${SCR:-/tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad/sls_vet}
GLX=${MOCK_DESC:-$TREE/tt_metal/third_party/tt-cluster-descriptors/blackhole/single_bh_galaxy_clus_desc/single_bh_galaxy_clus_desc.yaml}
V=$SCR/$MODE-$(basename $Y .yaml)
lib=$TREE/build_Release
rm -rf $V; mkdir -p $V
python3 - "$Y" "$V" <<'PY'
import sys, yaml
y, v = sys.argv[1:3]
d = yaml.safe_load(open(y))
for t in d["Tests"]:
    out = {k: val for k, val in d.items() if k != "Tests"}
    t = dict(t)
    t["benchmark_mode"] = False  # mock devices report 0 MHz: benchmark mode stops at the frequency check before traffic setup
    out["Tests"] = [t]
    yaml.safe_dump(out, open(f"{v}/{t['name']}.yaml", "w"), sort_keys=False)
PY
case $MODE in
    split) RB=fabric_exp/slotstudy/glx_2x2_split_rank_bindings.yaml; NR=2 ;;
    split4x4) RB=fabric_exp/slotstudy/glx_4x4_split_rank_bindings.yaml; NR=2 ;;
    split4x2) RB=fabric_exp/slotstudy/glx_4x2_split_rank_bindings.yaml; NR=4 ;;
esac
if [ $MODE != single ]; then
    { echo "rank_to_cluster_mock_cluster_desc:"; for r in $(seq 0 $((NR - 1))); do echo "  $r: \"$GLX\""; done; } > $V/mock_mapping.yaml
fi
one() {  # test-yaml
    local t=$1 n; shift; n=$(basename $t .yaml)
    local launch=($lib/test/tt_metal/tt_fabric/test_infra/test_tt_fabric --test_config $t)
    local mock=(TT_METAL_MOCK_CLUSTER_DESC_PATH=$GLX)
    if [ $MODE != single ]; then
        launch=($TREE/python_env/bin/python $TREE/ttnn/ttnn/distributed/ttrun.py --mock-cluster-rank-binding $V/mock_mapping.yaml
            --rank-binding $RB --mpi-args "--oversubscribe --mca btl self,sm" "${launch[@]}")
        mock=()
    fi
    (cd $TREE && env -u TT_METAL_SLOW_DISPATCH_MODE TT_METAL_HOME=$TREE TT_METAL_RUNTIME_ROOT=$TREE \
        LD_LIBRARY_PATH=$lib/tt_metal:$lib/tt_stl:$lib/lib:$lib/tt_metal/third_party/umd/lib \
        TT_METAL_CACHE=$V/cache_$n TT_METAL_LOGS_PATH=$V/logs_$n "${mock[@]}" \
        TT_FABRIC_KERNEL_VERSION=2 TT_FABRIC_EXP_FLAGS=13129088 TT_FABRIC_NO_BW_TELEMETRY=1 TT_FABRIC_EXP_LOG_SLOTS=1 \
        TT_FABRIC_EXP_SEQ_UPDATE_TIMEOUT=1000 TT_METAL_FABRIC_OPT_LEVEL=O3 "$@" timeout 1200 nice -n 10 \
        "${launch[@]}" > $V/$n.log 2>&1)
}
i=0
for t in $V/*.yaml; do
    [ "$(basename $t)" = mock_mapping.yaml ] && continue
    one $t "$@" &
    i=$((i + 1)); [ $((i % ${P:-3})) = 0 ] && wait
done
wait
for t in $V/*.yaml; do
    n=$(basename $t .yaml); [ $n = mock_mapping ] && continue; l=$V/$n.log
    k=$(sed 's/\x1b\[[0-9;]*m//g' $l | grep -ac "Launching programs")
    e=$(sed 's/\x1b\[[0-9;]*m//g' $l | grep -aE "TT_FATAL|TT_THROW|error:" | grep -v "g++" | head -2 | cut -c1-220)
    s=$(sed 's/\x1b\[[0-9;]*m//g' $l | grep -aoE 'VC[01] sender slots \{[^}]*\} receiver slots \{[^}]*\}' | sort | uniq -c | sort -rn | head -2 | tr -s ' \n' ' ')
    echo "$n: launch reached by $k process(es) ${e:+| $e} | slots: $s"
done
rm -rf $TREE/generated
