#!/bin/bash
# Cycle breakdown (fabric_exp/report/BREAKDOWN_PLAN.md): vet row yamls on a mock cluster (CPU only, no device) before any
# device run, as memory/tt-device-broker-quirks asks for new ring and torus yamls. Same method as
# slotstudy/sls_mock_vet.sh (one process per test, benchmark_mode off: the mock's 0 MHz clocks stop benchmark mode before
# the traffic setup; a mock run ends at the first program launch), but nothing is written into the shared tree: the run's
# TT_METAL_HOME is a symlink farm of the tree in $SCR with its own generated/ directory.
#   bd_mock_vet.sh p150|galaxy YAML... [-- ENV=VAL ...]
#   TREE (default /data/aagarwal/frs-glx/trees/sls_lead: post-merge host, the study router), SCR (default the session
#   scratchpad's bd_vet/), P parallel runs (default 3). Kernel v2 at the study flags unless ENV overrides it
#   (TT_FABRIC_KERNEL_VERSION=1 TT_FABRIC_EXP_FLAGS=0 vets main's router).
# Prints, per test: whether every process reached "Launching programs", the first fatal error.
set -uo pipefail
SYS=$1; shift
YAMLS=(); while [ $# -gt 0 ] && [ "$1" != "--" ]; do YAMLS+=("$(realpath "$1")"); shift; done
[ "${1:-}" = "--" ] && shift
TREE=${TREE:-/data/aagarwal/frs-glx/trees/sls_lead}
SCR=${SCR:-/tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad/bd_vet}
case $SYS in
    p150) DESC=/home/aagarwal/tt-metal/fabric_exp/a09u32_cluster_desc.yaml ;;
    galaxy) DESC=$TREE/tt_metal/third_party/tt-cluster-descriptors/blackhole/single_bh_galaxy_clus_desc/single_bh_galaxy_clus_desc.yaml ;;
    *) echo "usage: bd_mock_vet.sh p150|galaxy YAML... [-- ENV=VAL ...]"; exit 1 ;;
esac
[ -f "$DESC" ] || { echo "no cluster descriptor $DESC"; exit 1; }
FARM=$SCR/farm
if [ ! -d $FARM ]; then
    mkdir -p $FARM
    for e in $TREE/*; do [ "$(basename $e)" = generated ] || ln -s $e $FARM/$(basename $e); done
fi
lib=$TREE/build_Release
V=$SCR/$SYS
mkdir -p $V
for y in "${YAMLS[@]}"; do
    python3 - "$y" "$V" <<'PY'
import os, sys, yaml
y, v = sys.argv[1:3]
d = yaml.safe_load(open(y))
for t in d["Tests"]:
    out = {k: val for k, val in d.items() if k != "Tests"}
    t = dict(t)
    t["benchmark_mode"] = False
    out["Tests"] = [t]
    yaml.safe_dump(out, open(f"{v}/{os.path.basename(y)[:-5]}__{t['name']}.yaml", "w"), sort_keys=False)
PY
done
one() {  # test-yaml
    local t=$1 n; shift; n=$(basename $t .yaml)
    (cd $FARM && env -u TT_METAL_SLOW_DISPATCH_MODE TT_METAL_HOME=$FARM TT_METAL_RUNTIME_ROOT=$FARM \
        LD_LIBRARY_PATH=$lib/tt_metal:$lib/tt_stl:$lib/lib:$lib/tt_metal/third_party/umd/lib \
        TT_METAL_CACHE=$V/cache_$n TT_METAL_LOGS_PATH=$V/logs_$n TT_METAL_MOCK_CLUSTER_DESC_PATH=$DESC \
        TT_FABRIC_KERNEL_VERSION=2 TT_FABRIC_EXP_FLAGS=13129088 TT_FABRIC_NO_BW_TELEMETRY=1 \
        TT_FABRIC_EXP_SEQ_UPDATE_TIMEOUT=1000 TT_METAL_FABRIC_OPT_LEVEL=O3 "$@" timeout 1200 nice -n 10 \
        $lib/test/tt_metal/tt_fabric/test_infra/test_tt_fabric --test_config $t > $V/$n.log 2>&1)
}
i=0
for t in $V/*.yaml; do
    one $t "$@" &
    i=$((i + 1)); [ $((i % ${P:-3})) = 0 ] && wait
done
wait
for t in $V/*.yaml; do
    n=$(basename $t .yaml); l=$V/$n.log
    k=$(sed 's/\x1b\[[0-9;]*m//g' $l | grep -ac "Launching programs")
    e=$(sed 's/\x1b\[[0-9;]*m//g' $l | grep -aE "TT_FATAL|TT_THROW|error:" | grep -v "g++" | head -2 | cut -c1-220)
    echo "$n: launch reached by $k process(es)${e:+ | $e}"
done
