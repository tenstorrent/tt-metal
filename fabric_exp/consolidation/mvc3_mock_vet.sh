#!/bin/bash
# mvc3_mock_vet.sh SET VARIANT|main YAML... [-- ENV=VAL ...]: vet the multi-VC rows (mvc3_yamls.py) of the VC1 split on a
# mock of this host (CPU only, no device), before any device run. bd_mock_vet.sh's method (one process per test,
# benchmark_mode off: the mock's 0 MHz clocks stop benchmark mode before the traffic setup; a mock run ends at its first
# program launch) with vc1_mock.sh's two-rank launch (ttrun, --mock-cluster-rank-binding). Nothing is written into the
# shared tree, and a broker job's swap cannot reach the build: the runtime root is coverage_txo.py's pinned root (tt_metal/
# fabric, tt_metal/hw and the fabric tests copied at the tree's base state, VARIANT's files over them; `main` = no
# variant, kernel v1 at flags 0), everything else linked. Rank bindings: the standard ones, or (yaml name *_lag0*) the
# TT_RUN_RANK-swapped ones, with the run's launch delay on, so the log shows which process waits.
# Prints, per test: processes that reached "Launching programs", the delay lines, the slot rows, the first fatal error.
set -uo pipefail
SET=$1; VAR=$2; shift 2
YAMLS=(); while [ $# -gt 0 ] && [ "$1" != "--" ]; do YAMLS+=("$(realpath "$1")"); shift; done
[ "${1:-}" = "--" ] && shift
TREE=/home/aagarwal/tt-metal
SCR=/tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad/mvc3_vet
V=$SCR/$SET; ROOT=$SCR/root_$SET
mkdir -p $V
rm -f $V/*.yaml $V/*.log
python3 - "$ROOT" "$VAR" <<'PY'
import os, sys
sys.path.insert(0, "/home/aagarwal/tt-metal/fabric_exp/consolidation")
os.environ.setdefault("TT_COV_DIR", "/home/aagarwal/.tt-metal-v3r-builds/cov_m")
import coverage_txo as ct
root, var = sys.argv[1:3]
if var == "main":
    var = os.path.join(os.path.dirname(root), "novariant")
    os.makedirs(var, exist_ok=True)  # an empty variant: the base state only
    ct.make_pinned_root(root, var)
    os.rmdir(var)
else:
    ct.make_pinned_root(root, "/home/aagarwal/tt-metal/fabric_exp/variants/" + var)
PY
[ $? = 0 ] || { echo "no pinned root"; exit 1; }
for b in std lag0; do
    src=$TREE/fabric_exp/vc1/p150x8_2x2_split_rank_bindings.yaml
    [ $b = lag0 ] && src=$TREE/fabric_exp/vc1/mvc3/p150x8_2x2_split_rank_bindings_lag0.yaml
    sed "s#\"fabric_exp/#\"$TREE/fabric_exp/#" $src > $V/rb_$b.yml
done
printf 'rank_to_cluster_mock_cluster_desc:\n  0: "%s"\n  1: "%s"\n' $TREE/fabric_exp/a09u32_cluster_desc.yaml \
    $TREE/fabric_exp/a09u32_cluster_desc.yaml > $V/mock_map.yml
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
KV="TT_FABRIC_KERNEL_VERSION=2 TT_FABRIC_EXP_FLAGS=13129088 TT_FABRIC_EXP_SEQ_UPDATE_TIMEOUT=1000 TT_METAL_FABRIC_OPT_LEVEL=O3"
[ "$VAR" = main ] && KV="TT_FABRIC_KERNEL_VERSION=1 TT_FABRIC_EXP_FLAGS=0"
one() {  # test-yaml
    local t=$1 n b=std; shift; n=$(basename $t .yaml)
    [[ $n == *_lag0* ]] && b=lag0
    (cd $ROOT && env -u TT_METAL_SLOW_DISPATCH_MODE TT_METAL_HOME=$ROOT TT_METAL_RUNTIME_ROOT=$ROOT \
        TT_METAL_CACHE=$V/cache TT_METAL_LOGS_PATH=$V/logs_$n TT_FABRIC_NO_BW_TELEMETRY=1 TT_FABRIC_EXP_LOG_SLOTS=1 \
        TT_FABRIC_TEST_DIAG_LAUNCH_DELAY_MS=200 $KV "$@" timeout 900 nice -n 10 \
        $TREE/python_env/bin/python ttnn/ttnn/distributed/ttrun.py --mock-cluster-rank-binding $V/mock_map.yml \
        --rank-binding $V/rb_$b.yml --mpi-args "--oversubscribe" \
        ./build/test/tt_metal/tt_fabric/test_infra/test_tt_fabric --test_config $t > $V/$n.log 2>&1)
}
first=1
for t in $V/*.yaml; do
    if [ $first = 1 ]; then one $t "$@"; first=0; continue; fi  # the first run fills the router cache
    one $t "$@" &
    i=$(( ${i:-0} + 1 )); [ $((i % ${P:-4})) = 0 ] && wait
done
wait
for t in $V/*.yaml; do
    n=$(basename $t .yaml); l=$V/$n.log
    k=$(sed 's/\x1b\[[0-9;]*m//g' $l | grep -ac "Launching programs")
    e=$(sed 's/\x1b\[[0-9;]*m//g' $l | grep -aE "TT_FATAL|TT_THROW|error:" | grep -v "g++" | head -2 | cut -c1-220)
    echo "$n: launch reached by $k process(es)${e:+ | $e}"
    sed 's/\x1b\[[0-9;]*m//g' $l | grep -a "diag: rank" | sed 's/^\(\[[0-9,]*\]\).*diag: /   \1 diag: /' | cut -c1-160
done
echo "slot rows logged (first test):"
sed 's/\x1b\[[0-9;]*m//g' $(ls $V/*.log | head -1) | grep -aoE "(VC[0-9]) sender slots \{[^}]*\} receiver slots \{[^}]*\}" | sort | uniq -c
