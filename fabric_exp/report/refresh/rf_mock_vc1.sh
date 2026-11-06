#!/bin/bash
# Two-rank mock vet of the multi-VC split rows (CPU only; as port_slots/alloc_mock_vet.sh's vc1 case): one mvc3 yaml's
# first test, benchmark off, through tt-run with both ranks on the mock DESC; must reach "Launching programs" on both
# ranks. Logs the router compile levels and the kernel config buffer headroom (TT_METAL_LOG_JIT_TELEMETRY=1).
#   rf_mock_vc1.sh TAG TREE YAML RANK_BINDINGS DESC [ENV=VALUE ...]
set -uo pipefail
TAG=$1 PT=$2 Y=$3 RB=$4 DESC=$5; shift 5
V=/tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad/rfvet/vc1_$TAG
PY=/home/aagarwal/tt-metal/python_env/bin/python
rm -rf $V; mkdir -p $V
python3 - $Y $V/t.yaml <<'PY'
import sys, yaml
d = yaml.safe_load(open(sys.argv[1])); t = dict(d["Tests"][0]); t["benchmark_mode"] = False
t["parametrization_params"]["size"] = t["parametrization_params"]["size"][-1:]
d["Tests"] = [t]; yaml.safe_dump(d, open(sys.argv[2], "w"), sort_keys=False)
PY
printf 'rank_to_cluster_mock_cluster_desc:\n  0: "%s"\n  1: "%s"\n' $DESC $DESC > $V/mock_mapping.yaml
lib=$PT/build_Release
(cd $PT && env -u TT_METAL_SLOW_DISPATCH_MODE TT_METAL_HOME=$PT TT_METAL_RUNTIME_ROOT=$PT \
    LD_LIBRARY_PATH=$lib/tt_metal:$lib/tt_stl:$lib/lib:$lib/tt_metal/third_party/umd/lib \
    TT_METAL_CACHE=$V/cache TT_METAL_LOGS_PATH=$V/logs TT_FABRIC_NO_BW_TELEMETRY=1 TT_METAL_DEVICE_PROFILER=0 \
    TT_METAL_LOG_KERNELS_COMPILE_COMMANDS=1 TT_METAL_LOG_JIT_TELEMETRY=1 "$@" \
    timeout 900 nice -n 10 $PY $PT/ttnn/ttnn/distributed/ttrun.py --mock-cluster-rank-binding $V/mock_mapping.yaml \
    --rank-binding $RB --mpi-args --oversubscribe \
    $lib/test/tt_metal/tt_fabric/test_infra/test_tt_fabric --test_config $V/t.yaml > $V/run.log 2>&1)
sed 's/\x1b\[[0-9;]*m//g' $V/run.log > $V/run.clean
n=$(grep -ac "Launching programs" $V/run.clean)
lv=$(grep -a 'g++ compile cmd' $V/run.clean | grep -a 'fabric_erisc_router' | grep -oE ' -O[0-9s]+ ' | sort | uniq -c | tr -s ' \n' ' ')
echo "$TAG: $n of 2 ranks launched; router -O levels: ${lv:-none logged}"
grep -a "config buffer, least headroom" $V/run.clean | sed 's/^.*summary | /   /' | sort -u | head -4
[ "$n" -ge 2 ] || grep -aE "TT_FATAL|TT_THROW|what\(\)|could not fit|error:" $V/run.clean | head -4 | sed 's/^/   | /'
rm -rf $V/cache
