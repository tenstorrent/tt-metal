#!/bin/bash
# Mock vet of the receiver tuning sweep's suites (CPU only, no device; RECEIVER_TUNING.md): each suite's first test,
# benchmark off (mock clocks are 0 MHz), on the mock P150 (fabric_exp/a09u32_cluster_desc.yaml; vc1 / vc1i: two ranks
# through tt-run, each rank's mock the same descriptor) with the port build (/home/aagarwal/tt-metal-port-tune) and an
# arm's env (ARM_ENV, default the i1 arm's). Each must reach "Launching programs" and log the arm's router defines.
#   [V=<out dir>] [ARM_ENV="..."] mock_vet.sh [SUITE...]   (default: every suite of yamls/manifest.json)
set -uo pipefail
cd /home/aagarwal/tt-metal
PT=${PORT_TREE:-/home/aagarwal/tt-metal-port-tune}   # PORT_TREE: another built tree (e.g. the allocator follow-up's)
D=fabric_exp/port_tune
V=${V:-/tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad/tunevet}
ARM_ENV=${ARM_ENV:-TT_FABRIC_EXP_IDLE_PASSES=1 TT_FABRIC_EXP_PROOF_CHECKS=32}
DESC=/home/aagarwal/tt-metal/fabric_exp/a09u32_cluster_desc.yaml
PY=/home/aagarwal/tt-metal/python_env/bin/python
RB=$PWD/$D/cov/vc1/p150x8_2x2_split_rank_bindings.yaml
mkdir -p $V/yamls
suites=${*:-$(python3 -c "import json; print(' '.join(json.load(open('$D/yamls/manifest.json'))))")}
lib=$PT/build_Release
for s in $suites; do
    y=$(python3 -c "import json,sys; print(json.load(open('$D/yamls/manifest.json'))[sys.argv[1]]['yaml'])" $s)
    python3 - $y $V/yamls/$s.yaml <<'PY'
import sys, yaml
d = yaml.safe_load(open(sys.argv[1])); t = dict(d["Tests"][0]); t["benchmark_mode"] = False
d["Tests"] = [t]; yaml.safe_dump(d, open(sys.argv[2], "w"), sort_keys=False)
PY
    out=$V/$s; rm -rf "${out:?}"; mkdir -p $out
    cmd=($lib/test/tt_metal/tt_fabric/test_infra/test_tt_fabric --test_config $V/yamls/$s.yaml)
    mock=(TT_METAL_MOCK_CLUSTER_DESC_PATH=$DESC)
    case $s in vc1|vc1i)
        printf 'rank_to_cluster_mock_cluster_desc:\n  0: "%s"\n  1: "%s"\n' $DESC $DESC > $out/mock_mapping.yaml
        cmd=($PY $PT/ttnn/ttnn/distributed/ttrun.py --mock-cluster-rank-binding $out/mock_mapping.yaml --rank-binding $RB
             --mpi-args --oversubscribe "${cmd[@]}")
        mock=() ;;
    esac
    (cd $PT && env -u TT_METAL_SLOW_DISPATCH_MODE TT_METAL_HOME=$PT TT_METAL_RUNTIME_ROOT=$PT \
        LD_LIBRARY_PATH=$lib/tt_metal:$lib/tt_stl:$lib/lib:$lib/tt_metal/third_party/umd/lib \
        TT_METAL_CACHE=$out/cache TT_METAL_LOGS_PATH=$out/logs "${mock[@]}" \
        TT_FABRIC_NO_BW_TELEMETRY=1 TT_METAL_DEVICE_PROFILER=0 $ARM_ENV timeout 1200 nice -n 10 "${cmd[@]}" \
        > $out/run.log 2>&1)
    sed 's/\x1b\[[0-9;]*m//g' $out/run.log > $out/run.clean
    st=ok
    grep -aq "Launching programs" $out/run.clean || st="NO LAUNCH ($(grep -aE 'TT_FATAL|TT_THROW|error' $out/run.clean | head -1 | cut -c1-200))"
    if grep -aqE "build failed|Failed to generate binaries" $out/run.clean; then st="KERNEL BUILD FAILED ($(grep -aE "error:" $out/run.clean | head -1 | cut -c1-200))"; fi
    for e in $ARM_ENV; do
        v=${e#TT_FABRIC_EXP_}; v="FABRIC_EXP_${v%%=*}=${e#*=}"
        grep -aqF "port_tune: router define $v" $out/run.clean || st="$st; define $v not logged"
    done
    printf '%-9s %s\n' $s "$st"
    rm -rf $out/cache
done | tee $V/summary.txt
bad=$(grep -vc " ok$" $V/summary.txt)
[ "$bad" = 0 ] && echo TUNE_VET_OK || { echo "TUNE_VET: $bad not ok"; exit 1; }
