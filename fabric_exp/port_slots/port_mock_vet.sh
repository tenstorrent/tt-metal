#!/bin/bash
# Mock vet of the port slot measurements (SLOT_ALLOCATOR.md section 12.12; CPU only, no device): each arm's suite's
# first test on the mock P150 (fabric_exp/a09u32_cluster_desc.yaml) with the port host (PORT_TREE, default the
# worktree /home/aagarwal/tt-metal-port-slots: 22cdb1e55c3 + slot_override.patch) and the arm's env; it must reach
# program launch with the row's depths logged. The port's own settings (as its run library): BW telemetry off,
# profiler off, no study kernel or flag switches; TT_FABRIC_EXP_LOG_SLOTS=1 for the patch's slot log.
#   [PORT_TREE=...] [V=<out dir>] [JOBS=4] port_mock_vet.sh SET...   (sets of port_arms.sh)
set -uo pipefail
cd /home/aagarwal/tt-metal
PT=${PORT_TREE:-/home/aagarwal/tt-metal-port-slots}
V=${V:-/tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad/portvet}
D=fabric_exp/port_slots
source $D/port_arms.sh
mkdir -p $V/yamls
one() {  # set arm kind suite row [extra env]
    local set=$1 arm=$2 kind=$3 suite=$4 row=$5 ex=${6:--} exv=() sw=() want="" w f r out lib n
    [ "$ex" = - ] || IFS=';' read -ra exv <<< "$ex"
    python3 - $D/yamls/$suite.yaml $V/yamls/$set-$arm.yaml <<'PY'
import sys, yaml
d = yaml.safe_load(open(sys.argv[1])); t = dict(d["Tests"][0]); t["benchmark_mode"] = False  # mock clocks are 0 MHz
d["Tests"] = [t]; yaml.safe_dump(d, open(sys.argv[2], "w"), sort_keys=False)
PY
    if [ "$row" != - ]; then
        IFS=, read w f r <<< "$row"
        case $kind in 2d) sw=(TT_FABRIC_EXP_SLOTS_2D=$row); want="VC0 sender slots {$w, $f, $f, $f} receiver slots {$r}" ;;
                      1d) sw=(TT_FABRIC_EXP_SLOTS_1D=$row); want="VC0 sender slots {$w, $f} receiver slots {$r}" ;; esac
    fi
    out=$V/$set-$arm; rm -rf "${out:?}"; mkdir -p $out; lib=$PT/build_Release
    (cd $PT && env -u TT_METAL_SLOW_DISPATCH_MODE TT_METAL_HOME=$PT TT_METAL_RUNTIME_ROOT=$PT \
        LD_LIBRARY_PATH=$lib/tt_metal:$lib/tt_stl:$lib/lib:$lib/tt_metal/third_party/umd/lib \
        TT_METAL_CACHE=$out/cache TT_METAL_LOGS_PATH=$out/logs \
        TT_METAL_MOCK_CLUSTER_DESC_PATH=/home/aagarwal/tt-metal/fabric_exp/a09u32_cluster_desc.yaml \
        TT_FABRIC_NO_BW_TELEMETRY=1 TT_METAL_DEVICE_PROFILER=0 TT_FABRIC_EXP_LOG_SLOTS=1 TT_METAL_LOG_KERNELS_COMPILE_COMMANDS=1 \
        "${sw[@]}" "${exv[@]}" timeout 1200 nice -n 10 $lib/test/tt_metal/tt_fabric/test_infra/test_tt_fabric \
        --test_config $V/yamls/$set-$arm.yaml > $out/run.log 2>&1)
    sed 's/\x1b\[[0-9;]*m//g' $out/run.log > $out/run.clean
    local st=ok
    grep -aq "Launching programs" $out/run.clean || st="NO LAUNCH ($(grep -aE 'TT_FATAL|TT_THROW' $out/run.clean | head -1 | cut -c1-160))"
    [ -z "$want" ] || grep -aqF "$want" $out/run.clean || st="${st/ok/}MISSING '$want'"
    n=$(grep -aoE 'VC0 sender slots \{[^}]*\} receiver slots \{[^}]*\} \(channel buffer [0-9]+ B\)' $out/run.clean | sort | uniq -c | tr -s ' \n' ' ')
    printf '%-4s %-12s %-3s %-12s %-10s %s | %s\n' $set $arm $kind $suite $row "$st" "$n"
}
export -f one; export V D PT
sets=" $* "
port_arms | while read -r s a k su ro ex; do [[ $sets == *" $s "* ]] && echo "$s $a $k $su $ro ${ex:--}"; done |
    xargs -P ${JOBS:-4} -L 1 bash -c 'one "$@"' _ | tee $V/summary.txt
bad=$(grep -vc "  *ok |" $V/summary.txt)
echo "$(wc -l < $V/summary.txt) arms, $bad not ok"
[ "$bad" = 0 ] && echo PORT_VET_OK || { echo PORT_VET_MISMATCH; exit 1; }
