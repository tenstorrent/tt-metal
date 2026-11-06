#!/bin/bash
# Report refresh 2 at 400G (coordinator 2026-11-06): fused write + atomic on the 1D line, ring, XY torus (2x4) and
# neighbor exchange rows (refresh/yamls/rf2_fused_p150.yaml, benchmark mode), main against the port in one job, two
# passes in reversed router order, stock test infra, telemetry off. Runs on a09u32 through the tt-device broker, from
# the build worktrees themselves:
#   main  /home/aagarwal/tt-metal-rf2-main  b078540a262 (the PR's base) + the TT_FABRIC_NO_BW_TELEMETRY switch
#   port  /home/aagarwal/tt-metal-rf2-port  8047cfa60ca (the PR tip) + the same switch
# Pass a: a data-checked validation run per router (port_slots/yamls/pval2d_4352.yaml), then fused main, port; pass b:
# fused port, main. Each run: rf_lib.sh's rf_run (fast dispatch, TT_METAL_CLEAR_L1=1, profiler off, telemetry off,
# compile commands logged), a fresh kernel cache deleted once the run's results are copied. Labels rf2<a|b>_<router>_<row>.
# A hang (killed, captured into results/<label>_hangcap) stops the job; a failed run is reported and the job goes on.
cd /home/aagarwal/tt-metal || exit 1
out=fabric_exp/results
CON=fabric_exp/consolidation
source fabric_exp/report/refresh/rf_lib.sh || { echo "!!! rf_lib.sh"; exit 1; }
declare -A TREE=([main]=/home/aagarwal/tt-metal-rf2-main [port]=/home/aagarwal/tt-metal-rf2-port)
declare -A WANT=([main]=b078540a262 [port]=8047cfa60ca)
Y=fabric_exp/report/refresh/yamls/rf2_fused_p150.yaml
V=fabric_exp/port_slots/yamls/pval2d_4352.yaml
for r in main port; do
    [ "$(git -C ${TREE[$r]} rev-parse --short=11 HEAD)" = "${WANT[$r]}" ] || { echo "!!! $r tree is not at ${WANT[$r]}"; exit 1; }
    [ ${TREE[$r]}/build_Release/test/tt_metal/tt_fabric/test_infra/test_tt_fabric -nt ${TREE[$r]}/tests/tt_metal/tt_fabric/test_infra/test_tt_fabric.cpp ] ||
        { echo "!!! $r: test binary older than its source"; exit 1; }
    echo "$r: ${WANT[$r]} + $(git -C ${TREE[$r]} diff --stat | tail -1)"
done
one() {  # pass router row yaml timeout
    local lab=rf2$1_$2_$3 c=/home/aagarwal/.tt-metal-cache-rf2-$1-$2-$3 rc
    rm -rf "${c:?}"
    rf_run $lab ${TREE[$2]} $c $4 $5; rc=$?
    awk -F, -v l=$lab 'NR>1 {printf "   %s: %s %s B %.1f cycles/packet\n", l, $1, $7, 1.35e9/$10}' $out/$lab.csv 2>/dev/null
    rm -rf "${c:?}"
    [ $rc = 2 ] && { echo "!!! $lab hung (captured): stopping"; exit 1; }
    [ $rc = 0 ] || echo "!!! $lab failed (rc $rc): going on"
    return 0
}
echo "##### pass a (job at ${SECONDS} s)"
one a main val2d $V 300; one a port val2d $V 300
one a main fused $Y 600; one a port fused $Y 600
echo "##### pass b (job at ${SECONDS} s)"
one b port fused $Y 600; one b main fused $Y 600
echo "ALL_DONE (job at ${SECONDS} s)"
