#!/bin/bash
# Slack study offline check (NOTES.md; CPU only, no device): JIT-build the routers of every root of mk_roots.py on the mock
# P150 (fabric_exp/a09u32_cluster_desc.yaml), one process per slack_rows.yaml test (benchmark off: a mock run ends at the
# first program launch, after the routers are built), then img_check.py compares each patched root's router images
# with the plain root's (loadable bytes, functions with addresses masked, sizes).
#   offline_check.sh [ROUTER_ARM ...]   (default: every root under /home/aagarwal/tt-metal-slack-roots)
#   V (default the session scratchpad's slackvet/), P parallel runs (default 4).
set -uo pipefail
cd /home/aagarwal/tt-metal
S=fabric_exp/port_tune/slack
R=/home/aagarwal/tt-metal-slack-roots
V=${V:-/tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad/slackvet}
DESC=/home/aagarwal/tt-metal/fabric_exp/a09u32_cluster_desc.yaml
declare -A LIB=([main]=/home/aagarwal/tt-metal-main/build_Release [port]=/home/aagarwal/tt-metal-port-slack/build_Release)
roots=${*:-$(cd $R && ls -d main_* port_*)}
mkdir -p $V/yamls
python3 - $S/slack_rows.yaml $V/yamls <<'PY'
import sys, yaml
src, out = sys.argv[1:3]
d = yaml.safe_load(open(src))
for t in d["Tests"]:
    t = dict(t); t["benchmark_mode"] = False
    o = {k: v for k, v in d.items() if k != "Tests"}; o["Tests"] = [t]
    yaml.safe_dump(o, open(f"{out}/{t['name']}.yaml", "w"), sort_keys=False)
PY
one() {  # root test
    local r=$1 t=$2 lib=${LIB[${1%%_*}]} o=$V/$1/$2
    rm -rf "${o:?}"; mkdir -p $o
    (cd $R/$r && env -u TT_METAL_SLOW_DISPATCH_MODE TT_METAL_HOME=$R/$r TT_METAL_RUNTIME_ROOT=$R/$r \
        LD_LIBRARY_PATH=$lib/tt_metal:$lib/tt_stl:$lib/lib:$lib/tt_metal/third_party/umd/lib \
        TT_METAL_CACHE=$o/cache TT_METAL_LOGS_PATH=$o/logs TT_METAL_MOCK_CLUSTER_DESC_PATH=$DESC \
        TT_FABRIC_NO_BW_TELEMETRY=1 timeout 1200 nice -n 10 \
        $lib/test/tt_metal/tt_fabric/test_infra/test_tt_fabric --test_config $V/yamls/$t.yaml > $o/run.log 2>&1)
    local st=ok
    sed -i 's/\x1b\[[0-9;]*m//g' $o/run.log
    grep -aq "Launching programs" $o/run.log || st="NO LAUNCH ($(grep -aE 'TT_FATAL|TT_THROW|error' $o/run.log | head -1 | cut -c1-160))"
    grep -aqE "build failed|Failed to generate binaries" $o/run.log && st="KERNEL BUILD FAILED ($(grep -aE 'error:' $o/run.log | head -1 | cut -c1-160))"
    echo "$r $t: $st"
    # keep only what img_check.py reads (a full mock cache is ~0.7 GB; 2026-11-05 the unpruned set filled the disk)
    find $o/cache -type f ! \( -path "*/kernels/fabric_erisc_router/*" \( -name named_ct_arg_map_generated.h -o \
        \( -name "*.elf" ! -name "*.xip.elf" \) \) \) -delete 2>/dev/null
    find $o/cache -type d -empty -delete 2>/dev/null
}
i=0
{ for r in $roots; do
    for t in Mesh1HopEW NeighborExchangeUnicast; do
        one $r $t &
        i=$((i + 1)); [ $((i % ${P:-4})) = 0 ] && wait
    done
done; wait; } | tee $V/build_summary.txt
bad=$(grep -vc ": ok$" $V/build_summary.txt)
echo "builds: $(grep -c ': ok$' $V/build_summary.txt) ok, $bad not ok"
python3 $S/img_check.py $V | tee $V/img_check.txt
