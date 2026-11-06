#!/bin/bash
# Mock vet of the refresh's yamls (CPU only, no device; as pr_perf/pr_mock_vet.sh): each test of each yaml, one config
# (the size SIZE_PICK, default the first and 4096 B when present), benchmark off, on the mock DESC, on each TREES entry
# (name:path); each must reach "Launching programs".
#   [DESC=...] [TREES="main:/path port:/path"] [EXTRA_ENV="VAR=value ..."] [BENCH=1] rf_mock_vet.sh TAG YAML...
#   BENCH=1 keeps benchmark mode (compile check of the benchmark senders).
set -uo pipefail
cd /home/aagarwal/tt-metal
TAG=$1; shift
V=/tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad/rfvet/$TAG
DESC=${DESC:-/home/aagarwal/tt-metal/fabric_exp/a09u32_cluster_desc.yaml}
TREES=${TREES:-main:/home/aagarwal/tt-metal-main port:/home/aagarwal/tt-metal-rf-port}
mkdir -p $V
fail=0
for y in "$@"; do
    n=$(python3 -c "import yaml,sys; print(len(yaml.safe_load(open(sys.argv[1]))['Tests']))" $y)
    for i in $(seq 0 $((n - 1))); do
        b=$(basename $y .yaml)_t$i
        python3 - $y $i $V/$b.yaml ${BENCH:-0} <<'PY'
import sys, yaml
d = yaml.safe_load(open(sys.argv[1])); t = dict(d["Tests"][int(sys.argv[2])])
if sys.argv[4] != "1":
    t["benchmark_mode"] = False
pp = t.get("parametrization_params", {})
if isinstance(pp.get("size"), list):
    s = pp["size"]
    pp["size"] = sorted({s[0]} | ({4096} & set(s)))[:1] if len(s) == 1 else [s[0]]
for k in ("ntype", "num_links"):
    if isinstance(pp.get(k), list):
        pp[k] = pp[k][-1:]
d["Tests"] = [t]; yaml.safe_dump(d, open(sys.argv[3], "w"), sort_keys=False)
PY
        for tr in $TREES; do
            nm=${tr%%:*} T=${tr#*:} B=${tr#*:}/build_Release
            (cd $T && timeout 300 env -u TT_METAL_SLOW_DISPATCH_MODE TT_METAL_HOME=$T TT_METAL_RUNTIME_ROOT=$T \
                TT_METAL_CACHE=$V/cache_$nm TT_METAL_MOCK_CLUSTER_DESC_PATH=$DESC TT_FABRIC_NO_BW_TELEMETRY=1 ${EXTRA_ENV:-} \
                LD_LIBRARY_PATH=$B/tt_metal:$B/tt_stl:$B/lib:$B/tt_metal/third_party/umd/lib \
                $B/test/tt_metal/tt_fabric/test_infra/test_tt_fabric --test_config $V/$b.yaml > $V/${b}_$nm.log 2>&1)
            if grep -aq "Launching programs" $V/${b}_$nm.log; then r=ok; else r=FAIL; fail=1; fi
            echo "$TAG $b $nm: $r $(sed 's/\x1b\[[0-9;]*m//g' $V/${b}_$nm.log | grep -aoE 'Running Test: [^ ]+' | head -1)"
            [ $r = ok ] || sed 's/\x1b\[[0-9;]*m//g' $V/${b}_$nm.log | grep -aE "TT_FATAL|TT_THROW|what\(\)|Segmentation|error:" | head -3 | sed 's/^/   | /'
        done
    done
done
rm -rf $V/cache_*
exit $fail
