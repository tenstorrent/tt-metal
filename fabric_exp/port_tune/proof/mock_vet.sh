#!/bin/bash
# CPU-only mock vet of the proof-count rows (benchmark off, mock P150 a09u32) on a runtime root: every row must reach
# "Launching programs" with no kernel build failure. One JIT cache per root (V/<name>/cache) for the image compare.
#   mock_vet.sh ROOT NAME [V]
set -uo pipefail
H=$1; N=$2; V=${3:-/tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad/proofvet}
B=$H/build_Release; Y=${PROOF_YAMLS:-/home/aagarwal/tt-metal/fabric_exp/port_tune/proof/yamls}
mkdir -p $V/$N/yamls
for y in $Y/*.yaml; do
    r=$(basename $y .yaml)
    python3 - $y $V/$N/yamls/$r.yaml <<'PY'
import sys, yaml
d = yaml.safe_load(open(sys.argv[1]))
for t in d["Tests"]:
    t["benchmark_mode"] = False
    t.setdefault("defaults", {})
    if "repeat" in t["defaults"]: t["defaults"]["repeat"] = 100
    if "num_packets" in t["defaults"]: t["defaults"]["num_packets"] = 100
yaml.safe_dump(d, open(sys.argv[2], "w"), sort_keys=False)
PY
    out=$V/$N/$r; rm -rf $out; mkdir -p $out
    (cd $H && env -u TT_METAL_SLOW_DISPATCH_MODE TT_METAL_HOME=$H TT_METAL_RUNTIME_ROOT=$H \
        LD_LIBRARY_PATH=$B/tt_metal:$B/tt_stl:$B/lib:$B/tt_metal/third_party/umd/lib \
        TT_METAL_CACHE=$V/$N/cache TT_METAL_LOGS_PATH=$out/logs \
        TT_METAL_MOCK_CLUSTER_DESC_PATH=/home/aagarwal/tt-metal/fabric_exp/a09u32_cluster_desc.yaml \
        TT_FABRIC_NO_BW_TELEMETRY=1 TT_METAL_DEVICE_PROFILER=0 TT_METAL_LOG_KERNELS_COMPILE_COMMANDS=1 \
        timeout 1200 nice -n 10 $B/test/tt_metal/tt_fabric/test_infra/test_tt_fabric --test_config $V/$N/yamls/$r.yaml \
        > $out/run.log 2>&1) 2>/dev/null
    sed 's/\x1b\[[0-9;]*m//g' $out/run.log > $out/run.clean
    st=ok
    grep -aq "Launching programs" $out/run.clean || st="NO LAUNCH ($(grep -aE 'TT_FATAL|TT_THROW|error' $out/run.clean | head -1 | cut -c1-200))"
    if grep -aqE "build failed|Failed to generate binaries" $out/run.clean; then st="KERNEL BUILD FAILED ($(grep -aE 'error' $out/run.clean | head -1 | cut -c1-250))"; fi
    printf '%-10s %-26s %s\n' $N $r "$st"
done
