#!/bin/bash
# Mock vet of the PR perf refresh's new yamls (CPU only, no device): each yaml's tests, one config each, benchmark off,
# on the mock P150 (fabric_exp/a09u32_cluster_desc.yaml), on both trees; each must reach "Launching programs".
#   [DESC=cluster_desc.yaml] [EXTRA_ENV="VAR=value ..."] pr_mock_vet.sh [YAML...]   (default: the new yamls on the mock
#   P150; DESC: another mock, e.g. a Galaxy; EXTRA_ENV: e.g. a mesh graph descriptor)
set -uo pipefail
cd /home/aagarwal/tt-metal
V=${V:-/tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad/prvet}
DESC=${DESC:-/home/aagarwal/tt-metal/fabric_exp/a09u32_cluster_desc.yaml}
Y=${*:-fabric_exp/pr_perf/pr_2d_4096.yaml fabric_exp/pr_perf/pr_1d_4096.yaml fabric_exp/pr_perf/pr_ring_4096.yaml fabric_exp/pr_perf/pr_ring_se.yaml fabric_exp/pr_perf/pr_mux.yaml fabric_exp/pr_perf/pr_size_sweep.yaml fabric_exp/v4/exp_torus_y.yaml}
mkdir -p $V
fail=0
for y in $Y; do
    n=$(python3 -c "import yaml,sys; print(len(yaml.safe_load(open(sys.argv[1]))['Tests']))" $y)
    for i in $(seq 0 $((n - 1))); do
        b=$(basename $y .yaml)_t$i
        python3 - $y $i $V/$b.yaml <<'PY'
import sys, yaml
d = yaml.safe_load(open(sys.argv[1])); t = dict(d["Tests"][int(sys.argv[2])]); t["benchmark_mode"] = False
pp = t.get("parametrization_params", {})
for k in list(pp):
    if isinstance(pp[k], list) and k in ("size", "ntype", "num_links"):
        pp[k] = pp[k][-1:]  # one config: the last (largest size / second ntype / most links)
d["Tests"] = [t]; yaml.safe_dump(d, open(sys.argv[3], "w"), sort_keys=False)
PY
        for tr in main:/home/aagarwal/tt-metal-main port:/home/aagarwal/tt-metal-port-pr; do
            nm=${tr%%:*} T=${tr#*:} B=${tr#*:}/build_Release
            (cd $T && timeout 240 env -u TT_METAL_SLOW_DISPATCH_MODE TT_METAL_HOME=$T TT_METAL_RUNTIME_ROOT=$T TT_METAL_CACHE=$V/cache_$nm \
                TT_METAL_MOCK_CLUSTER_DESC_PATH=$DESC TT_FABRIC_NO_BW_TELEMETRY=1 ${EXTRA_ENV:-} \
                LD_LIBRARY_PATH=$B/tt_metal:$B/tt_stl:$B/lib:$B/tt_metal/third_party/umd/lib \
                $B/test/tt_metal/tt_fabric/test_infra/test_tt_fabric --test_config $V/$b.yaml > $V/${b}_$nm.log 2>&1)
            if grep -aq "Launching programs" $V/${b}_$nm.log; then r=ok; else r=FAIL; fail=1; fi
            echo "$b $nm: $r $(sed 's/\x1b\[[0-9;]*m//g' $V/${b}_$nm.log | grep -aoE 'Running Test: [^ ]+' | head -1)"
            [ $r = ok ] || sed 's/\x1b\[[0-9;]*m//g' $V/${b}_$nm.log | grep -aE "TT_FATAL|TT_THROW|what\(\)|Segmentation" | head -2 | sed 's/^/   | /'
        done
    done
done
exit $fail
