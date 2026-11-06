#!/bin/bash
# Mock vet of the flushed-atomic yamls (CPU only): every test's last config (benchmark off) on TREE's build (default the
# 3efd636e8df worktree), P150 mock for P150 yamls (4x2: the study MGD), single-Galaxy mock for glx yamls. Pass: "Launching
# programs" with no fatal error before it. ENV: extra environment (e.g. TT_METAL_DISABLE_FABRIC_TWO_ERISC=1).
set -uo pipefail
T=${TREE:-/home/aagarwal/tt-metal-gate-3efd}; B=$T/build_Release
D=/home/aagarwal/tt-metal/fabric_exp/pr_perf/flushda
V=/tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad/flushda_mock
P150=/home/aagarwal/tt-metal/fabric_exp/a09u32_cluster_desc.yaml
GLX=$T/tt_metal/third_party/tt-cluster-descriptors/blackhole/single_bh_galaxy_clus_desc/single_bh_galaxy_clus_desc.yaml
MGD42=/home/aagarwal/tt-metal/fabric_exp/torus/p150_x8_4x2_torus_xy_mgd.textproto
rm -rf $V; mkdir -p $V
fail=0
for y in ${*:-$D/flushda_val_ring.yaml $D/flushda_val_torus.yaml $D/flushda_bench_ring.yaml $D/flushda_bench_torus.yaml $D/flushda_bench_torus4x2.yaml $D/flushda_val_glx_ring.yaml $D/flushda_val_glx_torus.yaml $D/flushda_bench_glx.yaml}; do
    b=$(basename $y .yaml); desc=$P150; extra=""
    case $b in *glx*) desc=$GLX ;; esac
    case $b in *4x2*) extra="TT_MESH_GRAPH_DESC_PATH=$MGD42" ;; esac
    n=$(python3 -c "import yaml,sys; print(len(yaml.safe_load(open(sys.argv[1]))['Tests']))" $y)
    for i in $(seq 0 $((n - 1))); do
        python3 - $y $i $V/${b}_$i.yaml <<'PY'
import sys, yaml
d = yaml.safe_load(open(sys.argv[1])); t = dict(d["Tests"][int(sys.argv[2])]); t["benchmark_mode"] = False
pp = t.get("parametrization_params", {})
for k in ("size", "ftype"):
    if isinstance(pp.get(k), list):
        pp[k] = pp[k][-1:]
if isinstance(pp.get("ntype"), list):
    pp["ntype"] = pp["ntype"][-1:]
d["Tests"] = [t]; yaml.safe_dump(d, open(sys.argv[3], "w"), sort_keys=False)
PY
        (cd $T && timeout 600 env -u TT_METAL_SLOW_DISPATCH_MODE TT_METAL_HOME=$T TT_METAL_RUNTIME_ROOT=$T TT_METAL_CACHE=$V/cache \
            TT_METAL_MOCK_CLUSTER_DESC_PATH=$desc TT_FABRIC_NO_BW_TELEMETRY=1 $extra ${ENV:-} \
            LD_LIBRARY_PATH=$B/tt_metal:$B/tt_stl:$B/lib:$B/tt_metal/third_party/umd/lib \
            $B/test/tt_metal/tt_fabric/test_infra/test_tt_fabric --test_config $V/${b}_$i.yaml > $V/${b}_$i.log 2>&1)
        L=$(sed 's/\x1b\[[0-9;]*m//g' $V/${b}_$i.log)
        ok=$(echo "$L" | grep -ac "Launching programs")
        # any kernel build error anywhere (test kernels may compile after the launch line), and no fatal error before it
        fatal=$(( $(echo "$L" | grep -avE "compile cmd|link cmd" | awk '/Launching programs/{exit} {print}' | grep -acE "TT_FATAL|TT_THROW|error:|Segmentation") + $(echo "$L" | grep -avE "compile cmd|link cmd" | grep -acE "build failed|compile failure| error: ") ))
        r=ok; { [ "$ok" -ge 1 ] && [ "$fatal" = 0 ]; } || { r=FAIL; fail=1; }
        echo "$b test $i: $r ($(echo "$L" | grep -aoE 'Running Test: \S+' | head -1 | cut -c15-))"
        [ $r = ok ] || echo "$L" | grep -avE "compile cmd|link cmd" | grep -aE "TT_FATAL|TT_THROW|error" | head -3 | cut -c1-250 | sed 's/^/   | /'
        rm -rf $V/cache
    done
done
exit $fail
