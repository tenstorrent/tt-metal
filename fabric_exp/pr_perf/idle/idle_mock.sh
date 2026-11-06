#!/bin/bash
# Mock vet of the Galaxy IDLE_PASSES sweep (CPU only): trees/tune_e7fe as frs_glx_idle.sbatch runs it (a job farm, the
# arm's env), on the single-Galaxy mock, one config per test of each suite yaml (benchmark off). Pass: "Launching
# programs", the router define logged, no fatal error before the launch. Caches deleted afterwards.
set -uo pipefail
G=/data/aagarwal/frs-glx; Y=$G/yamls/pr_e7fe; T=$G/trees/tune_e7fe; lib=$T/build_Release
V=/tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad/idle_mock
DESC=/home/aagarwal/tt-metal-main/tt_metal/third_party/tt-cluster-descriptors/blackhole/single_bh_galaxy_clus_desc/single_bh_galaxy_clus_desc.yaml
rm -rf $V; mkdir -p $V/farm/generated
for e in $T/* $T/.[!.]*; do b=$(basename $e); [ "$b" = generated ] || ln -s $e $V/farm/$b; done
for v in $(env | grep -oE '^TT_[A-Za-z0-9_]+'); do unset $v; done
fail=0
for s in check nx mesh linear ring; do
    case $s in check) y=$Y/glx_check_2d.yaml ;; nx) y=$Y/pr_glx_nxab.yaml ;; *) y=$Y/pr_glx_idle_$s.yaml ;; esac
    n=$(python3 -c "import yaml,sys; print(len(yaml.safe_load(open(sys.argv[1]))['Tests']))" $y)
    for i in $(seq 0 $((n - 1))); do
        python3 - $y $i $V/${s}_$i.yaml <<'PY'
import sys, yaml
d = yaml.safe_load(open(sys.argv[1])); t = dict(d["Tests"][int(sys.argv[2])]); t["benchmark_mode"] = False
pp = t.get("parametrization_params", {})
for k in ("size", "ntype", "num_links"):
    if isinstance(pp.get(k), list):
        pp[k] = pp[k][-1:]
d["Tests"] = [t]; yaml.safe_dump(d, open(sys.argv[3], "w"), sort_keys=False)
PY
        (cd $V/farm && timeout 600 env TT_METAL_HOME=$V/farm TT_METAL_RUNTIME_ROOT=$V/farm TT_METAL_CACHE=$V/cache \
            TT_METAL_MOCK_CLUSTER_DESC_PATH=$DESC TT_FABRIC_NO_BW_TELEMETRY=1 TT_FABRIC_EXP_IDLE_PASSES=64 \
            LD_LIBRARY_PATH=$lib/tt_metal:$lib/tt_stl:$lib/lib:$lib/tt_metal/third_party/umd/lib \
            ./build/test/tt_metal/tt_fabric/test_infra/test_tt_fabric --test_config $V/${s}_$i.yaml > $V/${s}_$i.log 2>&1)
        L=$(sed 's/\x1b\[[0-9;]*m//g' $V/${s}_$i.log)
        ok=$(echo "$L" | grep -ac "Launching programs"); def=$(echo "$L" | grep -ac "port_tune: router define FABRIC_EXP_IDLE_PASSES=64")
        # any kernel build error anywhere (test kernels may compile after the launch line), and no fatal error before it
        fatal=$(( $(echo "$L" | grep -avE "compile cmd|link cmd" | awk '/Launching programs/{exit} {print}' | grep -acE "TT_FATAL|TT_THROW|error:|Segmentation") + $(echo "$L" | grep -avE "compile cmd|link cmd" | grep -acE "build failed|compile failure| error: ") ))
        r=ok; { [ "$ok" -ge 1 ] && [ "$def" -ge 1 ] && [ "$fatal" = 0 ]; } || { r=FAIL; fail=1; }
        echo "$s test $i: $r ($(echo "$L" | grep -aoE 'Running Test: \S+' | head -1 | cut -c15-))"
        [ $r = ok ] || echo "$L" | grep -avE "compile cmd|link cmd" | grep -aE "TT_FATAL|TT_THROW|error" | head -3 | sed 's/^/   | /'
        rm -rf $V/cache
    done
done
exit $fail
