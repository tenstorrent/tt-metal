#!/bin/bash
# Vet a test yaml on the mock Galaxy (CPU only, no device): one run per test (a mock run ends at program launch, so
# each test gets its own run), with this worktree's host and router. Checks parsing, host validation and kernel
# compiles. Usage: mock_vet.sh YAML [extra ENV=VAL ...]
set -uo pipefail
Y=$(realpath $1); shift
WT=/home/aagarwal/tt-metal-vc2line; C=/home/aagarwal/.tt-metal-vc2line-cov; V=$C/vet/$(basename $Y .yaml)
GLX=$WT/tt_metal/third_party/tt-cluster-descriptors/blackhole/single_bh_galaxy_clus_desc/single_bh_galaxy_clus_desc.yaml
rm -rf $V; mkdir -p $V
python3 - "$Y" "$V" <<'PY'
import sys, yaml
y, v = sys.argv[1:3]
d = yaml.safe_load(open(y))
for t in d["Tests"]:
    out = {k: val for k, val in d.items() if k != "Tests"}
    out["Tests"] = [t]
    yaml.safe_dump(out, open(f"{v}/{t['name']}.yaml", "w"), sort_keys=False)
PY
for t in $V/*.yaml; do
    n=$(basename $t .yaml)
    (cd $V && env -u TT_METAL_SLOW_DISPATCH_MODE TT_METAL_RUNTIME_ROOT=$C/root TT_METAL_HOME=$C/root \
        TT_METAL_CACHE=$V/cache_$n TT_METAL_LOGS_PATH=$V/logs_$n TT_METAL_MOCK_CLUSTER_DESC_PATH=$GLX \
        TT_FABRIC_KERNEL_VERSION=2 TT_FABRIC_EXP_FLAGS=13129088 TT_FABRIC_NO_BW_TELEMETRY=1 \
        TT_FABRIC_EXP_SEQ_UPDATE_TIMEOUT=1000 "$@" timeout 1200 nice -n 10 \
        $WT/build_Release/test/tt_metal/tt_fabric/test_infra/test_tt_fabric --test_config $t > $V/$n.log 2>&1) &
done
wait
for t in $V/*.yaml; do
    n=$(basename $t .yaml); l=$V/$n.log
    s=$(sed 's/\x1b\[[0-9;]*m//g' $l | grep -a "Launching programs" | head -1 >/dev/null && echo "reached launch" || echo "DID NOT REACH LAUNCH")
    e=$(sed 's/\x1b\[[0-9;]*m//g' $l | grep -aE "TT_FATAL|TT_THROW|error:" | grep -v "g++" | head -2 | cut -c1-200)
    echo "$n: $s ${e:+| $e}"
done
