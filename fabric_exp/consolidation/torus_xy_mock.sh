#!/bin/bash
# XY-torus vetting (2026-11-03; second-LB work, LB2_HARNESS.md): CPU-only mock runs of the fabric_exp/torus yamls on a
# P150x8 cluster descriptor, main_fw and v4rxg3, before any device run. No device, nothing written into the live tree:
# the runs use a scratch runtime root MR built from the lb2 stage (unswapped sources; build_Release, python_env and
# runtime linked read-only from the live tree), one kernel cache per router and shape under MR.
#
#   MR=<scratch dir> torus_xy_mock.sh [SHAPE...]        SHAPE: 2x4 (auto-discovered descriptor) | 4x2 (study descriptor
#                                                       DESC_4x2, default fabric_exp/torus/p150_x8_4x2_torus_xy_mgd.textproto)
# Each test group of torusxy_<shape>_sizes.yaml and val_torus_xy_p150.yaml runs on its own (a mock run ends at the first
# test's execution), benchmark_mode dropped so the test goes through its own setup (a benchmark yaml stops at the mock's
# frequency check first). Per run it prints: exit, the first error line, "could not be realized" warnings, the derived
# axis topologies; per router and shape, ENABLE_DEADLOCK_AVOIDANCE per direction (MY_DIRECTION 0..3 = E W N S) from the
# generated router args. "past setup" = the test started ("Running Test:") and the first error is the mock's own end
# (a sender or receiver that cannot run, result validation), not a setup abort.
# CLUSTER_DESC (default: a07u28's host-check descriptor) picks the mock cluster. JIT builds: taskset 48-63, nice 10.
set -uo pipefail
LIVE=/home/aagarwal/tt-metal
ST=/data/aagarwal/lb2/stage
MR=${MR:?set MR to a scratch directory}
DESC=${CLUSTER_DESC:-/data/aagarwal/lb2/nodes/bh-lb-120-a07u28/cluster_desc.yaml}
DESC_4x2=${DESC_4x2:-fabric_exp/torus/p150_x8_4x2_torus_xy_mgd.textproto}
CPUS=${MOCK_CPUS:-48-63}
mkdir -p $MR
cur=$(readlink $ST/current)
if [ "$(cat $MR/.stage 2>/dev/null)" != "$cur" ]; then
    rm -rf "${MR:?}/base" "${MR:?}/v4rxg3"
    mkdir -p $MR/base
    zstd -q -dc $ST/$cur/src.tar.zst | tar -x -C $MR/base || exit 1
    for d in build_Release python_env runtime; do ln -s $LIVE/$d $MR/base/$d; done
    ln -s build_Release $MR/base/build
    mkdir -p $MR/base/generated/fabric
    cp -a $MR/base $MR/v4rxg3
    python3 -B - $MR/v4rxg3 <<'EOF' || exit 1
import os, shutil, sys
sys.path.insert(0, "/home/aagarwal/tt-metal/fabric_exp/consolidation")
import offline_build as ob
root = sys.argv[1]
n = 0
for src, dst in ob.resolve(os.path.join(root, "fabric_exp/variants/v4rxg3")):
    t = os.path.join(root, dst)
    os.makedirs(os.path.dirname(t), exist_ok=True)
    if os.path.lexists(t):
        os.remove(t)
    shutil.copy(src, t)
    n += 1
print(f"v4rxg3 overlay: {n} files")
EOF
    echo $cur > $MR/.stage
fi
echo "mock root $MR (stage $cur), descriptor $DESC"

split_tests() {  # yaml outdir: one yaml per test group, benchmark_mode dropped
    python3 - "$1" "$2" <<'EOF'
import os, sys, yaml
y = yaml.safe_load(open(sys.argv[1]))
for t in y["Tests"]:
    t.pop("benchmark_mode", None)
    one = {k: v for k, v in y.items() if k != "Tests"}
    one["Tests"] = [t]
    yaml.safe_dump(one, open(os.path.join(sys.argv[2], t["name"] + ".yaml"), "w"), sort_keys=False)
    print(t["name"])
EOF
}

for shape in ${*:-2x4 4x2}; do
    mg=""; [ $shape = 4x2 ] && mg="TT_MESH_GRAPH_DESC_PATH=$MR/base/$DESC_4x2"
    for r in main_fw v4rxg3; do
        case $r in
            main_fw) tree=$MR/base; renv="TT_FABRIC_KERNEL_VERSION=1 TT_FABRIC_EXP_FLAGS=0" ;;
            v4rxg3) tree=$MR/v4rxg3; renv="TT_FABRIC_KERNEL_VERSION=2 TT_FABRIC_EXP_FLAGS=13129088 TT_FABRIC_EXP_SEQ_UPDATE_TIMEOUT=1000" ;;
        esac
        [ $shape = 4x2 ] && mg="TT_MESH_GRAPH_DESC_PATH=$tree/$DESC_4x2"
        c=$MR/cache_${r}_$shape; o=$MR/out_${r}_$shape
        rm -rf "${c:?}" "${o:?}"; mkdir -p $c $o/yamls
        names=$( { split_tests $tree/fabric_exp/torus/torusxy_${shape}_sizes.yaml $o/yamls; split_tests $tree/fabric_exp/torus/val_torus_xy_p150.yaml $o/yamls; } )
        echo "=== $shape $r: $(echo $names | wc -w) test groups"
        for t in $names; do
            (cd $tree && env -u TT_METAL_SLOW_DISPATCH_MODE TT_METAL_HOME=$tree TT_METAL_RUNTIME_ROOT=$tree \
                TT_METAL_MOCK_CLUSTER_DESC_PATH=$DESC TT_METAL_CACHE=$c TT_METAL_CLEAR_L1=1 TT_METAL_DEVICE_PROFILER=0 \
                TT_FABRIC_NO_BW_TELEMETRY=1 TT_METAL_LOG_KERNELS_COMPILE_COMMANDS=1 TT_METAL_FABRIC_OPT_LEVEL=O3 $renv $mg \
                timeout 900 nice -n 10 taskset -c $CPUS $LIVE/build/test/tt_metal/tt_fabric/test_infra/test_tt_fabric \
                --test_config $o/yamls/$t.yaml > $o/$t.log 2>&1)
            rc=$?
            L=$(sed 's/\x1b\[[0-9;]*m//g' $o/$t.log)
            started=$(echo "$L" | grep -ac "Running Test: ")
            err=$(echo "$L" | grep -aE "TT_FATAL|TT_THROW|critical|error  " | head -1 | sed 's/^.*| //' | cut -c1-150)
            warn=$(echo "$L" | grep -ac "could not be realized")
            axes=$(echo "$L" | grep -aoE "axis topologies derived as Y=[A-Z]+ X=[A-Z]+" | sort -u | sed 's/axis topologies derived as //' | tr '\n' ' ')
            printf '   %-26s exit %3s, %d started, realized-warn %d, axes %s| %s\n' $t $rc $started $warn "${axes:--}" "${err:-}"
        done
        # deadlock avoidance per router direction, from every generated router arg map of this cache
        python3 - $c <<'EOF'
import collections, glob, re, sys
da = collections.defaultdict(collections.Counter)
for f in glob.glob(sys.argv[1] + "/*/kernels/*router*/*/named_ct_arg_map_generated.h"):
    s = open(f).read()
    d = re.search(r'\{"MY_DIRECTION",(\d+)\}', s)
    e = re.search(r'\{"ENABLE_DEADLOCK_AVOIDANCE",(\d+)\}', s)
    if d and e:
        da["EWNS"[int(d.group(1))]][e.group(1)] += 1
print("   router configs, ENABLE_DEADLOCK_AVOIDANCE by direction: " +
      (", ".join(f"{k}: {dict(v)}" for k, v in sorted(da.items())) or "no named router args found"))
EOF
    done
done
