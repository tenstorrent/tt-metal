#!/bin/bash
# CPU-only mock vet of the study's test-infra changes on the port (test_infra_6dad.patch, README.md): each case is one
# test of a yaml, benchmark off (mock clocks are 0 MHz), on the mock P150 2x4 (fabric_exp/a09u32_cluster_desc.yaml).
#   base     6dad5fec638 as built: /home/aagarwal/tt-metal-port-gap's host binary, runtime root
#            /home/aagarwal/tt-metal-port-gap-roots/asbuilt (tests/ = git HEAD).
#   patched  /home/aagarwal/tt-metal-port-testinfra (6dad5fec638 + the patch), its own host build.
# A case passes when it reaches "Launching programs" with no kernel build failure. Main's yamls run on both arms and
# their dumped built tests (--dump-built-tests) are compared; the stream yaml runs on the patched arm only.
#   [V=<out dir>] vet_6dad.sh
set -uo pipefail
V=${V:-/tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad/testinfra/vet}
DESC=/home/aagarwal/tt-metal/fabric_exp/a09u32_cluster_desc.yaml
TY=tests/tt_metal/tt_fabric/test_infra/test_yamls
declare -A HOME_OF=([base]=/home/aagarwal/tt-metal-port-gap-roots/asbuilt [patched]=/home/aagarwal/tt-metal-port-testinfra)
declare -A BUILD_OF=([base]=/home/aagarwal/tt-metal-port-gap/build_Release [patched]=/home/aagarwal/tt-metal-port-testinfra/build_Release)
# case | yaml (relative to the patched tree, or absolute) | test name | arms
CASES="stream1|$TY/test_fabric_stream_sanity.yaml|StreamCommandTypesMesh|patched
stream2|$TY/test_fabric_stream_sanity.yaml|StreamSizesMesh|patched
stream3|$TY/test_fabric_stream_sanity.yaml|StreamCollectiveChunkMesh|patched
stream4|$TY/test_fabric_stream_sanity.yaml|StreamMulticastMesh|patched
stream5|$TY/test_fabric_stream_sanity.yaml|StreamLinear|patched
stream6|$TY/test_fabric_stream_sanity.yaml|StreamNoFlowControlMesh|patched
sc_lin|$TY/test_fabric_sanity_common.yaml|SimpleUnicastWriteLinear|base patched
sc_fc|$TY/test_fabric_sanity_common.yaml|FlowControlAllToAllLinear|base patched
sc_mesh|$TY/test_fabric_sanity_common.yaml|SimpleUnicastMesh|base patched
sc_types|$TY/test_fabric_sanity_common.yaml|TestNocSendTypesMesh|base patched
m2_mcast|$TY/test_fabric_sanity_at_least_2x2_mesh.yaml|MeshMulticast|base patched
m2_single|$TY/test_fabric_sanity_at_least_2x2_mesh.yaml|SingleSenderMeshUnicastAllDevices|base patched
mesh_fc|$TY/test_fabric_mesh_sanity_common.yaml|FlowControlMeshLowLatency|base patched
vc2_nx|$TY/test_fabric_vc2_at_least_2x2_mesh.yaml|VC2MeshNeighbourExchange|base patched
vc2_uni|$TY/test_fabric_vc2_at_least_2x2_mesh.yaml|MeshNeighbourExchangeVC0VC2Unidirectional|base patched
ub_nx|$TY/test_fabric_ubench_at_least_2x2_mesh.yaml|NeighborExchangeUnicast|base patched
pvs_ns|/home/aagarwal/tt-metal/fabric_exp/port_slots/yamls/pvs_ns.yaml|NS_Write|base patched"
mkdir -p $V/yamls /home/aagarwal/tt-metal-port-testinfra/generated/fabric   # --dump-built-tests needs generated/fabric
echo "$CASES" | while IFS='|' read -r c y t arms; do
    src=$y; [[ $src = /* ]] || src=/home/aagarwal/tt-metal-port-testinfra/$y
    python3 - "$src" "$t" $V/yamls/$c.yaml <<'PY'
import sys, yaml
d = yaml.safe_load(open(sys.argv[1]))
t = [x for x in d["Tests"] if x["name"] == sys.argv[2]]
assert t, sys.argv[2]
t = dict(t[0]); t["benchmark_mode"] = False   # the first test of that name (main's sanity yaml repeats one)
d["Tests"] = [t]; yaml.safe_dump(d, open(sys.argv[3], "w"), sort_keys=False)
PY
    for a in $arms; do
        H=${HOME_OF[$a]}; B=${BUILD_OF[$a]}; out=$V/$a/$c; rm -rf "${out:?}"; mkdir -p $out
        (cd $H && env -u TT_METAL_SLOW_DISPATCH_MODE TT_METAL_HOME=$H TT_METAL_RUNTIME_ROOT=$H \
            LD_LIBRARY_PATH=$B/tt_metal:$B/tt_stl:$B/lib:$B/tt_metal/third_party/umd/lib \
            TT_METAL_CACHE=$V/$a/cache TT_METAL_LOGS_PATH=$out/logs TT_METAL_MOCK_CLUSTER_DESC_PATH=$DESC \
            TT_FABRIC_NO_BW_TELEMETRY=1 TT_METAL_DEVICE_PROFILER=0 TT_METAL_LOG_KERNELS_COMPILE_COMMANDS=1 \
            timeout 1200 nice -n 10 $B/test/tt_metal/tt_fabric/test_infra/test_tt_fabric --test_config $V/yamls/$c.yaml \
            --dump-built-tests --built-tests-dump-file vet_$c.yaml > $out/run.log 2>&1)
        sed 's/\x1b\[[0-9;]*m//g' $out/run.log > $out/run.clean
        cp $H/generated/fabric/vet_$c.yaml $out/built.yaml 2>/dev/null
        st=ok
        grep -aq "Launching programs" $out/run.clean || st="NO LAUNCH ($(grep -aE 'TT_FATAL|TT_THROW|error' $out/run.clean | head -1 | cut -c1-220))"
        if grep -aqE "build failed|Failed to generate binaries" $out/run.clean; then
            st="KERNEL BUILD FAILED ($(grep -aE "error:" $out/run.clean | head -1 | cut -c1-220))"
        fi
        printf '%-8s %-9s %s\n' $a $c "$st"
    done
    if [[ $arms == *base* ]] && [ -f $V/base/$c/built.yaml ] && [ -f $V/patched/$c/built.yaml ]; then
        if cmp -s $V/base/$c/built.yaml $V/patched/$c/built.yaml; then echo "         $c built tests: identical"
        else echo "         $c built tests: differ ($(diff $V/base/$c/built.yaml $V/patched/$c/built.yaml | grep -c '^[<>]') lines)"; fi
    fi
done | tee $V/summary.txt
