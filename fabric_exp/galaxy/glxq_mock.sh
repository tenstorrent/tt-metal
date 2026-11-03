#!/bin/bash
# glxq_mock.sh [TREE_NAME] [FSD|none]: the multi-Galaxy plumbing on one host, CPU only (no device, no Slurm): tt-run in
# its rank-binding mode with a 4-rank mock cluster (a revC 32x4 subtorus quad: SC36_32x4_revC_subtorus_aisleD, hosts
# bh-glx-120-d03u02..u20, the four of tests/tt_metal/tt_fabric/mock_cluster_mappings/disconnected_sc36_quads_mapping.yaml),
# the quad MGD, glxq_rank.sh per rank (own tree farm, cache, log) and glxq_smoke.yaml. Each rank ends at its first
# program launch (mock). Pass: all four ranks reach "Launching programs" with no fatal error before it.
# Writes under the session scratchpad (glxq_mock/<stamp>); the caches are deleted at the end (size).
set -uo pipefail
G=/data/aagarwal/frs-glx
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TREE=$G/trees/${1:-v4txa11_1103}
FSD=${2:-none}
J=/tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad/glxq_mock/$(date -u +%H%M%S)
mkdir -p $J/ranks $J/cache $J/results
D=tt_metal/third_party/tt-cluster-descriptors/superclusters/blackhole/SC36_32x4_revC_subtorus_aisleD/SC36_32x4_revC_subtorus_aisleD_cluster_desc
{   echo "rank_to_cluster_mock_cluster_desc:"
    k=0; for u in 02:32 08:33 14:34 20:35; do
        echo "  $k: \"$TREE/$D/SC36_32x4_revC_subtorus_aisleD_cluster_desc_bh-glx-120-d03u${u%%:*}_rank_${u##*:}.yaml\""; k=$((k + 1)); done
} > $J/mock_mapping.yaml
{   echo "rank_bindings:"
    for r in 0 1 2 3; do printf '  - rank: %d\n    mesh_id: 0\n' $r; done
    echo "mesh_graph_desc_path: \"$TREE/tt_metal/fabric/mesh_graph_descriptors/32x4_quad_bh_galaxy_torus_xy_graph_descriptor.textproto\""
} > $J/rank_binding.yaml
python3 - $HERE/glxq_smoke.yaml $J/smoke_mock.yaml <<'PY'
import sys, yaml  # the mock stops benchmark mode before traffic setup; the smoke row is data-checked already
d = yaml.safe_load(open(sys.argv[1]))
for t in d["Tests"]:
    t["benchmark_mode"] = False
yaml.safe_dump(d, open(sys.argv[2], "w"), sort_keys=False)
PY
for v in $(env | grep -oE '^TT_[A-Za-z0-9_]+'); do unset $v; done
lib=$TREE/build_Release
export TT_METAL_HOME=$TREE TT_METAL_RUNTIME_ROOT=$TREE TT_GLXQ_JOB=$J TT_GLXQ_TREE=$TREE
export LD_LIBRARY_PATH=$lib/tt_metal:$lib/tt_stl:$lib/lib:$lib/tt_metal/third_party/umd/lib
export TT_FABRIC_NO_BW_TELEMETRY=1 TT_METAL_FABRIC_OPT_LEVEL=O3 TT_METAL_FABRIC_ROUTER_SYNC_TIMEOUT_MS=1000000
export TT_FABRIC_KERNEL_VERSION=2 TT_FABRIC_EXP_FLAGS=13129088 TT_FABRIC_EXP_SEQ_UPDATE_TIMEOUT=1000
FSD_ARGS=(); [ "$FSD" != none ] && FSD_ARGS=(--factory-system-descriptor $FSD) && export RELIABILITY_MODE=relaxed
echo "mock quad: tree ${TREE##*/}, FSD $FSD, out $J ($(date -u +%T))"
(cd $TREE && timeout 1500 nice -n 10 python_env/bin/python $TREE/ttnn/ttnn/distributed/ttrun.py \
    --mock-cluster-rank-binding $J/mock_mapping.yaml --rank-binding $J/rank_binding.yaml "${FSD_ARGS[@]}" \
    $HERE/glxq_rank.sh ./build/test/tt_metal/tt_fabric/test_infra/test_tt_fabric --test_config $J/smoke_mock.yaml) \
    > $J/results/ttrun.log 2>&1
echo "tt-run exit $? ($(date -u +%T)); per rank:"
ok=0
for f in $J/ranks/r*_*.log; do
    l=$(sed 's/\x1b\[[0-9;]*m//g' $f | grep -an "Launching programs" | head -1 | cut -d: -f1)
    e=$(sed 's/\x1b\[[0-9;]*m//g' $f | grep -anE "TT_FATAL|TT_THROW" | head -1)
    el=${e%%:*}
    if [ -n "$l" ] && { [ -z "$el" ] || [ "$el" -gt "$l" ]; }; then ok=$((ok + 1)); s="launch reached"; else s="NO LAUNCH"; fi
    echo "   $(basename $f): $s; $(sed 's/\x1b\[[0-9;]*m//g' $f | grep -aoE 'Mesh graph descriptor[^(]*|TT_MESH_HOST_RANK[^ ]*|host rank [0-9]+' | head -1)${e:+ | first fatal: $(echo "$e" | cut -d: -f2- | cut -c1-160)}"
done
grep -a "\[glxq\]" $J/results/ttrun.log | sed 's/^/   /'
echo "ranks reaching launch: $ok / 4"
rm -rf "${J:?}/cache"
