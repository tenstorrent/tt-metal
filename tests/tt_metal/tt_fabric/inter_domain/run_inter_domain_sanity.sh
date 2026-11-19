#!/usr/bin/env bash
# Inter-domain sanity on one host: two fabric domains, one process each.
#   run_inter_domain_sanity.sh <devices A> <devices B> [MGD]
#   2 chips: run_inter_domain_sanity.sh 0 1
#   4 chips: run_inter_domain_sanity.sh 0,1 2,3 tt_metal/fabric/mesh_graph_descriptors/p300_mesh_graph_descriptor.textproto
set -u
: "${TT_METAL_HOME:?}"
A=$1
B=$2
MGD=${3:-tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto}
[[ $MGD = /* ]] || MGD=$TT_METAL_HOME/$MGD
DIR=${OUT:-$(mktemp -d)}
mkdir -p "$DIR"
rm -f "$DIR"/d[12].meta "$DIR"/d[12].[0-9]*.*
BIN=$TT_METAL_HOME/build/test/tt_metal/tt_fabric/fabric_unit_tests
export TT_METAL_FABRIC_GRAPH_BUILDER=1 TT_METAL_FABRIC_INTER_DOMAIN=1 TT_INTER_DOMAIN_DIR=$DIR TT_MESH_GRAPH_DESC_PATH=$MGD

side() {  # devices id peer
    TT_VISIBLE_DEVICES=$1 TT_INTER_DOMAIN_ID=$2 TT_INTER_DOMAIN_PEER_ID=$3 TT_METAL_CACHE=$DIR/cache$2 \
        timeout "${TIMEOUT:-600}" "$BIN" --gtest_filter=InterDomain.Sanity > "$DIR/d$2.log" 2>&1
}
side "$A" 1 2 &
pa=$!
side "$B" 2 1 &
pb=$!
wait $pa
ra=$?
wait $pb
rb=$?
echo "domain 1 ($A) rc=$ra, domain 2 ($B) rc=$rb, logs in $DIR"
[[ $ra -eq 0 && $rb -eq 0 ]]
