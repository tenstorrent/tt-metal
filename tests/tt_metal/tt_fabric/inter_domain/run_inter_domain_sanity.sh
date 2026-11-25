#!/usr/bin/env bash
# Inter-domain sanity on one host: two fabric domains, one process per mesh of the MGD.
#   run_inter_domain_sanity.sh <devices A> <devices B> [MGD]
# Devices are one mesh's, or each mesh's in mesh order separated by ':'.
#   2 chips:            run_inter_domain_sanity.sh 0 1
#   4 chips, one mesh:  run_inter_domain_sanity.sh 0,1 2,3 tt_metal/fabric/mesh_graph_descriptors/p300_mesh_graph_descriptor.textproto
#   4 chips, 2 meshes:  run_inter_domain_sanity.sh 0:2 1:3 tests/tt_metal/tt_fabric/inter_domain/mesh_graph_descriptors/dual_1x1_domain.textproto
#   relaying routers (routing plane mismatch): TT_METAL_FABRIC_INTER_DOMAIN_DOWN_PLANES=2,3 TEST=RelayingPlanes
#     its scatter check needs bandwidth telemetry (it skips without, and the full set overflows the router binary):
#     TT_METAL_FABRIC_TELEMETRY="chips=all;eth=all;erisc=all;stats=BANDWIDTH"; add TT_METAL_FABRIC_INTER_DOMAIN_SCATTER_ADDRESS_BIT=10
#     to scatter by write address. One domain on fewer planes: MGD_1=tests/tt_metal/tt_fabric/inter_domain/p300_3_link_mesh_graph_descriptor.textproto
set -u
: "${TT_METAL_HOME:?}"
A=$1
B=$2
MGD=${3:-tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto}
[[ $MGD = /* ]] || MGD=$TT_METAL_HOME/$MGD
DIR=${OUT:-$(mktemp -d)}
mkdir -p "$DIR"
rm -f "$DIR"/d[12].*
BIN=$TT_METAL_HOME/build/test/tt_metal/tt_fabric/fabric_unit_tests
MPIRUN=$(command -v mpirun-ulfm || command -v mpirun)
export TT_METAL_FABRIC_GRAPH_BUILDER=1 TT_METAL_FABRIC_INTER_DOMAIN=1 TT_INTER_DOMAIN_DIR=$DIR TT_MESH_GRAPH_DESC_PATH=$MGD
export TT_INTER_DOMAIN_ROUNDS=${TT_INTER_DOMAIN_ROUNDS:-2} TT_METAL_FABRIC_TELEMETRY=${TT_METAL_FABRIC_TELEMETRY:-0} TT_METAL_OPERATION_TIMEOUT_SECONDS=${TT_METAL_OPERATION_TIMEOUT_SECONDS:-0}
TEST=${TEST:-Sanity}  # or AllAtOnce, with TT_METAL_FABRIC_TELEMETRY=1 to check VC2's routes

side() {  # devices id peer; MGD_1 / MGD_2 override the MGD per domain (e.g. one routing on fewer planes)
    local mgd_var=MGD_$2
    local mgd=${!mgd_var:-$MGD}
    [[ $mgd = /* ]] || mgd=$TT_METAL_HOME/$mgd
    export TT_MESH_GRAPH_DESC_PATH=$mgd TT_INTER_DOMAIN_ID=$2 TT_INTER_DOMAIN_PEER_ID=$3 TT_METAL_CACHE=$DIR/cache$2
    IFS=: read -ra meshes <<< "$1"
    if [[ ${#meshes[@]} -eq 1 ]]; then
        TT_VISIBLE_DEVICES=$1 timeout "${TIMEOUT:-600}" "$BIN" --gtest_filter=InterDomain.$TEST > "$DIR/d$2.log" 2>&1
        return
    fi
    local apps=()
    for m in "${!meshes[@]}"; do
        [[ $m -gt 0 ]] && apps+=(:)
        apps+=(-np 1 -x TT_MESH_ID=$m -x TT_VISIBLE_DEVICES=${meshes[$m]} "$BIN" --gtest_filter=InterDomain.$TEST)
    done
    timeout "${TIMEOUT:-600}" "$MPIRUN" --allow-run-as-root --oversubscribe --tag-output -x TT_METAL_HOME -x TT_METAL_RUNTIME_ROOT \
        -x TT_METAL_FABRIC_GRAPH_BUILDER -x TT_METAL_FABRIC_INTER_DOMAIN -x TT_INTER_DOMAIN_DIR -x TT_MESH_GRAPH_DESC_PATH \
        -x TT_INTER_DOMAIN_ID -x TT_INTER_DOMAIN_PEER_ID -x TT_METAL_CACHE -x TT_INTER_DOMAIN_ROUNDS -x TT_METAL_FABRIC_TELEMETRY -x TT_METAL_OPERATION_TIMEOUT_SECONDS -x TT_METAL_FABRIC_INTER_DOMAIN_DOWN_PLANES -x TT_METAL_FABRIC_INTER_DOMAIN_SCATTER_ADDRESS_BIT "${apps[@]}" \
        > "$DIR/d$2.log" 2>&1
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
[[ $ra -eq 0 && $rb -eq 0 ]] || grep -vh "^ --- " "$DIR"/d1.log "$DIR"/d2.log
[[ $ra -eq 0 && $rb -eq 0 ]]
