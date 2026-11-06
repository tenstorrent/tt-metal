#!/bin/bash
# Host reconciliation (RECONCILE_HOST.md): what each host passes the router, on the mock (CPU only, no device).
#   rc_capture.sh study|port [CONFIG...]      (default: every config below)
# study: the lead v4txa11 on the study host (TREE_STUDY, default /data/aagarwal/frs-glx/trees/v4txa11_1104cg: the
#        live a09u32 host, broker 769, whose router images equal the 649 host's with the new switches unset), with the
#        env the study's standard perf rows use (cap_lib.sh run_cap / jobs_lib.sh run): TT_FABRIC_KERNEL_VERSION=2,
#        TT_FABRIC_EXP_FLAGS=13129088, TT_FABRIC_EXP_SEQ_UPDATE_TIMEOUT=1000, TT_METAL_FABRIC_OPT_LEVEL=O3,
#        TT_FABRIC_NO_BW_TELEMETRY=1, profiler off.
# port:  the port, 22cdb1e55c3 (TREE_PORT, default the worktree /home/aagarwal/tt-metal-port-slots; its local slot
#        patch only adds the slot log and explicit-row switches, unset here), with the port's own run settings
#        (.scratch/val/val_lib.sh: BW telemetry off, profiler off; nothing else).
# RC_EXTRA_ENV: more NAME=VALUE words for the port side (e.g. a local switch of the tree under test).
# Both log the slot rows (TT_FABRIC_EXP_LOG_SLOTS=1) and the compile commands. Each config runs its yaml's first test
# up to program launch (benchmark_mode off: the mock's clocks are 0 MHz). Output: $RC/<side>/<config>/ (default RC
# in the session scratchpad).
set -uo pipefail
SIDE=${1:?study|port}; shift
cd /home/aagarwal/tt-metal
RC=${RC:-/tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad/rc}
P150=$PWD/fabric_exp/a09u32_cluster_desc.yaml
K=fabric_exp/consolidation/slotalloc/knee
MGD42M=TT_MESH_GRAPH_DESC_PATH=$PWD/$K/merge/p150_x8_4x2_mesh_mgd.textproto
MGD42T=TT_MESH_GRAPH_DESC_PATH=$PWD/fabric_exp/torus/p150_x8_4x2_torus_xy_mgd.textproto
case $SIDE in
    study) TREE=${TREE_STUDY:-/data/aagarwal/frs-glx/trees/v4txa11_1104cg}
           ENVS=(TT_FABRIC_KERNEL_VERSION=2 TT_FABRIC_EXP_FLAGS=13129088 TT_FABRIC_EXP_SEQ_UPDATE_TIMEOUT=1000
                 TT_METAL_FABRIC_OPT_LEVEL=O3) ;;
    port) TREE=${TREE_PORT:-/home/aagarwal/tt-metal-port-slots}; ENVS=() ;;
    *) echo "study|port"; exit 1 ;;
esac
# name | mode (single, split, split4x4) | yaml | system (p150, glx) | slot size (4352 = default) | extra env
CONFIGS="p150_line_4352|single|fabric_exp/slotstudy/sls_1d_p150.yaml|p150|4352|
p150_line_3072|single|fabric_exp/slotstudy/sls_1d_p150.yaml|p150|3072|
p150_ring_4352|single|fabric_exp/v4/val_ring_p150.yaml|p150|4352|
p150_ring_3072|single|fabric_exp/v4/val_ring_p150.yaml|p150|3072|
p150_mesh_4352|single|fabric_exp/slotstudy/sls_2d_p150.yaml|p150|4352|
p150_mesh_3072|single|fabric_exp/slotstudy/sls_2d_p150.yaml|p150|3072|
p150_mesh4x2_4352|single|fabric_exp/slotstudy/sls_2d_p150.yaml|p150|4352|$MGD42M
p150_torus2x4_4352|single|fabric_exp/torus/torusxy_2x4_sizes.yaml|p150|4352|
p150_torus2x4_3072|single|fabric_exp/torus/torusxy_2x4_sizes.yaml|p150|3072|
p150_torus4x2_4352|single|fabric_exp/torus/torusxy_4x2_sizes.yaml|p150|4352|$MGD42T
p150_torus4x2_3072|single|fabric_exp/torus/torusxy_4x2_sizes.yaml|p150|3072|$MGD42T
p150_torusy_4352|single|fabric_exp/v4/val_torus_y_p150.yaml|p150|4352|
glx_vc1split_4352|split|fabric_exp/slotstudy/sls_glx_vc1.yaml|glx|4352|
glx_vc1split_3072|split|fabric_exp/slotstudy/sls_glx_vc1_3k.yaml|glx|keep|
glx_vc1split4x4_4352|split4x4|fabric_exp/slotstudy/sls_glx_vc1_4x4.yaml|glx|4352|
glx_vc2_4352|single|fabric_exp/slotstudy/sls_vc2_glx_4k.yaml|glx|keep|TT_METAL_ENABLE_FABRIC_VC2=1
glx_vc2_3072|single|fabric_exp/slotstudy/sls_vc2_glx_3k.yaml|glx|keep|TT_METAL_ENABLE_FABRIC_VC2=1
glx_torus_4352|single|fabric_exp/galaxy/glx_sizes_torus.yaml|glx|4352|
glx_mesh_4352|single|fabric_exp/slotstudy/sls_2d_glx.yaml|glx|4352|
p150_ne_4352|single|fabric_exp/port_slots/yamls/pvs_c1ne.yaml|p150|4352|
p150_mesh_8704|single|fabric_exp/slotstudy/sls_2d_p150.yaml|p150|8704|
p150_line_8704|single|fabric_exp/slotstudy/sls_1d_p150.yaml|p150|8704|
glx_vc1split_8704|split|fabric_exp/slotstudy/sls_glx_vc1.yaml|glx|8704|
glx_vc2_8704|single|fabric_exp/slotstudy/sls_vc2_glx_4k.yaml|glx|8704|TT_METAL_ENABLE_FABRIC_VC2=1"
mkdir -p $RC/$SIDE $RC/yamls
while IFS='|' read -r name mode yaml sys slot extra; do
    [ $# -gt 0 ] && [[ " $* " != *" $name "* ]] && continue
    python3 - "$yaml" "$RC/yamls/$name.yaml" "$slot" <<'PY'
import sys, yaml
d = yaml.safe_load(open(sys.argv[1])); t = dict(d["Tests"][0]); slot = sys.argv[3]
t["benchmark_mode"] = False
if slot not in ("keep", "4352"):
    t["fabric_setup"] = dict(t.get("fabric_setup") or {}, max_packet_size=int(slot))
    pp = t.get("parametrization_params") or {}
    if "size" in pp:
        pp["size"] = [min(int(s), int(slot)) for s in pp["size"]][:1]
d["Tests"] = [t]
yaml.safe_dump(d, open(sys.argv[2], "w"), sort_keys=False)
PY
    desc=(); [ $sys = p150 ] && desc=(MOCK_DESC=$P150)
    ex=(); [ -n "$extra" ] && ex=($extra)
    rm -rf "${RC:?}/$SIDE/${name:?}"
    # sls_mock_vet.sh's launch with this side's env: its own env is the study's, so the port side goes through a
    # wrapper that unsets the study switches sls_mock_vet.sh adds (KERNEL_VERSION, EXP_FLAGS, keepalive, O3).
    if [ $SIDE = port ]; then
        env "${desc[@]}" SCR=$RC/$SIDE/$name TREE=$TREE P=1 RC_PORT=1 bash fabric_exp/consolidation/reconcile/rc_mock_vet.sh \
            $mode $RC/yamls/$name.yaml TT_METAL_LOG_KERNELS_COMPILE_COMMANDS=1 TT_METAL_DEVICE_PROFILER=0 "${ex[@]}" \
            ${RC_EXTRA_ENV:-} > $RC/$SIDE/$name.out 2>&1
    else
        env "${desc[@]}" SCR=$RC/$SIDE/$name TREE=$TREE P=1 bash fabric_exp/consolidation/reconcile/rc_mock_vet.sh \
            $mode $RC/yamls/$name.yaml TT_METAL_LOG_KERNELS_COMPILE_COMMANDS=1 TT_METAL_DEVICE_PROFILER=0 "${ENVS[@]}" "${ex[@]}" \
            > $RC/$SIDE/$name.out 2>&1
    fi
    echo "$SIDE $name: $(grep -c 'launch reached by [1-9]' $RC/$SIDE/$name.out) launched | $(tail -1 $RC/$SIDE/$name.out | cut -c1-150)"
done <<< "$CONFIGS"
