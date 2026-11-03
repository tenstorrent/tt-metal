#!/bin/bash
# glxq_run.sh: submit one command to a cabled BH Galaxy quad (glxq.sbatch; MULTI_GALAXY_HARNESS.md). Run on a09u32.
#
#   glxq_run.sh --quad Q --tree NAME [options] -- CMD...
#     --quad Q        pods (any free pod partition: pod3/6/7/8/9), pod3 | pod6 | pod7 | pod8 | pod9, or an SC16 rack
#                     a07 | a08 | a09 | a10 (partition bh_110a_sc16, those four nodes). Table: glxq_quads.tsv.
#     --tree NAME     /data/aagarwal/frs-glx/trees/NAME (lean study tree: mk_glx_base.sh / mk_glx_tree.sh)
#     --mgd PATH      mesh graph descriptor, tree-relative (default the 32x4 quad torus:
#                     tt_metal/fabric/mesh_graph_descriptors/32x4_quad_bh_galaxy_torus_xy_graph_descriptor.textproto)
#     --fsd PATH|none Factory System Descriptor for the link preflight (default: the published exabox FSD, production
#                     channel, /data/scaleout_configs/tt-cluster-config-sources/exabox-latest-main/factory_system_descriptor.textproto).
#                     "none": link counts printed only.
#     --fsd-run       also pass the FSD to tt-run (--factory-system-descriptor; the control plane keeps the allocated
#                     hosts' part). Off by default: with an FSD the control plane needs RELAXED system health for the
#                     RELAXED quad MGDs, and test_tt_fabric builds its first control plane in STRICT before reading its
#                     yaml (mock run, MULTI_GALAXY_HARNESS.md §5.1). For apps that open the fabric RELAXED themselves.
#     --router R      new (kernel v2 slot, flags 13129088, keepalive 1000; default) | main (kernel v1, flags 0)
#     --time T        Slurm time limit (default 01:30:00)      --timeout S   the command's timeout (default 1800)
#     --no-smoke      skip the data-checked smoke row before the command
#     --mpi-if IF     MPI NIC (default ens5f0np0, the Exabox one)
#     --dry           no submission: local checks, sbatch --test-only (when Slurm would start it, where), and
#                     tt-run --dry-run of the launch for the quad's ring (the mpirun command each rank gets)
#   CMD runs in every rank, in that rank's tree (a symlink farm of the tree, own generated/), e.g.
#     ./build/test/tt_metal/tt_fabric/test_infra/test_tt_fabric --test_config fabric_exp/galaxy/<yaml>
# Rules enforced here: bh-glx-110-a* quads only; no submission while any of this user's Galaxy jobs is running or
# pending (a quad takes all four study allocations); nothing is ever cancelled.
set -uo pipefail
G=/data/aagarwal/frs-glx
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LIVE=/home/aagarwal/tt-metal
QUAD=""; TREE_NAME=""; MGD=tt_metal/fabric/mesh_graph_descriptors/32x4_quad_bh_galaxy_torus_xy_graph_descriptor.textproto
FSD=/data/scaleout_configs/tt-cluster-config-sources/exabox-latest-main/factory_system_descriptor.textproto
ROUTER=new; TIME=01:30:00; RUN_TIMEOUT=1800; SMOKE=1; MPI_IF=ens5f0np0; DRY=0; FSD_RUN=0
while [ $# -gt 0 ]; do
    case $1 in
        --quad) QUAD=$2; shift 2 ;; --tree) TREE_NAME=$2; shift 2 ;; --mgd) MGD=$2; shift 2 ;; --fsd) FSD=$2; shift 2 ;;
        --router) ROUTER=$2; shift 2 ;; --time) TIME=$2; shift 2 ;; --timeout) RUN_TIMEOUT=$2; shift 2 ;;
        --no-smoke) SMOKE=0; shift ;; --mpi-if) MPI_IF=$2; shift 2 ;; --dry) DRY=1; shift ;; --fsd-run) FSD_RUN=1; shift ;;
        --) shift; break ;; *) echo "unknown option $1"; exit 1 ;;
    esac
done
[ $# -gt 0 ] || { echo "usage: glxq_run.sh --quad Q --tree NAME [options] -- CMD..."; exit 1; }
TREE=$G/trees/$TREE_NAME
die() { echo "!!! $*"; exit 1; }
[ -n "$TREE_NAME" ] && [ -d $TREE ] || die "no tree $TREE"
[ -f $TREE/$MGD ] || die "no MGD $TREE/$MGD"
[ "$FSD" = none ] || [ -f "$FSD" ] || die "no FSD $FSD"
[ $FSD_RUN = 0 ] || [ "$FSD" != none ] || die "--fsd-run needs an FSD"
case $ROUTER in new | main) ;; *) die "--router new | main" ;; esac
for f in build/test/tt_metal/tt_fabric/test_infra/test_tt_fabric python_env/bin/python ttnn/ttnn/distributed/ttrun.py; do
    [ -e $TREE/$f ] || die "tree lacks $f"
done

# the quad: partition(s), node list, ring order
PARTS=(); NODELIST=""; RINGS=()
while IFS=$'\t' read -r name slurm ring leaf; do
    [[ $name == \#* || -z $name ]] && continue
    if [ "$QUAD" = "$name" ] || { [ "$QUAD" = pods ] && [[ $name == pod* ]]; }; then
        PARTS+=("${slurm%%:*}"); [[ $slurm == *:* ]] && NODELIST=${slurm#*:}; RINGS+=("$name=$ring")
    fi
done < $HERE/glxq_quads.tsv
[ ${#PARTS[@]} -gt 0 ] || die "--quad $QUAD: not in glxq_quads.tsv (pods, pod3/6/7/8/9, a07-a10)"
SB=(--partition=$(IFS=,; echo "${PARTS[*]}") --time=$TIME --job-name=glxq-$QUAD)
[ -n "$NODELIST" ] && SB+=(--nodelist=$NODELIST)

# a quad takes all four study allocations: no other Galaxy job of this user, running or pending
mine=$(squeue -u $USER -h -o "%i %P %j %T" | awk '$2 ~ /^bh_sc36_6|^bh_110a_sc16/')
if [ -n "$mine" ]; then
    echo "$mine"
    [ $DRY = 1 ] && echo "(dry: a submission now would be refused)" || die "Galaxy jobs of $USER are queued or running: a quad needs all four allocations"
fi

# stage: everything the nodes read, on /data (/home is not shared with the Galaxies)
STAGE=$G/glxq/stage/$(date -u +%Y%m%d-%H%M%S)-$QUAD
mkdir -p $STAGE
cp $HERE/glxq_rank.sh $HERE/glxq_node.sh $HERE/glxq_links.py $HERE/glxq_smoke.yaml $HERE/glxq_quads.tsv $STAGE/
cp $LIVE/fabric_exp/read_eth_speed.py $LIVE/fabric_exp/consolidation/hang_capture.py $STAGE/
chmod +x $STAGE/glxq_rank.sh
{   echo "rank_bindings:"
    for r in 0 1 2 3; do printf '  - rank: %d\n    mesh_id: 0\n' $r; done
    echo "mesh_graph_desc_path: \"$TREE/$MGD\""
} > $STAGE/rank_binding.yaml
printf '%q ' "$@" > $STAGE/cmd.sh; echo >> $STAGE/cmd.sh
[ "$FSD" = none ] || sha256sum "$FSD" > $STAGE/fsd.sha256
echo "staged $STAGE (rank binding: 4 ranks on mesh 0, MGD $MGD; command: $(cat $STAGE/cmd.sh))"
export STAGE TREE FSD FSD_RUN ROUTER RUN_TIMEOUT SMOKE MPI_IF

if [ $DRY = 1 ]; then
    echo "=== sbatch --test-only ${SB[*]}"
    sbatch --test-only "${SB[@]}" $HERE/glxq.sbatch 2>&1 | sed 's/^/   /'
    for qr in "${RINGS[@]}"; do
        echo "=== tt-run --dry-run for ${qr%%=*} (ring ${qr#*=})"
        (cd $TREE && TT_METAL_HOME=$TREE TT_GLXQ_JOB=$G/jobs/DRY TT_GLXQ_TREE=$TREE TT_FABRIC_NO_BW_TELEMETRY=1 \
            timeout 120 python_env/bin/python $TREE/ttnn/ttnn/distributed/ttrun.py --dry-run \
            --rank-binding $STAGE/rank_binding.yaml --tcp-interface $MPI_IF \
            $([ $FSD_RUN = 0 ] || echo --factory-system-descriptor $FSD) \
            --mpi-args "--host ${qr#*=}" $STAGE/glxq_rank.sh bash $STAGE/cmd.sh) 2>&1 | grep -av "DEBUG" | sed 's/^/   /'
    done
    exit 0
fi
sbatch --export=ALL "${SB[@]}" $HERE/glxq.sbatch
