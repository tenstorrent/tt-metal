#!/bin/bash
# glxq_run.sh: submit one command to a cabled BH Galaxy quad or pair (glxq.sbatch; MULTI_GALAXY_HARNESS.md). Run on
# a09u32.
#
#   glxq_run.sh --quad Q --tree NAME [options] -- CMD...
#   glxq_run.sh --pair P --form F --tree NAME [options] -- CMD...
#     --pair P        a pair of glxq_pairs.tsv by name, or "any": every pair that supports the form, usable now (no
#                     standing exclude from fabric_exp/testplan/manifest.yaml, no node Slurm shows drained/down/failing).
#                     One job per candidate pair (a sibling group, job names glxp-<group>-<pair>); the first to start
#                     takes the group's lock and cancels the siblings still pending; one that starts later exits at once.
#                     The group counts as ONE study allocation.
#     --form F        16x4 (default; one mesh over both hosts, smoke glxp_smoke_mesh.yaml) | mm_exp | mm_zfb (two meshes,
#                     rank k on mesh k) | express16 | torusx16 (one 16x4 LINE x RING mesh); see glxq_pairs.tsv. The
#                     multi-mesh and express forms have no smoke row: --no-smoke, or a --suite.
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
#     --router R      new (kernel v2 slot, flags 13129088, keepalive 1000; default) | main (kernel v1, flags 0) |
#                     port (a mainline-port tree: no router selection)
#     --time T        Slurm time limit (default 01:30:00)      --timeout S   the command's timeout (default 1800)
#     --no-smoke      skip the data-checked smoke row before the command
#     --mpi-if IF     MPI NIC (default ens5f0np0, the Exabox one)
#     --dry           no submission: local checks, sbatch --test-only (when Slurm would start it, where), and
#                     tt-run --dry-run of the launch for the quad's ring (the mpirun command each rank gets)
#   CMD runs in every rank, in that rank's tree (a symlink farm of the tree, own generated/), e.g.
#     ./build/test/tt_metal/tt_fabric/test_infra/test_tt_fabric --test_config fabric_exp/galaxy/<yaml>
# Rules enforced here: bh-glx-110-a* quads and pairs only; a quad: no submission while any of this user's Galaxy jobs
# is running or pending (a quad takes all four study allocations); a pair group: at most four study allocations with it
# (a pair group counts once, a quad four, any other job once); /data needs free inodes before staging; nothing here
# cancels a job except a pair group's winner cancelling its own pending siblings.
set -uo pipefail
G=/data/aagarwal/frs-glx
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LIVE=/home/aagarwal/tt-metal
QUAD=""; PAIR_SEL=""; FORM=16x4; TREE_NAME=""; SUITE_SRC=""
MGD=tt_metal/fabric/mesh_graph_descriptors/32x4_quad_bh_galaxy_torus_xy_graph_descriptor.textproto
FSD=/data/scaleout_configs/tt-cluster-config-sources/exabox-latest-main/factory_system_descriptor.textproto
ROUTER=new; TIME=01:30:00; RUN_TIMEOUT=1800; SMOKE=1; MPI_IF=ens5f0np0; DRY=0; FSD_RUN=0
while [ $# -gt 0 ]; do
    case $1 in
        --quad) QUAD=$2; shift 2 ;; --tree) TREE_NAME=$2; shift 2 ;; --mgd) MGD=$2; shift 2 ;; --fsd) FSD=$2; shift 2 ;;
        --pair) PAIR_SEL=$2; shift 2 ;; --form) FORM=$2; shift 2 ;; --suite) SUITE_SRC=$2; shift 2 ;;
        --router) ROUTER=$2; shift 2 ;; --time) TIME=$2; shift 2 ;; --timeout) RUN_TIMEOUT=$2; shift 2 ;;
        --no-smoke) SMOKE=0; shift ;; --mpi-if) MPI_IF=$2; shift 2 ;; --dry) DRY=1; shift ;; --fsd-run) FSD_RUN=1; shift ;;
        --) shift; break ;; *) echo "unknown option $1"; exit 1 ;;
    esac
done
[ $# -gt 0 ] || [ -n "$SUITE_SRC" ] ||
    { echo "usage: glxq_run.sh --quad Q | --pair P [--form F | --suite FILE] --tree NAME [options] [-- CMD...]"; exit 1; }
[ -z "$SUITE_SRC" ] || [ -n "$PAIR_SEL" ] || { echo "--suite needs --pair"; exit 1; }
TREE=$G/trees/$TREE_NAME
die() { echo "!!! $*"; exit 1; }
[ -n "$TREE_NAME" ] && [ -d $TREE ] || die "no tree $TREE"
{ [ -n "$QUAD" ] && [ -z "$PAIR_SEL" ]; } || { [ -z "$QUAD" ] && [ -n "$PAIR_SEL" ]; } || die "one of --quad and --pair"
[ -n "$PAIR_SEL" ] || [ -f $TREE/$MGD ] || die "no MGD $TREE/$MGD"
[ "$FSD" = none ] || [ -f "$FSD" ] || die "no FSD $FSD"
[ $FSD_RUN = 0 ] || [ "$FSD" != none ] || die "--fsd-run needs an FSD"
case $ROUTER in new | main | port) ;; *) die "--router new | main | port" ;; esac
for f in build/test/tt_metal/tt_fabric/test_infra/test_tt_fabric python_env/bin/python ttnn/ttnn/distributed/ttrun.py; do
    [ -e $TREE/$f ] || die "tree lacks $f"
done

# ---------------------------------------------------------------------------------------------------------------------
# A pair: a sibling group, one job per candidate pair (glxq_pairs.tsv); the first job to start wins (glxq.sbatch step 0)
# ---------------------------------------------------------------------------------------------------------------------
if [ -n "$PAIR_SEL" ]; then
    form_info() {  # FORM: sets MGD_SRC, MESHES (meshes in the form) and FORM_SMOKE (its smoke row, or none)
        # Existing descriptors (tree paths) unless noted; glxq_pairs.tsv lists which pairs support which form.
        local T=$TREE/tests/tt_metal/tt_fabric/custom_mesh_descriptors
        MESHES=1; FORM_SMOKE=glxp_smoke_mesh.yaml
        case $1 in
            16x4) MGD_SRC=$TREE/tt_metal/fabric/mesh_graph_descriptors/16x4_dual_bh_galaxy_2d_mesh_graph_descriptor.textproto ;;
            mm_exp) MGD_SRC=$T/dual_bh_galaxy_experimental_mesh_graph_descriptor.textproto; MESHES=2; FORM_SMOKE=none ;;
            mm_zfb) MGD_SRC=$T/dual_4x8_z_fallback_mesh_graph_descriptor.textproto; MESHES=2; FORM_SMOKE=none ;;
            mm_exp8) MGD_SRC=$HERE/mgd/dual_bh_galaxy_experimental_8ch_derived_graph_descriptor.textproto; MESHES=2; FORM_SMOKE=none ;;
            mm_zfb8) MGD_SRC=$HERE/mgd/dual_4x8_z_fallback_8ch_derived_graph_descriptor.textproto; MESHES=2; FORM_SMOKE=none ;;
            express16) MGD_SRC=$T/express_links_16x4_mesh_graph_descriptor.textproto; FORM_SMOKE=none ;;
            torusx16) MGD_SRC=$HERE/mgd/torusx_16x4_derived_graph_descriptor.textproto; FORM_SMOKE=none ;;  # derived
            *) die "form $1: 16x4 | mm_exp | mm_exp8 | mm_zfb | mm_zfb8 | express16 | torusx16 (glxq_pairs.tsv)" ;;
        esac
        [ -f "$MGD_SRC" ] || die "no MGD $MGD_SRC"
    }
    write_rank_binding() {  # FORM OUT: ranks 0 and 1 on mesh 0 (one-mesh forms) or rank k on mesh k
        form_info $1
        {   echo "rank_bindings:"
            for r in 0 1; do printf '  - rank: %d\n    mesh_id: %d\n' $r $([ $MESHES = 2 ] && echo $r || echo 0); done
            echo "mesh_graph_desc_path: \"$STAGE/$(basename $MGD_SRC)\""
        } > $2
        cp $MGD_SRC $STAGE/
    }
    if [ -n "$SUITE_SRC" ]; then
        [ -f "$SUITE_SRC" ] || die "no suite $SUITE_SRC"
        FORMS=$(awk -F'\t' '!/^#/ && NF >= 6 && $1 != "config" {print $3}' "$SUITE_SRC" | sort -u | tr '\n' ' ')
        for f in $FORMS; do form_info $f; done
        SMOKE=0; SMOKE_YAML=none
    else
        FORMS=$FORM; form_info $FORM; SMOKE_YAML=$FORM_SMOKE
        [ "$SMOKE_YAML" != none ] || [ $SMOKE = 0 ] || die "--form $FORM has no smoke row yet: pass --no-smoke"
    fi
    # standing excludes (the test plan's), and the Slurm node states now
    EXCL=$(grep -m1 -E '^\s*exclude: \[' $LIVE/fabric_exp/testplan/manifest.yaml | sed 's/.*\[//; s/\].*//; s/,/ /g')
    [ -n "$EXCL" ] || die "no standing exclude list in fabric_exp/testplan/manifest.yaml"
    declare -A NODE_STATE=() PART_LIMIT=()
    to_s() { local t=$1 d=0; [[ $t == *-* ]] && { d=${t%%-*}; t=${t#*-}; }; IFS=: read -r a b c <<< "$t"
        [ -z "$c" ] && { c=$b; b=$a; a=0; }; echo $(( d * 86400 + 10#$a * 3600 + 10#${b:-0} * 60 + 10#${c:-0} )); }
    while read -r p l; do PART_LIMIT[${p%\*}]=$l; done < <(sinfo -h -o "%P %l")
    while read -r n st; do NODE_STATE[$n]="${NODE_STATE[$n]:-} $st"; done < <(sinfo -h -N -o "%N %T" | grep -E '^bh-glx-(110-a|120-c)')
    CAND=(); echo "pairs for --pair $PAIR_SEL, forms $FORMS(excludes: $EXCL):"
    while IFS=$'\t' read -r name slurm hosts leaf conns mgds state; do
        [[ $name == \#* || -z $name ]] && continue
        [ "$PAIR_SEL" = any ] || [ "$PAIR_SEL" = "$name" ] || continue
        all=1; for f in $FORMS; do [[ ",$mgds," == *",$f,"* ]] || all=0; done
        [ $all = 1 ] || continue
        why=""
        for h in ${hosts//,/ }; do
            if [[ $h == bh-glx-120-c* ]]; then  # the user's time-boxed 120-c window only (GLXQ_C120_EXCEPTION=1, GLXQ_C120_UNTIL)
                [ "${GLXQ_C120_EXCEPTION:-0}" = 1 ] && [ "$(date -u +%s)" -lt "${GLXQ_C120_UNTIL:-0}" ] || why+=" $h:no-120c-window"
            elif [[ $h != bh-glx-110-a* ]]; then why+=" $h:not-110a"; fi
            [[ " $EXCL " == *" $h "* ]] && why+=" $h:excluded"
            for st in ${NODE_STATE[$h]:-unknown}; do
                case $st in idle | allocated | mixed | completing | planned) ;; *) why+=" $h:$st" ;; esac
            done
        done
        [ "$slurm" != - ] || why+=" crosses-partitions"
        lim=${PART_LIMIT[${slurm%%:*}]:-}
        [ -z "$lim" ] || [ "$lim" = infinite ] || [ $(to_s $TIME) -le $(to_s $lim) ] || why+=" time-limit(${slurm%%:*}:$lim<$TIME)"
        if [ -z "$why" ]; then CAND+=("$name"$'\t'"$slurm"$'\t'"$hosts"); echo "   $name  $hosts  usable"
        else echo "   $name  skipped:$why"; fi
    done < $HERE/glxq_pairs.tsv
    [ ${#CAND[@]} -gt 0 ] || die "no usable pair for --pair $PAIR_SEL --form $FORM"
    # the cap: a pair group counts once, a quad (or any multi-node job but a pair) four, any other job once
    used=$(squeue -u $USER -h -o "%i|%P|%j|%T|%D" | awk -F'|' '
        $2 !~ /^bh_sc36|^bh_110a/ || $3 == "interactive" {next}
        $3 ~ /^glxp-/ {split($3, f, "-"); g[f[2]] = 1; next}
        $5 > 1 {n += 4; next}
        {n += 1}
        END {for (k in g) n++; print n + 0}')
    CAP=${GLXQ_CAP:-4}  # the user's study cap (4; raised only for a time-boxed window the user names)
    echo "study Galaxy allocations now: $used of $CAP (a pair group counts once); this group makes $((used + 1))"
    if [ $((used + 1)) -gt $CAP ]; then
        [ $DRY = 1 ] && echo "(dry: a submission now would be refused)" || die "at the Galaxy allocation cap: wait"
    fi
    inodes=$(df --output=iavail /data | tail -1 | tr -dc 0-9)
    echo "/data free inodes: $inodes (staging needs >= 100000)"
    [ "$inodes" -ge 100000 ] || die "/data is short of free inodes"
    GRP=$(date -u +%m%d%H%M%S)
    STAGE=$G/glxq/stage/$(date -u +%Y%m%d-%H%M%S)-pair-$([ -n "$SUITE_SRC" ] && basename "$SUITE_SRC" .tsv || echo $FORM)
    PAIR_LOCK=$G/glxq/locks/$GRP
    mkdir -p $STAGE $PAIR_LOCK
    cp $HERE/glxq_rank.sh $HERE/glxq_node.sh $HERE/glxq_links.py $HERE/glxq_pairs.tsv $HERE/glxq_quads.tsv $STAGE/
    [ "$SMOKE_YAML" = none ] || cp $HERE/$SMOKE_YAML $STAGE/
    cp $LIVE/fabric_exp/read_eth_speed.py $LIVE/fabric_exp/consolidation/hang_capture.py $STAGE/
    chmod +x $STAGE/glxq_rank.sh
    # glxq.sbatch sets TT_FABRIC_NO_BW_TELEMETRY=1; a port tree at ce2dee3cd6e or later reads it only with
    # fabric_exp/port_tune/local_patches/test_tt_fabric_no_bw_telemetry.patch applied (else its rows run telemetry-on)
    grep -q TT_FABRIC_NO_BW_TELEMETRY $TREE/tests/tt_metal/tt_fabric/test_infra/test_tt_fabric.cpp 2>/dev/null ||
        echo "WARNING: $TREE's test_tt_fabric does not read TT_FABRIC_NO_BW_TELEMETRY: benchmark rows run telemetry-on (apply port_tune/local_patches/test_tt_fabric_no_bw_telemetry.patch for perf runs)"
    for f in $FORMS; do write_rank_binding $f $STAGE/rank_binding_$f.yaml; done
    cp $STAGE/rank_binding_${FORMS%% *}.yaml $STAGE/rank_binding.yaml
    if [ -n "$SUITE_SRC" ]; then
        # the suite, its yamls resolved: tree-relative paths stay in the tree, any other file is staged
        : > $STAGE/suite.tsv
        while IFS=$'\t' read -r cfg label form to envs yaml; do
            [[ -z $cfg || $cfg == \#* || $cfg == config ]] && continue
            if [ -f "$TREE/$yaml" ]; then y=$TREE/$yaml
            elif [ -f "$yaml" ]; then cp "$yaml" $STAGE/; y=$STAGE/$(basename "$yaml")
            else die "suite $label: no yaml $yaml (tree-relative or a local file)"; fi
            lt=$(tr ',' '\n' <<< "$envs" | sed -n 's/^GLXQ_TREE=//p')  # a line's own tree (glxq.sbatch suite loop)
            [ -z "$lt" ] || [ -x "$(dirname $TREE)/$lt/build/test/tt_metal/tt_fabric/test_infra/test_tt_fabric" ] ||
                die "suite $label: GLXQ_TREE=$lt is not a built tree next to $TREE"
            printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$cfg" "$label" "$form" "$to" "$envs" "$y" >> $STAGE/suite.tsv
        done < "$SUITE_SRC"
        echo 'true' > $STAGE/cmd.sh
        SUITE=1; export SUITE
        echo "suite: $(wc -l < $STAGE/suite.tsv) steps, forms $FORMS"
        column -t -s $'\t' $STAGE/suite.tsv | sed 's/^/   /'
    else
        printf '%q ' "$@" > $STAGE/cmd.sh; echo >> $STAGE/cmd.sh
    fi
    [ "$FSD" = none ] || sha256sum "$FSD" > $STAGE/fsd.sha256
    printf '%s\n' "${CAND[@]}" > $PAIR_LOCK/candidates.tsv
    echo "staged $STAGE (forms $FORMS; command: $(cat $STAGE/cmd.sh)); group $GRP, lock $PAIR_LOCK"
    GLXQ_C120_EXCEPTION=${GLXQ_C120_EXCEPTION:-0}; PAIR_GROUP=$GRP
    export STAGE TREE FSD FSD_RUN ROUTER RUN_TIMEOUT SMOKE MPI_IF GLXQ_C120_EXCEPTION PAIR_GROUP PAIR_LOCK SMOKE_YAML
    for row in "${CAND[@]}"; do
        IFS=$'\t' read -r name slurm hosts <<< "$row"
        SB=(--partition=${slurm%%:*} --nodelist=${slurm#*:} --nodes=2 --time=$TIME --job-name=glxp-$GRP-$name)
        if [ $DRY = 1 ]; then
            echo "=== $name: sbatch --test-only ${SB[*]}"
            PAIR=$name sbatch --test-only --export=ALL "${SB[@]}" $HERE/glxq.sbatch 2>&1 | sed 's/^/   /' |
                sed 's/allocation failure: Requested node configuration is not available/& (expected while a pinned node is allocated: --test-only answers only for placeable nodes)/'
        else
            echo "=== $name: $(PAIR=$name sbatch --export=ALL "${SB[@]}" $HERE/glxq.sbatch 2>&1)"
        fi
    done
    if [ $DRY = 1 ]; then
        IFS=$'\t' read -r name slurm hosts <<< "${CAND[0]}"
        echo "=== tt-run --dry-run for $name (ranks on $hosts)"
        (cd $TREE && TT_METAL_HOME=$TREE TT_GLXQ_JOB=$G/jobs/DRY TT_GLXQ_TREE=$TREE TT_FABRIC_NO_BW_TELEMETRY=1 \
            timeout 120 python_env/bin/python $TREE/ttnn/ttnn/distributed/ttrun.py --dry-run \
            --rank-binding $STAGE/rank_binding.yaml --tcp-interface $MPI_IF \
            $([ $FSD_RUN = 0 ] || echo --factory-system-descriptor $FSD) \
            --mpi-args "--host $hosts" $STAGE/glxq_rank.sh bash $STAGE/cmd.sh) 2>&1 | grep -av "DEBUG" | sed 's/^/   /'
        rm -rf $PAIR_LOCK  # nothing submitted
    fi
    exit 0
fi

# the quad: partition(s), node list, ring order
PARTS=(); NODELIST=""; RINGS=()
while IFS=$'\t' read -r name slurm ring leaf; do
    [[ $name == \#* || -z $name ]] && continue
    if [[ $name == c120_* ]] && [ "${GLXQ_C120_EXCEPTION:-0}" != 1 ]; then continue; fi  # one-time user exception only
    if [ "$QUAD" = "$name" ] || { [ "$QUAD" = pods ] && [[ $name == pod* ]]; } ||
        { [ "$QUAD" = c120pods ] && [[ $name == c120_pod* ]]; } || { [ "$QUAD" = allpods ] && [[ $name == pod* || $name == c120_pod* ]]; }; then
        PARTS+=("${slurm%%:*}"); [[ $slurm == *:* ]] && NODELIST=${slurm#*:}; RINGS+=("$name=$ring")
    fi
done < $HERE/glxq_quads.tsv
[ ${#PARTS[@]} -gt 0 ] || die "--quad $QUAD: not in glxq_quads.tsv (pods, pod3/6/7/8/9, a07-a10)"
SB=(--partition=$(IFS=,; echo "${PARTS[*]}") --time=$TIME --job-name=glxq-$QUAD)
[ -n "$NODELIST" ] && SB+=(--nodelist=$NODELIST)

# a quad takes all four study allocations: no other Galaxy job of this user, running or pending
mine=$(squeue -u $USER -h -o "%i %P %j %T" | awk '$2 ~ /^bh_sc36_6|^bh_110a_sc16|^bh_sc36_2/')
if [ -n "$mine" ]; then
    echo "$mine"
    [ $DRY = 1 ] && echo "(dry: a submission now would be refused)" || die "Galaxy jobs of $USER are queued or running: a quad needs all four allocations"
fi

# stage: everything the nodes read, on /data (/home is not shared with the Galaxies)
STAGE=$G/glxq/stage/$(date -u +%Y%m%d-%H%M%S)-$QUAD
mkdir -p $STAGE
cp $HERE/glxq_rank.sh $HERE/glxq_node.sh $HERE/glxq_links.py $HERE/glxq_smoke.yaml $HERE/glxq_quads.tsv $STAGE/
cp $LIVE/fabric_exp/read_eth_speed.py $LIVE/fabric_exp/consolidation/hang_capture.py $STAGE/
grep -q TT_FABRIC_NO_BW_TELEMETRY $TREE/tests/tt_metal/tt_fabric/test_infra/test_tt_fabric.cpp 2>/dev/null ||  # as the pair path
    echo "WARNING: $TREE's test_tt_fabric does not read TT_FABRIC_NO_BW_TELEMETRY: benchmark rows run telemetry-on (apply port_tune/local_patches/test_tt_fabric_no_bw_telemetry.patch for perf runs)"
chmod +x $STAGE/glxq_rank.sh
{   echo "rank_bindings:"
    for r in 0 1 2 3; do printf '  - rank: %d\n    mesh_id: 0\n' $r; done
    echo "mesh_graph_desc_path: \"$TREE/$MGD\""
} > $STAGE/rank_binding.yaml
printf '%q ' "$@" > $STAGE/cmd.sh; echo >> $STAGE/cmd.sh
[ "$FSD" = none ] || sha256sum "$FSD" > $STAGE/fsd.sha256
echo "staged $STAGE (rank binding: 4 ranks on mesh 0, MGD $MGD; command: $(cat $STAGE/cmd.sh))"
GLXQ_C120_EXCEPTION=${GLXQ_C120_EXCEPTION:-0}
export STAGE TREE FSD FSD_RUN ROUTER RUN_TIMEOUT SMOKE MPI_IF GLXQ_C120_EXCEPTION

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
