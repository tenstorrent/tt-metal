#!/bin/bash
# Second-LB harness (LB2_HARNESS.md): submit one study command to an extra P150x8 LB through Slurm (lb2.sbatch).
#
#   lb2_run.sh -n a07u28|a08u28|a04u24|a04u32|a04u28|any [options] "<command>"  the command runs in /home/aagarwal/tt-metal on the
#                                                         node, exactly as a broker job's command would on a09u32
#   -n any           ONE job that starts on whichever approved LB frees first (-n a07u28,a04u24: the same, among the
#                    listed approved LBs only) (user, 2026-11-03): the approved nodes'
#                    partitions, every other node in them excluded (a09u32, the user's host, always), job name lb2-any,
#                    no per-node singleton (--exclusive keeps one job per node). The job runs on $SLURMD_NODENAME;
#                    its log is lb2-any-<job>.log and its SUMMARY / collect manifest name the node it ran on.
#     --test-only      submit nothing that can run: sbatch --test-only (where Slurm would start it; it answers
#                      "allocation failure" while every candidate is allocated), then the same job submitted held
#                      (--hold), Slurm's accepted partitions / excluded nodes / reason printed, and the held job
#                      cancelled (it never became eligible to start)
#   options:
#     -t HH:MM:SS      Slurm time limit (default 03:00:00); the command's own timeout is 20 min less (LB2_TIMEOUT)
#     --dry            sync + verify + preflight on the node, then print what would run (no device access)
#     --hostcheck      read-only host check (lb2_hostcheck.sh); no command, no reset
#     --no-reset       skip the tt-smi -r before the command (the link check still runs)
#     --inputs "P..."  more node-tree paths the preflight must find (e.g. a yaml a script reads)
#     --move-aside     first job on a node whose /home/aagarwal/tt-metal is not an lb2 tree: rename it (kept)
#     --stale-ok       submit even when a live input differs from the stage (default: refuse)
#     --after JOBID    start only after that lb2 job ended with exit 0 (Slurm afterok), e.g. dry -> hostcheck -> calibration
# Before submitting: the live tree's sources (unswapped view) against the current stage. Any changed script, yaml,
# variant, router source or any file named in the command refuses the submit: run lb2_stage.sh refresh first. Jobs on
# one node run one at a time (Slurm singleton per node: job name lb2-<node>). When a job ends, lb2_collect.sh JOBID copies
# its results into this tree's fabric_exp/results with a manifest.
set -uo pipefail
SRC=/home/aagarwal/tt-metal
CON=$SRC/fabric_exp/consolidation
G=/data/aagarwal/lb2
ST=$G/stage
# Approved nodes any-mode leaves out by default (a baseline link that does not train: LB2_HARNESS.md, "Node status").
ANY_SKIP_DEFAULT="a08u28"  # /home 13 GB free (171943), below the 25 GB floor (2026-11-04)
want="" node="" tl=03:00:00 mode=run reset=1 inputs="" aside="" stale="" after="" testonly=""
while [ $# -gt 1 ]; do
    case $1 in
        -n) node=$2; shift 2 ;;
        -t) tl=$2; shift 2 ;;
        --dry) mode=dry; shift ;;
        --hostcheck) mode=hostcheck; shift ;;
        --no-reset) reset=0; shift ;;
        --inputs) inputs=$2; shift 2 ;;
        --move-aside) aside=1; shift ;;
        --stale-ok) stale=1; shift ;;
        --after) after=$2; shift 2 ;;
        --test-only) testonly=1; shift ;;
        *) break ;;
    esac
done
cmd=${1:-}
[ "$cmd" = --hostcheck ] && { mode=hostcheck; cmd=""; }
case $node in
    a08u28|bh-lb-120-a08u28) node=bh-lb-120-a08u28; part=bh_lb_C67 ;;
    a07u28|bh-lb-120-a07u28) node=bh-lb-120-a07u28; part=bh_lb_C45 ;;
    a04u24|bh-lb-120-a04u24) node=bh-lb-120-a04u24; part=bh_lb_120_A45 ;;  # user, 2026-11-03
    a04u32|bh-lb-120-a04u32) node=bh-lb-120-a04u32; part=bh_lb_120_A456 ;;  # user, 2026-11-04
    a04u28|bh-lb-120-a04u28) node=bh-lb-120-a04u28; part=bh_lb_120_A45 ;;  # host check passed (172808), coordinator 2026-11-04
    # Candidate (coordinator, 2026-11-04): host check only (lb2.sbatch enforces it), not in any-mode until it passes.
    a06u24|bh-lb-120-a06u24) node=bh-lb-120-a06u24; part=bh_lb_120_A456 ;;
    any) node=any; want="" ;;
    *,*) want=",$node,"; node=any ;;
    *) echo "usage: lb2_run.sh -n a07u28|a08u28|a04u24|a04u32|a04u28|any [options] \"<command>\"  (only these nodes are approved)"; exit 1 ;;
esac
# -n any: the approved nodes Slurm lists in their partitions; every other node of those partitions excluded
if [ $node = any ]; then
    parts="" ok="" excl=""
    # LB2_ANY_SKIP="a07u28 ...": approved nodes left out of any-mode (e.g. a node with a baseline link that will not
    # train; LB2_HARNESS.md lists the current ones). An explicit -n list overrides it.
    for np in bh-lb-120-a07u28:bh_lb_C45 bh-lb-120-a08u28:bh_lb_C67 bh-lb-120-a04u24:bh_lb_120_A45 \
        bh-lb-120-a04u32:bh_lb_120_A456 bh-lb-120-a04u28:bh_lb_120_A45; do
        n=${np%%:*} p=${np#*:}
        if [ -n "$want" ] && [[ $want != *",${n#bh-lb-120-},"* ]] && [[ $want != *",$n,"* ]]; then
            continue
        fi
        if [ -z "$want" ] && [[ " ${LB2_ANY_SKIP-$ANY_SKIP_DEFAULT} " == *" ${n#bh-lb-120-} "* ]]; then
            echo "   $n: left out of any-mode (LB2_ANY_SKIP / the default list)"; continue
        fi
        if sinfo -h -N -p $p -o %N 2>/dev/null | grep -qx $n; then
            ok="$ok $n"; case ",$parts," in *",$p,"*) ;; *) parts=${parts:+$parts,}$p ;; esac
        else
            echo "   $n: not listed in partition $p by sinfo: left out"
        fi
    done
    [ -n "$ok" ] || { echo "!!! no approved (and requested) LB is listed by sinfo"; exit 1; }
    for n in $(sinfo -h -N -p $parts -o %N | sort -u); do
        case " $ok " in *" $n "*) ;; *) excl=${excl:+$excl,}$n ;; esac
    done
    case ",$excl," in *,bh-lb-120-a09u32,*) ;; *)
        sinfo -h -N -p $parts -o %N | grep -qx bh-lb-120-a09u32 &&
            { echo "!!! a09u32 is in $parts but not excluded"; exit 1; } ;;
    esac
    case " $ok " in *" bh-lb-120-a09u32 "*) echo "!!! a09u32 among the candidates"; exit 1 ;; esac
    echo "   any: candidates$ok; partitions $parts; excluded $excl"
fi
[ $mode = hostcheck ] || [ -n "$cmd" ] || { echo "!!! no command"; exit 1; }
[ "$(hostname)" = bh-lb-120-a09u32 ] || { echo "!!! submit from bh-lb-120-a09u32"; exit 1; }
cur=$(readlink $ST/current) && [ -f $ST/$cur/REF_src.sha256 ] || { echo "!!! no stage: lb2_stage.sh full"; exit 1; }

# the live sources against the stage (unswapped view): what the node would run differently
python3 -B - $SRC $ST/$cur "$cmd" <<'EOF'
import hashlib, os, re, sys
sys.path.insert(0, os.path.join(sys.argv[1], "fabric_exp/consolidation"))
import lb2_pack
src, st, cmd = sys.argv[1:]
class A: pass
a = A(); a.backup = "/home/aagarwal/.tt-metal-p1-swap-backup"; a.toplevel = os.path.join(st, "toplevel")
a.dirs = open(os.path.join(st, "SCOPE")).read().split('"')[1].split()
ref = lb2_pack.load_ref(os.path.join(st, "REF_src.sha256"))
now = dict(lb2_pack.hashes(src, a))
diff = sorted(k for k in set(ref) | set(now) if ref.get(k) != now.get(k))
# what the job can read: the files the command names, the consolidation scripts they call (transitively), the
# libraries every study job sources, the variants named, every yaml / descriptor, and all router / host sources
CONS = "fabric_exp/consolidation/"
named = set(re.findall(r"fabric_exp/[^\s;&|'\"]+", cmd)) | {CONS + w for w in re.findall(r"\$C/([^\s;&|'\"]+)", cmd)}
named |= {CONS + x for x in ("jobs_lib.sh", "cap_lib.sh", "vc1_lib.sh", "vc1_capture_lib.sh", "swap.py",
          "offline_build.py", "hang_capture.py", "hang_decode.py", "pc_sampler.py")} | {"fabric_exp/run_exp.sh"}
todo = [n for n in named if n.endswith((".sh", ".py"))]
while todo:
    f = os.path.join(src, todo.pop())
    if not os.path.isfile(f):
        continue
    for w in re.findall(r"(?:\$CON|\$C|\$\{CON\}|fabric_exp/consolidation)/([A-Za-z0-9_.-]+\.(?:sh|py))", open(f, errors="replace").read()):
        if CONS + w not in named:
            named.add(CONS + w); todo.append(CONS + w)
text = cmd + "".join(open(os.path.join(src, n), errors="replace").read() for n in named
                     if n.endswith(".sh") and n.startswith(CONS) and os.path.isfile(os.path.join(src, n)))
variants = {w for w in re.findall(r"[A-Za-z0-9_-][A-Za-z0-9_.-]*", text) if os.path.isdir(os.path.join(src, "fabric_exp/variants", w))}
def blocking(k):
    if any(k == n or k.startswith(n.rstrip("/") + "/") for n in named):
        return True
    m = re.match(r"fabric_exp/variants/([^/]+)/", k)
    if m:
        return m.group(1) in variants
    if k.endswith((".yaml", ".yml", ".textproto")):
        return lb2_pack.is_input(k)
    return not k.startswith("fabric_exp/")  # router and host sources, tests, top level
blk = [k for k in diff if blocking(k)]
print(f"stage {os.path.basename(st)}: {len(diff)} live files differ from it, {len(blk)} of them this job could read "
      f"({len(named)} scripts/files named or called, variants {sorted(variants) or 'none'})")
for k in blk[:25]:
    print(f"   {'new' if k not in ref else 'gone' if k not in now else 'changed'}: {k}")
sys.exit(3 if blk else 0)
EOF
r=$?
if [ $r = 3 ] && [ -z "$stale" ]; then echo "!!! stage is stale for this job: run lb2_stage.sh refresh, or --stale-ok"; exit 1; fi
[ $r = 0 ] || [ $r = 3 ] || { echo "!!! stage check failed"; exit 1; }

n=$(squeue -u $USER -h -o '%j' | grep -c '^lb2-')
[ $n -lt ${LB2_MAX_QUEUE:-6} ] || { echo "!!! $n lb2 jobs already running or pending"; exit 1; }
mkdir -p $G/cmds $G/logs
cf=$G/cmds/$(date -u +%Y%m%d-%H%M%S)-$$.cmd
printf '%s\n' "$cmd" > $cf
h=${tl%%:*}; m=${tl#*:}; m=${m%%:*}
to=$(( 10#$h * 3600 + 10#$m * 60 - 1200 )); [ $to -gt 600 ] || to=600
short=${node#bh-lb-120-}
if [ $node = any ]; then
    where=(-p $parts -N 1 --exclude=$excl)
    dep=""; [ -n "$after" ] && dep="--dependency=afterok:$after"
else
    where=(-p $part -w $node)
    dep=singleton; [ -n "$after" ] && dep="afterok:$after,singleton"; dep="--dependency=$dep"
fi
args=("${where[@]}" -J lb2-$short $dep -t $tl
    --export=ALL,LB2_MODE=$mode,LB2_CMD_FILE=$cf,LB2_TIMEOUT=$to,LB2_RESET=$reset,LB2_INPUTS="$inputs",LB2_MOVE_ASIDE=$aside)
if [ -n "$testonly" ]; then
    echo "   sbatch ${args[*]} $CON/lb2.sbatch"
    sbatch --test-only "${args[@]}" $CON/lb2.sbatch 2>&1 | sed 's/^/   test-only: /'
    hj=$(sbatch --parsable --hold "${args[@]}" $CON/lb2.sbatch) || { echo "!!! held sbatch rejected"; rm -f $cf; exit 1; }
    scontrol show job $hj | grep -oE '(JobState|Reason|Partition|ReqNodeList|ExcNodeList|NumNodes|Dependency)=[^ ]*' |
        sed "s/^/   held $hj: /"
    if [ "$(squeue -j $hj -h -o %T)" = PENDING ]; then scancel $hj && echo "   held $hj cancelled (never eligible)"; else
        echo "!!! held $hj is $(squeue -j $hj -h -o %T): not cancelled"; fi
    rm -f $cf; exit 0
fi
jid=$(sbatch --parsable "${args[@]}" $CON/lb2.sbatch) || { echo "!!! sbatch failed"; exit 1; }
echo "submitted $jid: lb2-$short ($mode, time $tl, command timeout ${to}s, stage $cur)"
echo "   log: $G/logs/lb2-$short-$jid.log   results: $G/jobs/$jid/   then: lb2_collect.sh $jid"
