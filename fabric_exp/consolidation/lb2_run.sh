#!/bin/bash
# Second-LB harness (LB2_HARNESS.md): submit one study command to an extra P150x8 LB through Slurm (lb2.sbatch).
#
#   lb2_run.sh -n a07u28|a08u28|a04u24 [options] "<command>"     the command runs in /home/aagarwal/tt-metal on the node,
#                                                         exactly as a broker job's command would on a09u32
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
node="" tl=03:00:00 mode=run reset=1 inputs="" aside="" stale="" after=""
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
        *) break ;;
    esac
done
cmd=${1:-}
[ "$cmd" = --hostcheck ] && { mode=hostcheck; cmd=""; }
case $node in
    a08u28|bh-lb-120-a08u28) node=bh-lb-120-a08u28; part=bh_lb_C67 ;;
    a07u28|bh-lb-120-a07u28) node=bh-lb-120-a07u28; part=bh_lb_C45 ;;
    a04u24|bh-lb-120-a04u24) node=bh-lb-120-a04u24; part=bh_lb_120_A45 ;;  # user, 2026-11-03
    *) echo "usage: lb2_run.sh -n a07u28|a08u28|a04u24 [options] \"<command>\"  (only these nodes are approved)"; exit 1 ;;
esac
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
dep=singleton; [ -n "$after" ] && dep="afterok:$after,singleton"
jid=$(sbatch --parsable -p $part -w $node -J lb2-$short --dependency=$dep -t $tl \
    --export=ALL,LB2_MODE=$mode,LB2_CMD_FILE=$cf,LB2_TIMEOUT=$to,LB2_RESET=$reset,LB2_INPUTS="$inputs",LB2_MOVE_ASIDE=$aside \
    $CON/lb2.sbatch) || { echo "!!! sbatch failed"; exit 1; }
echo "submitted $jid: lb2-$short ($mode, time $tl, command timeout ${to}s, stage $cur)"
echo "   log: $G/logs/lb2-$short-$jid.log   results: $G/jobs/$jid/   then: lb2_collect.sh $jid"
