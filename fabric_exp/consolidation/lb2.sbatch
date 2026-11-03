#!/bin/bash
# Second-LB harness (LB2_HARNESS.md): one study command on an extra Blackhole P150x8 LoudBox, through Slurm. Submit with
# lb2_run.sh (it sets the partition, the node and the LB2_* settings); never sbatch this directly.
#
# Steps, in order (any failure stops the job before the device is touched):
#   guard    the node must be bh-lb-120-a07u28, a08u28 or a04u24 (user, 2026-11-03); never a09u32
#   busy     other users' device processes (ps) or a running job in the node's own tt-device broker: stop
#   space    /home free space; our oldest node-local JIT caches pruned below LB2_MIN_FREE_GB (default 25)
#   restore  a leftover swap in the node-local tree (swap.py restore ~/.tt-metal-p1-swap-backup)
#   sync     /data/aagarwal/lb2/stage -> /home/aagarwal/tt-metal (sources every job; build_Release, python_env, runtime
#            and the uv Python when the stage's full stamp changed), then sha256 against the stage's reference lists
#            (= the live a09u32 tree, unswapped view) and git status / diff checksums against the live tree's
#   preflight test binaries, sfpi, python modules, MPI, tt-smi; every fabric_exp path, script and variant named in
#            LB2_CMD (plus LB2_INPUTS) present in the node tree
#   reset    tt-smi -r in this allocation (LB2_RESET=0 skips; host checks), then the link check: every Ethernet core
#            that was UP / PASS / 400G in the node's host-check baseline still is
#   run      LB2_CMD in the node tree, in its own session, SLURM_* removed from its environment (mpirun would read
#            the 1-CPU Slurm allocation), TT_FABRIC_NO_BW_TELEMETRY=1, output streamed, timeout LB2_TIMEOUT
#   hang     past LB2_TIMEOUT: the session is killed (TERM, then KILL), the router state captured (hang_capture.py)
#            into the job's results; after any hang (this timeout, or a study script's own *_hangcap) the devices are
#            reset and the links checked again
#   end      swap.py restore; new results, saved builds and the router ELFs/headers of the caches the job wrote copied
#            to /data/aagarwal/lb2/jobs/<job>/ with MANIFEST.tsv (lb2_collect.sh then copies them into a09u32's tree)
# Modes (LB2_MODE): run (default); hostcheck (read-only host check, lb2_hostcheck.sh, no reset); dry (everything up to
# and including preflight, then prints what it would run; no device access).
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --job-name=lb2
#SBATCH --output=/data/aagarwal/lb2/logs/%x-%j.log
#SBATCH --chdir=/data/aagarwal/lb2

set -uo pipefail
G=/data/aagarwal/lb2
ST=$G/stage
T=/home/aagarwal/tt-metal
BK=/home/aagarwal/.tt-metal-p1-swap-backup
UVDST=/home/aagarwal/.local/share/uv/python
NODE=${SLURMD_NODENAME:-$(hostname)}
JOB=${SLURM_JOB_ID:-manual}
J=$G/jobs/$JOB
MODE=${LB2_MODE:-run}
CMD=${LB2_CMD:-$( [ -n "${LB2_CMD_FILE:-}" ] && cat "$LB2_CMD_FILE" )}
TO=${LB2_TIMEOUT:-5400}
t0=$SECONDS
say() { echo "[lb2 $(date -u +%T) +$((SECONDS - t0))s] $*"; }
die() { say "!!! $*"; echo "rc=1 stage=${STAGE_NAME:-?} reason=$*" > $J/SUMMARY 2>/dev/null; exit 1; }
STAGE_NAME=guard

# ---- guard
case "$NODE" in
    bh-lb-120-a07u28|bh-lb-120-a08u28|bh-lb-120-a04u24) ;;
    *) echo "!!! node $NODE is not an approved extra LB (bh-lb-120-a07u28, a08u28, a04u24): stopping"; exit 1 ;;
esac
[ "$(hostname)" = "$NODE" ] || { echo "!!! hostname $(hostname) != SLURMD_NODENAME $NODE"; exit 1; }
case $MODE in run|hostcheck|dry) ;; *) echo "!!! LB2_MODE $MODE"; exit 1 ;; esac
mkdir -p $J/results || exit 1
say "node=$NODE job=$JOB mode=$MODE start=$(date -u +%FT%TZ) timeout=${TO}s"
[ $MODE = hostcheck ] || [ -n "$CMD" ] || die "no LB2_CMD"
unset $(env | grep -oE '^TT_[A-Za-z0-9_]*' | tr '\n' ' ')   # a clean study environment, as the broker's jobs get
[ -n "$CMD" ] && { say "command: $CMD"; printf '%s\n' "$CMD" > $J/COMMAND; }
CUR=$(readlink $ST/current)                    # resolved once: a refresh during the job does not affect it
S=$ST/$CUR
[ -n "$CUR" ] && [ -f $S/src.tar.zst ] && [ -f $S/REF_src.sha256 ] && [ -f $S/FULL ] || die "no complete stage in $ST"
FULLD=$(cat $S/FULL); F=$ST/$FULLD
source $S/SCOPE                                # SRC_DIRS, FULL_DIRS
UVPY=cpython-3.10.19-linux-x86_64-gnu
say "stage $CUR (full $FULLD), staged $(cat $S/STAGE_TIME)"
NB=$G/nodes/$NODE
mkdir -p $NB

# ---- busy: other users' processes that could hold the device; the node's broker queue
STAGE_NAME=busy
busy=$(ps -eo user:20,uid,comm,args --no-headers | awk -v me="$USER" '$1 != me && $1 != "root" && $2 >= 1000 && $2 < 60000 &&
    $3 ~ /^(python|pytest|test_|unit_tests|tt-smi|tt_|ttnn|prte|prted|mpirun|orted|run_|tt-train|fabric_)/' | cut -c1-200 | head -8)
if [ -n "$busy" ] && [ -z "${LB2_IGNORE_BUSY:-}" ]; then
    echo "$busy"; die "other users' processes that may hold the device"
fi
bs=$(timeout 30 /usr/local/bin/tt-device-mcp status 5 2>&1)
echo "$bs" | sed -n '/^RUNNING/,/^RECENT/p' | grep -v '^RECENT' | sed 's/^/   broker | /'
if echo "$bs" | sed -n '/^RUNNING/,/^$/p' | grep -qE '^ +[0-9]{3,} '; then
    [ -z "${LB2_IGNORE_BUSY:-}" ] && die "the node's tt-device broker is running a job"
fi
say "busy check: clear"

# ---- space
STAGE_NAME=space
free_gb() { df -BG --output=avail /home | tail -1 | tr -dc 0-9; }
min=${LB2_MIN_FREE_GB:-25}
while [ "$(free_gb)" -lt $min ]; do
    old=$(ls -1dtr /home/aagarwal/.tt-metal-cache-* 2>/dev/null | head -1)
    [ -n "$old" ] || break
    say "space: $(free_gb) GB free < $min: removing our oldest cache $old"
    rm -rf "${old:?}"
done
[ "$(free_gb)" -ge $min ] || die "/home has $(free_gb) GB free, below $min, and no cache of ours is left to prune"
say "space: $(free_gb) GB free on /home"

# ---- restore a leftover swap, then sync from the stage
STAGE_NAME=sync
if [ -f $BK/manifest.json ] && [ -f $T/fabric_exp/consolidation/swap.py ]; then
    (cd $T && python3 fabric_exp/consolidation/swap.py restore $BK) || die "leftover swap could not be restored"
fi
# an lb2 tree carries .lb2_owned (written when the harness creates it), .lb2_full or .lb2_node
if [ -e $T ] && [ ! -f $T/.lb2_owned ] && [ ! -f $T/.lb2_full ] && [ ! -f $T/.lb2_node ]; then
    [ -n "${LB2_MOVE_ASIDE:-}" ] || die "$T exists and is not an lb2 tree; LB2_MOVE_ASIDE=1 moves it to $T.pre-lb2-<date>"
    mv $T $T.pre-lb2-$(date +%Y%m%d) || die "move-aside failed"
    say "moved the existing $T aside to $T.pre-lb2-$(date +%Y%m%d) (kept, not deleted)"
fi
[ -e $T ] || { mkdir -p $T && echo "created by lb2 job $JOB $(date -u +%FT%TZ)" > $T/.lb2_owned; }
mkdir -p $BK $UVDST
IN=/home/aagarwal/.lb2-incoming
rm -rf "${IN:?}"; mkdir -p $IN/src $IN/full $IN/uv
stats() { awk -v l="$1" '/Number of regular files transferred/ {gsub(",", "", $NF); n=$NF}
    /Total transferred file size/ {gsub(",", "", $5); b=$5} /Number of deleted files/ {gsub(",", "", $5); d=$5}
    END {printf "   sync %s: %d files, %.1f MB moved, %d deleted\n", l, n, b/1e6, d}'; }
# node-local outputs (results, saved builds, generated CSVs) are excluded, so --delete never touches them
# (the directories themselves too: the stage has no results dir, so --delete would otherwise try to remove them)
EXCL=(--exclude=/fabric_exp/results/ --exclude=/fabric_exp/consolidation/builds/ --exclude=/fabric_exp/variants/*/elf/
      --exclude=__pycache__/ --exclude=*.pyc)
bad=0
zstd -q -dc $S/src.tar.zst | tar -x -C $IN/src || die "src.tar.zst did not extract"
rsync -a --stats --files-from=$S/toplevel $IN/src/ $T/ | stats top-level || bad=1
for d in $SRC_DIRS; do rsync -a --delete --stats "${EXCL[@]}" $IN/src/$d $T/ | stats $d || bad=1; done
if [ "$(cat $T/.lb2_full 2>/dev/null)" != "$FULLD" ] || [ -n "${LB2_FULL_SYNC:-}" ]; then
    for d in $FULL_DIRS; do
        zstd -q -dc $F/$d.tar.zst | tar -x -C $IN/full || die "$d.tar.zst did not extract"
        rsync -a --delete --stats --exclude=__pycache__/ --exclude=*.pyc $IN/full/$d $T/ | stats $d || bad=1
        rm -rf "${IN:?}/full/$d"
    done
    zstd -q -dc $F/uv-python.tar.zst | tar -x -C $IN/uv || die "uv-python.tar.zst did not extract"
    rsync -a --delete --stats --exclude=__pycache__/ --exclude=*.pyc $IN/uv/$UVPY $UVDST/ | stats uv-python || bad=1
fi
rm -rf "${IN:?}"
[ $bad = 0 ] || die "an rsync from the stage failed"
mkdir -p $T/generated/fabric $T/fabric_exp/results $T/fabric_exp/consolidation/builds
vf() {  # dir ref
    (cd $1 && sha256sum -c --quiet $2 > $J/verify_$(basename $2 .sha256).txt 2>&1) ||
        { head -20 $J/verify_$(basename $2 .sha256).txt; die "node copy differs from the stage reference $(basename $2)"; }
    say "verify: $(wc -l < $2) files of $(basename $2) match (live a09u32 tree, unswapped view, at $(cat $S/STAGE_TIME))"
}
vf $T $S/REF_src.sha256
for d in $FULL_DIRS; do vf $T $F/REF_$d.sha256; done
vf $UVDST $F/REF_uv-python.sha256
echo $FULLD > $T/.lb2_full
source $S/GIT     # HEAD of the live tree at stage time
xa=(); while read -r p; do [ -n "$p" ] && xa+=(--exclude "$p"); done < $S/GIT_EXCLUDES
python3 -B $T/fabric_exp/consolidation/lb2_pack.py gitsum $ST/git $T $J/git_node.json "${xa[@]}" $SRC_DIRS > /dev/null ||
    die "git status on the node copy failed"
(cd $T && python3 -B fabric_exp/consolidation/lb2_pack.py gitcmp $S/GIT.json $J/git_node.json) ||
    die "git status/diff of the node copy differ from the live tree's (HEAD ${HEAD:0:11}) in input files"
say "git: HEAD ${HEAD:0:11}; status and per-file diff of the node copy = the live tree's at stage time"
echo "$CUR $(date -u +%FT%TZ)" > $T/.lb2_node

# ---- preflight
STAGE_NAME=preflight
cd $T || die "no $T"
miss=""
for f in build/test/tt_metal/tt_fabric/test_infra/test_tt_fabric build/test/tt_metal/tt_fabric/fabric_unit_tests \
    runtime/sfpi/compiler/bin/riscv-tt-elf-g++ python_env/bin/python ttnn/ttnn/distributed/ttrun.py \
    fabric_exp/consolidation/jobs_lib.sh fabric_exp/consolidation/cap_lib.sh fabric_exp/consolidation/swap.py \
    fabric_exp/consolidation/hang_capture.py fabric_exp/read_eth_speed.py /usr/local/bin/tt-smi /usr/local/bin/mpirun-ulfm; do
    [ -e "$f" ] || miss="$miss $f"
done
ldd build/test/tt_metal/tt_fabric/test_infra/test_tt_fabric | grep -q "not found" && miss="$miss (test_tt_fabric: unresolved libraries)"
python_env/bin/python -c "import tt_umd, ttexalens, yaml, click, loguru, pydantic" 2>/dev/null || miss="$miss (python_env modules)"
# every fabric_exp path, $C/<script> and variant name in the command, plus LB2_INPUTS
for w in $(printf '%s\n' "$CMD" | tr -s " \t;&|()'\"=" '\n' | sort -u); do
    case $w in
        fabric_exp/*) [ -e "$w" ] || miss="$miss $w" ;;
        \$C/*|\${C}/*) f=fabric_exp/consolidation/${w#*/}; [ -e "$f" ] || miss="$miss $f" ;;
        [A-Za-z0-9_-]*) grep -qF "  fabric_exp/variants/$w/" $S/REF_src.sha256 && { [ -d fabric_exp/variants/$w ] || miss="$miss variants/$w"; } ;;
    esac
done
for f in ${LB2_INPUTS:-}; do [ -e "$f" ] || miss="$miss $f"; done
[ -z "$miss" ] || die "preflight: missing$miss"
for w in $(printf '%s\n' "$CMD" | grep -oE '(fabric_exp/[^ ;&|]*|\$C/[^ ;&|]*)\.sh' | sort -u); do
    f=$w; case $w in \$C/*) f=fabric_exp/consolidation/${w#*/} ;; esac
    bash -n "$f" || die "preflight: $f does not parse"
done
vs=$(printf '%s\n' "$CMD" | tr -s " \t;&|()'\"=" '\n' | sort -u | while read w; do [[ $w =~ ^[A-Za-z0-9_-] ]] && [ -d fabric_exp/variants/$w ] && echo $w; done | tr '\n' ' ')
say "preflight: binaries, sfpi, python modules, MPI and tt-smi present; command inputs present (variants: ${vs:-none})"
ncpu=$(nproc)
say "cpus: nproc $ncpu, $(grep Cpus_allowed_list /proc/self/status | tr -s '\t' ' '), memory $(free -g | awk '/Mem:/ {print $7}') GB available"
[ $ncpu -ge 8 ] || [ -n "${LB2_ALLOW_FEW_CPUS:-}" ] || die "only $ncpu CPUs in this allocation (JIT builds and MPI ranks would be starved)"
# the study commands run as on a09u32's broker: outside Slurm (mpirun would otherwise read the 1-CPU allocation)
unset $(env | grep -oE '^(SLURM|SBATCH|SRUN|PMI|PMIX|OMPI)_[A-Za-z0-9_]*' | tr '\n' ' ')

# ---- the node's host-check baseline (link states) and the link check
eth_check() {  # label: read_eth_speed.py now vs the baseline; returns 1 when a link that was UP/PASS/400 is not
    local now=$J/eth_$1.txt
    timeout 300 python_env/bin/python -I fabric_exp/read_eth_speed.py 8 > $now 2>&1 || { tail -5 $now; return 1; }
    [ -f $NB/eth_baseline.txt ] || { say "link check $1: no baseline yet ($NB/eth_baseline.txt)"; return 1; }
    python3 - $NB/eth_baseline.txt $now <<'EOF'
import re, sys
def load(p):
    d = {}
    for l in open(p):
        m = re.match(r"chip (\d+) eth \(\s*(\d+),\s*(\d+)\): port_status=(\d+) train_status=(\d+) train_speed=(\d+)", l)
        if m:
            d[m.group(1, 2, 3)] = m.group(4, 5, 6)
    return d
b, n = load(sys.argv[1]), load(sys.argv[2])
good = [k for k, v in b.items() if v == ("1", "2", "400")]
lost = [k for k in good if n.get(k) != ("1", "2", "400")]
print(f"   link check: {len(n)} cores read, {len(good)} UP/PASS/400 in the baseline, {len(lost)} of them not now")
for k in lost:
    print(f"   !!! chip {k[0]} eth ({k[1]},{k[2]}): baseline {b[k]}, now {n.get(k)}")
sys.exit(1 if lost or len(n) != len(b) else 0)
EOF
}
reset_and_check() {  # label: tt-smi -r, then the link check; links retrain after a reset, so up to 8 tries 15 s apart
    local try
    say "reset ($1): tt-smi -r"
    timeout 600 /usr/local/bin/tt-smi -r > $J/reset_$1.txt 2>&1; local rc=$?
    tail -3 $J/reset_$1.txt | sed 's/^/   | /'
    [ $rc = 0 ] || { say "!!! tt-smi -r exit $rc"; return 1; }
    for try in 1 2 3 4 5 6 7 8; do
        eth_check $1 > $J/eth_$1_check.txt 2>&1 && { cat $J/eth_$1_check.txt; return 0; }
        sleep 15
    done
    cat $J/eth_$1_check.txt; return 1
}

# ---- hostcheck / dry
if [ $MODE = hostcheck ]; then
    STAGE_NAME=hostcheck
    bash fabric_exp/consolidation/lb2_hostcheck.sh $NODE $J 2>&1; rc=$?
    say "host check exit $rc"
    echo "rc=$rc stage=hostcheck" > $J/SUMMARY
    exit $rc
fi
if [ $MODE = dry ]; then
    say "DRY: would reset (tt-smi -r: ${LB2_RESET:-1}), check links against $NB/eth_baseline.txt ($([ -f $NB/eth_baseline.txt ] && echo present || echo 'not yet'))"
    say "DRY: would run in $T with SLURM_* removed, TT_FABRIC_NO_BW_TELEMETRY=1, timeout ${TO}s: $CMD"
    echo "rc=0 stage=dry" > $J/SUMMARY
    exit 0
fi

# ---- reset + link check
STAGE_NAME=reset
if [ "${LB2_RESET:-1}" = 1 ]; then
    reset_and_check start || die "reset or link check failed before the command"
else
    eth_check start || die "link check failed before the command"
fi

# ---- run
STAGE_NAME=run
M=$J/.start_marker; touch $M; sleep 1
log=$J/command.log
say "run: $CMD"
setsid bash -c "export TT_FABRIC_NO_BW_TELEMETRY=1; cd $T && $CMD" > >(stdbuf -oL tee $log) 2>&1 < /dev/null &
root=$!
sid=$root
hung=0
while kill -0 $root 2>/dev/null; do
    sleep 5
    if [ $((SECONDS - t0)) -ge $TO ]; then hung=1; break; fi
done
if [ $hung = 1 ]; then
    say "!!! timeout ${TO}s: killing session $sid"
    pkill -TERM -s $sid; sleep 15; pkill -KILL -s $sid; sleep 5
    say "capture -> results/lb2_${JOB}_hangcap"
    timeout 300 python_env/bin/python fabric_exp/consolidation/hang_capture.py fabric_exp/results/lb2_${JOB}_hangcap \
        --time-budget 240 --meta job=$JOB --meta node=$NODE 2>&1 | tail -5 | sed 's/^/   | /'
    rc=124
else
    wait $root; rc=$?
fi
say "command exit $rc"
caps=$(find fabric_exp/results -maxdepth 1 -name '*_hangcap' -newer $M 2>/dev/null | tr '\n' ' ')
[ -n "$caps" ] && { hung=1; say "hang captures written: $caps"; }

# ---- end: restore, reset after a hang, copy back
STAGE_NAME=end
python3 fabric_exp/consolidation/swap.py restore $BK || say "!!! swap restore failed"
post=""
if [ $hung = 1 ]; then
    if reset_and_check posthang; then post="reset+links OK after the hang"; else post="RESET OR LINK CHECK FAILED after the hang"; fi
    say "$post"
fi
(cd fabric_exp/results && find . -type f -newer $M -printf '%P\n' | sort) > $J/.newres
rsync -a --files-from=$J/.newres fabric_exp/results/ $J/results/
# other new files under fabric_exp (a script writing outside results): kept, since the next sync would delete them
(cd fabric_exp && find . -type f -newer $M ! -path './results/*' ! -path './consolidation/builds/*' ! -name '*.pyc' -printf '%P\n' | sort) > $J/.newother
[ -s $J/.newother ] && { mkdir -p $J/other; rsync -a --files-from=$J/.newother fabric_exp/ $J/other/; }
nb=$(cd fabric_exp/consolidation/builds && find . -mindepth 1 -maxdepth 1 -newer $M -printf '%P\n' | wc -l)
(cd fabric_exp/consolidation/builds && find . -mindepth 1 -maxdepth 1 -newer $M -printf '%P\n') > $J/.newbuilds
[ $nb -gt 0 ] && { mkdir -p $J/builds; rsync -a --files-from=$J/.newbuilds -r fabric_exp/consolidation/builds/ $J/builds/; }
nc=0
for c in $(find /home/aagarwal -maxdepth 1 -name '.tt-metal-cache-*' 2>/dev/null); do
    n=$(basename $c); n=${n#.}   # visible name on /data
    (cd $c && find . -path '*/kernels/fabric_erisc_router_v2/*' \( -name '*.elf' -o -name '*.h' -o -name '*.hpp' \) -newer $M -print) > $J/.cl
    [ -s $J/.cl ] || continue
    mkdir -p $J/caches/$n && rsync -a --files-from=$J/.cl $c/ $J/caches/$n/ && nc=$((nc + 1))
done
(cd $J && find results builds caches other -type f 2>/dev/null | sort | while read f; do
    printf '%s\t%s\t%s\t%s\t%s\n' "$f" "$(stat -c %s "$f")" "$(sha256sum "$f" | cut -c1-16)" "$NODE" "$JOB"; done) > $J/MANIFEST.tsv
say "copied back: $(wc -l < $J/.newres) result files, $(wc -l < $J/.newother) other fabric_exp files, $nb builds, router ELFs/headers of $nc caches -> $J (MANIFEST.tsv)"
printf 'rc=%s hung=%s node=%s job=%s post=%s start=%s end=%s\n' $rc $hung $NODE $JOB "${post:-none}" \
    "$(stat -c %y $M | cut -c1-19)" "$(date -u +%FT%TZ)" > $J/SUMMARY
say "done: rc $rc, hung $hung; $(free_gb) GB free on /home"
exit $rc
