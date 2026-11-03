#!/bin/bash
# Second-LB harness (LB2_HARNESS.md): snapshot of the live study tree on bh-lb-120-a09u32 into /data, which lb2.sbatch
# syncs onto each extra LB node at /home/aagarwal/tt-metal. Read-only on the live tree and its caches.
#
#   lb2_stage.sh refresh   (default) the source scope: top-level files, tt_metal, ttnn, tt_stl, tests, tools, scripts,
#                          third_party, fabric_exp (no results, consolidation/builds, variants/*/elf). Every dispatch
#                          whose inputs changed (a new variant, script or yaml) needs one.
#   lb2_stage.sh full      also build_Release, python_env, runtime (sfpi), the uv Python that python_env links to, and
#                          .git (kept on /data for the node's git status check; never copied to a node). Needed once,
#                          and again only after a host-library rebuild on a09u32 (refresh warns when build_Release drifted).
#
# Layout (/data/aagarwal/lb2/stage): src-<time>/ {src.tar.zst, REF_src.sha256, GIT, SCOPE, FULL, STAGE_TIME, CHANGED},
# full-<time>/ {build_Release, python_env, runtime, uv-python}.tar.zst + REF_<x>.sha256 + FULL_STAMP, git/ (rsync of
# .git), and `current` -> the newest src-<time> (switched only after every check passed; the two newest src-* kept).
# NFS creates small files slowly (~10/s on 2026-11-03), so each scope is one zstd tarball; the node extracts it to local
# disk and rsyncs it into its tree, which prints the files and bytes that changed there.
#
# A broker job may have a router variant swapped into the live tree (swap.py, backup ~/.tt-metal-p1-swap-backup). The
# snapshot is the unswapped tree: files in the active swap's manifest come from the backup, files the swap created are
# left out; a pack is repeated when a swap was applied or restored during it. Checks: each archive read back against
# its reference list (written from the same bytes), and the live tree hashed again against it.
set -uo pipefail
SRC=/home/aagarwal/tt-metal
ST=${LB2_STAGE:-/data/aagarwal/lb2/stage}
BK=/home/aagarwal/.tt-metal-p1-swap-backup
UVP=/home/aagarwal/.local/share/uv/python
UVPY=cpython-3.10.19-linux-x86_64-gnu
CON=$SRC/fabric_exp/consolidation
PK="python3 -B $CON/lb2_pack.py"
MODE=${1:-refresh}
case $MODE in refresh|full) ;; *) echo "usage: lb2_stage.sh [refresh|full]"; exit 1 ;; esac
[ "$(hostname)" = bh-lb-120-a09u32 ] || { echo "!!! run on bh-lb-120-a09u32 (the live tree's host)"; exit 1; }
SRC_DIRS="tt_metal ttnn tt_stl tests tools scripts third_party fabric_exp"
FULL_DIRS="build_Release python_env runtime"
TS=$(date -u +%Y%m%d-%H%M%S)
t0=$SECONDS
mkdir -p $ST
prev=$(readlink $ST/current 2>/dev/null)
if [ $MODE = refresh ]; then
    if [ -n "$prev" ]; then FULLD=$(cat $ST/$prev/FULL)
    else  # no src stage yet: the newest complete full stage (an interrupted first run)
        FULLD=$(ls -1d $ST/full-*/FULL_STAMP 2>/dev/null | sort | tail -1 | xargs -r dirname | xargs -r basename)
        [ -n "$FULLD" ] || { echo "!!! no stage yet: run lb2_stage.sh full first"; exit 1; }
    fi
else
    FULLD=full-$TS
fi
S=$ST/src-$TS
mkdir -p $S $ST/$FULLD
trap '[ -L $ST/current ] && [ "$(readlink $ST/current)" = src-$TS ] || rm -rf "${S:?}"' EXIT   # a failed run leaves no src dir
echo "== lb2_stage $MODE $(date -u +%FT%TZ): $SRC -> $S (full: $FULLD)"

swap_state() { { stat -c %Y $BK 2>/dev/null; cat $BK/manifest.json 2>/dev/null; } | sha256sum | cut -c1-16; }
quiet() {  # run "$@" until no swap apply/restore happened during it (6 tries)
    local try s0 s1
    for try in 1 2 3 4 5 6; do
        s0=$(swap_state); "$@" || return 1; s1=$(swap_state)
        [ "$s0" = "$s1" ] && return 0
        echo "   a swap was applied or restored meanwhile: again"
    done
    echo "!!! no quiet pass in 6 tries"; return 1
}

if [ $MODE = full ]; then
    F=$ST/$FULLD
    for d in $FULL_DIRS; do
        $PK pack $SRC $F/$d.tar.zst $F/REF_$d.sha256 $d || exit 1
        $PK check $F/$d.tar.zst $F/REF_$d.sha256 || exit 1
    done
    $PK pack $UVP $F/uv-python.tar.zst $F/REF_uv-python.sha256 $UVPY || exit 1
    $PK check $F/uv-python.tar.zst $F/REF_uv-python.sha256 || exit 1
    sha256sum $SRC/build_Release/lib/*.so* $SRC/build_Release/tt_metal/*.so* \
        $SRC/build_Release/test/tt_metal/tt_fabric/test_infra/test_tt_fabric 2>/dev/null | sed "s#$SRC/##" > $F/FULL_STAMP
    mkdir -p $ST/git
    rsync -a --delete --stats $SRC/.git/ $ST/git/ | awk '/Number of regular files transferred/ {n=$NF}
        /Total transferred file size/ {b=$5} END {gsub(",", "", n); gsub(",", "", b); printf "   .git: %s files, %.1f MB moved\n", n, b/1e6}'
fi

# source scope: top-level files + SRC_DIRS, unswapped
(cd $SRC && find . -maxdepth 1 \( -type f -o -type l \) -printf '%P\n' | sort) > $S/toplevel
quiet $PK pack $SRC $S/src.tar.zst $S/REF_src.sha256 --backup $BK --toplevel $S/toplevel $SRC_DIRS || exit 1
$PK check $S/src.tar.zst $S/REF_src.sha256 || exit 1
quiet $PK live $SRC $S/REF_src.sha256 --backup $BK --toplevel $S/toplevel $SRC_DIRS ||
    { echo "!!! a live file changed after the pack (an agent edit?): run lb2_stage.sh again"; exit 1; }
# what changed since the previous stage (files whose content differs, added or removed)
if [ -n "$prev" ] && [ -f $ST/$prev/REF_src.sha256 ]; then
    python3 - $ST/$prev/REF_src.sha256 $S/REF_src.sha256 $SRC <<'PYEOF' | tee $S/CHANGED
import os, sys
def load(p):
    return {l.rstrip("\n").split("  ", 1)[1]: l.split("  ", 1)[0] for l in open(p)}
a, b = load(sys.argv[1]), load(sys.argv[2])
ch = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
nb = sum(os.path.getsize(os.path.join(sys.argv[3], k)) for k in ch if k in b and os.path.exists(os.path.join(sys.argv[3], k)))
print(f"   changed since the previous stage: {len(ch)} files ({sum(k in b and k not in a for k in ch)} new, "
      f"{sum(k in a and k not in b for k in ch)} removed), {nb / 1e6:.2f} MB")
for k in ch[:40]:
    print(f"     {k}")
if len(ch) > 40:
    print(f"     ... {len(ch) - 40} more")
PYEOF
fi
# git status lines and per-file diff hashes over the source scope (submodules ignored: their .git files point into
# the parent's .git/modules and their content is in the sha256 list; fabric_exp/results not staged; the active swap's
# files left out). lb2.sbatch takes the same summary of the node copy with the staged .git and compares (gitcmp).
python3 -c "import json,sys; m=json.load(open(sys.argv[1])); print('\\n'.join(m['saved']+m['created']))" $BK/manifest.json \
    > $S/GIT_EXCLUDES 2>/dev/null || : > $S/GIT_EXCLUDES
xa=(); while read -r p; do [ -n "$p" ] && xa+=(--exclude "$p"); done < $S/GIT_EXCLUDES
$PK gitsum $SRC/.git $SRC $S/GIT.json "${xa[@]}" $SRC_DIRS || exit 1
echo "HEAD=$(git -C $SRC rev-parse HEAD)" > $S/GIT
printf '%s\n' "SRC_DIRS=\"$SRC_DIRS\"" "FULL_DIRS=\"$FULL_DIRS\"" > $S/SCOPE
echo $FULLD > $S/FULL
date -u +%FT%TZ > $S/STAGE_TIME
(cd $SRC && sha256sum -c --quiet $ST/$FULLD/FULL_STAMP > /dev/null 2>&1) ||
    echo "!!! build_Release on a09u32 changed since $FULLD (host-library rebuild?): run lb2_stage.sh full"
ln -sfn src-$TS $ST/current.new && mv -T $ST/current.new $ST/current
for old in $(ls -1d $ST/src-* | sort | head -n -2); do rm -rf "${old:?}"; done
for old in $(ls -1d $ST/full-* | sort); do
    grep -qx "$(basename $old)" $ST/src-*/FULL 2>/dev/null || rm -rf "${old:?}"
done
echo "== current -> src-$TS ($(du -sh $S | cut -f1)); done in $((SECONDS - t0)) s"
