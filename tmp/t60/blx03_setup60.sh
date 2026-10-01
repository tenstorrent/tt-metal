#!/bin/bash
# CPU-only setup on blx03 for the #60 decode A/B (no device access): a worktree of the shared repo at
# the t60 commit, submodules from the shared repo's objects, and a Release build (~2.5 GB, a few min).
# Usage (from g15blx02): ssh g14blx03 'bash -s' < tmp/t60/blx03_setup60.sh   (returns at once; worktree at the
# pushed branch tip; for a given commit use 'bash -s -- <sha>')
# Another branch/worktree: ssh g14blx03 'BR=<branch> W=/home/smarton/fasth3/<name> bash -s' < tmp/t60/blx03_setup60.sh
# Done when ~/fasth3/<name>-setup.log ends with SETUP60_DONE rc=0 (<name> = basename of W, default t60).
# Remove afterwards: git -C ~/fasth3/tt-metal worktree remove --force ~/fasth3/<name>; rm ~/fasth3/<name>-setup.*
set -u
REV=${1:-FETCH_HEAD}
BASE=/home/smarton/fasth3/tt-metal; W=${W:-/home/smarton/fasth3/t60}
LOG=/home/smarton/fasth3/$(basename $W)-setup.log; SH=/home/smarton/fasth3/$(basename $W)-setup.sh
BR=${BR:-ttp/t60-fold-per-conv-pad-mask-mul-and-temporal-}
cat > $SH <<EOF
set -x
run() {
  git -C $BASE fetch origin $BR && git -C $BASE cat-file -e $REV^{commit} || return 11
  REV=\$(git -C $BASE rev-parse $REV)
  [ -d $W ] || git -C $BASE worktree add --detach $W \$REV || return 12
  git -C $W checkout --detach \$REV || return 13
  for n in tracy umd tt-cluster-descriptors; do
    git -C $W submodule update --init --depth 1 -- tt_metal/third_party/\$n || return 14
  done
  cd $W || return 15
  CPM=; [ -d $BASE/.cpmcache ] && CPM="--cpm-source-cache $BASE/.cpmcache"
  bash build_metal.sh --release \$CPM || return 16
  test -f $W/ttnn/ttnn/_ttnn.so || return 17
}
run; echo "SETUP60_DONE rc=\$?"
EOF
setsid nohup bash $SH > $LOG 2>&1 < /dev/null &
echo "started; log $LOG"
