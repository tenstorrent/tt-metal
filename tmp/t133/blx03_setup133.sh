#!/bin/bash
# CPU-only setup on blx03 for the #133 NoC MID-skip A/B (no device access): two worktrees of the shared repo,
# A = t48 tip without #56023, B = the t133 branch tip (A + the #56023 cherry-pick + these scripts), each with a lean Release build (~2.5 GB each).
# Usage (from g15blx02): ssh g14blx03 'bash -s' < tmp/t133/blx03_setup133.sh
# Done when ~/fasth3/t133-setup.log ends with SETUP133_DONE rc=0.
# Remove afterwards: for w in t133a t133b; do git -C ~/fasth3/tt-metal worktree remove --force ~/fasth3/$w; done
set -u
BASE=/home/smarton/fasth3/tt-metal
BR=ttp/t133-cherry-pick-56023-bh-noc-mid-register-sk
REV_A=${REV_A:-c4409b1fa24f841bc84c207da5baa2f17e3257b1}
LOG=/home/smarton/fasth3/t133-setup.log; SH=/home/smarton/fasth3/t133-setup.sh
cat > $SH <<EOF
set -x
one() {  # \$1 worktree, \$2 rev
  local W=\$1 REV=\$(git -C $BASE rev-parse \$2) || return 11
  [ -d \$W ] || git -C $BASE worktree add --detach \$W \$REV || return 12
  git -C \$W checkout --detach \$REV || return 13
  for n in tracy umd tt-cluster-descriptors; do
    git -C \$W submodule update --init --depth 1 -- tt_metal/third_party/\$n || return 14
  done
  cd \$W || return 15
  CPM=; [ -d $BASE/.cpmcache ] && CPM="--cpm-source-cache $BASE/.cpmcache"
  bash build_metal.sh --release \$CPM || return 16
  test -f \$W/ttnn/ttnn/_ttnn.so || return 17
}
run() {
  git -C $BASE fetch origin $BR || return 10
  REV_B=\$(git -C $BASE rev-parse FETCH_HEAD)
  git -C $BASE merge-base --is-ancestor $REV_A \$REV_B || return 18
  one /home/smarton/fasth3/t133a $REV_A || return \$?
  one /home/smarton/fasth3/t133b \$REV_B || return \$?
  # The 4x8 test exists only on B; A runs the same file untracked.
  mkdir -p /home/smarton/fasth3/t133a/tmp/t133 &&
    cp /home/smarton/fasth3/t133b/tmp/t133/test_t133_4x8.py /home/smarton/fasth3/t133a/tmp/t133/ || return 19
}
run; echo "SETUP133_DONE rc=\$?"
EOF
setsid nohup bash $SH > $LOG 2>&1 < /dev/null &
echo "started; log $LOG"
