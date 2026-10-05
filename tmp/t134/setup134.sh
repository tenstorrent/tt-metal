#!/bin/bash
# blx03 off-device setup for #134: worktree ~/fasth3/t134 at the pushed branch tip, lean Release build.
# Run detached on blx03; ~/fasth3/t134-setup.log ends with "SETUP134_DONE rc=<n>".
B=ttp/t134-cherry-pick-sdpa-chain-57979-58032-58223
R=/home/smarton/fasth3/tt-metal; W=/home/smarton/fasth3/t134
set -x
run() {
  git -C $R fetch origin $B && git -C $R cat-file -e FETCH_HEAD^{commit} || return 11
  REV=$(git -C $R rev-parse FETCH_HEAD)
  [ -d $W ] || git -C $R worktree add --detach $W $REV || return 12
  git -C $W checkout --detach $REV || return 13
  for n in tracy umd tt-cluster-descriptors; do
    git -C $W submodule update --init --depth 1 -- tt_metal/third_party/$n || return 14
  done
  cd $W || return 15
  CPM=; [ -d $R/.cpmcache ] && CPM="--cpm-source-cache $R/.cpmcache"
  bash build_metal.sh --release $CPM || return 16
  test -f $W/ttnn/ttnn/_ttnn.so || return 17
}
run; echo "SETUP134_DONE rc=$?"
