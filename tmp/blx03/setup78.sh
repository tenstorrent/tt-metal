#!/bin/bash
# Off-device build of the t78 branch on blx03 into its own worktree ~/fasth3/t78 (the halo-only path changes the
# conv3d reader kernel + host validation, so the shared t48 build can't run it). Run detached on blx03:
#   setsid nohup bash tmp/blx03/setup78.sh > ~/fasth3/t78-setup.log 2>&1 &
# Done marker: "SETUP78_DONE rc=0" in ~/fasth3/t78-setup.log. Remove ~/fasth3/t78 (git worktree remove) after the A/B.
set -x
B=ttp/t78-conv3d-halo-only-input-for-ltx-vae-decod; R=/home/smarton/fasth3/tt-metal; W=/home/smarton/fasth3/t78
run() {
  free_g=$(df -BG --output=avail /home | tail -1 | tr -dc 0-9)
  [ "$free_g" -ge 150 ] || { echo "only ${free_g}G free on /home (<150G)"; return 10; }
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
run; echo "SETUP78_DONE rc=$?"
