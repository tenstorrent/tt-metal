#!/bin/bash
# t315 (non-device): worktree + release build of ltx-rt f6547442b30 on blx03, for the standard LTX-2.3 8+3 e2e timing.
R=/home/smarton/fasth3/tt-metal; W=/home/smarton/fasth3/t315; V=/var/tmp/fasth3/t315
WANT=f6547442b304d744711e80e2281f6bd368291673
mkdir -p $V
run() {
  git -C $R fetch origin ltx-rt || return 11
  [ "$(git -C $R rev-parse FETCH_HEAD)" = $WANT ] || git -C $R cat-file -e $WANT^{commit} || return 12
  [ -d $W ] || git -C $R worktree add --detach $W $WANT || return 13
  git -C $W checkout --detach $WANT || return 14
  for n in tracy umd tt-cluster-descriptors; do
    git -C $W submodule update --init --depth 1 -- tt_metal/third_party/$n || return 15
  done
  cd $W || return 16
  source $R/python_env/bin/activate || return 17
  CPM=; [ -d $R/.cpmcache ] && CPM="--cpm-source-cache $R/.cpmcache"
  bash build_metal.sh --release $CPM || return 18
  test -f $W/ttnn/ttnn/_ttnn.so || return 19
}
run > $V/build.log 2>&1; rc=$?
echo "T315_DRIVER_DONE setup $rc $(date -u '+%F %T')" >> $V/driver.log; exit $rc
