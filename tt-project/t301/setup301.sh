#!/bin/bash
# t301 host setup on blx01 (no device): lean release build of ttp/ltx23-main-pr (#194 draft PR head) under
# /var/tmp/fasth3/t301/pr. The main arm reuses t293's clean origin/main build (/var/tmp/fasth3/t293/main @ 80b1cd689d0).
# Writes $D/setup.rc when done (0 = ok). Run detached (setsid nohup).
set -eo pipefail
F=/var/tmp/fasth3; D=$F/t301; R=$F/t48; PR=df9e5ecaac6
trap 'echo $? > $D/setup.rc' EXIT
mkdir -p $D; rm -f $D/setup.rc
echo "[setup] $(date -u '+%F %T') UTC df /: $(df -h / | tail -1)"
git -C $R fetch -q origin ttp/ltx23-main-pr
if [ ! -e $D/pr/ttnn/ttnn/_ttnn.so ]; then
  [ -d $D/pr ] || git -C $R worktree add -q --detach $D/pr "$PR"
  cd $D/pr
  [ "$(git rev-parse --short=11 HEAD)" = $PR ]
  git submodule update --init --recursive -q
  source $R/python_env/bin/activate
  ./build_metal.sh --release
fi
test -e $D/pr/ttnn/ttnn/_ttnn.so
git -C $D/pr rev-parse --short=11 HEAD
du -sh $D/pr
echo "[setup] done $(date -u '+%F %T') UTC df /: $(df -h / | tail -1)"
