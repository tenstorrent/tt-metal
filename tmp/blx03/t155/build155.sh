#!/bin/bash
# #155 runner job on blx03 (no device use): apply the t152 vol2col_rm sizing fix (be371d08a9f, C++ only)
# to ~/fasth3/t48 as a detached commit and rebuild it incrementally. Runs through the runner so it
# serializes with every other project job that uses the t48 tree. Marker: BUILD155_DONE rc=<n>.
B=/home/smarton/fasth3/t48; V=/var/tmp/fasth3/t155; L=$V/build.log
F=ttnn/cpp/ttnn/operations/experimental/conv3d/device/conv3d_program_factory.cpp
run() {
  cd $B || return 11
  echo "[t155] $(date -u '+%F %T') before: $(git log -1 --format='%H %s')"
  git status --short | grep -v '^??' && return 14
  if ! grep -q 'Unaligned blocks get exactly num_patches pages' $F; then
    git apply ~/fasth3/t155/fix.diff || return 12
    git -c user.name='Steve Marton' -c user.email=smarton@tenstorrent.com commit -q -a \
      -m 'conv3d: size vol2col_rm at num_patches pages for unaligned blocks' \
      -m 'C++ part of ttp/t152 be371d08a9f, for the #155 device check.' || return 13
  fi
  echo "[t155] after: $(git log -1 --format='%H %s')"
  bash build_metal.sh --release --cpm-source-cache /home/smarton/fasth3/tt-metal/.cpmcache || return 16
  test -f $B/ttnn/ttnn/_ttnn.so || return 17
  echo "[t155] $(date -u '+%F %T') build ok"
}
mkdir -p $V; run >> $L 2>&1; rc=$?; echo "BUILD155_DONE rc=$rc" >> $L; exit $rc
