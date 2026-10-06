#!/bin/bash
# #151 on blx03: apply the t149 conv3d vol2col_rm straddle guard (aa4ade43a28, C++ only) to the existing
# ~/fasth3/t48 build as a detached commit, then rebuild it incrementally. Marker: BUILD151_DONE rc=<n>.
# guard.diff = `git diff c4409b1fa24 aa4ade43a28 -- ttnn/` (c4409b1fa24 is the t48 build HEAD; C++ differs only by the guard).
B=/home/smarton/fasth3/t48; L=/var/tmp/fasth3/t151/build.log
mkdir -p /var/tmp/fasth3/t151
run() {
  cd $B || return 11
  if ! git log -1 --format=%s | grep -q 'vol2col_rm'; then
    git apply ~/fasth3/t151drv/guard.diff || return 12
    git -c user.name='Steve Marton' -c user.email=smarton@tenstorrent.com commit -q -a \
      -m 'conv3d: reject unaligned blocks over 64 patches that overrun vol2col_rm' \
      -m 'C++ part of ttp/t149 aa4ade43a28, for the #151 device check.' || return 13
  fi
  git log --oneline -2
  bash build_metal.sh --release --cpm-source-cache /home/smarton/fasth3/tt-metal/.cpmcache || return 16
  test -f $B/ttnn/ttnn/_ttnn.so || return 17
}
run >> $L 2>&1; echo "BUILD151_DONE rc=$?" >> $L
