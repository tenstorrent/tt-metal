#!/bin/bash
# Stage this branch's Python tree for run115.sh on blx03 (off-device, /var/tmp only). Run from g15blx02:
#   bash tmp/blx03/t115/stage114.sh [rev]
set -e
REV=$(git rev-parse --short "${1:-HEAD}")
S=/var/tmp/fasth3/t115/src
git archive --format=tar "$REV" models conftest.py pytest.ini tmp/blx03 |
  ssh g14blx03 "rm -rf $S && mkdir -p $S && tar -x -C $S && echo $REV > $S/REV && du -sh $S"
