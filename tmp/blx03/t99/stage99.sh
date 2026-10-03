#!/bin/bash
# Stage the Python tree for run99.sh on blx03 (off-device, /var/tmp only). Run from g15blx02:
#   bash tmp/blx03/t99/stage99.sh [rev]   (rev = t99 + the t96 trace harness, branch ttp/t99-stage)
set -e
REV=$(git rev-parse --short "${1:-HEAD}")
S=/var/tmp/fasth3/t99/src
git archive --format=tar "$REV" models conftest.py pytest.ini tmp/blx03 \
  tests/ttnn/nightly/unit_tests/operations/experimental/test_dit_rms_norm_unary_fused.py |
  ssh g14blx03 "rm -rf $S && mkdir -p $S && tar -x -C $S && echo $REV > $S/REV && du -sh $S"
