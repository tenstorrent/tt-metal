#!/bin/bash
cd /tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad/shtest
CON=/home/aagarwal/tt-metal/fabric_exp/consolidation
BK=/nonexistent-bk
out=results4
(while sleep 60; do :; done) & HB=$!
trap 'kill $HB 2>/dev/null; echo ORIGINAL-TRAP-RAN' EXIT
export FAKE_TTRUN=$PWD/fake_ttrun.py FAKE_COUNTER=$PWD/counter FAKE_TTX_SRC=/tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad/synth_cap PYTHONPATH=/tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad/fake_ttx
export VC1CAP_RUN_TIMEOUT=8 VC1CAP_CACHE_ROOT=$PWD/caches/c
source lib_test.sh
rm -f counter
vc1_capture_selftest t1 tree O3
echo "selftest rc=$?"
rm -f counter; export FAKE_HANG_AT=2
vc1_repeat_capture t2 tree t2lab O3 3
echo "repeat rc=$?"
echo "trap now: $(trap -p EXIT)"
