#!/bin/bash
S=/tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad/v3v
cd $S
( ./diagcheck.sh v3vc_d20 v3vcK > check_v3vc_d20.txt 2>&1 ) &
( ./diagcheck.sh v3vc_d50 v3vcK > check_v3vc_d50.txt 2>&1 ) &
wait
( ./diagcheck.sh v3vc_d100 v3vcK > check_v3vc_d100.txt 2>&1 ) &
( ./diagcheck.sh v3tk_d50 v3tk > check_v3tk_d50.txt 2>&1 ) &
wait
echo round5-done
