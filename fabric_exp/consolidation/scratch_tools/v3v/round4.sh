#!/bin/bash
S=/tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad/v3v
cd $S
( ./fullcheck.sh v3v4 v4 > $S/check_v3v4.txt 2>&1 ) &
( ./fullcheck.sh v3v5 v5 > $S/check_v3v5.txt 2>&1 ) &
wait
cd /home/aagarwal/tt-metal/fabric_exp/consolidation
python3 coverage.py build --variant ../variants/v3v4 --opt O3 --name v3v4c > $S/cov_v3v4c.log 2>&1 &
python3 coverage.py build --variant ../variants/v3v5 --opt O3 --name v3v5c > $S/cov_v3v5c.log 2>&1 &
wait
python3 coverage.py build --variant ../variants/v3v4 --opt O2 --name v3v4c --cases bh_2d_vc2 > $S/cov_v3v4c_O2.log 2>&1 &
python3 coverage.py build --variant ../variants/v3v4 --opt Os --name v3v4c --cases bh_2d_vc2 > $S/cov_v3v4c_Os.log 2>&1 &
python3 coverage.py build --variant ../variants/v3v5 --opt O2 --name v3v5c --cases bh_2d_vc2 > $S/cov_v3v5c_O2.log 2>&1 &
python3 coverage.py build --variant ../variants/v3v5 --opt Os --name v3v5c --cases bh_2d_vc2 > $S/cov_v3v5c_Os.log 2>&1 &
wait
python3 coverage.py report v3tk v3v2c v3v4c v3v5c > $S/cov_report_r4.txt 2>&1
echo round4-done
