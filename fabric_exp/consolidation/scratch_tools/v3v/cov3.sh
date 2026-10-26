#!/bin/bash
S=/tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad/v3v
cd /home/aagarwal/tt-metal/fabric_exp/consolidation
for p in "v3v:v3vapi" "v3v2:v3v2c" "v3ve:v3vec" "v3vc:v3vcc" "v3vd:v3vdc" "v3v3:v3v3c"; do
  d=${p%%:*}; n=${p#*:}
  python3 coverage.py build --variant ../variants/$d --opt O3 --name $n > $S/cov_$n.log 2>&1 &
  sleep 2
done
wait
python3 coverage.py build --variant ../variants/v3v2 --opt O2 --name v3v2c --cases bh_2d_vc2 > $S/cov_v3v2c_O2.log 2>&1 &
python3 coverage.py build --variant ../variants/v3v2 --opt Os --name v3v2c --cases bh_2d_vc2 > $S/cov_v3v2c_Os.log 2>&1 &
wait
python3 coverage.py report v3tk v3vapi v3v2c v3vec v3vcc v3vdc v3v3c > $S/cov_report_r3.txt 2>&1
echo cov-done
