#!/bin/bash
# Waits (detached, up to 2 h) for the blx01 fast5 driver marker, then scores fast5 on g15 with score_g15.sh.
D=/home/smarton/fasth3/tt-metal/tt-project/data/g15/t170
for i in $(seq 240); do
  timeout 30 ssh blx01 test -e /var/tmp/fasth3/t170/driver_fast/DRIVER.done && break
  sleep 30
done
timeout 30 ssh blx01 cat /var/tmp/fasth3/t170/driver_fast/DRIVER.done > $D/fast_driver.done 2>&1 || { echo "1 no driver marker" > $D/SCORE.done; exit 1; }
grep -q '^0 ' $D/fast_driver.done || { echo "2 driver: $(cat $D/fast_driver.done)" > $D/SCORE.done; exit 2; }
exec bash /home/smarton/fasth3/tt-metal/tt-project/worktrees/t170/tmp/t170/score_g15.sh fast5
