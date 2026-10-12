#!/bin/bash
# t430 step 2 (CPU only, blx01, no device): assemble_clip_parts old (t423/b = b0f45278d72^ code) vs new
# (t427/b = b0f45278d72), unpinned, on cores 32-63 (outside tt-metal's round-robin dispatch threads 0-31),
# and on one core. Alternates old/new per setting.
set -eo pipefail
F=/var/tmp/fasth3; VENV=$F/t48/python_env
export HOME=$F/home XDG_CACHE_HOME=$F/home/.cache TMPDIR=$F/tmp
source $VENV/bin/activate
cd $F/t430
# Busy box (other builds) skews CPU timings: wait up to 30 min for the 1-min load to drop below 20.
for i in $(seq 180); do l=$(cut -d' ' -f1 /proc/loadavg); [ "${l%.*}" -lt 20 ] && break; sleep 10; done
echo "load at start: $(cat /proc/loadavg)"
for secs in 5 15; do
  for pin in "" "32-63" "40"; do
    for tree in t423 t427 t423 t427; do
      W=$F/$tree/b
      pre=(); [ -n "$pin" ] && pre=(taskset -c $pin)
      echo -n "pin=${pin:-none} "
      SECONDS_=$secs TT_METAL_HOME=$W PYTHONPATH=$W:$W/ttnn nice -n 5 "${pre[@]}" python -u bench_assemble.py $tree 2>&1 | grep ASSEMBLE
    done
  done
done
echo "load at end: $(cat /proc/loadavg)"
echo BENCH430_DONE
