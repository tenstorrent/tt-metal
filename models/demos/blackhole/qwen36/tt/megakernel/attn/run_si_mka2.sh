#!/bin/bash
# usage (inside a dev pod): run_si_mka2.sh <tag> [env...]   -- single chip via MK_CHIP=<N>, JIT kernels only, hang watchdog MK_HANG_S (default 900 s, no kill)
TAG=$1; shift
MK2=${MK2:-/tmp/mk2}
TM=/home/container_app_user/tt-metal
export TT_METAL_HOME=$TM PYTHONPATH=$TM:$TM/ttnn ARCH_NAME=blackhole TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_DIR=$MK2/prof/$TAG
export MKA_KD=$MK2/megakernel/attn/kernels MK_MIXN=${MK_MIXN:-1}
unset TT_METAL_SIMULATOR MESH_DEVICE TT_VISIBLE_DEVICES
for kv in "$@"; do export "$kv"; done
mkdir -p $TT_METAL_PROFILER_DIR; cd $MK2
[ -n "${MK_PCACHE:-}" ] && export TT_METAL_CACHE=$MK2/cache   # private JIT cache (per-run binary sizes)
# MK_CHIP=<N> (device policy 23:20Z): single-chip run on chip N under the SHARED host lock + an exclusive per-chip lock
if [ -n "${MK_CHIP:-}" ]; then LCK="flock -w 3600 -s /tmp/tt_device.lock flock -w 3600 -x /tmp/tt_chip${MK_CHIP}.lock env TT_VISIBLE_DEVICES=${MK_CHIP}"; else LCK="flock -w 1800 /tmp/tt_device.lock"; fi
# whole-host request (device rule, RELAUNCH_BRIEF): single-chip runs wait while a whole-host run has asked for the host;
# a whole-host run (no MK_CHIP) raises the request before its exclusive flock and drops it on exit (trap)
if [ -n "${MK_CHIP:-}" ]; then while [ -e /tmp/tt_host_exclusive.req ]; do sleep 5; done
else touch /tmp/tt_host_exclusive.req; trap 'rm -f /tmp/tt_host_exclusive.req' EXIT; fi
# No hard kills of persistent kernels (RELAUNCH_BRIEF, main 01:20Z): no `timeout` on the device process. A watchdog reports a hang
# after MK_HANG_S seconds of run time (the clock starts once the locks are held), marks the chip suspect and leaves the process
# running for a proper reset. A chip with a suspect marker is refused.
SUS=/tmp/tt_chip${MK_CHIP:-host}.suspect
if [ -e "$SUS" ]; then echo "REFUSED: $SUS exists (chip suspect after a hang; reset it first)"; exit 4; fi
rm -f $MK2/$TAG.started
s=$(date +%s); $LCK bash -c "touch $MK2/$TAG.started; exec $TM/python_env/bin/python $MK2/megakernel/attn/bench_mka2.py" > $MK2/$TAG.log 2>&1 &
PID=$!; t=0
while kill -0 $PID 2>/dev/null; do
  sleep 5; [ -f $MK2/$TAG.started ] || continue; t=$((t+5))
  if [ $t -ge ${MK_HANG_S:-${MK_TIMEOUT:-900}} ]; then
    echo "HANG after ${t}s: pid $PID left RUNNING (not killed); chip ${MK_CHIP:-host} marked suspect ($SUS). Report to main + the host's whole-host owner; log it in DEVICE_LOG.md."
    echo "$(date -u +%FT%TZ) $TAG pid $PID" > "$SUS"; exit 2
  fi
done
wait $PID; rc=$?
echo "rc=$rc wall=$(( $(date +%s)-s ))s"; grep -E "\[mka2\]|\[mk1\] iter|\[fm11\] iter|TT_THROW|TT_FATAL|Error|Timeout" $MK2/$TAG.log | head -24
CSV=$(find $TT_METAL_PROFILER_DIR -name profile_log_device.csv 2>/dev/null | head -1)
[ -n "$CSV" ] && $TM/python_env/bin/python $MK2/megakernel/attn/mka_prof.py "$CSV" | tail -24
