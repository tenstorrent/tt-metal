#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Blackhole chip telemetry from sysfs (tt-kmd; opens no device, so it runs beside a server): a CSV row every PERIOD s
# with the unix time, then per chip its AICLK (MHz), and per chip ASIC temperature (C), power (W) and current (A).
# A sustained decode can drop one chip's AICLK (see serve/README.md); this shows which chip and when.
#
# usage: chip_telemetry.sh OUT.csv [PERIOD_S]   (runs until killed)
OUT=${1:?OUT.csv}
PERIOD=${2:-0.5}
H=$(for h in /sys/class/hwmon/hwmon*; do [ "$(cat $h/name)" = blackhole ] && echo $h; done)
C=$(ls -d /sys/class/tenstorrent/tenstorrent!*)
{ printf "t"; for d in $C; do i=${d##*!}; printf ",aiclk%s" $i; done
  for h in $H; do p=$(basename $(readlink $h/device)); printf ",temp_%s,power_%s,curr_%s" $p $p $p; done; echo; } > $OUT
while true; do
  line=$(date +%s.%N | cut -c1-14)
  for d in $C; do line+=",$(cat $d/tt_aiclk)"; done
  for h in $H; do line+=",$(( $(cat $h/temp1_input) / 1000 )),$(( $(cat $h/power1_input) / 1000000 )),$(( $(cat $h/curr1_input) / 1000 ))"; done
  echo "$line" >> $OUT
  sleep $PERIOD
done
