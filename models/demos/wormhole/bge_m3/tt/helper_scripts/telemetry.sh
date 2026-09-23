#!/bin/sh
# Log tt-smi telemetry every ~1 s until killed: time voltage current power aiclk temp
while true; do
  tt-smi -s 2>/dev/null | python3 -c "import json,sys,time; t=json.load(sys.stdin)['device_info'][0]['telemetry']; print('%.1f'%time.time(), *(t[k].strip() for k in ('voltage','current','power','aiclk','asic_temperature')), flush=True)" 2>/dev/null
  sleep 0.5
done
