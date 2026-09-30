exec 9>/data/kmabee/runs_sasha/queue.lock; flock 9
source /data/kmabee/runs_sasha/env.sh; waitchips
cd $W && echo "=== clk probe $(date +%T) sha=$(git rev-parse --short HEAD) $(tdp)"
for M in 1024 256; do
  env TT_METAL_HOME=$W PYTHONPATH=$W/ttnn:$W $PY $O/mm_sweep/mm_loop.py --m $M --seconds 40 > $O/mm_sweep/loop_$M.log 2>&1 &
  P=$!
  until grep -q LOOP_START $O/mm_sweep/loop_$M.log 2>/dev/null || ! kill -0 $P 2>/dev/null; do sleep 1; done
  sleep 8
  for i in 1 2 3 4 5 6; do tt-smi -s 2>/dev/null | python3 -c "
import json,sys; d=json.load(sys.stdin)
t=[x['telemetry'] for x in d['device_info']]; b=max(t,key=lambda x: float(x['power']))
print('M=$M busiest chip aiclk',b['aiclk'].strip(),'power',b['power'].strip(),'W temp',b['asic_temperature'],'| idle-chip aiclk',sorted(float(x['aiclk']) for x in t)[0])"; sleep 3; done
  wait $P; grep LOOP_DONE $O/mm_sweep/loop_$M.log
done
echo "=== clk done $(date +%T)"
