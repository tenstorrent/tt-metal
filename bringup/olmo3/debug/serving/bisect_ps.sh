#!/bin/bash
# for each OLMO_PS_SKIP setting: fresh server, measure doubled-word glitches on the device-sampled path
cd /home/ubuntu/ssinghal/olmo3
source tt-metal/python_env/bin/activate
for skip in "$@"; do
  ps -eo pid,args | awk '/run.py --model allenai|run_vllm_api_server|EngineCore/ && !/awk/ {print $1}' | xargs -r kill -9
  sleep 3; tt-smi -glx_reset > /dev/null 2>&1
  (OLMO_PS_SKIP=$skip OLMO_PREFILL_SAMPLING_CCL_RESET=0 nohup ./run_ttis.sh > ~/ssinghal/olmo3-logs/ttis_bis_$skip.log 2>&1 &)
  sleep 2; ./scratch/wait_server.sh > /dev/null || { echo "$skip: server failed"; continue; }
  echo "SKIP=$skip: $(python scratch/glitch_rate.py 96 300 2>&1 | head -1)"
done
echo bisect-done
