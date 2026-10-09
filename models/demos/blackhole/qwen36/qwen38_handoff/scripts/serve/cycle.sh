#!/bin/bash
# usage: cycle.sh C5 [ENV=VAL]   (run from anywhere)
S=/home/ttuser/atupe/qwen38_work/serve; N=$1; shift; D=$S/$N; mkdir -p $D
T1=/home/ttuser/atupe/qwen38_work/runs/T1
$T1/gate2.sh 360 > $D/gate.out 2>&1 || { echo GATE_FAIL > $D/status.txt; exit 1; }
WT=/home/ttuser/atupe/tt-metal/.claude/worktrees/qwen38-optimizations
cd /home/ttuser/atupe/tt-inference-server
date +%s > $D/launch_ts
env "$@" HF_TOKEN=dummy TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0 setsid python3 run.py --model Qwen/Qwen3.6-27B --tt-device p300x2 --workflow server --local-server --dev-mode \
  --tt-metal-home $WT --tt-metal-python-venv-dir /home/ttuser/atupe/python_env_vllm \
  --host-weights-dir /home/runara/models/Qwen3.6-27B --skip-system-sw-validation --no-auth </dev/null > $D/run.out 2>&1 &
echo $! > $D/server_pid
echo LAUNCHED > $D/status.txt
for i in $(seq 1 360); do
  curl -sf http://localhost:8000/v1/models >/dev/null 2>&1 && break
  if ! kill -0 $(cat $D/server_pid) 2>/dev/null; then SP=$(grep -o 'Created local server process PID: [0-9]*' $D/run.out | grep -o '[0-9]*$'); { [ -n "$SP" ] && kill -0 $SP 2>/dev/null; } || { grep -q 'rc=0' $D/run.out 2>/dev/null && [ -n "$SP" ] || { echo SERVER_DIED > $D/status.txt; exit 2; }; }; fi
  sleep 5
done
curl -sf http://localhost:8000/v1/models >/dev/null 2>&1 || { echo NOT_READY > $D/status.txt; exit 3; }
echo READY > $D/status.txt
grep -o 'Created local server process PID: [0-9]*' $D/run.out | grep -o '[0-9]*$' > $D/srv_pid; L=$(grep -o '/home/[^ ]*vllm_local_[^ ]*\.log' $D/run.out | head -1); echo $L > $D/server_log_path; ln -sf $L $D/server.log
cd $S
bash ./bench.sh $N > $D/bench_rc.txt 2>&1
bash ./bench5.sh $N >> $D/bench_rc.txt 2>&1
./correct.sh $D > $D/correct.txt
python3 sum.py $N > $D/sum.txt; python3 sum5.py $N R5 > $D/sum5.txt
echo BENCH_DONE > $D/status.txt
