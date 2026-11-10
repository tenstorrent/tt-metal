#!/bin/bash
# MK1 silicon checkpoint for the -mk-tail track (d27u08 dev0). usage: run_si.sh <tag> <NL> <smix_us> <sar_us> <mixn> [extra env]
# Pushes kernels + bench, runs with the device profiler under a timeout, pulls the profiler CSV to the laptop, prints zones.
set -u
export KUBECONFIG=~/.kube/yyz2-cloud-kubeconfig.yaml; NS=ttop-q38-test; TAG=$1; XENV=${6:-}
P=$(kubectl -n $NS get pod -l qwen38-dev-d27u08=true -o name | head -1); P=${P#pod/}
T=$(cd "$(dirname "$0")" && pwd)
kubectl -n $NS exec $P -c inference-server -- bash -c "pgrep -fa '[p]ytest|[s]im_mk1|[b]ench|[v]llm' | head -2; mkdir -p /tmp/r16/mk/kernels"
for f in $T/mk1/kernels/*; do kubectl -n $NS cp $f $P:/tmp/r16/mk/kernels/$(basename $f) -c inference-server; done
for f in sim_mk1.py zones.py profparse.py; do kubectl -n $NS cp $T/$f $P:/tmp/r16/mk/$f -c inference-server; done
kubectl -n $NS exec $P -c inference-server -- bash -c "
TM=/home/container_app_user/tt-metal; export PYTHONPATH=\$TM:\$TM/ttnn TT_METAL_HOME=\$TM ARCH_NAME=blackhole TT_VISIBLE_DEVICES=0
export TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_DIR=/tmp/r16/mk/prof_$TAG MK_KD=/tmp/r16/mk/kernels FM_NL=$2 MK_SMIX_US=$3 MK_SAR_US=$4 MK_MIXN=$5 FM_ITERS=2 $XENV
cd /tmp/r16/mk && sed -i 's#/work/mk1/kernels#/tmp/r16/mk/kernels#' sim_mk1.py
timeout 900 \$TM/python_env/bin/python sim_mk1.py > run_$TAG.log 2>&1; echo rc=\$?
grep -E '\[(mk1|fm11)\] (iter|MIXN)|PCC|pcc' run_$TAG.log | tail -6
CSV=\$(find /tmp/r16/mk/prof_$TAG \$TM/generated -name profile_log_device.csv -newer sim_mk1.py 2>/dev/null | head -1); echo csv=\$CSV; cp \$CSV /tmp/r16/mk/csv_$TAG.csv 2>/dev/null"
mkdir -p $T/csv; kubectl -n $NS cp $P:/tmp/r16/mk/csv_$TAG.csv $T/csv/$TAG.csv -c inference-server >/dev/null 2>&1 && echo "pulled $T/csv/$TAG.csv"
kubectl -n $NS cp $P:/tmp/r16/mk/run_$TAG.log $T/csv/run_$TAG.log -c inference-server >/dev/null 2>&1
