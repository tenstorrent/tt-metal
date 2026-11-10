#!/bin/bash
# Silicon checkpoint of MK1 on a dev pod (device 0 only). usage (from laptop): run_si_mk1.sh <pod> <tag> <NL> <smix_us> <sar_us> <mixn>
# Copies kernels + bench to the pod's /tmp/r16/mk, runs with the device profiler, prints PCC and kernel/zone times.
set -u
export KUBECONFIG=~/.kube/yyz2-cloud-kubeconfig.yaml; NS=ttop-q38-test; P=$1; TAG=$2
W=$(cd "$(dirname "$0")" && pwd)
kubectl -n $NS exec $P -c inference-server -- mkdir -p /tmp/r16/mk/kernels
for f in $W/mk1/kernels/*; do kubectl -n $NS cp $f $P:/tmp/r16/mk/kernels/$(basename $f) -c inference-server; done
for f in sim_mk1.py zones.py profparse.py; do kubectl -n $NS cp $W/$f $P:/tmp/r16/mk/$f -c inference-server; done
kubectl -n $NS exec $P -c inference-server -- bash -c "
TM=/home/container_app_user/tt-metal; export PYTHONPATH=\$TM:\$TM/ttnn TT_METAL_HOME=\$TM ARCH_NAME=blackhole TT_VISIBLE_DEVICES=0
export TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_DIR=/tmp/r16/mk/prof_$TAG MK_KD=/tmp/r16/mk/kernels FM_NL=$3 MK_SMIX_US=$4 MK_SAR_US=$5 MK_MIXN=$6 FM_ITERS=2
cd /tmp/r16/mk && sed -i 's#/work/mk1/kernels#/tmp/r16/mk/kernels#' sim_mk1.py
timeout 1800 \$TM/python_env/bin/python sim_mk1.py > run_$TAG.log 2>&1; echo rc=\$?
grep -E '\[(mk1|fm11)\] (iter|MIXN)' run_$TAG.log
CSV=\$(find /tmp/r16/mk/prof_$TAG \$TM/generated -name profile_log_device.csv -newer sim_mk1.py 2>/dev/null | head -1); echo csv=\$CSV
python3 profparse.py \$CSV | tail -2; python3 zones.py \$CSV \$(python3 profparse.py \$CSV | tail -1 | awk '{print \$2}') | grep -E '#1'"
