#!/bin/bash
# Silicon mixed-group checkpoint (from the laptop). usage: [XDEF=A=1,B=1] [XENV="K=V"] run_si_gdnext.sh <pod> <tag> <NL> <AMASK> [ctx]
# Pushes the megakernel dir to the pod's /tmp/mk2/megakernel, runs bench_mka2 on chip 0, prints PCCs + per-layer µs.
set -u
export KUBECONFIG=~/.kube/yyz2-cloud-kubeconfig.yaml; NS=ttop-q38-test; P=$1; TAG=$2; NLV=$3; AM=$4; CTXV=${5:-32768}
D=$(cd "$(dirname "$0")/.." && pwd)
tar -C $(dirname $D) -czf /tmp/mk_mixed_push.tgz megakernel
kubectl -n $NS exec $P -c inference-server -- mkdir -p /tmp/mk2
kubectl -n $NS cp /tmp/mk_mixed_push.tgz $P:/tmp/mk2/mk.tgz -c inference-server
kubectl -n $NS exec $P -c inference-server -- bash -c "cd /tmp/mk2 && rm -rf megakernel && tar xzf mk.tgz && \
  bash megakernel/attn/run_si_mka2.sh $TAG FM_NL=$NLV MK_AMASK=$AM MK_SGDN_US=64 MK_SAR_US=18 MKA_SWO_US=10 MKA_CTX=$CTXV \
  MKA_KVDT=bfp8 MKA_NBB=1 MKA_BUDGET=16384 FM_ITERS=2 MK_DEFINES=MKA_GUPRE=1${XDEF:+,$XDEF} ${XENV:-}; \
  CSV=\$(find /tmp/mk2/prof/$TAG -name profile_log_device.csv | head -1); \
  /home/container_app_user/tt-metal/python_env/bin/python megakernel/attn/mka2_layers.py \$CSV | head -14"
kubectl -n $NS cp $P:/tmp/mk2/prof/$TAG ~/Documents/tt-orchestration/.claude/tmp/r16/mega/si_$TAG -c inference-server >/dev/null 2>&1; echo "csv copied to .claude/tmp/r16/mega/si_$TAG"
