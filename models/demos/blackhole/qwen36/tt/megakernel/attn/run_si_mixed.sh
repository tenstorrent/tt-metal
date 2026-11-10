#!/bin/bash
# Silicon mixed-group checkpoint (from the laptop). usage: run_si_mixed.sh <pod> <tag> <NL> <AMASK> [ctx] [extra KEY=VAL ...]
# Pushes the megakernel dir to the pod's /tmp/mk2/megakernel, runs bench_mka2 on chip 0, prints PCCs + per-layer µs.
set -u
export KUBECONFIG=~/.kube/yyz2-cloud-kubeconfig.yaml; NS=ttop-q38-test; P=$1; TAG=$2; NLV=$3; AM=$4; CTXV=${5:-32768}; shift 5 2>/dev/null; EXTRA="$*"
D=$(cd "$(dirname "$0")/.." && pwd)
TGZ=$(mktemp -t mkpush.XXXXXX); trap 'rm -f "$TGZ"' EXIT   # unique per run: concurrent agents shared a fixed /tmp path
RD=/tmp/mk2r/$TAG.$(date +%s).$$   # unique pod run dir
STAMP="$(git -C $D rev-parse --short HEAD) dirty=$(git -C $D diff HEAD -- . | md5 -q | cut -c1-8)"; echo "pushed-tree: $STAMP"
tar -C $(dirname $D) -czf "$TGZ" megakernel
LH=$(cd $D && find attn/kernels attn/bench_mka2.py -type f | LC_ALL=C sort | xargs cat | md5 -q)
kubectl -n $NS exec $P -c inference-server -- mkdir -p $RD
kubectl -n $NS cp "$TGZ" $P:$RD/mk.tgz -c inference-server
kubectl -n $NS exec $P -c inference-server -- bash -c "cd $RD && tar xzf mk.tgz && echo $RD > /tmp/mk2r/$TAG.dir && echo \"$STAMP\" > $RD/PUSH_STAMP && \
  RH=\$(cd megakernel && find attn/kernels attn/bench_mka2.py -type f | LC_ALL=C sort | xargs cat | md5sum | cut -d' ' -f1) && \
  if [ \"\$RH\" != \"$LH\" ]; then echo TREE MISMATCH local $LH pod \$RH; exit 3; fi; echo TREE $LH; \
  MK2=$RD bash megakernel/attn/run_si_mka2.sh $TAG FM_NL=$NLV MK_AMASK=$AM MK_SGDN_US=64 MK_SAR_US=18 MKA_SWO_US=10 MKA_CTX=$CTXV \
  MKA_KVDT=bfp8 MKA_NBB=1 MKA_BUDGET=16384 FM_ITERS=2 MKA_EXT=1 MK_DEFINES=MKA_GUPRE=1,MK_DNPIPE=1,MK_FLAG_FLUSH=1,MKA_MIXTRID=1 $EXTRA; \
  CSV=\$(find $RD/prof/$TAG -name profile_log_device.csv | head -1); \
  /home/container_app_user/tt-metal/python_env/bin/python megakernel/attn/mka2_layers.py \$CSV | head -14; grep -E '\\[(mk1|mka2|fm11)\\] (iter|defines)' $RD/$TAG.log | head -12"
kubectl -n $NS cp $P:$RD/prof/$TAG ~/Documents/tt-orchestration/.claude/tmp/r16/mega/si_$TAG -c inference-server >/dev/null 2>&1; echo "csv copied to .claude/tmp/r16/mega/si_$TAG"
