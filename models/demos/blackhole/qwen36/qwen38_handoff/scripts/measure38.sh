#!/bin/bash
# usage: [SUBSET="perf1 acc ..."] measure38.sh LABEL   (flags via env, e.g. QWEN36_BFP4_GDN_IN=1)
LABEL=$1
SUBSET=${SUBSET:-"perf1 perf2 acc acc1536 t4k b8"}
T3=/home/ttuser/atupe/qwen38_work/runs/T3
MAIN=/home/ttuser/atupe/tt-metal; NEW=$MAIN/.claude/worktrees/qwen38-optimizations
OUTD=$T3/$LABEL; mkdir -p $OUTD
cd $NEW
source $T3/env38.sh || exit 5
env | grep -E '^QWEN' | sort > $OUTD/flags.txt
$MAIN/python_env/bin/python -c "import models.demos.blackhole.qwen36.tt.tp_common as t; print('MODELS_FROM', t.__file__)" > $OUTD/models_from.txt 2>&1
if ! grep -q "MODELS_FROM $NEW/" $OUTD/models_from.txt; then echo "MODELS_FROM not under NEW: $(cat $OUTD/models_from.txt)" >&2; exit 2; fi
busy() { if lsof /dev/tenstorrent/* 2>/dev/null | grep -q .; then echo "DEVICE BUSY before $1" | tee -a $OUTD/exits.txt; return 0; fi; return 1; }
run() { name=$1; shift
  case " $SUBSET " in *" $name "*) ;; *) return 0;; esac
  /home/ttuser/atupe/qwen38_work/runs/T1/gate2.sh 360 >> $OUTD/gate.log 2>&1 || exit 4
  busy $name && exit 3
  s=$(date +%s)
  $MAIN/python_env/bin/python -m pytest --timeout 3600 models/demos/blackhole/qwen36/demo/text_demo.py -k "$*" > $OUTD/$name.log 2>&1
  echo "$name EXIT=$? secs=$(( $(date +%s)-s ))" >> $OUTD/exits.txt; }
run perf1 "traced_128 and not traced_128k"
run perf2 "traced_128 and not traced_128k"
run acc "accuracy_512"
run acc1536 "accuracy_1536"
run t4k "traced_4k"
run b8 "batched_128_b8"
