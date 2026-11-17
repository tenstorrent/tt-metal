#!/bin/bash
# product13 gate chain: runs on .44 against the server on .45 (tag a1)
cd /mnt/tt-data/ssinghal/hy4-logs/program/product13; H=10.82.97.45; TAG=${1:-a1}
S=../state/product13.status.md; st() { echo "$(date -u +%FT%TZ) $*" >> $S; }
. ../serve-long/evals2/env.sh
PP=../product-integ/probes/product_probe.py
until curl -s -m 5 http://$H:8000/health >/dev/null; do sleep 20; if grep -q "RC=" logs/server_$TAG.log; then st BOOT_FAILED; echo BOOT_FAILED; exit 1; fi; done
st "server up"; mkdir -p ev; echo 12288 > hybrid_cap; [ -f pfx_file ] || echo 1 > pfx_file
for i in 1 2 3; do python3 $PP $H paris > ev/paris$i.out 2>&1; python3 $PP $H tool > ev/tool$i.out 2>&1; done; st "paris/tool done"
python3 $PP $H itl 1000 1 1024 > ev/itl_u1.out 2>&1; python3 $PP $H itl 1000 4 1024 > ev/itl_u4.out 2>&1; st "itl done: $(tail -n 2 ev/itl_u1.out | tr '\n' ' ')"
NTOK=1500 REPS=3 python3 val10.py $H short ev/short1500.json > ev/short1500.out 2>&1
NTOK=6000 REPS=2 python3 val10.py $H short ev/short6000.json > ev/short6000.out 2>&1; st "ttft short done"
python3 burst.py $H 4000 3 > ev/burst4k.out 2>&1; st "burst4k: $(tail -n1 ev/burst4k.out)"
python3 burst.py $H 6000 3 > ev/burst6k.out 2>&1; st "burst6k: $(tail -n1 ev/burst6k.out)"
export HY4_GATE_GSM8K=0.9577; O=$PWD/ev; LIMIT=50 bash ../serve-long/evals2/gsm8k_remote_lim.sh $H $O 4 > $O/gsm8k.log 2>&1
python3 /mnt/tt-data/ssinghal/wt/hy4_ttis/scripts/hy4_agentx/gsm8k_rescore.py $O/gsm8k/lm_out --json-out $O/gsm8k/rescore.json > $O/rescore.log 2>&1; st "gsm done: $(tail -n 3 $O/rescore.log | tr '\n' ' ')"
python3 val10.py $H story ev/story.json 500 0 > ev/story.out 2>&1
PROMPT_BASE=4 python3 val10.py $H story ev/story2.json 500 0 > ev/story2.out 2>&1; st "story done"
python3 val10.py $H gpqa ev/gpqa.json 9000 0.9 > ev/gpqa.out 2>&1; GPQA_OFF=4 python3 val10.py $H gpqa ev/gpqa2.json 9000 0.9 > ev/gpqa2.out 2>&1; st "gpqa 8q done (prefix cache on)"
python3 multiturn.py $H 12 190 > ev/multiturn.out 2>&1; st "multiturn: $(tail -n1 ev/multiturn.out)"
python3 toolflow.py $H 6 > ev/toolflow.out 2>&1; st "toolflow: $(tail -n1 ev/toolflow.out)"
./soak13.sh $H; st "soak done: $(tr '\n' ' ' < logs/soak.log)"
python3 burst.py $H 4400 2 > ev/burst44k.out 2>&1; st "burst4.4k: $(tail -n1 ev/burst44k.out)"
k=0; for n in 64000 64000 384000 384000; do k=$((k+1)); python3 $PP $H needle $n 0.5 > ev/needle_${n}_$k.out 2>&1; st "needle $n run$k: $(tail -n1 ev/needle_${n}_$k.out)"; done
python3 $PP $H itl 1000 1 1024 > ev/itl_u1_end.out 2>&1; python3 $PP $H itl 1000 4 1024 > ev/itl_u4_end.out 2>&1; st "itl end done"
st CHAIN_DONE; echo CHAIN_DONE
