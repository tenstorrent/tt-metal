#!/bin/bash
cd /localdev/wransom/tt-metal
CSV="${CB_CSV:-/localdev/wransom/tt-metal/conv_bench_data.csv}"
if [ ! -f "$CSV" ]; then echo "timestamp,model,label,batch,in_ch,out_ch,H,W,filter,stride,shard,abh,fp32_accum,bias,mode,per_core_M,per_core_N,sbm,trm,used,median_ns,min_ns,max_ns,n,spread_pct,pcc,status" > "$CSV"; fi
REPS="$1"; shift; MODES="$*"
for MODE in $MODES; do
  durs=""
  for r in $(seq 1 "$REPS"); do
    env TT_CONV_BENCH_MODE="$MODE" TT_CONV_BENCH_FORCE_TRM="$( [ "$MODE" = helper_trm ] && echo 1 )" timeout 500 python -m tracy -r -p -v -m pytest \
      tests/ttnn/unit_tests/operations/conv/test_conv_bench.py > /tmp/cb_last.log 2>&1
    d=$(python /tmp/cb_warm_conv.py 2>/dev/null); durs="$durs $d"
  done
  used=$(grep -oE "USING out_subblock=[0-9]+x[0-9]+" /tmp/cb_last.log | head -1 | grep -oE "[0-9]+x[0-9]+")
  sbm=$(grep -oE "SubblockMajor=[0-9]+x[0-9]+" /tmp/cb_last.log | head -1 | grep -oE "[0-9]+x[0-9]+")
   trm=$(grep -oE "trm_pin=true" /tmp/cb_last.log | head -1)
  pM=$(grep -oE "per_core_M=[0-9]+" /tmp/cb_last.log | head -1 | grep -oE "[0-9]+")
  pN=$(grep -oE "per_core_N=[0-9]+" /tmp/cb_last.log | head -1 | grep -oE "[0-9]+")
  pcc=$(grep -iE "PCC = " /tmp/cb_last.log | grep -oE "[0-9]\.[0-9]{4,}" | head -1)
  if grep -q "1 passed" /tmp/cb_last.log; then st=ok; elif grep -qiE "beyond max L1|Out of Memory" /tmp/cb_last.log; then st=OOM; elif grep -qiE "completion reader queue is not empty" /tmp/cb_last.log; then st=HANG; elif grep -qiE "TT_FATAL" /tmp/cb_last.log; then st=FATAL; else st=FAIL; fi
  MODE="$MODE" durs="$durs" used="$used" sbm="$sbm" trm="$trm" pM="$pM" pN="$pN" pcc="$pcc" st="$st" CSV="$CSV" python - <<'PY'
import os, statistics, csv, datetime
durs=[float(x) for x in os.environ["durs"].split() if x and x.replace('.','',1).isdigit()]
med=mn=mx=spread=""; n=len(durs)
if durs:
    med=f"{statistics.median(durs):.0f}"; mn=f"{min(durs):.0f}"; mx=f"{max(durs):.0f}"; spread=f"{100*(max(durs)-min(durs))/statistics.median(durs):.1f}"
row=[datetime.datetime.now().isoformat(timespec='seconds'),os.environ.get("MODEL",""),os.environ.get("LABEL",""),os.environ.get("CB_BATCH","1"),os.environ.get("CB_IN_CH",""),os.environ.get("CB_OUT_CH",""),os.environ.get("CB_H",""),os.environ.get("CB_W",""),os.environ.get("CB_FILTER",""),os.environ.get("CB_STRIDE",""),os.environ.get("CB_SHARD",""),os.environ.get("CB_ABH",""),os.environ.get("CB_FP32_ACCUM",""),os.environ.get("CB_BIAS",""),os.environ["MODE"],os.environ["pM"],os.environ["pN"],os.environ["sbm"],os.environ["trm"],os.environ["used"],med,mn,mx,n,spread,os.environ["pcc"],os.environ["st"]]
with open(os.environ["CSV"],"a",newline="") as f: csv.writer(f).writerow(row)
print(f"SUMMARY | {os.environ.get('LABEL',''):28s} | {os.environ['MODE']:11s} | used={os.environ['used']} sbm/trm={os.environ['sbm']}/{os.environ['trm']} | median={med}ns spread={spread}% n={n} | pcc={os.environ['pcc']} | {os.environ['st']}")
PY
done
