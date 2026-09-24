#!/bin/bash
# sustained_run.sh <bs> <chip> <iters> <tag> "<ENV>" : one e2e run with tt-smi sampling; prints cold best, sustained median (last half) and clock/power
BS=$1; CHIP=$2; IT=$3; TAG=$4; ENV_=$5; S=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$S/../../../../.." && pwd)
cd "$REPO"; export TT_METAL_HOME=$PWD PYTHONPATH=$PWD HF_MODEL=perplexity-ai/pplx-embed-v1-4b MESH_DEVICE=P150
D=/tmp/smi_$TAG; rm -rf $D; mkdir -p $D
( while [ ! -f $D/STOP ]; do ts=$(date +%s.%N); env -u TT_VISIBLE_DEVICES timeout 20 ./python_env/bin/tt-smi -s --snapshot_no_tty > $D/$ts.json 2>/dev/null; sleep 0.3; done ) &
SP=$!
env TT_VISIBLE_DEVICES=$CHIP $ENV_ timeout 2400 ./python_env/bin/python $S/e2e_run_fp.py $BS $IT > /tmp/sus_${TAG}.log 2>&1
sleep 2; touch $D/STOP; wait $SP 2>/dev/null
./python_env/bin/python - "$D" "$CHIP" "/tmp/sus_${TAG}.log" "$TAG" <<'PY'
import json,glob,sys,re,os,statistics,datetime
D,dev,logf,tag=sys.argv[1],int(sys.argv[2]),sys.argv[3],sys.argv[4]
its=[float(v) for v in re.findall(r"Iteration \d+: ([0-9.]+)ms", open(logf,'rb').read().decode('utf-8','ignore'))]
tsl=[datetime.datetime.strptime(m,"%Y-%m-%d %H:%M:%S.%f").timestamp() for m in re.findall(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}).*Iteration \d+:", open(logf,'rb').read().decode('utf-8','ignore'))]
rows=[]
for f in sorted(glob.glob(D+"/*.json")):
    raw=open(f).read(); i=raw.find("{")
    if i<0: continue
    try: d=json.loads(raw[i:])
    except Exception: continue
    di=d.get("device_info") or []
    if len(di)<=dev: continue
    t=di[dev].get("telemetry",{})
    def num(v):
        try: return float(str(v).split()[0])
        except Exception: return None
    rows.append((float(os.path.basename(f)[:-5]), num(t.get("aiclk")), num(t.get("power")), num(t.get("current")), num(t.get("asic_temperature"))))
if its and tsl:
    a,b=tsl[0],tsl[-1]; win=[r for r in rows if a-0.5<=r[0]<=b+0.3]
    half=its[len(its)//2:]
    clk=[r[1] for r in win if r[1]]; pw=[r[2] for r in win if r[2]]; cur=[r[3] for r in win if r[3]]; tmp=[r[4] for r in win if r[4]]
    f=lambda v: f"{statistics.median(v):.0f} ({min(v):.0f}-{max(v):.0f})" if v else "n/a"
    print(f"RES sus {tag} bs{len(its) and sys.argv[0] and ''}: cold_best={min(its):.1f} sustained_median={statistics.median(half):.1f} (it{len(its)//2}-{len(its)-1}) aiclk={f(clk)} MHz power={f(pw)} W current={f(cur)} A temp={f(tmp)} C samples={len(win)}")
else:
    print(f"RES sus {tag}: no iterations parsed ({len(its)}) samples={len(rows)}")
PY
