#!/bin/bash
# t164 phase 2 on g15blx02, after the pack driver (TAG=pack2): pick the 1-2 configs with the lowest mean
# gen1/gen2 warm e2e that beat the baseline by 20 ms and pass a loose "not broken" gate against the baseline
# clips (PCC >= 0.95, PSNR >= 25 dB on gen1 and gen2; VBench and the visual check do the real judging).
# Then run baseline5 and <config>5 as 5-seed jobs (default prompt, seeds 0-4, the ref_dv145 protocol), and
# VBench each 5-seed set against ref_dv145 (ltx_eval batch --vbench-ref; CPU, niced, 3 clips at a time).
# Marker: $P/PHASE2.done = "<code> <reason>"; 20 = no config qualified (needs a judgment, nothing run).
set -eo pipefail
T=/home/smarton/fasth3/tt-metal/tt-project/worktrees/t164/tmp/t164
S=/home/smarton/fasth3/tt-metal/tt-project/worktrees/t164
DATA=/home/smarton/fasth3/tt-metal/tt-project/data/g15
REF=/home/smarton/fasth3/tt-metal/tt-project/baselines/ltx25_1080p_6s/ref_dv145
PACK=$DATA/t164/driver_pack2/DRIVER.done; PACK_RC=${PACK_RC:?rc file of the detached pack driver}
P=$DATA/t164/phase2; mkdir -p $P
reason="died"
trap 'echo "$? $reason" > $P/PHASE2.done' EXIT
log() { echo "$(date -u '+%F %T') $*" >> $P/phase2.log; }

log "waiting for $PACK"
until [ -e $PACK ] || [ -e $PACK_RC ]; do sleep 60; done
sleep 5
[ -e $PACK ] || { reason="pack driver exited without a marker"; exit 12; }
read -r pcode prest < $PACK
log "pack: $pcode $prest"
[ "$pcode" = 0 ] || { reason="pack driver ended: $pcode $prest"; exit 11; }

reason="config pick failed"
COMMON="LTX_FRESH_PROMPTS=0 LTX_E2E_SEEDS=0,1,2,3,4 LTX_E2E_EXTRA_REPLAYS=0"
python3 - $DATA/t164/summary_configs.json $T/configs.txt "$COMMON" > $P/configs5.txt <<'EOF'
import json, sys
res, lines, common = json.load(open(sys.argv[1])), open(sys.argv[2]).read().splitlines(), sys.argv[3]
flags = {l.split()[0]: " ".join(l.split()[1:]) for l in lines if l.strip() and not l.startswith("#")}
def mean_e2e(v):
    t = [v["gens"][g]["e2e"] for g in ("1", "2") if "e2e" in v.get("gens", {}).get(g, {})]
    return sum(t) / len(t) if len(t) == 2 else None
def sane(v):
    q = [v["gens"].get(g, {}).get("quality") for g in ("1", "2")]
    return all(x and x["pcc"] >= 0.95 and x["psnr"] >= 25 for x in q)
b = res.get("baseline", {})
base = mean_e2e(b) if b.get("exit") == "0" else None
if base is None:
    sys.exit(0)
picks = sorted((mean_e2e(v), c) for c, v in res.items()
               if c != "baseline" and v.get("exit") == "0" and mean_e2e(v) and mean_e2e(v) < base - 0.02 and sane(v))
if picks:
    print(f"baseline5 {common}")
    for _, c in picks[:2]:
        print(f"{c}5 {flags[c]} {common}")
EOF
log "configs5: $(tr '\n' ';' < $P/configs5.txt)"
[ -s $P/configs5.txt ] || { reason="no config beat the baseline and passed the gate; see summary_configs.md"; exit 20; }

CONFIGS=$P/configs5.txt TAG=seeds bash $T/driver.sh
read -r scode srest < $DATA/t164/driver_seeds/DRIVER.done
log "seeds: $scode $srest"
[ "$scode" = 0 ] || { reason="seeds driver ended: $scode $srest"; exit 13; }

cd $S
source /home/smarton/fasth3/tt-metal/python_env/bin/activate
export PYTHONPATH=$S:$S/../t158/ttnn:$S/../t158/tools LTX_EVAL_THREADS=8 HF_HUB_OFFLINE=1
while read -r label _; do
  d=$DATA/t164/$label
  grep -q "T164_EXIT\[$label\]=0" $d/run.log || { log "$label: no clean run, no VBench"; continue; }
  mkdir -p $d/seeds
  for i in 0 1 2 3 4; do ln -sfn ../ltx_av_fast_1920x1088_$((i + 1)).mp4 $d/seeds/seed$i.mp4; done
  vrc=0
  nice -n 19 timeout 3000 python -m models.tt_dit.tests.models.ltx.tools.ltx_eval batch --cand-dir $d/seeds \
    --ref-dir $REF --out $d/vbench --jobs 3 --vbench-ref < /dev/null > $d/vbench.log 2>&1 || vrc=$?
  log "$label VBench rc=$vrc: $(grep -E '^BATCH' $d/vbench.log | cut -c1-400)"
done < $P/configs5.txt
reason="ok"
