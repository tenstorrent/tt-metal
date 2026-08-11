#!/bin/bash
# How repeatable is the generator harness? Three IDENTICAL runs on unchanged HEAD.
# quality_baseline and biasBefore are the same code path and read 37.47 vs 39.93 ms/frame, so the
# harness spread may exceed the 1.918 ms the bias change delivers. If so, no RTF quoted from a
# single run means anything, and the interleaved block A/B is the only timing instrument of record.
cd "${TT_METAL_HOME:?set TT_METAL_HOME to the repo root}"
export PYTHONPATH="$TT_METAL_HOME/ttnn:$TT_METAL_HOME/tools:$TT_METAL_HOME"   # all three -- §2
V=models/experimental/voxtral_tts
for r in 1 2 3; do
  for s in 0 1; do
    python3 $V/scripts/generate_quality_set.py --tag "rep${r}s${s}" --seed $s >/dev/null 2>&1
  done
  echo "run $r done"
done
python3 - <<'PY'
import json, statistics
V="models/experimental/voxtral_tts/generated"
print(f"  {'run':>5} {'ms/frame':>10} {'RTF':>8}   (long-form cases, warmup case 0 excluded)")
allm=[]
for r in (1,2,3):
    rows=[x for s in (0,1) for x in json.load(open(f"{V}/resultsrep{r}s{s}.json"))]
    lf=[x for x in rows if len(x["text"].split())>=20 and x["case"]!=0]
    m=statistics.mean(x["gen_ms_per_frame"] for x in lf)
    t=statistics.mean(x["rtf"] for x in lf)
    allm.append((m,t)); print(f"  {r:>5} {m:>10.3f} {t:>8.4f}")
ms=[a for a,_ in allm]; rt=[b for _,b in allm]
print(f"\n  mean {statistics.mean(ms):.3f} ms/frame, RTF {statistics.mean(rt):.4f}")
print(f"  SPREAD {max(ms)-min(ms):.3f} ms/frame  (the bias change is worth 1.918 ms/step)")
PY
echo REPDONE
