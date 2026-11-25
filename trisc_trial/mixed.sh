#!/bin/bash
# usage: mixed.sh <test_config.py> <label> <trials per seed> <outdir> [compiler]; 8 seeds in parallel, no 'diag' forms.
TC=$1; L=$2; N=$3; OUT=$4; CXX=${5:-c++}
POOL=$(python3 -c "import forms; print(' '.join(f for f in sorted(forms.F) if forms.F[f][1]!='diag'))")
mkdir -p "$OUT"
for seed in 1 2 3 4 5 6 7 8; do
  python3 harness.py "$TC" "$N" $seed "$OUT/$L$seed/w" "$CXX" $POOL > "$OUT/$L$seed.json" &
done
wait
python3 - "$OUT" "$L" <<'PY'
import json, glob, sys, collections
out, l = sys.argv[1], sys.argv[2]
wrong = trials = reused = 0
by = collections.Counter()
for f in sorted(glob.glob(f"{out}/{l}[0-9].json")):
    d = json.load(open(f))
    wrong += d["wrong"]
    for k, s in d["stats"].items():
        by[s["cls"]] += s["trials"]
    for ce in json.load(open(f.replace(f"/{l}", f"/counterexamples-").replace(".json", ".json"))) if False else []:
        pass
print(l, "wrong", wrong, "form-uses by class", dict(by))
PY
