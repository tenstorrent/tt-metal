#!/usr/bin/env bash
# Main's device and kernel files (MAIN) against this head (V7) at op level, with an A/A copy of main (MAIN2): three blocks
# of MAIN V7 MAIN2 MAIN2 V7 MAIN, own kernel cache per side; device time and output bits.
source $(dirname $0)/common.sh
export TRACY_NO_WEB_SERVER=1
rm -f /tmp/hr_out/*.pt
for b in 1 2 3; do for side in MAIN V7 MAIN2 MAIN2 V7 MAIN; do
  n=$(( $(ls -d $O/mh_${side}_* 2>/dev/null | grep -v '\.log$' | wc -l) + 1 )); out=$O/mh_${side}_$n
  base=$side; [ $side = MAIN2 ] && base=MAIN; use $base; export TT_METAL_CACHE=/tmp/hr_cache_mh_$side; mkdir -p $TT_METAL_CACHE
  rm -rf $out; t0=$(date +%s)
  TT_METAL_PROFILER_DIR=/tmp/praw_mh_${side}_$n timeout -s INT -k 60 1500 python3 -m tracy -r -p --no-web-server -o $out $S/tools/prof_mh.py $side $n > $out.log 2>&1
  echo "== mh $side $n rc=$? $(( $(date +%s) - t0 )) s $(grep -c 'CASE .* ok' $out.log) ok $(grep -c 'CASE .* FAIL' $out.log) fail"; grep -E "CASE .* FAIL" $out.log | head -3 | cut -c1-300
done; done
use V7
python3 $S/tools/kelf.py /tmp/hr_cache_mh_MAIN /tmp/hr_cache_mh_V7 MAIN V7 2>&1 | grep -v "KELF same" | head -30
mkdir -p $O/mhd; for d in $O/mh_*; do [ -d $d ] && ln -sfn $d $O/mhd/prof_${d#$O/mh_}; done
DT_BASE=MAIN DT_AA=MAIN2 python3 $S/tools/devtime.py $O/mhd V7 2>&1 | sed 's/^/MH /'
python3 - <<'PY' 2>&1 | sed 's/^/MHBITS /'
import glob, os, collections, torch
d = collections.defaultdict(dict)
for p in glob.glob("/tmp/hr_out/*.pt"):
    side, run, name = os.path.basename(p)[:-3].split("_", 2)
    d[name][(side, run)] = torch.load(p)
for name, runs in sorted(d.items()):
    ref = runs.get(("MAIN", "1"))
    if ref is None:
        continue
    eq = {k: bool(torch.equal(v.view(torch.int32), ref.view(torch.int32))) for k, v in runs.items()}
    print(name, "equal to MAIN run 1:", sum(eq.values()), "of", len(eq))
PY
exit 0
