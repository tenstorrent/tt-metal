#!/usr/bin/env bash
# Constant flag writes in the row max (V7) against the head (H) and an A/A copy (H2): three blocks of H V7 H2 H2 V7 H, each side with its
# own kernel cache; device time, output bits, and per-kernel ELF identity of the two caches.
source $(dirname $0)/common.sh
export TRACY_NO_WEB_SERVER=1
for b in 1 2 3; do for side in H V7 H2 H2 V7 H; do
  n=$(( $(ls -d $O/prof_${side}_* 2>/dev/null | grep -v '\.log$' | wc -l) + 1 )); out=$O/prof_${side}_$n
  base=$side; [ $side = H2 ] && base=H; use $base; export TT_METAL_CACHE=/tmp/hr_cache_$side; mkdir -p $TT_METAL_CACHE
  rm -rf $out /tmp/praw_${side}_$n; mkdir -p /tmp/praw_${side}_$n; t0=$(date +%s)
  TT_METAL_PROFILER_DIR=/tmp/praw_${side}_$n timeout -s INT -k 60 1500 python3 -m tracy -r -p --no-web-server -o $out $S/tools/prof7.py $side $n > $out.log 2>&1
  echo "== prof $side $n rc=$? $(( $(date +%s) - t0 )) s $(grep -c 'CASE .* ok' $out.log) ok $(grep -c 'CASE .* FAIL' $out.log) fail"; grep -E "CASE .* FAIL" $out.log | head -3 | cut -c1-300
  grep -E "^CASE .* ok BITS" $out.log | sed 's/^/   /'
done; done
use V7
python3 $S/tools/kelf.py /tmp/hr_cache_H /tmp/hr_cache_V7 H V7 2>&1 | head -60
python3 $S/tools/kelf.py /tmp/hr_cache_H /tmp/hr_cache_H2 H H2 2>&1 | head -3
python3 $S/tools/kelf.py /tmp/hr_cache_H /tmp/hr_cache_V7 H V7 | grep -q "^KELF differ" || { echo "== VOID A/B: H and V7 compiled the same kernels"; exit 1; }
python3 $S/tools/devtime.py $O V7 2>&1 | sed 's/^/PROF /'
python3 - <<'PY' 2>&1 | sed 's/^/OUTBITS /'
import glob, os, collections, torch
d = collections.defaultdict(dict)
for p in glob.glob("/tmp/hr_out/*.pt"):
    side, run, name = os.path.basename(p)[:-3].split("_", 2)
    d[name][(side, run)] = torch.load(p)
for name, runs in sorted(d.items()):
    ref = runs.get(("H", "1"))
    if ref is None:
        continue
    eq = {k: bool(torch.equal(v.view(torch.int32), ref.view(torch.int32))) for k, v in runs.items()}
    print(name, "equal to H run 1:", " ".join(f"{s}{r}={'yes' if e else 'NO'}" for (s, r), e in sorted(eq.items())))
PY
exit 0
