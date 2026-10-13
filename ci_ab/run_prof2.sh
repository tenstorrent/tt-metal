#!/usr/bin/env bash
# Reduce over H with a 1x32 output tile and a two-face-row writer: H (head) against R (TT_HR_REDUCE_H_ROW set), A/A copy
# H2; three blocks of H R H2 H2 R H.
set +e +o pipefail
R=/work; [ -d $R/ci_ab ] || R=$(git rev-parse --show-toplevel); cd $R; S=$R/ci_ab; O=/tmp/hr; mkdir -p $O /tmp/hr_out
export TT_METAL_HOME=$R TRACY_NO_WEB_SERVER=1
python3 -c "import tracy" 2>/dev/null || export PYTHONPATH=$R:$R/tools
echo "== runner $(hostname) $(date -u +%FT%TZ) head $(git log -1 --format='%h %s' | cut -c1-80)"
for b in 1 2 3; do for side in H R H2 H2 R H; do
  n=$(( $(ls -d $O/prof_${side}_* 2>/dev/null | grep -v '\.log$' | wc -l) + 1 )); out=$O/prof_${side}_$n
  rm -rf $out; t0=$(date +%s)
  if [ $side = R ]; then export TT_HR_REDUCE_H_ROW=1; else unset TT_HR_REDUCE_H_ROW; fi
  TT_METAL_CACHE=/tmp/hr_cache_$side TT_METAL_PROFILER_DIR=/tmp/praw_${side}_$n timeout -s INT -k 60 900 python3 -m tracy -r -p --no-web-server -o $out $S/tools/prof2.py $side $n > $out.log 2>&1
  echo "== prof $side $n rc=$? $(( $(date +%s) - t0 )) s $(grep -c 'CASE .* ok' $out.log) ok"; grep -E "Error|TT_THROW|TT_FATAL" $out.log | head -3 | cut -c1-300
  grep -E "^CASE .* ok BITS" $out.log | sed 's/^/   /'
done; done
unset TT_HR_REDUCE_H_ROW
a=$(find /tmp/hr_cache_H -name 'trisc*.elf' | sort | xargs -r sha256sum | awk '{print $1}' | sort | sha256sum | cut -c1-16)
b=$(find /tmp/hr_cache_R -name 'trisc*.elf' | sort | xargs -r sha256sum | awk '{print $1}' | sort | sha256sum | cut -c1-16)
echo "== guard H $a R $b"; [ "$a" != "$b" ] || { echo "== VOID A/B: H and R compiled the same kernels"; exit 1; }
python3 $S/tools/devtime.py $O R 2>&1 | sed 's/^/PROF /'
python3 - <<'PY2' 2>&1 | sed 's/^/OUTBITS /'
import glob, os, collections, torch
d = collections.defaultdict(dict)
for p in glob.glob("/tmp/hr_out/*.pt"):
    side, run, name = os.path.basename(p)[:-3].split("_", 2)
    d[name][(side, run)] = torch.load(p)
for name, runs in sorted(d.items()):
    ref = runs.get(("H", "1"))
    eq = {k: bool(torch.equal(v.view(torch.int32), ref.view(torch.int32))) for k, v in runs.items()}
    print(name, "equal to H run 1:", " ".join(f"{s}{r}={'yes' if e else 'NO'}" for (s, r), e in sorted(eq.items())))
PY2
exit 0
