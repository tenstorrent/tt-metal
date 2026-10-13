#!/usr/bin/env bash
# Reduce over W at production shapes: #60453's head (HEAD, A/A copy HEAD2) against the faces 0 and 2 writer (W2), the
# faces 0 to 2 writer (W3) and main (MAIN: main's device files, the H 1x32 path off); two blocks of
# HEAD W2 W3 MAIN HEAD2 HEAD2 MAIN W3 W2 HEAD, fresh kernel cache per side; device time, output and consumer bits.
source $(dirname $0)/common.sh
export TRACY_NO_WEB_SERVER=1
rm -rf /tmp/hr_cache_* /tmp/hr_out/*.pt
for b in 1 2; do for side in HEAD W2 W3 MAIN HEAD2 HEAD2 MAIN W3 W2 HEAD; do
  n=$(( $(ls -d $O/prof_${side}_* 2>/dev/null | grep -v '\.log$' | wc -l) + 1 )); out=$O/prof_${side}_$n
  base=$side; [ $side = HEAD2 ] && base=HEAD; case $side in W2|W3) base=HEAD;; esac
  use $base; export TT_METAL_CACHE=/tmp/hr_cache_$side; mkdir -p $TT_METAL_CACHE
  unset TT_HR_W_FACES TT_HR_NO_H_ROW
  case $side in W2) export TT_HR_W_FACES=2;; W3) export TT_HR_W_FACES=3;; MAIN) export TT_HR_NO_H_ROW=1;; esac
  rm -rf $out; t0=$(date +%s)
  TT_METAL_PROFILER_DIR=/tmp/praw_${side}_$n timeout -s INT -k 60 1500 python3 -m tracy -r -p --no-web-server -o $out $S/tools/prof_w.py $side $n > $out.log 2>&1
  echo "== prof $side $n rc=$? $(( $(date +%s) - t0 )) s $(grep -c 'CASE .* ok' $out.log) ok"; grep -E "Error|TT_THROW|TT_FATAL" $out.log | head -3 | cut -c1-300
  grep -E "^CASE .* ok BITS" $out.log | sed 's/^/   /'
done; done
unset TT_HR_W_FACES TT_HR_NO_H_ROW
use HEAD
for s in HEAD W2 W3 MAIN; do echo "== writer kernels $s: col2 $(find /tmp/hr_cache_$s -path '*writer_reduce_w_col_interleaved*' -name '*.elf' | wc -l) col3 $(find /tmp/hr_cache_$s -path '*writer_reduce_w_col3_interleaved*' -name '*.elf' | wc -l) h_row $(find /tmp/hr_cache_$s -path '*writer_reduce_h_row_interleaved*' -name '*.elf' | wc -l)"; done
[ $(find /tmp/hr_cache_W2 -path '*writer_reduce_w_col_interleaved*' -name '*.elf' | wc -l) -gt 0 ] && [ $(find /tmp/hr_cache_HEAD -path '*writer_reduce_w_col*' -name '*.elf' | wc -l) -eq 0 ] || echo "== VOID A/B: W2 writer not compiled or present on HEAD"
guard HEAD MAIN
python3 $S/tools/kelf.py /tmp/hr_cache_HEAD /tmp/hr_cache_HEAD2 HEAD HEAD2 2>&1 | grep -v "KELF same" | head -10
DT_BASE=HEAD DT_AA=HEAD2 python3 $S/tools/devtime.py $O W2,W3,MAIN 2>&1 | sed 's/^/PROF /'
python3 - <<'PY' 2>&1 | sed 's/^/OUTBITS /'
import glob, os, collections, torch
d = collections.defaultdict(dict)
for p in glob.glob("/tmp/hr_out/*.pt"):
    side, run, name = os.path.basename(p)[:-3].split("_", 2)
    d[name][(side, run)] = torch.load(p)
for name, runs in sorted(d.items()):
    ref = runs.get(("HEAD", "1"))
    if ref is None:
        continue
    bad = [f"{s}{r}" for (s, r), v in sorted(runs.items()) if not torch.equal(v.view(torch.int32) if v.dtype == torch.float32 else v, ref.view(torch.int32) if ref.dtype == torch.float32 else ref)]
    print(name, "equal to HEAD run 1:", len(runs) - len(bad), "of", len(runs), ("differ: " + " ".join(bad)) if bad else "")
PY
exit 0
