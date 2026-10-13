#!/usr/bin/env bash
# 1x32 reduced output tile bound: H (head) against P (TT_HR_REDUCE_H_1X32 set), A/A copy H2; two blocks of H P H2 H2 P H.
set +e +o pipefail
R=/work; [ -d $R/ci_ab ] || R=$(git rev-parse --show-toplevel); cd $R; S=$R/ci_ab; O=/tmp/hr; mkdir -p $O /tmp/hr_out
export TT_METAL_HOME=$R TRACY_NO_WEB_SERVER=1
python3 -c "import tracy" 2>/dev/null || export PYTHONPATH=$R:$R/tools
echo "== runner $(hostname) $(date -u +%FT%TZ) head $(git log -1 --format='%h %s' | cut -c1-80)"
for b in 1 2; do for side in H P H2 H2 P H; do
  n=$(( $(ls -d $O/prof_${side}_* 2>/dev/null | grep -v '\.log$' | wc -l) + 1 )); out=$O/prof_${side}_$n
  rm -rf $out; t0=$(date +%s)
  if [ $side = P ]; then export TT_HR_REDUCE_H_1X32=1; else unset TT_HR_REDUCE_H_1X32; fi
  TT_METAL_CACHE=/tmp/hr_cache_$side TT_METAL_PROFILER_DIR=/tmp/praw_${side}_$n timeout -s INT -k 60 900 python3 -m tracy -r -p --no-web-server -o $out $S/tools/prof2.py $side $n > $out.log 2>&1
  echo "== prof $side $n rc=$? $(( $(date +%s) - t0 )) s $(grep -c 'CASE .* ok' $out.log) ok"; grep -E "Error|TT_THROW|TT_FATAL" $out.log | head -3 | cut -c1-300
  grep -E "^CASE .* ok BITS" $out.log | sed 's/^/   /'
done; done
unset TT_HR_REDUCE_H_1X32
a=$(find /tmp/hr_cache_H -name 'trisc*.elf' | sort | xargs -r sha256sum | awk '{print $1}' | sort | sha256sum | cut -c1-16)
b=$(find /tmp/hr_cache_P -name 'trisc*.elf' | sort | xargs -r sha256sum | awk '{print $1}' | sort | sha256sum | cut -c1-16)
echo "== guard H $a P $b"; [ "$a" != "$b" ] || { echo "== VOID A/B: H and P compiled the same kernels"; exit 1; }
python3 $S/tools/devtime.py $O P 2>&1 | sed 's/^/PROF /'
exit 0
