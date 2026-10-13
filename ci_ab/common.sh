# Shared by run_bits.sh and run_prof.sh: side switching of device files in /work and the installed runtime root.
set +e +o pipefail
R=/work; [ -d $R/ci_ab ] || R=$(git rev-parse --show-toplevel); cd $R; S=$R/ci_ab; O=/tmp/hr; mkdir -p $O /tmp/hr_out
export TT_METAL_HOME=$R
python3 -c "import tracy" 2>/dev/null || export PYTHONPATH=$R:$R/tools
export PYTHONPATH=$S/tools:$PYTHONPATH
RT=$(cd /tmp && env -u PYTHONPATH python3 -c "import ttnn, os; print(os.path.dirname(ttnn.__file__))" 2>/dev/null)
echo "== runner $(hostname) $(date -u +%FT%TZ) RT=$RT head $(git log -1 --format='%h %s' | cut -c1-80)"
(cd $S/variants && find . -path '*/tree/*' -type f ! -name .keep | sed -E 's#^\./[^/]+/tree/##' | sort -u) > $O/paths.txt
rm -rf $O/base; mkdir -p $O/base
while read p; do [ -f "$R/$p" ] && { mkdir -p $O/base/$(dirname $p); cp $R/$p $O/base/$p; }; done < $O/paths.txt
echo "== overlay paths $(wc -l < $O/paths.txt), under the runtime root $(while read p; do [ -f "$RT/$p" ] && echo x; done < $O/paths.txt | wc -l)"
use() {
  local side=$1 f
  (cd $O/base && find . -type f) | while read f; do cp "$O/base/$f" "$R/$f"; [ -f "$RT/$f" ] && cp "$O/base/$f" "$RT/$f"; done
  if [ -d $S/variants/$side/tree ]; then
    (cd $S/variants/$side/tree && find . -type f ! -name .keep) | while read f; do cp "$S/variants/$side/tree/$f" "$R/$f"; [ -f "$RT/$f" ] && cp "$S/variants/$side/tree/$f" "$RT/$f"; done
  fi
  export TT_METAL_CACHE=/tmp/hr_cache_$side; mkdir -p $TT_METAL_CACHE
  echo "   use $side: reduce header work $(sha256sum $R/tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_math_reduce.h | cut -c1-12) runtime $(sha256sum $RT/tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_math_reduce.h 2>/dev/null | cut -c1-12)"
}
elfsum() { # digest of every compute (trisc1) ELF a side's cache holds
  find /tmp/hr_cache_$1 -name "trisc1*.elf" 2>/dev/null | sort | xargs -r sha256sum | awk '{print $1}' | sort | sha256sum | cut -c1-16
}
guard() { # guard <A> <B>: the A/B is void when the two sides compiled the same compute ELFs
  local a=$(elfsum $1) b=$(elfsum $2)
  echo "== guard $1 $a $(find /tmp/hr_cache_$1 -name 'trisc1*.elf' | wc -l) ELFs, $2 $b $(find /tmp/hr_cache_$2 -name 'trisc1*.elf' | wc -l) ELFs"
  [ "$a" != "$b" ] || { echo "== VOID A/B: $1 and $2 compiled the same compute ELFs"; exit 1; }
}
