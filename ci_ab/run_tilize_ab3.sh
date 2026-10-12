#!/usr/bin/env bash
# Regression audit of #59143 on one CI Blackhole runner (knowledge/process/17_CI_AB.md). Variants (ci_ab/variants/<v>/tree,
# the files of ci_ab/variants/files.txt): main (origin/main f0832760bf5), mrg (the PR head merged with it, 5230cc64dd7),
# head (the audit fix, 4574191022b). Before each run the variant's files go into the checkout and into the installed
# ttnn package (the JIT reads LLK and API headers from the package and kernel sources from the checkout); each run gets
# a fresh kernel cache of its own name (main2 is main's files again under another cache: the A/A pair).
#   job 2: device time of the fast tilize callers where job 1 left a doubt (conv2d, group norm, the small sharded tilize),
#   variants rotated over three blocks; variant fx2 is fe3c8c6a978 (the fix without the API commit).
#   1. bits: test_r3_audit_dumps.py T2/T3/T4 on main and head, every output compared as raw words;
#   2. device time of the fast tilize callers (ci_ab/ids/prof_ft.txt), tracy, 4 repetitions per test (the first
#      dropped), runs main mrg head main2 | main2 head mrg main;
#   3. tt-llk harness: the functional modules of the change with LLK asserts on (head), then perf_fast_tilize_prod
#      main head head main (L1_TO_L1), each side with its own build directory.
set +e +o pipefail
export HWLOCK_HELD=ci TRACY_NO_WEB_SERVER=1
R=$(git rev-parse --show-toplevel); S=$R/ci_ab; cd $R
SP=$(python -c "import ttnn, os; print(os.path.dirname(os.path.dirname(os.path.abspath(ttnn.__file__))))")
echo "== checkout $R, package root $SP, $(git log --oneline -1)"
OBJ=$(find / -name riscv-tt-elf-objcopy -type f 2>/dev/null | head -1); echo "== objcopy $OBJ"
OUTD=$R/generated/tab; mkdir -p $OUTD
place() {  # place <variant>
  local d=$S/variants/$1
  while read -r rel; do
    [[ -z $rel ]] && continue
    cp "$d/tree/$rel" "$R/$rel"
    local suf; suf=$(echo "$rel" | awk -F/ '{n=NF; print $(n-3)"/"$(n-2)"/"$(n-1)"/"$n}')
    while read -r p; do [[ -n $p && $p != "$R/$rel" ]] && cp "$d/tree/$rel" "$p"; done < <(find "$SP" -path "*/$suf" -type f 2>/dev/null)
  done < $S/variants/files.txt
}
while read -r rel; do suf=$(echo "$rel" | awk -F/ '{n=NF; print $(n-3)"/"$(n-2)"/"$(n-1)"/"$n}'); echo "== $rel -> $(find "$SP" -path "*/$suf" -type f 2>/dev/null | tr '\n' ' ')"; done < $S/variants/files.txt
elfsum() {  # combined hash of each kernel's compute ELF code in a cache (positive control of the swap)
  local c=$1
  for k in $(ls -d $c/*/kernels/* 2>/dev/null | xargs -n1 basename | sort -u); do
    h=$(for e in $(ls $c/*/kernels/$k/*/trisc[012]/trisc[012].elf 2>/dev/null | sort); do
          if [[ -n $OBJ ]]; then $OBJ -O binary -j .text "$e" /tmp/elf.bin 2>/dev/null && md5sum < /tmp/elf.bin; else md5sum < "$e"; fi
        done | md5sum | cut -c1-12)
    echo "   $k $h"
  done
}
cache() {  # cache <name>: a fresh kernel cache
  export TT_METAL_CACHE=$HOME/tabcache_$1; rm -rf $TT_METAL_CACHE; mkdir -p $TT_METAL_CACHE
}

# 2. device time
MODS=$(cut -d: -f1 $S/ids/prof_c3.txt | sort -u | tr '\n' ' ')
i=0
for v in main mrg head main2 main2 head mrg main main head main2 mrg; do
  i=$((i+1)); f=$v; [[ $v == main2 ]] && f=main
  place $f
  if [[ " ${PFRESH:-} " != *" $v "* ]]; then cache prof_$v; PFRESH="${PFRESH:-} $v"; else export TT_METAL_CACHE=$HOME/tabcache_prof_$v; fi
  OUT=$OUTD/out_pd_${i}_$v; rm -rf $OUT; mkdir -p $OUT
  echo "== prof $v ($i) $(date -u +%T)"
  MM_REPS=6 FR_SELECT=$S/ids/prof_c3.txt PYTHONPATH="$S/tools:$R${PYTHONPATH:+:$PYTHONPATH}" timeout -s INT -k 60 1500 \
    python -m tracy -r -p --no-web-server -o $OUT -m pytest -p mm_prof_plugin -p no:cacheprovider -o timeout_method=thread -q -rfE $S/plugin $MODS \
    > $OUTD/log_pd_${i}_$v.txt 2>&1
  echo "== prof $v ($i) rc=$? $(date -u +%T): $(grep -E 'passed|failed' $OUTD/log_pd_${i}_$v.txt | tail -1)"
  grep -a -E "^FAILED|^ERROR" $OUTD/log_pd_${i}_$v.txt | head -3 | cut -c1-300
  if [[ $i -le 5 ]]; then echo "== compute ELFs of $v:"; elfsum $TT_METAL_CACHE | grep -E "cache|tilize"; fi
done
echo "== reduce pd (base main)"; PROF_BASE=main python $S/tools/prof_reduce.py $OUTD pd 2>&1
echo "== reduce pd (base main2)"; PROF_BASE=main2 python $S/tools/prof_reduce.py $OUTD pd 2>&1

echo "== reduce pd (base mrg)"; PROF_BASE=mrg python $S/tools/prof_reduce.py $OUTD pd 2>&1
cd $R; place head
echo "== DONE $(date -u +%T)"
