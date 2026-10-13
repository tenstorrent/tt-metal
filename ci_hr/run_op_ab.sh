#!/usr/bin/env bash
# Op-level A/B on one CI Blackhole runner (test-dispatch with tracy): each variant's device headers (opvariants/<v>/, the
# paths of opfiles.txt) are copied into the checkout and the installed ttnn package (where the JIT reads them) before its
# run, each variant with its own fresh kernel cache; eb_prof_plugin repeats every test EB_REPS times (first dropped) under
# signposts, eb_bits_plugin hashes every output. Plan lines: "<tag> <variant>". Guard: per-variant compute ELF sums, and the
# variants must differ from the base on some kernel. Results: per op configuration tables and the bits comparison.
set +e +o pipefail
export HWLOCK_HELD=ci TRACY_NO_WEB_SERVER=1
R=$(git rev-parse --show-toplevel); S=$R/ci_hr; cd $R
PLAN=$S/${1:-opplan.txt}
SP=$(python -c "import ttnn, os; print(os.path.dirname(os.path.dirname(os.path.abspath(ttnn.__file__))))")
echo "== start $(date -u +%T) checkout $R package root $SP"
export PYTHONPATH="$S/ops:$R${PYTHONPATH:+:$PYTHONPATH}" EB_REPS=${EB_REPS:-4}
OBJ=$(find / -name riscv-tt-elf-objcopy -type f 2>/dev/null | head -1)
place() {
  while read -r rel; do
    [[ -z $rel ]] && continue
    cp "$S/opvariants/$1/$rel" "$R/$rel"
    local suf; suf=$(echo "$rel" | awk -F/ '{n=NF; print $(n-3)"/"$(n-2)"/"$(n-1)"/"$n}')
    while read -r p; do [[ -n $p && $p != "$R/$rel" ]] && cp "$S/opvariants/$1/$rel" "$p"; done < <(find "$SP" -path "*/$suf" -type f 2>/dev/null)
  done < $S/opfiles.txt
}
while read -r rel; do suf=$(echo "$rel" | awk -F/ '{n=NF; print $(n-3)"/"$(n-2)"/"$(n-1)"/"$n}'); echo "== $rel -> $(find "$SP" -path "*/$suf" -type f 2>/dev/null | tr '\n' ' ')"; done < $S/opfiles.txt
elfsum() {
  local c=$1
  for k in $(ls -d $c/*/kernels/* 2>/dev/null | xargs -n1 basename | sort -u); do
    for t in 0 1 2; do
      h=$(for e in $(ls $c/*/kernels/$k/*/trisc$t/trisc$t.elf 2>/dev/null | sort); do
            if [[ -n $OBJ ]]; then $OBJ -O binary -j .text "$e" /tmp/elf.bin 2>/dev/null && md5sum < /tmp/elf.bin; else md5sum < "$e"; fi
          done | md5sum | cut -c1-12)
      echo "$k/trisc$t $h"
    done
  done
}
OUTD=$R/generated/opab; rm -rf $OUTD; mkdir -p $OUTD
i=0
while read -r tag v; do
  [[ -z ${tag:-} || $tag == \#* ]] && continue
  i=$((i + 1))
  place $v
  export TT_METAL_CACHE=$HOME/hr_opcache_$v
  if [[ " ${FRESHED:-} " != *" $v "* ]]; then rm -rf $TT_METAL_CACHE; FRESHED="${FRESHED:-} $v"; fi
  mkdir -p $TT_METAL_CACHE
  OUT=$OUTD/out_${tag}_${i}_$v
  echo "== $tag $v ($i) $(date -u +%T)"
  EB_DET_OUT=$OUTD/det_${i}_$v.jsonl EB_HASH_OUT=$OUTD/hash_${i}_$v.json timeout -s INT -k 60 ${CI_LIMIT:-1500} python -m tracy -r -p --no-web-server -o $OUT -m pytest -p eb_prof_plugin -p eb_bits_plugin -p no:cacheprovider -o timeout_method=thread -q -rfE $S/ops/test_eb_hr_ops.py > $OUTD/log_${tag}_${i}_$v.txt 2>&1
  rc=$?; echo "== $tag $v ($i) rc=$rc $(date -u +%T): $(grep -E 'passed|failed' $OUTD/log_${tag}_${i}_$v.txt | tail -1)"
  [[ $rc != 0 ]] && grep -a -E "^(FAILED|ERROR)" $OUTD/log_${tag}_${i}_$v.txt | head -8 | cut -c1-220
  elfsum $TT_METAL_CACHE > $OUTD/elfsum_$v.txt
  for e in $(ls $TT_METAL_CACHE/*/kernels/*/*/trisc[012]/trisc[012].elf 2>/dev/null); do
    k=$(echo $e | awk -F/ '{print $(NF-3)"/"$(NF-1)}'); [[ -n $OBJ ]] && $OBJ -O binary -j .text "$e" /tmp/elf1.bin 2>/dev/null && echo "$k $(md5sum < /tmp/elf1.bin | cut -c1-12)"
  done | sort -u > $OUTD/elfeach_$v.txt
done < $PLAN
for v in $(awk '!/^#/ && NF {print $2}' $PLAN | sort -u | grep -vx ${PROF_BASE:-head}); do
  echo "== per-ELF: $v ELFs whose .text is not among ${PROF_BASE:-head}'s, by kernel/thread (of the kernel/thread's count):"
  awk 'NR==FNR {h[$2]=1; next} {n[$1]++; if (!($2 in h)) d[$1]++} END {for (k in n) if (d[k]) print "   " k, d[k], "of", n[k]}' $OUTD/elfeach_${PROF_BASE:-head}.txt $OUTD/elfeach_$v.txt
  echo "   total $(wc -l < $OUTD/elfeach_$v.txt) distinct (kernel/thread, .text) pairs"
done
echo "== compute ELF sums per variant"; for f in $OUTD/elfsum_*.txt; do echo "   $(basename $f): $(wc -l < $f) kernel threads, set $(md5sum < $f | cut -c1-12)"; done
B=${PROF_BASE:-head}
for v in $(awk '!/^#/ && NF {print $2}' $PLAN | sort -u | grep -vx $B); do
  echo "== kernel threads whose compute ELFs differ, $v against $B:"; join <(sort $OUTD/elfsum_$B.txt) <(sort $OUTD/elfsum_$v.txt) | awk '$2 != $3 {print "   " $1}'
  n=$(join <(sort $OUTD/elfsum_$B.txt) <(sort $OUTD/elfsum_$v.txt) | awk '$2 != $3' | wc -l); [[ $n == 0 ]] && echo "== GUARD FAIL: $v compiled to $B's ELFs"
done
for a in ${GUARD_PAIRS:-}; do x=${a%%-*}; y=${a##*-}; echo "== kernel threads whose compute ELFs differ, $y against $x:"; join <(sort $OUTD/elfsum_$x.txt) <(sort $OUTD/elfsum_$y.txt) | awk '$2 != $3 {print "   " $1}'; done
for tag in $(awk '!/^#/ && NF {print $1}' $PLAN | sort -u); do echo "== reduce $tag (base $B)"; PROF_BASE=$B python $S/ops/prof_reduce_cfg.py $OUTD $tag 2>&1; done
echo "== bits"; python $S/ops/bits_cmp.py $OUTD $B 2>&1 | tail -40
echo "== det"; python $S/ops/det_cmp.py $OUTD $B 2>&1 | tail -40
echo "== done $(date -u +%T)"
