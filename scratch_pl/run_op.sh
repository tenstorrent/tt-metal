#!/usr/bin/env bash
# Pack-lever round, scratch: device-profiler A/B of the #59143 head and the fast tilize W-counter L1 step, on one
# CI Blackhole runner (knowledge/process/17_CI_AB.md). Each variant's files (scratch_pl/variants/<v>/tree, every file of
# scratch_pl/files.txt) are copied into the checkout and the installed ttnn package before its run; each variant has its
# own fresh JIT cache. usage: run_op2.sh <plan file>, lines "<tag> <variant> <tests>", tests comma separated or @<ids file>
set -uo pipefail
export HWLOCK_HELD=ci TRACY_NO_WEB_SERVER=1
R=$(git rev-parse --show-toplevel); S=$R/scratch_pl; PLAN=$(readlink -f "$1"); cd $R
SP=$(python -c "import ttnn, os; print(os.path.dirname(os.path.dirname(os.path.abspath(ttnn.__file__))))")
echo "== checkout $R head $(git rev-parse HEAD), package root $SP"
export PYTHONPATH="$S/tools:$R${PYTHONPATH:+:$PYTHONPATH}"
OBJ=$(find / -name riscv-tt-elf-objcopy -type f 2>/dev/null | head -1); echo "== objcopy $OBJ"
place() {
  local d=$1
  while read -r rel; do
    [[ -z $rel ]] && continue
    local src=$d/tree/$rel
    cp "$src" "$R/$rel"
    local suf; suf=$(echo "$rel" | awk -F/ '{n=NF; s=$(n-3)"/"$(n-2)"/"$(n-1)"/"$n; print s}')
    while read -r p; do [[ -n $p && $p != "$R/$rel" ]] && cp "$src" "$p"; done < <(find "$SP" -path "*/$suf" -type f 2>/dev/null)
  done < $S/files.txt
}
while read -r rel; do suf=$(echo "$rel" | awk -F/ '{n=NF; print $(n-3)"/"$(n-2)"/"$(n-1)"/"$n}'); echo "== $rel -> $(find "$SP" -path "*/$suf" -type f 2>/dev/null | wc -l) package copies"; done < $S/files.txt
elfsum() {
  local c=$1
  for k in $(ls -d $c/*/kernels/* 2>/dev/null | xargs -r -n1 basename | sort -u); do
    h=$(for e in $(ls $c/*/kernels/$k/*/trisc[012]/trisc[012].elf 2>/dev/null | sort); do
          if [[ -n $OBJ ]]; then $OBJ -O binary -j .text "$e" /tmp/elf.bin 2>/dev/null && md5sum < /tmp/elf.bin; else md5sum < "$e"; fi
        done | md5sum | cut -c1-12)
    echo "$k $h"
  done
}
OUTD=$R/generated/plci; mkdir -p $OUTD
i=0
while read -r tag v tests; do
  [[ -z ${tag:-} || $tag == \#* ]] && continue
  i=$((i+1))
  place $S/variants/$v
  export TT_METAL_CACHE=$HOME/plcache_$v
  if [[ " ${FRESHED:-} " != *" $v "* ]]; then rm -rf $TT_METAL_CACHE; FRESHED="${FRESHED:-} $v"; fi
  mkdir -p $TT_METAL_CACHE
  OUT=$OUTD/out_${tag}_${i}_$v; rm -rf $OUT; mkdir -p $OUT
  echo "== $tag $v ($i) $tests $(date -u +%T)"
  T=()
  if [[ $tests == @* ]]; then mapfile -t T < "$S/${tests#@}"; else
    IFS=, read -ra TL <<< "$tests"; for t in "${TL[@]}"; do if [[ $t == tests/* || $t == models/* ]]; then T+=("$R/$t"); else T+=("$S/tools/$t"); fi; done
  fi
  printf '%s\n' "${T[@]}" > $OUTD/nodes_${i}.txt
  HR_NODES=$OUTD/nodes_${i}.txt timeout -s INT -k 60 ${CI_LIMIT:-1800} python -m tracy -r -p --no-web-server -o $OUT $S/tools/run_nodes.py > $OUTD/log_${tag}_${i}_$v.txt 2>&1
  rc=$?; echo "== $tag $v ($i) rc=$rc $(date -u +%T): $(grep -E 'passed|failed' $OUTD/log_${tag}_${i}_$v.txt | tail -1)"
  if [[ $rc -ne 0 ]] || ! grep -q passed $OUTD/log_${tag}_${i}_$v.txt; then grep -a -m8 -E "error:|undefined reference|Error" $OUTD/log_${tag}_${i}_$v.txt; grep -a -E "^(FAILED|ERROR)" $OUTD/log_${tag}_${i}_$v.txt | head -20; tail -40 $OUTD/log_${tag}_${i}_$v.txt; fi
done < "$PLAN"
place $S/variants/new
echo "== GUARD: compute ELF hashes per variant (17_CI_AB.md)"
for v in main fin new new2; do elfsum $HOME/plcache_$v > $OUTD/elfsum_$v.txt; echo "== ELFs of $v: $(wc -l < $OUTD/elfsum_$v.txt) kernels"; cat $OUTD/elfsum_$v.txt; done
for p in "main new" "fin new"; do
  set -- $p; n=$(join $OUTD/elfsum_$1.txt $OUTD/elfsum_$2.txt | awk '$2 != $3' | wc -l)
  echo "GUARD $1 against $2: $n kernels differ"; [[ $n == 0 ]] && echo "GUARD FAIL $1 and $2 read the same builds"
done
n=$(join $OUTD/elfsum_new.txt $OUTD/elfsum_new2.txt | awk '$2 != $3' | wc -l); echo "A/A new against new2: $n kernels differ (expect 0)"
for tag in $(awk '!/^#/ && NF {print $1}' "$PLAN" | sort -u); do
  echo "== reduce $tag (base main)"; PROF_BASE=main python $S/tools/prof_reduce.py $OUTD $tag 2>&1
done
tar -czf $R/generated/plci.tgz -C $R/generated plci 2>/dev/null; echo "== results $(stat -c %s $R/generated/plci.tgz) bytes"
