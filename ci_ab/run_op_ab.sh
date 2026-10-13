#!/usr/bin/env bash
# Op-level A/B on one CI Blackhole runner (test-dispatch with tracy): each variant's LLK headers (ci_ab/opvariants/<v>/,
# the paths of ci_ab/opfiles.txt) are copied into the checkout and into the installed ttnn package (where the JIT reads
# them) before its run, each variant with its own fresh kernel cache; eb_prof_plugin repeats every test (first repetition
# dropped) under signposts. Plan lines: "<tag> <variant>". Guard: per-variant compute ELF hashes, and the variants must
# differ on the opted-in kernels. Results: prof_reduce tables in the log.
set +e +o pipefail
export HWLOCK_HELD=ci TRACY_NO_WEB_SERVER=1
R=$(git rev-parse --show-toplevel); S=$R/ci_ab; cd $R
PLAN=$S/${1:-opplan.txt}
SP=$(python -c "import ttnn, os; print(os.path.dirname(os.path.dirname(os.path.abspath(ttnn.__file__))))")
echo "== start $(date -u +%T) checkout $R package root $SP"
export PYTHONPATH="$S/ops:$R${PYTHONPATH:+:$PYTHONPATH}" SEL_FILE=$S/ops/sel.txt EB_REPS=${EB_REPS:-4}
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
    h=$(for e in $(ls $c/*/kernels/$k/*/trisc[012]/trisc[012].elf 2>/dev/null | sort); do
          if [[ -n $OBJ ]]; then $OBJ -O binary -j .text "$e" /tmp/elf.bin 2>/dev/null && md5sum < /tmp/elf.bin; else md5sum < "$e"; fi
        done | md5sum | cut -c1-12)
    echo "$k $h"
  done
}
OUTD=$R/generated/opab; rm -rf $OUTD; mkdir -p $OUTD
i=0
while read -r tag v; do
  [[ -z ${tag:-} || $tag == \#* ]] && continue
  i=$((i + 1))
  place $v
  export TT_METAL_CACHE=$HOME/eb_opcache_$v
  if [[ " ${FRESHED:-} " != *" $v "* ]]; then rm -rf $TT_METAL_CACHE; FRESHED="${FRESHED:-} $v"; fi
  mkdir -p $TT_METAL_CACHE
  OUT=$OUTD/out_${tag}_${i}_$v
  echo "== $tag $v ($i) $(date -u +%T)"
  timeout -s INT -k 60 ${CI_LIMIT:-2400} python -m tracy -r -p --no-web-server -o $OUT -m pytest -p eb_prof_plugin -p sel_plugin -p no:cacheprovider -o timeout_method=thread -q -rfE $S/ops/test_eb_r3_ops.py $S/ops/test_eb_r3_bng.py $S/ops/test_eb_ra_softmax.py > $OUTD/log_${tag}_${i}_$v.txt 2>&1
  rc=$?; echo "== $tag $v ($i) rc=$rc $(date -u +%T): $(grep -E 'passed|failed' $OUTD/log_${tag}_${i}_$v.txt | tail -1)"
  [[ $rc != 0 ]] && grep -a -E "^(FAILED|ERROR)" $OUTD/log_${tag}_${i}_$v.txt | head -8 | cut -c1-220
  elfsum $TT_METAL_CACHE > $OUTD/elfsum_$v.txt
done < $PLAN
echo "== compute ELF sets per variant"; for f in $OUTD/elfsum_*.txt; do echo "   $(basename $f): $(wc -l < $f) kernels, set $(md5sum < $f | cut -c1-12)"; done
B=${PROF_BASE:-head}
for v in $(awk '!/^#/ && NF {print $2}' $PLAN | sort -u | grep -vx $B); do
  echo "== kernels whose compute ELFs differ, $v against $B:"; join <(sort $OUTD/elfsum_$B.txt) <(sort $OUTD/elfsum_$v.txt) | awk '$2 != $3 {print "   " $1}'
  n=$(join <(sort $OUTD/elfsum_$B.txt) <(sort $OUTD/elfsum_$v.txt) | awk '$2 != $3' | wc -l); [[ $n == 0 ]] && echo "== GUARD FAIL: $v compiled to $B's ELFs"
done
for tag in $(awk '!/^#/ && NF {print $1}' $PLAN | sort -u); do echo "== reduce $tag (base $B)"; PROF_BASE=$B python $S/ops/prof_reduce.py $OUTD $tag 2>&1; done
echo "== done $(date -u +%T)"
