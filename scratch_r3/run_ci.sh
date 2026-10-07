#!/usr/bin/env bash
# Round 3 matmul (#58817 second review): device-profiler A/B runs on one CI Blackhole runner. Each variant's files
# (scratch_r3/variants/<v>/tree) are copied into the checkout and into the installed ttnn package (the JIT compiler
# reads the LLK headers from the package and the kernel sources from the checkout), each variant runs with its own
# empty JIT cache, and the head's files are put back after every run. Not for our card: the tests' HWLOCK_HELD guard is
# for scripts/hwlock.sh there, so it is set to "ci" here.
# usage: run_ci.sh <plan file>, lines "<tag> <variant> <test,test...> [conf] [pytest -k expr]"
set -uo pipefail
export HWLOCK_HELD=ci TRACY_NO_WEB_SERVER=1
R=$(git rev-parse --show-toplevel); S=$R/scratch_r3; cd $R
SP=$(python -c "import ttnn, os; print(os.path.dirname(os.path.dirname(os.path.abspath(ttnn.__file__))))")
echo "== checkout $R, package root $SP, $(python -c 'import ttnn; print(ttnn.__file__)')"
export PYTHONPATH="$S/tools:$R${PYTHONPATH:+:$PYTHONPATH}"
OBJ=$(find / -name riscv-tt-elf-objcopy -type f 2>/dev/null | head -1); echo "== objcopy $OBJ"
place() {  # place <variant dir>: copy every file of the head's union from <dir>/tree (or the head's) into checkout and package
  local d=$1
  while read -r rel; do
    [[ -z $rel ]] && continue
    local src=$S/variants/head/tree/$rel; [[ -f $d/tree/$rel ]] && src=$d/tree/$rel
    cp "$src" "$R/$rel"
    local suf; suf=$(echo "$rel" | awk -F/ '{n=NF; s=$(n-3)"/"$(n-2)"/"$(n-1)"/"$n; print s}')
    while read -r p; do [[ -n $p && $p != "$R/$rel" ]] && cp "$src" "$p"; done < <(find "$SP" -path "*/$suf" -type f 2>/dev/null)
  done < $S/variants/head/files.txt
}
if [[ ${CI_SHOW_PATHS:-1} == 1 ]]; then
  while read -r rel; do suf=$(echo "$rel" | awk -F/ '{n=NF; print $(n-3)"/"$(n-2)"/"$(n-1)"/"$n}'); echo "== $rel -> $(find "$SP" -path "*/$suf" -type f 2>/dev/null | tr '\n' ' ')"; done < $S/variants/head/files.txt
fi
elfsum() {  # combined hash per kernel name of the compute ELFs' loaded bytes in a cache
  local c=$1
  for k in $(ls -d $c/*/kernels/* 2>/dev/null | xargs -n1 basename | sort -u); do
    h=$(for e in $(ls $c/*/kernels/$k/*/trisc[012]/trisc[012].elf 2>/dev/null | sort); do
          if [[ -n $OBJ ]]; then $OBJ -O binary -j .text "$e" /tmp/elf.bin 2>/dev/null && md5sum < /tmp/elf.bin; else md5sum < "$e"; fi
        done | md5sum | cut -c1-12)
    echo "   $k $h"
  done
}
OUTD=$R/generated/r3ci; mkdir -p $OUTD
i=0
while read -r tag v tests conf kexp; do
  [[ -z ${tag:-} || $tag == \#* ]] && continue
  i=$((i+1))
  if [[ $v == head ]]; then place $S/variants/head; else place $S/variants/$v; fi
  export TT_METAL_CACHE=$HOME/r3cache_$v; mkdir -p $TT_METAL_CACHE
  T=(); IFS=, read -ra TL <<< "$tests"; for t in "${TL[@]}"; do T+=("$S/tools/$t"); done
  C=(); [[ ${conf:-} == conf ]] && C=(-p conftest)
  K=(); [[ -n ${kexp:-} && ${kexp:-} != - ]] && K=(-k "$kexp")
  OUT=$OUTD/out_${tag}_${i}_$v; rm -rf $OUT; mkdir -p $OUT
  echo "== $tag $v ($i) $tests $(date -u +%T)"
  if [[ $tag == bits* ]]; then  # outputs as raw bits, no profiler
    B=$OUTD/bits_$v; mkdir -p $B
    JSDPA_OUT=$B V12_OUT=$B SYNC_OUT=$B TT_METAL_LLK_ASSERTS=${BITS_ASSERTS:-1} timeout -s INT -k 60 ${CI_LIMIT:-900} python -m pytest "${C[@]}" -p no:cacheprovider -o timeout_method=thread -q -rfE "${K[@]}" "${T[@]}" > $OUTD/log_${tag}_${i}_$v.txt 2>&1
    rc=$?; echo "== $tag $v ($i) rc=$rc $(date -u +%T): $(grep -E 'passed|failed' $OUTD/log_${tag}_${i}_$v.txt | tail -1)"
    if [[ $rc -ne 0 ]] || ! grep -q passed $OUTD/log_${tag}_${i}_$v.txt; then tail -40 $OUTD/log_${tag}_${i}_$v.txt; fi
    continue
  fi
  timeout -s INT -k 60 ${CI_LIMIT:-900} python -m tracy -r -p --no-web-server -o $OUT -m pytest "${C[@]}" -p mm_prof_plugin -p no:cacheprovider -o timeout_method=thread -q -rfE "${K[@]}" "${T[@]}" > $OUTD/log_${tag}_${i}_$v.txt 2>&1
  rc=$?; echo "== $tag $v ($i) rc=$rc $(date -u +%T): $(grep -E 'passed|failed' $OUTD/log_${tag}_${i}_$v.txt | tail -1)"
  if [[ $rc -ne 0 ]] || ! grep -q passed $OUTD/log_${tag}_${i}_$v.txt; then tail -40 $OUTD/log_${tag}_${i}_$v.txt; fi
  echo "== compute ELFs of $v:"; elfsum $TT_METAL_CACHE
done < "$1"
place $S/variants/head
if ls -d $OUTD/bits_* >/dev/null 2>&1; then
  echo "== bits: every saved output of each variant against the head's"
  python - "$OUTD" <<'PY'
import glob, os, sys, torch
d = sys.argv[1]
head = {os.path.basename(f): f for f in glob.glob(os.path.join(d, "bits_head", "*.pt"))}
for vd in sorted(glob.glob(os.path.join(d, "bits_*"))):
    v = os.path.basename(vd)[5:]
    for f in sorted(glob.glob(os.path.join(vd, "*.pt"))):
        n = os.path.basename(f)
        a = torch.load(f)
        if v == "head":
            continue
        if n not in head:
            print(f"   {v} {n}: no head output"); continue
        b = torch.load(head[n])
        ta = a if isinstance(a, tuple) else (a,)
        tb = b if isinstance(b, tuple) else (b,)
        same = all(torch.equal(x, y) for x, y in zip(ta, tb))
        diff = sum(int((x != y).sum()) for x, y in zip(ta, tb))
        total = sum(x.numel() for x in ta)
        print(f"   {v} {n}: {'bit-identical' if same else 'DIFFERENT'} ({diff} of {total} elements differ)")
    if v == "head":
        names = sorted(head)
        for n in names:
            if n.endswith("_full.pt"):
                h = n[:-8] + "_half.pt"
                if h in head:
                    x, y = torch.load(head[n]), torch.load(head[h])
                    print(f"   head {n[:-8]} full against half: {'bit-identical' if torch.equal(x, y) else 'DIFFERENT (%d of %d)' % (int((x != y).sum()), x.numel())}")
PY
fi
for tag in $(awk '!/^#/ && NF {print $1}' "$1" | sort -u); do
  echo "== reduce $tag (base head)"; PROF_BASE=head python $S/tools/prof_reduce.py $OUTD $tag 2>&1
  if [[ -n ${CI_PAIRS:-} ]]; then echo "== pairs $tag ($CI_PAIRS)"; PROF_PAIRS=$CI_PAIRS python $S/tools/prof_reduce.py $OUTD $tag 2>&1; fi
done
tar -czf $R/generated/r3ci.tgz -C $R/generated r3ci 2>/dev/null; ls -la $R/generated/r3ci.tgz
