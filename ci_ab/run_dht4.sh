#!/usr/bin/env bash
# #59143 fast tilize, two chunks per DEST section (job T4: why max_pool2d's tiled output reads slower) (knowledge/process/17_CI_AB.md), one P150. Variants (ci_ab/variants/<v>/tree,
# files of ci_ab/variants/files.txt, placed in the checkout and in the installed ttnn package before each build or run):
# base (the PR head merged with main), fin (pairing, out-of-line helper), fnp (fin with the pairing switched off at run time,
# the code kept), floop (pairing in one loop with a run-time slot, no second copy of the loop), base2 (A/A).
#   1. device time of the fast tilize callers (ids/prof_ft.txt), tracy, 4 repetitions (the first dropped), the variants
#      forward then backward
#   2. bits: test_r3_audit_dumps.py T3 (every bf16 pattern through ttnn.tilize), base and floop
set +e +o pipefail
export HWLOCK_HELD=ci TRACY_NO_WEB_SERVER=1
R=$(git rev-parse --show-toplevel); S=$R/ci_ab; cd $R
SP=$(python -c "import ttnn, os; print(os.path.dirname(os.path.dirname(os.path.abspath(ttnn.__file__))))")
echo "== checkout $R, package root $SP, $(git log --oneline -1)"
OBJ=$(find / -name riscv-tt-elf-objcopy -type f 2>/dev/null | head -1); echo "== objcopy $OBJ"
OUTD=$R/generated/tab; mkdir -p $OUTD
src() { case $1 in base2) echo base ;; *) echo $1 ;; esac; }
place() {
  local d=$S/variants/$(src $1)
  while read -r rel; do
    [[ -z $rel ]] && continue
    cp "$d/tree/$rel" "$R/$rel"
    local suf; suf=$(echo "$rel" | awk -F/ '{n=NF; print $(n-3)"/"$(n-2)"/"$(n-1)"/"$n}')
    while read -r p; do [[ -n $p && $p != "$R/$rel" ]] && cp "$d/tree/$rel" "$p"; done < <(find "$SP" -path "*/$suf" -type f 2>/dev/null)
  done < $S/variants/files.txt
}
while read -r rel; do suf=$(echo "$rel" | awk -F/ '{n=NF; print $(n-3)"/"$(n-2)"/"$(n-1)"/"$n}'); echo "== $rel -> $(find "$SP" -path "*/$suf" -type f 2>/dev/null | wc -l) package copies"; done < $S/variants/files.txt
elfsum() {
  local c=$1
  for k in $(ls -d $c/*/kernels/* 2>/dev/null | xargs -n1 basename | sort -u); do
    h=$(for e in $(ls $c/*/kernels/$k/*/trisc[012]/trisc[012].elf 2>/dev/null | sort); do
          if [[ -n $OBJ ]]; then $OBJ -O binary -j .text "$e" /tmp/elf.bin 2>/dev/null && md5sum < /tmp/elf.bin; else md5sum < "$e"; fi
        done | md5sum | cut -c1-12)
    echo "   $k $h"
  done
}
cache() { export TT_METAL_CACHE=$HOME/tabcache_$1; rm -rf $TT_METAL_CACHE; mkdir -p $TT_METAL_CACHE; }

# 1. device time
MODS=$(cut -d: -f1 $S/ids/prof_ft.txt | sort -u | tr '\n' ' ')
i=0
for v in base fin fnp floop base2 base2 floop fnp fin base; do
  i=$((i+1))
  place $v
  if [[ " ${PFRESH:-} " != *" $v "* ]]; then cache prof_$v; PFRESH="${PFRESH:-} $v"; else export TT_METAL_CACHE=$HOME/tabcache_prof_$v; fi
  OUT=$OUTD/out_pt_${i}_$v; rm -rf $OUT; mkdir -p $OUT
  MM_REPS=4 FR_SELECT=$S/ids/prof_ft.txt PYTHONPATH="$S/tools:$R${PYTHONPATH:+:$PYTHONPATH}" timeout -s INT -k 60 1800 \
    python -m tracy -r -p --no-web-server -o $OUT -m pytest -p mm_prof_plugin -p no:cacheprovider -o timeout_method=thread -q -rfE $S/plugin $MODS \
    > $OUTD/log_pt_${i}_$v.txt 2>&1
  echo "== prof $v ($i) rc=$? $(date -u +%T): $(grep -E 'passed|failed' $OUTD/log_pt_${i}_$v.txt | tail -1)"
  [[ $i == 1 ]] && grep -a -E "Error|error|FAILED|No module|Traceback" $OUTD/log_pt_${i}_$v.txt | head -8 | cut -c1-240
  if [[ $i -le 5 ]]; then echo "== compute ELFs of $v:"; elfsum $TT_METAL_CACHE | grep -E 'tilize|pool|compute|groupnorm|rotary|conv|cache|sdpa'; fi
done
for b in base base2; do echo "== reduce pt (base $b)"; PROF_BASE=$b python $S/tools/prof_reduce.py $OUTD pt 2>&1; done
# 2. bits
for v in base floop; do
  place $v; cache bits_$v
  AUDIT_DUMP_DIR=$OUTD/dump AUDIT_TAG=$v timeout -s INT -k 60 1200 python -m pytest -p no:cacheprovider -o timeout_method=thread -q -rfE \
    -k "t3" tests/ttnn/unit_tests/operations/pool/test_r3_audit_dumps.py > $OUTD/log_bits_$v.txt 2>&1
  echo "== bits $v rc=$? $(date -u +%T): $(grep -E 'passed|failed' $OUTD/log_bits_$v.txt | tail -1)"
  echo "== compute ELFs of bits $v:"; elfsum $TT_METAL_CACHE | grep -E 'tilize|compute'
done
python - $OUTD/dump <<'PY'
import glob, os, sys, torch
d = sys.argv[1]
names = sorted(os.path.basename(p) for p in glob.glob(f"{d}/base/*.pt"))
iv = {1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}
same = diff = miss = 0
for n in names:
    p = f"{d}/floop/{n}"
    if not os.path.exists(p):
        miss += 1; print("   missing on floop:", n); continue
    a, b = torch.load(f"{d}/base/{n}", weights_only=True), torch.load(p, weights_only=True)
    wa = a.contiguous().view(iv[a.element_size()]) if a.is_floating_point() else a
    wb = b.contiguous().view(iv[b.element_size()]) if b.is_floating_point() else b
    k = int((wa != wb).sum()) if wa.shape == wb.shape else -1
    if k == 0: same += 1
    else: diff += 1; print(f"   {n}: {k} words differ" if k > 0 else f"   {n}: shape {tuple(b.shape)} vs {tuple(a.shape)}")
print(f"== bits: {len(names)} outputs of base, {same} identical on floop (raw words), {diff} differ, {miss} missing")
PY

place base
echo "== DONE $(date -u +%T)"
