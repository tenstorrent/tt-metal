#!/usr/bin/env bash
# Regression audit of #59143 on one CI Blackhole runner (knowledge/process/17_CI_AB.md). Variants (ci_ab/variants/<v>/tree,
# the files of ci_ab/variants/files.txt): main (origin/main f0832760bf5), mrg (the PR head merged with it, 5230cc64dd7),
# head (the audit fix, 4574191022b). Before each run the variant's files go into the checkout and into the installed
# ttnn package (the JIT reads LLK and API headers from the package and kernel sources from the checkout); each run gets
# a fresh kernel cache of its own name (main2 is main's files again under another cache: the A/A pair).
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

# 1. bits
for v in main head; do
  place ${v}; cache bits_$v
  echo "== bits $v $(date -u +%T)"
  AUDIT_DUMP_DIR=$OUTD/dump AUDIT_TAG=$v timeout -s INT -k 60 1200 python -m pytest -p no:cacheprovider -o timeout_method=thread -q -rfE \
    -k "t2 or t3 or t4" tests/ttnn/unit_tests/operations/pool/test_r3_audit_dumps.py > $OUTD/log_bits_$v.txt 2>&1
  echo "== bits $v rc=$? $(date -u +%T): $(grep -E 'passed|failed' $OUTD/log_bits_$v.txt | tail -1)"
  echo "== compute ELFs of bits $v:"; elfsum $TT_METAL_CACHE | grep -E 'tilize|pool|bilinear|compute'
done
python - $OUTD/dump <<'PY'
import glob, os, sys, torch
d = sys.argv[1]
names = sorted(os.path.basename(p) for p in glob.glob(f"{d}/main/*.pt"))
iv = {1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}
same = diff = miss = 0
for n in names:
    p = f"{d}/head/{n}"
    if not os.path.exists(p):
        miss += 1; print("   missing on head:", n); continue
    a, b = torch.load(f"{d}/main/{n}", weights_only=True), torch.load(p, weights_only=True)
    wa = a.contiguous().view(iv[a.element_size()]) if a.is_floating_point() else a
    wb = b.contiguous().view(iv[b.element_size()]) if b.is_floating_point() else b
    k = int((wa != wb).sum()) if wa.shape == wb.shape else -1
    if k == 0: same += 1
    else: diff += 1; print(f"   {n}: {k} words differ" if k > 0 else f"   {n}: shape {tuple(b.shape)} vs {tuple(a.shape)}")
print(f"== bits: {len(names)} outputs of main, {same} identical on head (raw words), {diff} differ, {miss} missing; head has {len(glob.glob(f'{d}/head/*.pt'))}")
PY

# 2. device time
MODS=$(cut -d: -f1 $S/ids/prof_ft.txt | sort -u | tr '\n' ' ')
i=0
for v in main mrg head main2 main2 head mrg main; do
  i=$((i+1)); f=${v%2}
  place $f
  if [[ " ${PFRESH:-} " != *" $v "* ]]; then cache prof_$v; PFRESH="${PFRESH:-} $v"; else export TT_METAL_CACHE=$HOME/tabcache_prof_$v; fi
  OUT=$OUTD/out_pt_${i}_$v; rm -rf $OUT; mkdir -p $OUT
  echo "== prof $v ($i) $(date -u +%T)"
  MM_REPS=4 FR_SELECT=$S/ids/prof_ft.txt PYTHONPATH="$S/tools:$R${PYTHONPATH:+:$PYTHONPATH}" timeout -s INT -k 60 1500 \
    python -m tracy -r -p --no-web-server -o $OUT -m pytest -p mm_prof_plugin -p no:cacheprovider -o timeout_method=thread -q -rfE $S/plugin $MODS \
    > $OUTD/log_pt_${i}_$v.txt 2>&1
  echo "== prof $v ($i) rc=$? $(date -u +%T): $(grep -E 'passed|failed' $OUTD/log_pt_${i}_$v.txt | tail -1)"
  if [[ $i -le 3 ]]; then echo "== compute ELFs of $v:"; elfsum $TT_METAL_CACHE | grep -E 'tilize|pool|compute|groupnorm|rotary|conv|cache|sdpa'; fi
done
echo "== reduce pt (base main)"; PROF_BASE=main python $S/tools/prof_reduce.py $OUTD pt 2>&1
echo "== reduce pt (base main2)"; PROF_BASE=main2 python $S/tools/prof_reduce.py $OUTD pt 2>&1

# 3. tt-llk harness
place head
echo "== harness setup $(date -u +%T)"
python -m venv $HOME/llkvenv && source $HOME/llkvenv/bin/activate
pip install -q -r tt_metal/tt-llk/tests/requirements.txt > $OUTD/log_pip.txt 2>&1; echo "== pip rc=$? $(tail -1 $OUTD/log_pip.txt)"
(cd tt_metal/tt-llk/tests && ./setup_testing_env.sh > $OUTD/log_setup.txt 2>&1); echo "== setup rc=$? $(tail -2 $OUTD/log_setup.txt | tr '\n' ' ')"
export CHIP_ARCH=blackhole LLK_HOME=$R/tt_metal/tt-llk LLK_PERF_L1_MUX_GROUP=0
cd $R/tt_metal/tt-llk/tests/python_tests
mapfile -t F < $S/ids/func_t.txt
echo "== harness functional, LLK asserts on, head $(date -u +%T)"
rm -rf $HOME/llkb_func
TT_LLK_DISABLE_ASSERTS=0 TMPDIR=$HOME/llkb_func timeout -s INT -k 60 1500 python -m pytest --compile-producer -n 8 -p no:cacheprovider -q "${F[@]}" > $OUTD/log_func_prod.txt 2>&1
echo "== harness functional build rc=$? $(date -u +%T): $(grep -E 'passed|failed' $OUTD/log_func_prod.txt | tail -1)"
TT_LLK_DISABLE_ASSERTS=0 TMPDIR=$HOME/llkb_func timeout -s INT -k 60 1500 python -m pytest --compile-consumer -p no:cacheprovider -q -rfE "${F[@]}" > $OUTD/log_func.txt 2>&1
echo "== harness functional rc=$? $(date -u +%T): $(grep -E 'passed|failed' $OUTD/log_func.txt | tail -1)"
grep -a -m10 -E "FAILED|LLK_ASSERT|assert" $OUTD/log_func.txt | head -10
mapfile -t P < $S/ids/prod.txt
j=0
for v in main head head main; do
  j=$((j+1)); (cd $R && place $v)
  [[ $j == 1 || $j == 2 ]] && rm -rf $HOME/llkb_prod_$v
  echo "== harness prod $v ($j) $(date -u +%T)"
  if [[ ! -d $HOME/llkb_prod_$v ]]; then
    TT_LLK_DISABLE_ASSERTS=1 TMPDIR=$HOME/llkb_prod_$v LLK_PERF_RUN_TYPES=L1_TO_L1 timeout -s INT -k 60 900 \
      python -m pytest --compile-producer -n 8 -p no:cacheprovider -q "${P[@]}" > $OUTD/log_prodb_${j}_$v.txt 2>&1
    echo "== harness prod build $v rc=$? $(grep -E 'passed|failed' $OUTD/log_prodb_${j}_$v.txt | tail -1)"
  fi
  TT_LLK_DISABLE_ASSERTS=1 TMPDIR=$HOME/llkb_prod_$v PERF_RUN_TAG=tab_${j}_$v LLK_PERF_RUN_TYPES=L1_TO_L1 timeout -s INT -k 60 900 \
    python -m pytest --compile-consumer -p no:cacheprovider -q -rfE "${P[@]}" > $OUTD/log_prod_${j}_$v.txt 2>&1
  echo "== harness prod $v ($j) rc=$? $(date -u +%T): $(grep -E 'passed|failed' $OUTD/log_prod_${j}_$v.txt | tail -1)"
  f=$(ls $R/tt_metal/tt-llk/perf_data/runs/tab_${j}_$v/perf_fast_tilize_prod/perf_fast_tilize_prod.csv 2>/dev/null)
  [[ -n $f ]] && { echo "== csv prod $v ($j)"; cat "$f"; echo "== end csv"; }
done
cd $R; place head
tar -czf $R/generated/tab.tgz -C $R/generated tab 2>/dev/null; ls -la $R/generated/tab.tgz
echo "== DONE $(date -u +%T)"
