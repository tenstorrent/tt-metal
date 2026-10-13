#!/usr/bin/env bash
# DEST hand-off job T7 (knowledge/process/17_CI_AB.md), one Blackhole runner: the math-only DEST clear (#58816) opted in by
# binary_ng's block-pack sections (#58818) and the ttnn.tilize kernels (#59143). Checkout: #58818 + #59143 heads merged.
# Variants (ci_ab/variants/<v>/tree, files of files.txt, placed in the checkout and the installed ttnn package):
# base (the heads before the opt-in), fin (with it), base2 (base under its own cache or build: the A/A pair), lay1/lay2/lay4
# (base with 1, 2 or 4 NOPs at the top of each harness thread: layout controls).
#   1. tilize callers' device time (ids/prof_ft.txt), tracy, 4 repetitions (first dropped): base fin base2 | base2 fin base
#   2. binary_ng block sections at SDXL shapes, bits and in-process determinism (ebops/test_eb_hr_ops.py): same order
#   3. bits: test_r3_audit_dumps.py T3 (ttnn.tilize, every bf16 pattern), base and fin
#   4. tt-llk harness: functional fast tilize with asserts (fin); perf modules NC base fin base2 lay1 lay2 lay4, WC base fin
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


# 1. tilize callers, device time
MODS=$(cut -d: -f1 $S/ids/prof_ft.txt | sort -u | tr '\n' ' ')
i=0
for v in base fin base2 base2 fin base; do
  i=$((i+1))
  place $v
  if [[ " ${PFRESH:-} " != *" $v "* ]]; then cache prof_$v; PFRESH="${PFRESH:-} $v"; else export TT_METAL_CACHE=$HOME/tabcache_prof_$v; fi
  OUT=$OUTD/out_pt_${i}_$v; rm -rf $OUT; mkdir -p $OUT
  MM_REPS=4 FR_SELECT=$S/ids/prof_ft.txt PYTHONPATH="$S/tools:$R${PYTHONPATH:+:$PYTHONPATH}" timeout -s INT -k 60 1800 \
    python -m tracy -r -p --no-web-server -o $OUT -m pytest -p mm_prof_plugin -p no:cacheprovider -o timeout_method=thread -q -rfE $S/plugin $MODS \
    > $OUTD/log_pt_${i}_$v.txt 2>&1
  echo "== prof $v ($i) rc=$? $(date -u +%T): $(grep -E 'passed|failed' $OUTD/log_pt_${i}_$v.txt | tail -1)"
  if [[ $i -le 3 ]]; then echo "== compute ELFs of $v:"; elfsum $TT_METAL_CACHE | grep -E 'tilize|pool|compute|groupnorm|rotary|conv|cache|sdpa'; fi
done
for b in base base2; do echo "== reduce pt (base $b)"; PROF_BASE=$b python $S/tools/prof_reduce.py $OUTD pt 2>&1; done

# 2. binary_ng block sections at SDXL shapes
EB=$OUTD/eb; mkdir -p $EB; i=0
for v in base fin base2 base2 fin base; do
  i=$((i+1))
  place $v
  export TT_METAL_CACHE=$HOME/ebcache_$v
  if [[ " ${EFRESH:-} " != *" $v "* ]]; then rm -rf $TT_METAL_CACHE; EFRESH="${EFRESH:-} $v"; fi
  mkdir -p $TT_METAL_CACHE
  OUT=$EB/out_o_${i}_$v
  EB_REPS=4 EB_DET_OUT=$EB/det_${i}_$v.jsonl EB_HASH_OUT=$EB/hash_${i}_$v.json PYTHONPATH="$S/ebops:$R${PYTHONPATH:+:$PYTHONPATH}" timeout -s INT -k 60 1800 \
    python -m tracy -r -p --no-web-server -o $OUT -m pytest -p eb_prof_plugin -p eb_bits_plugin -p no:cacheprovider -o timeout_method=thread -q -rfE $S/ebops/test_eb_hr_ops.py \
    > $EB/log_o_${i}_$v.txt 2>&1
  rc=$?; echo "== eb $v ($i) rc=$rc $(date -u +%T): $(grep -E 'passed|failed' $EB/log_o_${i}_$v.txt | tail -1)"
  [[ $rc != 0 ]] && grep -a -E "^(FAILED|ERROR)|error:" $EB/log_o_${i}_$v.txt | head -8 | cut -c1-240
  if [[ $i -le 3 ]]; then echo "== eb compute ELFs of $v:"; elfsum $TT_METAL_CACHE | grep -E 'eltwise|binary|compute'; fi
done
for b in base base2; do echo "== reduce eb (base $b)"; PROF_BASE=$b python $S/ebops/prof_reduce_cfg.py $EB o 2>&1; done
echo "== eb bits"; python $S/ebops/bits_cmp.py $EB base 2>&1 | tail -40
echo "== eb det"; python $S/ebops/det_cmp.py $EB base 2>&1 | tail -40

# 3. bits, ttnn.tilize
for v in base fin; do
  place $v; cache bits_$v
  AUDIT_DUMP_DIR=$OUTD/dump AUDIT_TAG=$v timeout -s INT -k 60 1200 python -m pytest -p no:cacheprovider -o timeout_method=thread -q -rfE \
    -k "t3" tests/ttnn/unit_tests/operations/pool/test_r3_audit_dumps.py > $OUTD/log_bits_$v.txt 2>&1
  echo "== bits $v rc=$? $(date -u +%T): $(grep -E 'passed|failed' $OUTD/log_bits_$v.txt | tail -1)"
done
python - $OUTD/dump <<'PYX'
import glob, os, sys, torch
d = sys.argv[1]
names = sorted(os.path.basename(p) for p in glob.glob(f"{d}/base/*.pt"))
iv = {1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}
same = diff = miss = 0
for n in names:
    p = f"{d}/fin/{n}"
    if not os.path.exists(p):
        miss += 1; print("   missing on fin:", n); continue
    a, b = torch.load(f"{d}/base/{n}", weights_only=True), torch.load(p, weights_only=True)
    wa = a.contiguous().view(iv[a.element_size()]) if a.is_floating_point() else a
    wb = b.contiguous().view(iv[b.element_size()]) if b.is_floating_point() else b
    k = int((wa != wb).sum()) if wa.shape == wb.shape else -1
    if k == 0: same += 1
    else: diff += 1; print(f"   {n}: {k} words differ" if k > 0 else f"   {n}: shape {tuple(b.shape)} vs {tuple(a.shape)}")
print(f"== bits: {len(names)} outputs of base, {same} identical on fin (raw words), {diff} differ, {miss} missing")
PYX

# 4. harness
place fin
python -m venv $HOME/llkvenv && source $HOME/llkvenv/bin/activate
pip install -q -r tt_metal/tt-llk/tests/requirements.txt > $OUTD/log_pip.txt 2>&1; echo "== pip rc=$? $(tail -1 $OUTD/log_pip.txt)"
(cd tt_metal/tt-llk/tests && ./setup_testing_env.sh > $OUTD/log_setup.txt 2>&1); echo "== setup rc=$? $(tail -2 $OUTD/log_setup.txt | tr '\n' ' ')"
export CHIP_ARCH=blackhole LLK_HOME=$R/tt_metal/tt-llk LLK_PERF_L1_MUX_GROUP=0 PERF_KEEP_RUNS=400
cd $R/tt_metal/tt-llk/tests/python_tests
mapfile -t F < $S/ids/func_ft.txt
rm -rf $HOME/llkb_func
TT_LLK_DISABLE_ASSERTS=0 RUNNER_TEMP=$HOME/llkb_func TMPDIR=$HOME/llkb_func timeout -s INT -k 60 1500 python -m pytest --compile-producer -n 8 -p no:cacheprovider -q "${F[@]}" > $OUTD/log_func_prod.txt 2>&1
echo "== harness functional build rc=$? $(date -u +%T): $(grep -E 'passed|failed' $OUTD/log_func_prod.txt | tail -1)"
TT_LLK_DISABLE_ASSERTS=0 RUNNER_TEMP=$HOME/llkb_func TMPDIR=$HOME/llkb_func timeout -s INT -k 60 1500 python -m pytest --compile-consumer -p no:cacheprovider -q -rfE "${F[@]}" > $OUTD/log_func.txt 2>&1
echo "== harness functional (fin, asserts on) rc=$? $(date -u +%T): $(grep -E 'passed|failed' $OUTD/log_func.txt | tail -1)"
grep -a -m10 -E "^FAILED|LLK_ASSERT" $OUTD/log_func.txt | cut -c1-240
mapfile -t P < $S/ids/prod.txt
build() {  # build <variant> <dir> [extra]
  (cd $R && place $1); rm -rf $2
  TT_LLK_DISABLE_ASSERTS=1 RUNNER_TEMP=$2 TMPDIR=$2 timeout -s INT -k 60 1500 python -m pytest --compile-producer -n 8 -p no:cacheprovider -q $3 "${P[@]}" > $OUTD/log_b_$1$3.txt 2>&1
  echo "== harness build $1 $3 rc=$? $(date -u +%T): $(grep -aE '[0-9]+ (passed|failed|error)' $OUTD/log_b_$1$3.txt | tail -1)"
}
for v in base fin base2 lay1 lay2 lay4; do build $v $HOME/llkb_$v; done
for v in base fin; do build $v $HOME/llkw_$v --dump-perf-counters; done
python $S/tools/elfcheck.py base=$HOME/llkb_base fin=$HOME/llkb_fin base2=$HOME/llkb_base2 lay1=$HOME/llkb_lay1 lay2=$HOME/llkb_lay2 lay4=$HOME/llkb_lay4 | tee $OUTD/elf.txt | grep ELFSUM
python - $OUTD/elf.txt <<'PY' || { echo "== GUARD FAIL"; exit 1; }
import re, sys
s = {m.group(1): tuple(map(int, m.group(2, 3, 4, 5, 6))) for m in (re.match(r"ELFSUM (\S+) dirs=(\d+) common=(\d+) differ_from_first unpack=(\d+) math=(\d+) pack=(\d+)", l) for l in open(sys.argv[1])) if m}
ok = s["base2"][2:] == (0, 0, 0) and s["fin"][4] > 0 and s["fin"][3] > 0 and min(s["lay1"][3], s["lay2"][3], s["lay4"][3]) > 0
print("== GUARD", "ok" if ok else "FAIL", s); sys.exit(0 if ok else 1)
PY
run() {  # run <variant> <tag> <dir> [extra]
  (cd $R && place $1)
  TT_LLK_DISABLE_ASSERTS=1 RUNNER_TEMP=$3 TMPDIR=$3 PERF_RUN_TAG=$2 timeout -s INT -k 60 1500 python -m pytest --compile-consumer -p no:cacheprovider -q -rfE $4 "${P[@]}" > $OUTD/log_r_$2.txt 2>&1
  echo "== harness run $1 $2 rc=$? $(date -u +%T): $(grep -aE '[0-9]+ (passed|failed)' $OUTD/log_r_$2.txt | tail -1)"
  grep -a -m3 -E "^FAILED|^ERROR" $OUTD/log_r_$2.txt | cut -c1-240
  for f in $(find $LLK_HOME/perf_data/runs/$2 -name "*.csv" 2>/dev/null | grep -v -e post -e counters); do echo "== csv $2 $(basename $f)"; cat "$f"; echo "== end csv"; done
  for f in $(find $LLK_HOME/perf_data/runs/$2 -name "*.counters.csv" 2>/dev/null); do echo "== ccsv $2 $(basename $f)"; python $S/tools/cfilter.py "$f"; echo "== end ccsv"; done
}
VS="base fin base2 lay1 lay2 lay4"; RV="lay4 lay2 lay1 base2 fin base"; j=0
for blk in 1 2; do for v in $VS $RV; do j=$((j + 1)); run $v h_${j}_$v $HOME/llkb_$v ""; done; done
for v in base fin; do j=$((j + 1)); run $v hw_${j}_$v $HOME/llkw_$v --dump-perf-counters; done
deactivate; cd $R

place base
echo "== DONE $(date -u +%T)"
