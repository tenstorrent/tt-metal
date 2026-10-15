#!/usr/bin/env bash
# Scratch A/B job (#60646 regression), not for merge. The job's tree is main plus #60646's test_grid_sample.py. Arms (ci_ab/v/<arm>/,
# files in ci_ab/files.txt, copied into the checkout and the installed package before every run, fresh JIT cache per arm):
#   M  main: kernels and LLK
#   S  #60646 head 9d48d7149b5: pool and bilinear kernels with the pack untilize init around the narrower last chunk
#   X  main's kernels, LLK _llk_pack_untilize_ runs the MOP with the block's tile count in the MOP word
#   X2 A/A copy of X
# 1. op level: the pool, upsample, grid sample and untilize nodes of ci_ab/nodes.txt under the device profiler, M S X X2 X2 X S M
#    in two blocks, R3_REPS calls per test (the first a warm-up), per op code sums (sum_op19.py), table by rg_table.py.
#    Guard: the compute_pool_2d and bilinear trisc .text hash of an arm is the same in all its runs, X = X2, M, S, X distinct.
# 2. words: every kernel ELF of M's cache against X's (paired by kernel name and generated headers): differing .text words,
#    each decoded (worddiff.py); X keeps main's kernels, so the only expected difference is the MOP word of a pack untilize.
# 3. bits: M, S and X run the pool family selection (pool_ids.txt) with every ttnn.to_torch result hashed, then grid_diag.py.
set +e +o pipefail
cd /work; export TT_METAL_HOME=/work TRACY_NO_WEB_SERVER=1 R3_REPS=${R3_REPS:-8} AB_NODES_FILE=/work/ci_ab/nodes.txt
S=/work/ci_ab; OUT=/work/generated/rg; rm -rf $OUT; mkdir -p $OUT
python3 -c "import tracy" 2>/dev/null || export PYTHONPATH=/work:/work/tools
SP=$(cd /tmp && env -u PYTHONPATH python3 -c "import ttnn, os; print(os.path.dirname(os.path.dirname(os.path.abspath(ttnn.__file__))))")
echo "== $(date -u +%FT%TZ) $(git log -1 --format='%h %s') package root $SP $(hostname)"
OBJ=/work/runtime/sfpi/compiler/bin/riscv-tt-elf-objcopy; [[ -x $OBJ ]] || OBJ=$(which riscv-tt-elf-objcopy 2>/dev/null)
[[ -x $OBJ ]] || OBJ=$(find /work/runtime /opt -name riscv-tt-elf-objcopy -type f 2>/dev/null | head -1)
export RG_OBJCOPY=$OBJ
suffix() { echo "$1" | awk -F/ '{n=NF; print $(n-3)"/"$(n-2)"/"$(n-1)"/"$n}'; }
declare -A PK PD
while read -r rel; do
  PK[$rel]=$(find "$SP" -path "*/$(suffix $rel)" -type f 2>/dev/null)
  [[ -z ${PK[$rel]} ]] && { s=$(dirname "$rel" | awk -F/ '{n=NF; print $(n-2)"/"$(n-1)"/"$n}'); PD[$rel]=$(find "$SP" -type d -path "*/$s" 2>/dev/null); }
  echo "RAB FILE $rel package copies: $(echo ${PK[$rel]} | wc -w) dirs: $(echo ${PD[$rel]:-} | wc -w); tree equals main: $(git show HEAD:$rel 2>/dev/null | cmp -s - /work/$rel && echo yes || echo no)"
done < $S/files.txt
use() {  # use <cache tag> <files dir>
  while read -r rel; do
    cp "$S/v/$2/$rel" "/work/$rel"
    for p in ${PK[$rel]}; do cp "$S/v/$2/$rel" "$p"; done
    for d in ${PD[$rel]:-}; do cp "$S/v/$2/$rel" "$d/"; done
  done < $S/files.txt
  echo "RAB FILES $1 $(while read -r rel; do cat /work/$rel; done < $S/files.txt | md5sum | cut -c1-8)"
  export TT_METAL_CACHE=$OUT/cache_$1; mkdir -p $TT_METAL_CACHE
}
elfsum() {  # .text hash of every compute_pool_2d and bilinear trisc ELF of a cache, order independent
  for e in $(ls $1/*/kernels/compute_pool_2d/*/trisc*/*.elf $1/*/kernels/bilinear/*/trisc*/*.elf 2>/dev/null); do
    $OBJ -O binary -j .text "$e" /tmp/e.bin 2>/dev/null && md5sum < /tmp/e.bin | cut -c1-12
  done | sort | md5sum | cut -c1-12
}

# 1. op level
declare -A ES
r=0; fail=0
for blk in 1 2; do for a in M S X X2 X2 X S M; do
  r=$((r+1)); use $a ${a%2}
  o=$OUT/${a}_r$r; mkdir -p $o; t0=$(date +%s)
  timeout -s INT -k 60 1500 python3 -m tracy -r -p --no-web-server -o $o $S/run_nodes.py > $o.log 2>&1
  rc=$?
  echo "== run $r $a rc=$rc $(( $(date +%s) - t0 )) s: $(grep -aE '[0-9]+ (passed|failed)' $o.log | tail -1)"
  [[ $rc -ne 0 ]] && { fail=1; grep -a -m5 -E '^(FAILED|ERROR)|Error' $o.log | cut -c1-200; }
  python3 $S/sum_op19.py $o $a $r
  e=$(elfsum $TT_METAL_CACHE); ES[$a]+=" $e"
  echo "RAB ELF $a r$r $e $(ls -d $TT_METAL_CACHE/*/kernels/compute_pool_2d/* $TT_METAL_CACHE/*/kernels/bilinear/* 2>/dev/null | wc -l) builds"
done; done
uniq1() { echo $1 | tr ' ' '\n' | sort -u | wc -l; }
one() { echo $1 | tr ' ' '\n' | grep . | head -1; }
gs=ok
for a in M S X X2; do [[ $(uniq1 "${ES[$a]}") == 1 ]] || gs=UNSTABLE; done
[[ -n $OBJ ]] || gs=VOID
gx=$([[ $(one "${ES[X]}") == $(one "${ES[X2]}") ]] && echo ok || echo DIFF)
gd=$([[ $(one "${ES[M]}") != $(one "${ES[X]}") && $(one "${ES[M]}") != $(one "${ES[S]}") && $(one "${ES[S]}") != $(one "${ES[X]}") ]] && echo ok || echo VOID)
echo "RAB GUARD stable=$gs X_vs_X2=$gx M_S_X_distinct=$gd"; [[ $gs == ok && $gx == ok && $gd == ok ]] || fail=1

# 2. words, M against X (and X against X2: no difference expected)
python3 $S/worddiff.py $OUT/cache_M $OUT/cache_X M X
python3 $S/worddiff.py $OUT/cache_X $OUT/cache_X2 X X2 | grep -a -E "^WORDS (SUM|NOPAIR)"
for a in M S X; do python3 $S/elfhash.py $a $OUT/cache_$a compute_pool_2d bilinear untilize; done

# 3. bits and the #59368 readback
P=tests/ttnn/unit_tests/operations/pool
PMODS="$P/test_maxpool2d.py $P/test_avgpool2d.py $P/test_upsample.py $P/test_grid_sample.py"
PT="python3 -m pytest -s -q -p no:cacheprovider -o timeout_method=thread -o addopts= -rfE $S/plugin"
for a in M S X; do
  use bits_$a $a
  HWLOCK_HELD=ci BITS_TAG=$a SEL_FILE=$S/pool_ids.txt timeout -s INT -k 60 1500 $PT $PMODS > $OUT/bits_$a.log 2>&1
  echo "== bits $a rc=$? $(grep -aE '[0-9]+ (passed|failed)' $OUT/bits_$a.log | tail -1) $(grep -a '^SEL' $OUT/bits_$a.log | tail -1)"
  grep -a -E '^(FAILED|ERROR)' $OUT/bits_$a.log | cut -c1-250
  grep -a -o "BITS $a .*" $OUT/bits_$a.log | sed "s/^BITS $a //" | sort > $OUT/bits_$a.txt
  echo "== BITS $a lines $(wc -l < $OUT/bits_$a.txt)"
  [[ -s $OUT/bits_$a.txt ]] || { echo "== BITSCMP VOID: $a printed no BITS lines"; fail=1; }
  HWLOCK_HELD=ci timeout -s INT -k 60 900 python3 $S/grid_diag.py $OUT/gd_$a $a > $OUT/gd_$a.log 2>&1
  echo "== grid_diag $a rc=$?"; grep -a -E "^$a c|DIAG DONE" $OUT/gd_$a.log | cut -c1-600
done
cmpbits() {  # cmpbits <A> <B>
  awk -v A=$1 -v B=$2 'function key(l){sub(/ [^ ]+$/,"",l); return l} function hv(l){sub(/^.* /,"",l); return l}
    NR==FNR{h[key($0)]=hv($0); next}
    {k=key($0); seen[k]=1; if(k in h){c++; if(h[k]==hv($0)) e++; else {d++; print "DIFF " A "_" B " " k}} else ob++}
    END{for(k in h) if(!(k in seen)) oa++; print "== BITSKEY " A " " B " common " c+0 ", equal " e+0 ", differ " d+0 ", " A " only " oa+0 ", " B " only " ob+0}' \
    $OUT/bits_$1.txt $OUT/bits_$2.txt | cut -c1-300
}
cmpbits S X; cmpbits M X; cmpbits M S
use M M
echo "== done $(date -u +%FT%TZ) fail=$fail"
exit $fail
