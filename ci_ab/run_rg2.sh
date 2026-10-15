#!/usr/bin/env bash
# Scratch confirmation job (#60646 regression), not for merge. Same setup as run_rg_ab.sh. Arms M main, X the MOP word fix,
# X2 A/A copy of X, on the 12 narrower-last-chunk nodes (ci_ab/nodes2.txt), M X X2 X2 X M three times, R3_REPS 8; guard as
# before; then the pack (trisc2) disassembly of every compute_pool_2d and bilinear build, M against X, as a diff with the
# addresses dropped (disasm_diff.py), to show what the longer X builds hold.
set +e +o pipefail
cd /work; export TT_METAL_HOME=/work TRACY_NO_WEB_SERVER=1 R3_REPS=${R3_REPS:-8} AB_NODES_FILE=/work/ci_ab/nodes2.txt
S=/work/ci_ab; OUT=/work/generated/rg2; rm -rf $OUT; mkdir -p $OUT
python3 -c "import tracy" 2>/dev/null || export PYTHONPATH=/work:/work/tools
SP=$(cd /tmp && env -u PYTHONPATH python3 -c "import ttnn, os; print(os.path.dirname(os.path.dirname(os.path.abspath(ttnn.__file__))))")
echo "== $(date -u +%FT%TZ) $(git log -1 --format='%h %s') package root $SP $(hostname)"
OBJ=/work/runtime/sfpi/compiler/bin/riscv-tt-elf-objcopy; [[ -x $OBJ ]] || OBJ=$(which riscv-tt-elf-objcopy 2>/dev/null)
[[ -x $OBJ ]] || OBJ=$(find /work/runtime /opt -name riscv-tt-elf-objcopy -type f 2>/dev/null | head -1)
export RG_OBJCOPY=$OBJ RG_OBJDUMP=${OBJ%objcopy}objdump
suffix() { echo "$1" | awk -F/ '{n=NF; print $(n-3)"/"$(n-2)"/"$(n-1)"/"$n}'; }
declare -A PK
while read -r rel; do PK[$rel]=$(find "$SP" -path "*/$(suffix $rel)" -type f 2>/dev/null); echo "RAB FILE $rel package copies: $(echo ${PK[$rel]} | wc -w)"; done < $S/files.txt
use() {
  while read -r rel; do cp "$S/v/$2/$rel" "/work/$rel"; for p in ${PK[$rel]}; do cp "$S/v/$2/$rel" "$p"; done; done < $S/files.txt
  echo "RAB FILES $1 $(while read -r rel; do cat /work/$rel; done < $S/files.txt | md5sum | cut -c1-8)"
  export TT_METAL_CACHE=$OUT/cache_$1; mkdir -p $TT_METAL_CACHE
}
elfsum() {
  for e in $(ls $1/*/kernels/compute_pool_2d/*/trisc*/*.elf $1/*/kernels/bilinear/*/trisc*/*.elf 2>/dev/null); do
    $OBJ -O binary -j .text "$e" /tmp/e.bin 2>/dev/null && md5sum < /tmp/e.bin | cut -c1-12
  done | sort | md5sum | cut -c1-12
}
declare -A ES
r=0; fail=0
for blk in 1 2 3; do for a in M X X2 X2 X M; do
  r=$((r+1)); use $a ${a%2}
  o=$OUT/${a}_r$r; mkdir -p $o; t0=$(date +%s)
  timeout -s INT -k 60 1500 python3 -m tracy -r -p --no-web-server -o $o $S/run_nodes.py > $o.log 2>&1
  rc=$?
  echo "== run $r $a rc=$rc $(( $(date +%s) - t0 )) s: $(grep -aE '[0-9]+ (passed|failed)' $o.log | tail -1)"
  [[ $rc -ne 0 ]] && { fail=1; grep -a -m5 -E '^(FAILED|ERROR)|Error' $o.log | cut -c1-200; }
  python3 $S/sum_op19.py $o $a $r
  e=$(elfsum $TT_METAL_CACHE); ES[$a]+=" $e"
  echo "RAB ELF $a r$r $e"
done; done
uniq1() { echo $1 | tr ' ' '\n' | grep . | sort -u | wc -l; }
one() { echo $1 | tr ' ' '\n' | grep . | head -1; }
gs=ok; for a in M X X2; do [[ $(uniq1 "${ES[$a]}") == 1 ]] || gs=UNSTABLE; done
gx=$([[ $(one "${ES[X]}") == $(one "${ES[X2]}") ]] && echo ok || echo DIFF)
gd=$([[ $(one "${ES[M]}") != $(one "${ES[X]}") && -n $OBJ ]] && echo ok || echo VOID)
echo "RAB GUARD stable=$gs X_vs_X2=$gx M_vs_X=$gd"; [[ $gs == ok && $gx == ok && $gd == ok ]] || fail=1
python3 $S/worddiff.py $OUT/cache_M $OUT/cache_X M X | grep -a -E "^WORDS (SUM|.* LEN)"
python3 $S/disasm_diff.py $OUT/cache_M $OUT/cache_X
use M M
echo "== done $(date -u +%FT%TZ) fail=$fail"
exit $fail
