#!/usr/bin/env bash
# Scratch, not for merge. Pack stack 4e60eb8f2e4 on main: the device files of arm S (ci_ab/v/S) against arm M (main, ci_ab/v/M) copied into the checkout and the installed package before each run, fresh JIT cache per arm; M2 is an A/A copy of M; order M S M2 then reversed. Guard: M stable, M == M2, S != M.
set +e +o pipefail
cd /work; export TT_METAL_HOME=/work TRACY_NO_WEB_SERVER=1 MM_N=${MM_N:-12}
S=/work/ci_ab; OUT=/work/generated/mm_pk; rm -rf $OUT; mkdir -p $OUT
python3 -c "import tracy" 2>/dev/null || export PYTHONPATH=/work:/work/tools
SP=$(cd /tmp && env -u PYTHONPATH python3 -c "import ttnn, os; print(os.path.dirname(os.path.dirname(os.path.abspath(ttnn.__file__))))")
echo "== $(date -u +%FT%TZ) $(git log -1 --format='%h %s' | cut -c1-60) package root $SP"
OBJ=$(find /work/runtime /opt -name riscv-tt-elf-objcopy -type f 2>/dev/null | head -1)
suffix() { echo "$1" | awk -F/ '{n=NF; print $(n-3)"/"$(n-2)"/"$(n-1)"/"$n}'; }
declare -A PK
while read -r rel; do PK[$rel]=$(find "$SP" -path "*/$(suffix $rel)" -type f 2>/dev/null); echo "== file $rel package copies: $(echo ${PK[$rel]} | wc -w)"; done < $S/files.txt
# a file only the stack adds goes next to a sibling's package copy
while read -r rel; do
  [[ -z ${PK[$rel]} ]] && { sib=$(grep -m1 "^$(dirname $rel)/" $S/files.txt | grep -v "^$rel\$"); for p in ${PK[$sib]}; do PK[$rel]+=" $(dirname $p)/$(basename $rel)"; done; echo "== new file $rel -> ${PK[$rel]}"; }
done < $S/files.txt
use() {
  while read -r rel; do [[ -e "$S/v/$2/$rel" ]] || continue; cp "$S/v/$2/$rel" "/work/$rel"; for p in ${PK[$rel]}; do cp "$S/v/$2/$rel" "$p"; done; done < $S/files.txt
  echo "== files $1 $(while read -r rel; do cat /work/$rel; done < $S/files.txt | md5sum | cut -c1-8)"
  export TT_METAL_CACHE=$OUT/cache_$1; mkdir -p $TT_METAL_CACHE
}
elfsum() {
  for e in $(ls $1/*/kernels/bmm_large_block_zm_fused_bias_activation_metal2/*/trisc*/*.elf 2>/dev/null | grep -v xip); do
    $OBJ -O binary -j .text "$e" /tmp/e.bin 2>/dev/null && md5sum < /tmp/e.bin | cut -c1-12
  done | sort | md5sum | cut -c1-12
}
declare -A ES; r=0; fail=0
for a in M S M2 M2 S M; do
  r=$((r+1)); src=${a%2}; use $a $src
  o=$OUT/${a}_r$r; mkdir -p $o; t0=$(date +%s)
  timeout -s INT -k 60 1800 python3 -m tracy -r -p --no-web-server -o $o $S/mm_pk.py > $o.log 2>&1; rc=$?
  echo "== run $r $a rc=$rc $(( $(date +%s) - t0 )) s: $(grep -c 'case done' $o.log) done $(grep -c 'case FAIL' $o.log) fail"
  grep -a 'case done\|case FAIL' $o.log | sed "s/^/== out $a r$r /"
  [[ $rc -ne 0 ]] && { fail=1; grep -a -m5 -E 'Error|error:' $o.log | cut -c1-200; }
  python3 $S/sum_mm.py $o $a $r
  e=$(elfsum $TT_METAL_CACHE); ES[$a]+=" $e"; echo "== elf $a r$r $e"
done
uniq1() { echo $1 | tr ' ' '\n' | grep . | sort -u | wc -l; }
one() { echo $1 | tr ' ' '\n' | grep . | head -1; }
gs=ok; for a in M S M2; do [[ $(uniq1 "${ES[$a]}") == 1 ]] || gs=UNSTABLE; done
gx=$([[ $(one "${ES[M]}") == $(one "${ES[M2]}") ]] && echo ok || echo DIFF)
gd=ok; [[ $(one "${ES[M]}") != $(one "${ES[S]}") && -n $OBJ ]] || gd=VOID
echo "== GUARD stable=$gs M_vs_M2=$gx S_vs_M=$gd"
use M M
echo "== DONE $(date -u +%FT%TZ) fail=$fail"
