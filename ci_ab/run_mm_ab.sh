#!/usr/bin/env bash
# Scratch, not for merge. #58817 bmm regression attribution: the 14 device files of each arm (ci_ab/v/<arm>) copied into
# the checkout and the installed package before each run, fresh JIT cache per arm; arms M (main 7208b885b33), H (head
# 4f1e6db5716), H2 (A/A copy of H), B1 (7751048e2ca, first #58817 commit on main), Ht (H, ttsync unpack off), Hr (H, row MOP off), Htm (Ht and Hr, and the bmm pack-format argument Invalid);
# order M H F H2 then reversed; F = H plus the HiFi2 one-tile main-form key (branch mvlahovic/mm-hifi2-1x1). Guard: H == H2, H != M, H != F.
set +e +o pipefail
cd /work; export TT_METAL_HOME=/work TRACY_NO_WEB_SERVER=1 MM_N=${MM_N:-12}
S=/work/ci_ab; OUT=/work/generated/mm_ab; rm -rf $OUT; mkdir -p $OUT
python3 -c "import tracy" 2>/dev/null || export PYTHONPATH=/work:/work/tools
SP=$(cd /tmp && env -u PYTHONPATH python3 -c "import ttnn, os; print(os.path.dirname(os.path.dirname(os.path.abspath(ttnn.__file__))))")
echo "== $(date -u +%FT%TZ) $(git log -1 --format='%h %s' | cut -c1-60) package root $SP"
OBJ=$(find /work/runtime /opt -name riscv-tt-elf-objcopy -type f 2>/dev/null | head -1)
suffix() { echo "$1" | awk -F/ '{n=NF; print $(n-3)"/"$(n-2)"/"$(n-1)"/"$n}'; }
declare -A PK
while read -r rel; do PK[$rel]=$(find "$SP" -path "*/$(suffix $rel)" -type f 2>/dev/null); echo "== file $rel package copies: $(echo ${PK[$rel]} | wc -w)"; done < $S/files.txt
use() {
  while read -r rel; do cp "$S/v/$2/$rel" "/work/$rel"; for p in ${PK[$rel]}; do cp "$S/v/$2/$rel" "$p"; done; done < $S/files.txt
  echo "== files $1 $(while read -r rel; do cat /work/$rel; done < $S/files.txt | md5sum | cut -c1-8)"
  export TT_METAL_CACHE=$OUT/cache_$1; mkdir -p $TT_METAL_CACHE
}
elfsum() {
  for e in $(ls $1/*/kernels/bmm_large_block_zm_fused_bias_activation_metal2/*/trisc*/*.elf 2>/dev/null | grep -v xip); do
    $OBJ -O binary -j .text "$e" /tmp/e.bin 2>/dev/null && md5sum < /tmp/e.bin | cut -c1-12
  done | sort | md5sum | cut -c1-12
}
declare -A ES; r=0; fail=0
for a in M H F H2 H2 F H M; do
  r=$((r+1)); src=${a%2}; use $a $src
  o=$OUT/${a}_r$r; mkdir -p $o; t0=$(date +%s)
  timeout -s INT -k 60 1800 python3 -m tracy -r -p --no-web-server -o $o $S/mm_ci.py > $o.log 2>&1; rc=$?
  echo "== run $r $a rc=$rc $(( $(date +%s) - t0 )) s: $(grep -c 'case done' $o.log) done $(grep -c 'case FAIL' $o.log) fail"
  grep -a 'case done\|case FAIL' $o.log | sed "s/^/== out $a r$r /"
  [[ $rc -ne 0 ]] && { fail=1; grep -a -m5 -E 'Error|error:' $o.log | cut -c1-200; }
  python3 $S/sum_mm.py $o $a $r
  e=$(elfsum $TT_METAL_CACHE); ES[$a]+=" $e"; echo "== elf $a r$r $e"
done
uniq1() { echo $1 | tr ' ' '\n' | grep . | sort -u | wc -l; }
one() { echo $1 | tr ' ' '\n' | grep . | head -1; }
gs=ok; for a in M H H2 F; do [[ $(uniq1 "${ES[$a]}") == 1 ]] || gs=UNSTABLE; done
gx=$([[ $(one "${ES[H]}") == $(one "${ES[H2]}") ]] && echo ok || echo DIFF)
gd=ok; for a in M F; do [[ $(one "${ES[H]}") != $(one "${ES[$a]}") && -n $OBJ ]] || gd=VOID; done
echo "== GUARD stable=$gs H_vs_H2=$gx H_vs_M_Ht_Hr=$gd"
use H H
echo "== DONE $(date -u +%FT%TZ) fail=$fail"
