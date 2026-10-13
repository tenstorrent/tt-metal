#!/usr/bin/env bash
# Scratch A/B job (TopK headroom round), not for merge. Device time of the production ops that run the bitonic TopK network
# (prof_ops.py) under the device profiler: A = the stack's current tt-llk ckernel_sfpu_topk.h, M = with #58254 merged,
# B = the round's head (as built), B2 = an A/A copy of B. Each arm its own fresh JIT cache, its file swapped into the
# checkout and the installed package; A B M B2 B2 M B A in two blocks. Guard: the arms' compute ELF .text hashes are
# printed; the job fails if A equals B or B differs from B2.
set +e +o pipefail
cd /work; export TT_METAL_HOME=/work TRACY_NO_WEB_SERVER=1 OPS_NMEAS=${OPS_NMEAS:-8} HWLOCK_HELD=ci
S=/work/ci_ab; OUT=/work/generated/ops; rm -rf $OUT; mkdir -p $OUT
python3 -c "import tracy" 2>/dev/null || export PYTHONPATH=/work:/work/tools
SP=$(cd /tmp && python3 -c "import ttnn, os; print(os.path.dirname(os.path.dirname(os.path.abspath(ttnn.__file__))))")
echo "== $(date -u +%FT%TZ) $(git log -1 --format='%h %s') package root $SP"
OBJ=$(find /work/runtime /opt /usr/local "$SP" -name riscv-tt-elf-objcopy -type f 2>/dev/null | head -1)
suffix() { echo "$1" | awk -F/ '{n=NF; print $(n-3)"/"$(n-2)"/"$(n-1)"/"$n}'; }
while read -r rel; do echo "== $rel -> $(find "$SP" -path "*/$(suffix $rel)" -type f 2>/dev/null | tr '\n' ' ')"; done < $S/files.txt
use() {
  while read -r rel; do
    cp "$S/variants/$2/$rel" "/work/$rel"
    while read -r p; do [[ -n $p && $p != "/work/$rel" ]] && cp "$S/variants/$2/$rel" "$p"; done < <(find "$SP" -path "*/$(suffix $rel)" -type f 2>/dev/null)
  done < $S/files.txt
  echo "OPS FILES $1 $(while read -r rel; do cat /work/$rel; done < $S/files.txt | md5sum | cut -c1-8)"
  export TT_METAL_CACHE=$OUT/cache_$1; mkdir -p $TT_METAL_CACHE
}
elfsum() { for e in $(find $1 -path '*/kernels/*' -name 'trisc*.elf' ! -name '*.xip.elf' 2>/dev/null | sort); do
  $OBJ -O binary -j .text "$e" /tmp/e.bin 2>/dev/null && md5sum < /tmp/e.bin | cut -c1-12; done | sort | md5sum | cut -c1-12; }
declare -A ELF
r=0; fail=0
for blk in 1 2; do for a in A B M B2 B2 M B A; do
  r=$((r+1)); use $a $a
  o=$OUT/${a}_r$r; mkdir -p $o; t0=$(date +%s)
  timeout -s INT -k 60 1800 python3 -m tracy -r -p --no-web-server -o $o $S/prof_ops.py > $o.log 2>&1
  rc=$?
  echo "== run $r $a rc=$rc $(( $(date +%s) - t0 )) s: $(grep -c 'CASE .* ok' $o.log) ok, $(grep -c 'CASE .* FAILED' $o.log) failed"
  [[ $rc -ne 0 ]] && { fail=1; grep -a -m5 -E 'FAILED|Error' $o.log | cut -c1-240; }
  grep -a 'CASE .* FAILED' $o.log | head -5 | cut -c1-240
  python3 $S/sum_ops.py $o $a $r
  o2=$OUT/${a}_r${r}_row; mkdir -p $o2
  OPS_ROW_DISPATCH=1 timeout -s INT -k 60 600 python3 -m tracy -r -p --no-web-server -o $o2 $S/prof_ops.py > $o2.log 2>&1
  echo "== run $r $a row rc=$? $(grep -c 'CASE .* ok' $o2.log) ok, $(grep -c 'CASE .* FAILED' $o2.log) failed"
  grep -a 'CASE .* FAILED' $o2.log | head -2 | cut -c1-240
  python3 $S/sum_ops.py $o2 $a $r
  if [[ -z ${ELF[$a]:-} ]]; then ELF[$a]=$(elfsum $TT_METAL_CACHE)
    echo "OPS ELF $a ${ELF[$a]} $(find $TT_METAL_CACHE -path '*/kernels/*' -name 'trisc*.elf' ! -name '*.xip.elf' | wc -l) compute ELFs"; fi
done; done
use B B
echo "OPS GUARD A=${ELF[A]} M=${ELF[M]} B=${ELF[B]} B2=${ELF[B2]}"
[[ ${ELF[A]} == "${ELF[B]}" ]] && { echo "OPS GUARD FAIL: A and B built the same kernels"; fail=1; }
[[ ${ELF[B]} != "${ELF[B2]}" ]] && { echo "OPS GUARD FAIL: the A/A arms built different kernels"; fail=1; }
echo "== done $(date -u +%FT%TZ)"
exit $fail
