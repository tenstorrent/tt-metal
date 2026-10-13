#!/usr/bin/env bash
# Scratch A/B job (MoE gate, the SFPU-only route for one ungrouped block), not for merge. generalized_moe_gate device time
# under the device profiler: A = the route off (the unified kernel with sfpu_sort false), B = the head (route on), B2 = an
# A/A copy of B; own fresh JIT cache per arm; A B B2 B2 B A twice. Then the outputs of 9216 tokens per arm (3 stimulus kinds,
# 12 configurations) for the offline compare, and the gate tests on B. Guard: compute ELF hashes; fails if A equals B.
set +e +o pipefail
cd /work; export TT_METAL_HOME=/work TRACY_NO_WEB_SERVER=1 GATE_NMEAS=${GATE_NMEAS:-8} HWLOCK_HELD=ci
S=/work/ci_ab; OUT=/work/generated/gate; rm -rf $OUT; mkdir -p $OUT
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
for blk in 1 2; do for a in A B B2 B2 B A; do
  r=$((r+1)); use $a $a
  o=$OUT/${a}_r$r; mkdir -p $o; t0=$(date +%s)
  GATE_MODE=prof timeout -s INT -k 60 1200 python3 -m tracy -r -p --no-web-server -o $o $S/gate_ops.py > $o.log 2>&1
  rc=$?
  echo "== run $r $a rc=$rc $(( $(date +%s) - t0 )) s: $(grep -c 'CASE .* ok' $o.log) ok"
  [[ $rc -ne 0 ]] && { fail=1; grep -a -m5 -E 'FAILED|Error' $o.log | cut -c1-240; }
  python3 $S/sum_ops.py $o $a $r
  if [[ -z ${ELF[$a]:-} ]]; then ELF[$a]=$(elfsum $TT_METAL_CACHE); echo "OPS ELF $a ${ELF[$a]}"; fi
done; done
for a in A B; do
  use $a $a; export TT_METAL_CACHE=$OUT/cache_${a}_dump; mkdir -p $TT_METAL_CACHE
  GATE_MODE=dump timeout -s INT -k 60 1800 python3 $S/gate_ops.py $OUT/dump_$a > $OUT/dump_$a.log 2>&1
  echo "== dump $a rc=$? $(grep -a 'DUMP DONE' $OUT/dump_$a.log)"
  python3 $S/gate_cmp.py $OUT/dump_$a/results.pt $a
done
python3 $S/gate_cmp.py $OUT/dump_A/results.pt A $OUT/dump_B/results.pt
use B B; export TT_METAL_CACHE=$OUT/cache_T; mkdir -p $TT_METAL_CACHE
for t in models/common/tests/modules/moe/test_generalized_moe_gate.py models/common/tests/modules/moe/test_tt_moe_gate.py; do
  t0=$(date +%s); timeout -s INT -k 60 2400 python3 -m pytest -q -p no:cacheprovider -o timeout_method=thread $t > $OUT/ut.log 2>&1
  echo "== $(basename $t) on B rc=$? $(( $(date +%s) - t0 )) s: $(grep -aE '[0-9]+ (passed|failed)' $OUT/ut.log | tail -1)"
  grep -aE '^(FAILED|ERROR)' $OUT/ut.log | head -10 | cut -c1-240
done
echo "OPS GUARD A=${ELF[A]} B=${ELF[B]} B2=${ELF[B2]}"
[[ ${ELF[A]} == "${ELF[B]}" ]] && { echo "OPS GUARD FAIL: A and B built the same kernels"; fail=1; }
[[ ${ELF[B]} != "${ELF[B2]}" ]] && { echo "OPS GUARD FAIL: the A/A arms differ"; fail=1; }
echo "== done $(date -u +%FT%TZ)"
exit $fail
