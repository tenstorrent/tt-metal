#!/usr/bin/env bash
# Scratch A/B job (TopK merge fix F4 against the stack head e7081bed1a1), not for merge. A = head, B = head + fix, B2 = A/A of B.
# the bitonic TopK network (prof_ops.py, 55 shapes) under the device profiler: A = main's two TopK LLK headers, B = #60287's
# (host code and ttnn kernels are main's in both arms), B2 = an A/A copy of B. Each arm its own fresh JIT cache, its files
# copied into the checkout and the installed package; A B B2 B2 B A in two blocks. Guard: the arms' compute ELF .text hashes
# are printed; the job fails if A equals B or B differs from B2.
set +e +o pipefail
cd /work; export TT_METAL_HOME=/work TRACY_NO_WEB_SERVER=1 OPS_NMEAS=${OPS_NMEAS:-8} HWLOCK_HELD=ci
S=/work/ci_ab; OUT=/work/generated/ops; rm -rf $OUT; mkdir -p $OUT
python3 -c "import tracy" 2>/dev/null || export PYTHONPATH=/work:/work/tools
SP=$(cd /tmp && python3 -c "import ttnn, os; print(os.path.dirname(os.path.dirname(os.path.abspath(ttnn.__file__))))")
echo "== $(date -u +%FT%TZ) $(git log -1 --format='%h %s') package root $SP"
OBJ=$(find /work/runtime /opt /usr/local "$SP" -name riscv-tt-elf-objcopy -type f 2>/dev/null | head -1)
suffix() { echo "$1" | awk -F/ '{n=NF; print $(n-3)"/"$(n-2)"/"$(n-1)"/"$n}'; }
RT=$(cd /tmp && env -u PYTHONPATH python3 -c "import ttnn, os; print(os.path.dirname(os.path.abspath(ttnn.__file__)))")
nopkg=0; while read -r rel; do [[ -f $RT/$rel ]] || { nopkg=$((nopkg+1)); echo "OPS NOPKG $rel"; }; cmp -s /work/$rel $S/variants/A/$rel || echo "OPS TREE_NOT_MAIN $rel"; done < $S/files_ops.txt
echo "OPS FILES $(wc -l < $S/files_ops.txt) files, $nopkg without a package copy, runtime root $RT"
use() {
  while read -r rel; do
    cp "$S/variants/${2%2}/$rel" "/work/$rel"; [[ -f $RT/$rel ]] && cp "$S/variants/${2%2}/$rel" "$RT/$rel"
  done < $S/files_ops.txt
  echo "OPS USE $1 $(while read -r rel; do cat /work/$rel; done < $S/files_ops.txt | md5sum | cut -c1-8) $(while read -r rel; do [[ -f $RT/$rel ]] && cat $RT/$rel; done < $S/files_ops.txt | md5sum | cut -c1-8)"
  export TT_METAL_CACHE=$OUT/cache_$1; mkdir -p $TT_METAL_CACHE
}
elfsum() { for e in $(find $1 -path '*/kernels/*' -name 'trisc*.elf' ! -name '*.xip.elf' 2>/dev/null | sort); do
  $OBJ -O binary -j .text "$e" /tmp/e.bin 2>/dev/null && md5sum < /tmp/e.bin | cut -c1-12; done | sort | md5sum | cut -c1-12; }
declare -A ELF
r=0; fail=0
for blk in 1 2; do for a in ${ARMS:-A B B2 B2 B A}; do
  r=$((r+1)); use $a $a
  o=$OUT/${a}_r$r; mkdir -p $o; t0=$(date +%s)
  timeout -s INT -k 60 1800 python3 -m tracy -r -p --no-web-server -o $o $S/prof_ops.py > $o.log 2>&1
  rc=$?
  echo "== run $r $a rc=$rc $(( $(date +%s) - t0 )) s: $(grep -c 'CASE .* ok' $o.log) ok, $(grep -c 'CASE .* FAILED' $o.log) failed"
  [[ $rc -ne 0 ]] && { fail=1; grep -a -m5 -E 'FAILED|Error' $o.log | cut -c1-240; }
  grep -a 'CASE .* FAILED' $o.log | head -5 | cut -c1-240
  python3 $S/sum_ops.py $o $a $r
  if [[ -z ${ELF[$a]:-} ]]; then ELF[$a]=$(elfsum $TT_METAL_CACHE)
    echo "OPS ELF $a ${ELF[$a]} $(find $TT_METAL_CACHE -path '*/kernels/*' -name 'trisc*.elf' ! -name '*.xip.elf' | wc -l) compute ELFs"; fi
done; done
# The noisiest row again: OPS_FOCUS cases, OPS_NMEAS 32, A B B A four times.
if [[ -n ${OPS_FOCUS:-} ]]; then for blk in 1 2 3 4; do for a in A B B A; do
  r=$((r+1)); use $a $a; o=$OUT/${a}_f$r; mkdir -p $o
  OPS_CASES="$OPS_FOCUS" OPS_NMEAS=32 timeout -s INT -k 60 900 python3 -m tracy -r -p --no-web-server -o $o $S/prof_ops.py > $o.log 2>&1
  echo "== focus $r $a rc=$? $(grep -c 'CASE .* ok' $o.log) ok"
  GATE_NMEAS=32 OPS_NMEAS=32 python3 $S/sum_ops.py $o ${a}f $r
done; done; fi
use A A
echo "OPS GUARD A=${ELF[A]} B=${ELF[B]} B2=${ELF[B2]} A_vs_B=$([[ ${ELF[A]} != "${ELF[B]}" && -n ${ELF[A]} ]] && echo ok || echo VOID) B_vs_B2=$([[ ${ELF[B]} == "${ELF[B2]}" ]] && echo ok || echo DIFF)"
[[ ${ELF[A]} == "${ELF[B]}" ]] && { echo "OPS GUARD FAIL: A and B built the same kernels"; fail=1; }
[[ ${ELF[B]} != "${ELF[B2]}" ]] && { echo "OPS GUARD FAIL: the A/A arms built different kernels"; fail=1; }
use B B; cd /work; timeout -s INT -k 60 2700 python3 -m pytest -p no:cacheprovider -q tests/ttnn/unit_tests/operations/reduce/test_topk.py tests/ttnn/unit_tests/operations/reduce/test_moe.py > $OUT/ut_B.log 2>&1; echo "OPS UT B rc=$? $(grep -E "passed|failed|error" $OUT/ut_B.log | tail -1 | cut -c1-200)"; grep -a -E "^(FAILED|ERROR) " $OUT/ut_B.log | head -5 | cut -c1-250
echo "== done $(date -u +%FT%TZ)"
exit $fail
