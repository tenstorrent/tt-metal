#!/usr/bin/env bash
# Scratch bits job (TopK headroom round), not for merge. The family's dump scripts and the gate dump on the card, on A (the
# stack's current tt-llk ckernel_sfpu_topk.h swapped in) and B (the round's head as built), each arm its own fresh JIT
# cache; per case the sha256 of every output (TBITS lines), compared offline. Guard: the arms' compute ELF hashes are
# printed; the job fails if A equals B. Then the TopK, sort, sampling, moe and gate unit tests on B.
set +e +o pipefail
cd /work; export TT_METAL_HOME=/work HWLOCK_HELD=ci
S=/work/ci_ab; OUT=/work/generated/bits; rm -rf $OUT; mkdir -p $OUT
SP=$(cd /tmp && python3 -c "import ttnn, os; print(os.path.dirname(os.path.dirname(os.path.abspath(ttnn.__file__))))")
echo "== $(date -u +%FT%TZ) $(git log -1 --format='%h %s') package root $SP"
OBJ=$(find /work/runtime /opt /usr/local "$SP" -name riscv-tt-elf-objcopy -type f 2>/dev/null | head -1)
suffix() { echo "$1" | awk -F/ '{n=NF; print $(n-3)"/"$(n-2)"/"$(n-1)"/"$n}'; }
use() {
  while read -r rel; do
    cp "$S/variants/$2/$rel" "/work/$rel"
    while read -r p; do [[ -n $p && $p != "/work/$rel" ]] && cp "$S/variants/$2/$rel" "$p"; done < <(find "$SP" -path "*/$(suffix $rel)" -type f 2>/dev/null)
  done < $S/files.txt
  echo "TBITS FILES $1 $(while read -r rel; do cat /work/$rel; done < $S/files.txt | md5sum | cut -c1-8)"
  export TT_METAL_CACHE=$OUT/cache_$1; mkdir -p $TT_METAL_CACHE
}
elfsum() { for e in $(find $1 -path '*/kernels/*' -name 'trisc*.elf' ! -name '*.xip.elf' 2>/dev/null | sort); do
  $OBJ -O binary -j .text "$e" /tmp/e.bin 2>/dev/null && md5sum < /tmp/e.bin | cut -c1-12; done | sort | md5sum | cut -c1-12; }
fail=0
dump() {  # dump <arm> <script> [env]
  local o=$OUT/$1/$2${3:+_row}; mkdir -p $o; local t0=$(date +%s)
  env ${3:-} timeout -s INT -k 60 1500 python3 $S/dumps/$2.py $o > $o.log 2>&1
  local rc=$?
  echo "== $1 $2${3:+ row} rc=$rc $(( $(date +%s) - t0 )) s: $(grep -c 'CASE .* ok' $o.log) ok, $(grep -c 'CASE .* FAILED' $o.log) failed"
  [[ $rc -ne 0 ]] && { fail=1; tail -5 $o.log | cut -c1-240; }
  grep -a 'CASE .* FAILED' $o.log | head -3 | cut -c1-200
  [[ -f $o/results.pt ]] && python3 $S/bits_hash.py $o $1 $2${3:+_row}
}
DUMPS="ab_dump ab_dump_fp32 ab_dump_extra ab_dump_uniad ab_dump_sort ab_dump_audit ab_dump_nan ab_dump_samp ab_dump_gates"
declare -A E
for a in B A; do
  use $a $a
  for s in $DUMPS; do dump $a $s; done
  dump $a ab_dump_gates DUMP_ROW_DISPATCH=1
  E[$a]=$(elfsum $TT_METAL_CACHE)
done
echo "TBITS GUARD A=${E[A]} B=${E[B]}"
[[ ${E[A]} == "${E[B]}" ]] && { echo "TBITS GUARD FAIL: A and B built the same compute kernels"; fail=1; }
use B B
export TT_METAL_CACHE=$OUT/cache_T; mkdir -p $TT_METAL_CACHE
for t in tests/ttnn/unit_tests/operations/reduce/test_topk.py tests/ttnn/nightly/unit_tests/operations/reduction/test_topk_stable_adversarial.py \
         tests/ttnn/unit_tests/operations/data_movement/test_sort.py tests/ttnn/unit_tests/operations/reduce/test_moe.py \
         tests/ttnn/unit_tests/operations/reduce/test_sampling.py tests/ttnn/nightly/unit_tests/operations/reduction/test_deepseek_grouped_gate.py \
         tests/ttnn/nightly/unit_tests/operations/experimental/test_topk_router_gpt.py; do
  t0=$(date +%s)
  timeout -s INT -k 60 2400 python3 -m pytest -q -p no:cacheprovider -o timeout_method=thread $t > $OUT/ut.log 2>&1
  echo "== $(basename $t) on B rc=$? $(( $(date +%s) - t0 )) s: $(grep -aE '[0-9]+ (passed|failed)' $OUT/ut.log | tail -1)"
  grep -aE '^(FAILED|ERROR)' $OUT/ut.log | head -10 | cut -c1-240
done
echo "== done $(date -u +%FT%TZ)"
exit $fail
