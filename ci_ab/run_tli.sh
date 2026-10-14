#!/usr/bin/env bash
# Scratch job (#60291 merge round M4), not for merge. topk_large_indices device time under the device profiler:
# A = the head (#57396 38edb652319 merged, DEST placement, K thresholds, no split without a whole row) with the split
# compiled out, A2 its copy; G = the head, G2 its copy; own fresh JIT cache per arm; A G G2 A2 A2 G2 G A twice. Then
# the indices of every case A against G, and test_topk_large_indices.py on G.
set +e +o pipefail
cd /work; export TT_METAL_HOME=/work TRACY_NO_WEB_SERVER=1 OPS_NMEAS=${OPS_NMEAS:-8} HWLOCK_HELD=ci
S=/work/ci_ab; OUT=/work/generated/tli; rm -rf $OUT; mkdir -p $OUT
python3 -c "import tracy" 2>/dev/null || export PYTHONPATH=/work:/work/tools
SP=$(cd /tmp && python3 -c "import ttnn, os; print(os.path.dirname(os.path.dirname(os.path.abspath(ttnn.__file__))))")
echo "== $(date -u +%FT%TZ) $(git log -1 --format='%h %s') package root $SP"
OBJ=$(find /work/runtime /opt /usr/local "$SP" -name riscv-tt-elf-objcopy -type f 2>/dev/null | head -1)
suffix() { echo "$1" | awk -F/ '{n=NF; print $(n-3)"/"$(n-2)"/"$(n-1)"/"$n}'; }
while read -r rel; do echo "== $rel -> $(find "$SP" -path "*/$(suffix $rel)" -type f 2>/dev/null | tr '\n' ' ')"; done < $S/files.txt
use() {
  while read -r rel; do
    cp "$S/${VARIANTS:-variants_m4}/$2/$rel" "/work/$rel"
    while read -r p; do [[ -n $p && $p != "/work/$rel" ]] && cp "$S/${VARIANTS:-variants_m4}/$2/$rel" "$p"; done < <(find "$SP" -path "*/$(suffix $rel)" -type f 2>/dev/null)
  done < $S/files.txt
  echo "OPS FILES $1 $(while read -r rel; do cat /work/$rel; done < $S/files.txt | md5sum | cut -c1-8)"
  export TT_METAL_CACHE=$OUT/cache_$1; mkdir -p $TT_METAL_CACHE
}
elfsum() { for e in $(find $1 -path '*/kernels/*' -name 'trisc*.elf' ! -name '*.xip.elf' 2>/dev/null | sort); do
  $OBJ -O binary -j .text "$e" /tmp/e.bin 2>/dev/null && md5sum < /tmp/e.bin | cut -c1-12; done | sort | md5sum | cut -c1-12; }
declare -A ELF
r=0; fail=0
for blk in 1 2; do for a in ${ARMS:-A G G2 A2 A2 G2 G A}; do
  r=$((r+1)); use $a $a
  o=$OUT/${a}_r$r; mkdir -p $o; t0=$(date +%s)
  timeout -s INT -k 60 1800 python3 -m tracy -r -p --no-web-server -o $o $S/prof_tli.py > $o.log 2>&1
  rc=$?
  echo "== run $r $a rc=$rc $(( $(date +%s) - t0 )) s: $(grep -c 'CASE .* ok' $o.log) ok, $(grep -c 'CASE .* FAILED' $o.log) failed"
  [[ $rc -ne 0 ]] && { fail=1; grep -a -m5 -E 'FAILED|Error' $o.log | cut -c1-240; }
  grep -a 'CASE .* FAILED' $o.log | head -5 | cut -c1-240
  python3 $S/sum_ops.py $o $a $r
  if [[ -z ${ELF[$a]:-} ]]; then ELF[$a]=$(elfsum $TT_METAL_CACHE)
    echo "OPS ELF $a ${ELF[$a]} $(find $TT_METAL_CACHE -path '*/kernels/*' -name 'trisc*.elf' ! -name '*.xip.elf' | wc -l) compute ELFs"; fi
done; done
for a in A G; do
  use $a $a; export TT_METAL_CACHE=$OUT/cache_${a}_dump; mkdir -p $TT_METAL_CACHE
  TLI_DUMP=$OUT/dump_$a.pt timeout -s INT -k 60 1800 python3 $S/prof_tli.py > $OUT/dump_$a.log 2>&1
  echo "== dump $a rc=$? $(grep -a 'DUMP DONE' $OUT/dump_$a.log) $(grep -ac 'CASE .* FAILED' $OUT/dump_$a.log) failed"
done
python3 $S/tli_cmp.py $OUT/dump_A.pt $OUT/dump_G.pt
for a in G; do
  use $a $a; export TT_METAL_CACHE=$OUT/cache_T$a; mkdir -p $TT_METAL_CACHE
  t0=$(date +%s); timeout -s INT -k 60 3000 python3 -m pytest -q -p no:cacheprovider -o timeout_method=thread tests/ttnn/nightly/unit_tests/operations/experimental/test_topk_large_indices.py -k "not production_perf" > $OUT/ut_$a.log 2>&1
  echo "== test_topk_large_indices.py on $a rc=$? $(( $(date +%s) - t0 )) s: $(grep -aE '[0-9]+ (passed|failed)' $OUT/ut_$a.log | tail -1)"
  grep -aE '^(FAILED|ERROR)' $OUT/ut_$a.log | head -10 | cut -c1-240
done
use G G
echo "OPS GUARD A=${ELF[A]} A2=${ELF[A2]} G=${ELF[G]} G2=${ELF[G2]}"
[[ ${ELF[A]} == "${ELF[G]}" ]] && { echo "OPS GUARD FAIL: A and G built the same kernels"; fail=1; }
[[ ${ELF[A]} != "${ELF[A2]}" ]] && { echo "OPS GUARD FAIL: the A/A arms built different kernels"; fail=1; }
[[ ${ELF[G]} != "${ELF[G2]}" ]] && { echo "OPS GUARD FAIL: the A/A arms built different kernels"; fail=1; }
echo "== done $(date -u +%FT%TZ)"
exit $fail
