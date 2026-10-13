#!/usr/bin/env bash
# Scratch A/B job (TopK, #60291 merged with #57396), not for merge. topk_large_indices device time under the device
# profiler: A = #57396 merged with main (its device files swapped in), B = the merged head (as built), B2 = an A/A copy of
# B. Each arm its own fresh JIT cache, files swapped into the checkout and the installed package; A B B2 B2 B A in two
# blocks. Guard: the arms' topk_large_indices compute ELF hashes are printed; the job fails if A equals B or B differs
# from B2.
set +e +o pipefail
cd /work; export TT_METAL_HOME=/work TRACY_NO_WEB_SERVER=1 TLI_NMEAS=${TLI_NMEAS:-8} HWLOCK_HELD=ci
S=/work/ci_ab; OUT=/work/generated/ab; rm -rf $OUT; mkdir -p $OUT
python3 -c "import tracy" 2>/dev/null || export PYTHONPATH=/work:/work/tools
SP=$(cd /tmp && python3 -c "import ttnn, os; print(os.path.dirname(os.path.dirname(os.path.abspath(ttnn.__file__))))")
echo "== $(date -u +%FT%TZ) $(git log -1 --format='%h %s') package root $SP"
OBJ=$(find /work/runtime /opt /usr/local "$SP" -name riscv-tt-elf-objcopy -type f 2>/dev/null | head -1)
echo "== objcopy $OBJ"
suffix() { echo "$1" | awk -F/ '{n=NF; print $(n-3)"/"$(n-2)"/"$(n-1)"/"$n}'; }
while read -r rel; do echo "== $rel -> $(find "$SP" -path "*/$(suffix $rel)" -type f 2>/dev/null | tr '\n' ' ')"; done < $S/files.txt
use() {  # use <arm> <files>: the files into the checkout and every copy under the package root, the arm's own JIT cache
  while read -r rel; do
    cp "$S/v/$2/$rel" "/work/$rel"
    while read -r p; do [[ -n $p && $p != "/work/$rel" ]] && cp "$S/v/$2/$rel" "$p"; done < <(find "$SP" -path "*/$(suffix $rel)" -type f 2>/dev/null)
  done < $S/files.txt
  echo "TAB FILES $1 $(while read -r rel; do cat /work/$rel; done < $S/files.txt | md5sum | cut -c1-8)"
  export TT_METAL_CACHE=$OUT/cache_$1; mkdir -p $TT_METAL_CACHE
}
elfsum() { for e in $(find $1 -path '*/kernels/compute/*' -name 'trisc*.elf' ! -name '*.xip.elf' 2>/dev/null | sort); do
  $OBJ -O binary -j .text "$e" /tmp/e.bin 2>/dev/null && md5sum < /tmp/e.bin | cut -c1-12; done | sort | md5sum | cut -c1-12; }
declare -A ELF
r=0; fail=0
for blk in 1 2; do for a in A B B2 B2 B A; do
  r=$((r+1)); use $a ${a%2}
  o=$OUT/${a}_r$r; mkdir -p $o; t0=$(date +%s)
  timeout -s INT -k 60 1500 python3 -m tracy -r -p --no-web-server -o $o $S/prof_tli.py > $o.log 2>&1
  rc=$?
  echo "== run $r $a rc=$rc $(( $(date +%s) - t0 )) s: $(grep -c 'CASE .* ok' $o.log) ok, $(grep -c 'CASE .* FAILED' $o.log) failed"
  [[ $rc -ne 0 ]] && { fail=1; grep -a -m5 -E 'FAILED|Error' $o.log | cut -c1-240; }
  python3 $S/sum_tli.py $o $a $r
  if [[ -z ${ELF[$a]:-} ]]; then ELF[$a]=$(elfsum $TT_METAL_CACHE)
    echo "TAB ELF $a ${ELF[$a]} $(find $TT_METAL_CACHE -path '*/kernels/compute/*' -name 'trisc*.elf' ! -name '*.xip.elf' | wc -l) compute ELFs"; fi
done; done
use B B
echo "TAB GUARD A=${ELF[A]} B=${ELF[B]} B2=${ELF[B2]}"
[[ ${ELF[A]} == "${ELF[B]}" ]] && { echo "TAB GUARD FAIL: A and B built the same compute kernels"; fail=1; }
[[ ${ELF[B]} != "${ELF[B2]}" ]] && { echo "TAB GUARD FAIL: the A/A arms built different compute kernels"; fail=1; }
echo "== done $(date -u +%FT%TZ)"
exit $fail
