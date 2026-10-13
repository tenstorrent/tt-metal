#!/usr/bin/env bash
# Scratch A/B job (TopK stack against #57396's tree merge), not for merge. ttnn.topk multi-core (tree merge) device time
# under the device profiler: A = #57396 merged with main (its TopK LLK and API files swapped in), B = #57396 with the TopK
# stack (the merged #60291 head as built), B2 = an A/A copy of B, C = #57396 with #60287's LLK only (#60288's topk.h not
# applied). Each arm its own fresh JIT cache, files swapped into the checkout and the installed package; A B C B2 B2 C B A
# in two blocks. Guard: the arms' tree-merge and final compute ELF hashes; the job fails if A equals B or B differs from B2.
set +e +o pipefail
cd /work; export TT_METAL_HOME=/work TRACY_NO_WEB_SERVER=1 TREE_NMEAS=${TREE_NMEAS:-8} HWLOCK_HELD=ci
S=/work/ci_ab; OUT=/work/generated/tree; rm -rf $OUT; mkdir -p $OUT
python3 -c "import tracy" 2>/dev/null || export PYTHONPATH=/work:/work/tools
SP=$(cd /tmp && python3 -c "import ttnn, os; print(os.path.dirname(os.path.dirname(os.path.abspath(ttnn.__file__))))")
echo "== $(date -u +%FT%TZ) $(git log -1 --format='%h %s') package root $SP"
OBJ=$(find /work/runtime /opt /usr/local "$SP" -name riscv-tt-elf-objcopy -type f 2>/dev/null | head -1)
suffix() { echo "$1" | awk -F/ '{n=NF; print $(n-3)"/"$(n-2)"/"$(n-1)"/"$n}'; }
while read -r rel; do echo "== $rel -> $(find "$SP" -path "*/$(suffix $rel)" -type f 2>/dev/null | tr '\n' ' ')"; done < $S/files.txt
use() {
  while read -r rel; do
    cp "$S/v/$2/$rel" "/work/$rel"
    while read -r p; do [[ -n $p && $p != "/work/$rel" ]] && cp "$S/v/$2/$rel" "$p"; done < <(find "$SP" -path "*/$(suffix $rel)" -type f 2>/dev/null)
  done < $S/files.txt
  echo "TRE FILES $1 $(while read -r rel; do cat /work/$rel; done < $S/files.txt | md5sum | cut -c1-8)"
  export TT_METAL_CACHE=$OUT/cache_$1; mkdir -p $TT_METAL_CACHE
}
elfsum() { for e in $(find $1 \( -path '*/kernels/topk_local_tree_merge/*' -o -path '*/kernels/topk_final/*' \) -name 'trisc*.elf' ! -name '*.xip.elf' 2>/dev/null | sort); do
  $OBJ -O binary -j .text "$e" /tmp/e.bin 2>/dev/null && md5sum < /tmp/e.bin | cut -c1-12; done | sort | md5sum | cut -c1-12; }
declare -A ELF
r=0; fail=0
for blk in 1 2; do for a in A B C B2 B2 C B A; do
  r=$((r+1)); use $a ${a%2}
  o=$OUT/${a}_r$r; mkdir -p $o; t0=$(date +%s)
  timeout -s INT -k 60 1500 python3 -m tracy -r -p --no-web-server -o $o $S/prof_tree.py > $o.log 2>&1
  rc=$?
  echo "== run $r $a rc=$rc $(( $(date +%s) - t0 )) s: $(grep -c 'CASE .* ok' $o.log) ok, $(grep -c 'CASE .* FAILED' $o.log) failed"
  [[ $rc -ne 0 ]] && { fail=1; grep -a -m5 -E 'FAILED|Error' $o.log | cut -c1-240; }
  python3 $S/sum_tree.py $o $a $r
  if [[ -z ${ELF[$a]:-} ]]; then ELF[$a]=$(elfsum $TT_METAL_CACHE)
    echo "TRE ELF $a ${ELF[$a]} $(find $TT_METAL_CACHE \( -path '*/kernels/topk_local_tree_merge/*' -o -path '*/kernels/topk_final/*' \) -name 'trisc*.elf' ! -name '*.xip.elf' | wc -l) ELFs"; fi
done; done
use B B
echo "TRE GUARD A=${ELF[A]} B=${ELF[B]} B2=${ELF[B2]} C=${ELF[C]}"
[[ ${ELF[A]} == "${ELF[B]}" ]] && { echo "TRE GUARD FAIL: A and B built the same kernels"; fail=1; }
[[ ${ELF[B]} != "${ELF[B2]}" ]] && { echo "TRE GUARD FAIL: the A/A arms built different kernels"; fail=1; }
echo "== done $(date -u +%FT%TZ)"
exit $fail
