#!/usr/bin/env bash
# Scratch bits job (TopK, #60291 merged with #57396), not for merge. The family's dump scripts on the card: every dump
# on B (the merged head as built), the topk_large_indices dumps on A (#57396 merged with main, its device files swapped
# in), each arm its own fresh JIT cache; per case the sha256 of every output (TBITS lines), compared offline with main's
# dumps. Guard: the arms' topk_large_indices compute ELF hashes are printed; the job fails if A equals B.
set +e +o pipefail
cd /work; export TT_METAL_HOME=/work HWLOCK_HELD=ci
S=/work/ci_ab; OUT=/work/generated/bits; rm -rf $OUT; mkdir -p $OUT
SP=$(cd /tmp && python3 -c "import ttnn, os; print(os.path.dirname(os.path.dirname(os.path.abspath(ttnn.__file__))))")
echo "== $(date -u +%FT%TZ) $(git log -1 --format='%h %s') package root $SP"
OBJ=$(find /work/runtime /opt /usr/local "$SP" -name riscv-tt-elf-objcopy -type f 2>/dev/null | head -1)
suffix() { echo "$1" | awk -F/ '{n=NF; print $(n-3)"/"$(n-2)"/"$(n-1)"/"$n}'; }
use() {
  while read -r rel; do
    cp "$S/v/$2/$rel" "/work/$rel"
    while read -r p; do [[ -n $p && $p != "/work/$rel" ]] && cp "$S/v/$2/$rel" "$p"; done < <(find "$SP" -path "*/$(suffix $rel)" -type f 2>/dev/null)
  done < $S/files.txt
  echo "TBITS FILES $1 $(while read -r rel; do cat /work/$rel; done < $S/files.txt | md5sum | cut -c1-8)"
  export TT_METAL_CACHE=$OUT/cache_$1; mkdir -p $TT_METAL_CACHE
}
elfsum() { for e in $(find $1 -path '*/kernels/compute/*' -name 'trisc*.elf' ! -name '*.xip.elf' 2>/dev/null | sort); do
  $OBJ -O binary -j .text "$e" /tmp/e.bin 2>/dev/null && md5sum < /tmp/e.bin | cut -c1-12; done | sort | md5sum | cut -c1-12; }
fail=0
dump() {  # dump <arm> <script>
  local o=$OUT/$1/$2; mkdir -p $o; local t0=$(date +%s)
  timeout -s INT -k 60 1500 python3 $S/dumps/$2.py $o > $o.log 2>&1
  local rc=$?
  echo "== $1 $2 rc=$rc $(( $(date +%s) - t0 )) s: $(grep -c 'CASE .* ok' $o.log) ok, $(grep -c 'CASE .* FAILED' $o.log) failed"
  [[ $rc -ne 0 ]] && { fail=1; tail -5 $o.log | cut -c1-240; }
  grep -a 'CASE .* FAILED' $o.log | head -3 | cut -c1-200
  [[ -f $o/results.pt ]] && python3 $S/bits_hash.py $o $1 $2
}
use B B
for s in ab_dump ab_dump_fp32 ab_dump_extra ab_dump_uniad ab_dump_sort ab_dump_audit ab_dump_nan ab_dump_tli ab_dump_tli2 \
         ab_dump_tli5 ab_dump_samp ab_dump_tli1024; do dump B $s; done
EB=$(elfsum $TT_METAL_CACHE)
use A A
for s in ab_dump_tli ab_dump_tli2 ab_dump_tli5 ab_dump_tli1024; do dump A $s; done
EA=$(elfsum $TT_METAL_CACHE)
use B B
echo "TBITS GUARD A=$EA B=$EB"
[[ $EA == "$EB" ]] && { echo "TBITS GUARD FAIL: A and B built the same compute kernels"; fail=1; }
echo "== done $(date -u +%FT%TZ)"
exit $fail
