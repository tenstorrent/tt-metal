#!/usr/bin/env bash
# Scratch bits job (TopK, #57396 merged with main f0832760bf5 as built, arm P), not for merge. The family's dump scripts
# on the card with a fresh JIT cache; per case the sha256 of every output (TBITS lines), compared offline with main's and
# with the merged head's (test-dispatch 38330227874). Guard: the topk_large_indices compute ELF hash is printed.
set +e +o pipefail
cd /work; export TT_METAL_HOME=/work HWLOCK_HELD=ci
S=/work/ci_ab; OUT=/work/generated/bits; rm -rf $OUT; mkdir -p $OUT
echo "== $(date -u +%FT%TZ) $(git log -1 --format='%h %s')"
OBJ=$(find /work/runtime /opt /usr/local -name riscv-tt-elf-objcopy -type f 2>/dev/null | head -1)
elfsum() { for e in $(find $1 -path '*/kernels/compute/*' -name 'trisc*.elf' ! -name '*.xip.elf' 2>/dev/null | sort); do
  $OBJ -O binary -j .text "$e" /tmp/e.bin 2>/dev/null && md5sum < /tmp/e.bin | cut -c1-12; done | sort | md5sum | cut -c1-12; }
export TT_METAL_CACHE=$OUT/cache_P; mkdir -p $TT_METAL_CACHE
fail=0
for s in ab_dump ab_dump_fp32 ab_dump_extra ab_dump_uniad ab_dump_sort ab_dump_audit ab_dump_nan ab_dump_tli ab_dump_tli2 \
         ab_dump_tli5 ab_dump_samp ab_dump_tli1024; do
  o=$OUT/P/$s; mkdir -p $o; t0=$(date +%s)
  timeout -s INT -k 60 1500 python3 $S/dumps/$s.py $o > $o.log 2>&1
  rc=$?
  echo "== P $s rc=$rc $(( $(date +%s) - t0 )) s: $(grep -c 'CASE .* ok' $o.log) ok, $(grep -c 'CASE .* FAILED' $o.log) failed"
  [[ $rc -ne 0 ]] && { fail=1; tail -5 $o.log | cut -c1-240; }
  [[ -f $o/results.pt ]] && python3 $S/bits_hash.py $o P $s
done
echo "TBITS GUARD P=$(elfsum $TT_METAL_CACHE)"
echo "== done $(date -u +%FT%TZ)"
exit $fail
