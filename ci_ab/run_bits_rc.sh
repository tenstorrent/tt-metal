#!/usr/bin/env bash
# Scratch bits job: every case of ut_cases.py (interleaved, sharded, unpadding, update_cache, fill_cache, paged_fused_update_cache,
# wide-range data) and the perf shapes, arms A B C, output hash per readback; test count per arm printed and checked.
source /work/ci_ab/common.sh
OUT=/work/generated/utbrc; rm -rf $OUT; mkdir -p $OUT; fail=0
T=tests/ttnn/unit_tests/operations/data_movement/test_ut_ab.py
for a in A B C; do
  use bits_$a $a
  BITS_TAG=$a timeout -s INT -k 60 3300 python3 -m pytest -s -q -p no:cacheprovider -o timeout_method=thread -o addopts= -rfE -k "test_cov and rc_" $T > $OUT/bits_$a.log 2>&1
  echo "== bits $a rc=$? $(grep -aE '[0-9]+ (passed|failed)' $OUT/bits_$a.log | tail -1)"
  grep -a -E '^(FAILED|ERROR)' $OUT/bits_$a.log | cut -c1-250 | head -20
  grep -a -o "BITS $a .*" $OUT/bits_$a.log | sed "s/^BITS $a //" | sort > $OUT/bits_$a.txt
  echo "== BITS $a lines $(wc -l < $OUT/bits_$a.txt) PASS $(grep -c ' PASS ' $OUT/bits_$a.txt) FAIL $(grep -c ' FAIL ' $OUT/bits_$a.txt) ERR $(grep -c ' ERR ' $OUT/bits_$a.txt)"
  [[ -s $OUT/bits_$a.txt ]] || { echo "== BITSCMP VOID: $a printed no BITS lines"; fail=1; }
  echo "RAB ELF $a $(elfsum $TT_METAL_CACHE) $(ls -d $TT_METAL_CACHE/*/kernels/*/* 2>/dev/null | wc -l) builds"
done
# Bfp4_b and every block-float exponent: fast untilize against pack untilize of raw tiles, arm C only (A has no face-pass flag, B refuses block-float passes)
use fvp_C C
timeout -s INT -k 60 3000 python3 -m pytest -q -p no:cacheprovider -o timeout_method=thread -o addopts= -rfE tests/ttnn/unit_tests/operations/data_movement/test_untilize_fast_vs_pack.py > $OUT/fvp_C.log 2>&1
echo "== fast_vs_pack C rc=$? $(grep -aE '[0-9]+ (passed|failed)' $OUT/fvp_C.log | tail -1)"
grep -a -E '^(FAILED|ERROR)' $OUT/fvp_C.log | cut -c1-250 | head -20
cmpbits() {
  awk -v A=$1 -v B=$2 'function key(l){sub(/ [^ ]+$/,"",l); return l} function hv(l){sub(/^.* /,"",l); return l}
    NR==FNR{h[key($0)]=hv($0); next}
    {k=key($0); seen[k]=1; if(k in h){c++; if(h[k]==hv($0)) e++; else {d++; print "DIFF " A "_" B " " k}} else ob++}
    END{for(k in h) if(!(k in seen)) oa++; print "== BITSKEY " A " " B " common " c+0 ", equal " e+0 ", differ " d+0 ", " A " only " oa+0 ", " B " only " ob+0}' \
    $OUT/bits_$1.txt $OUT/bits_$2.txt | cut -c1-300
}
cmpbits A B; cmpbits A C; cmpbits B C
use A A
echo "== done $(date -u +%FT%TZ) fail=$fail"
exit $fail
