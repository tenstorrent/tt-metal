#!/usr/bin/env bash
# Output bits of ttnn.gelu (accurate; bf16 patterns, fp32 widened and seeded fp32) and ttnn.linear with a fused gelu
# (16- and 32-bit DEST) for three ckernel_sfpu_gelu.h texts on one build and machine: prev (#58821 merged with main),
# fix (the 32-bit gelu's Prgm1/Prgm2 set in the call), ctl (fix with a wrong Prgm2, the positive control). Each variant
# is copied over every copy of the header the JIT can read (/work and the installed runtime) and gets a fresh cache.
set +e +o pipefail
cd /work
H=tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_gelu.h
mapfile -t COPIES < <( { echo /work/$H; python3 -c "import ttnn, os; print(os.path.dirname(os.path.dirname(ttnn.__file__)))" 2>/dev/null | xargs -I{} find {} -path "*$H" 2>/dev/null; find / -xdev -path "*site-packages*$H" 2>/dev/null; } | sort -u )
echo "== $(date -u +%FT%TZ) $(git log -1 --format='%h %s') header copies: ${#COPIES[@]}"; printf '   %s\n' "${COPIES[@]}"
export HWLOCK_HELD=ci
for v in prev fix ctl prev2; do
  src=ci_ab/gelu/${v%2}.h
  for c in "${COPIES[@]}"; do cp "$src" "$c"; done
  out=/tmp/bits/$v; rm -rf $out; mkdir -p $out
  export TT_METAL_CACHE=/tmp/ttcache_$v; rm -rf $TT_METAL_CACHE
  SFPU_FIX_DUMP_DIR=$out R3_DUMP_OPS=gelu timeout 900 python3 -m pytest -q -p no:cacheprovider ci_ab/tests/test_sfpu_unary_r3_dump.py > $out.dump.log 2>&1
  r1=$?
  SFPU_FIX_DUMP_DIR=$out R3_MM_MODE=dump R3_MM_ACTS=gelu timeout 900 python3 -m pytest -q -p no:cacheprovider ci_ab/tests/test_sfpu_unary_r3_mm.py > $out.mm.log 2>&1
  r2=$?
  echo "== $v dump rc=$r1 $(grep -E '[0-9]+ (passed|failed)' $out.dump.log | tail -1) mm rc=$r2 $(grep -E '[0-9]+ (passed|failed)' $out.mm.log | tail -1) files $(ls $out | wc -l) $(date -u +%T)"
  [[ $r1 -eq 0 && $r2 -eq 0 ]] || { tail -30 $out.dump.log; tail -30 $out.mm.log; }
  ls $out; find $TT_METAL_CACHE -name '*.elf' -path '*trisc1*' 2>/dev/null | xargs -r sha256sum | sort | awk '{print "   elf", $1}' | sort -u | head -40
done
for c in "${COPIES[@]}"; do cp ci_ab/gelu/fix.h "$c"; done
echo "== prev against fix";  python3 ci_ab/tests/compare_r2_dumps.py /tmp/bits/prev /tmp/bits/fix
echo "== prev against prev2 (A/A)";  python3 ci_ab/tests/compare_r2_dumps.py /tmp/bits/prev /tmp/bits/prev2
echo "== prev against ctl (positive control, must differ in the 32-bit DEST files)";  python3 ci_ab/tests/compare_r2_dumps.py /tmp/bits/prev /tmp/bits/ctl
exit 0
