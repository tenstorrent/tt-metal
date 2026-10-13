#!/usr/bin/env bash
# FusedRope gtests (unit_tests_llk) per variant: the variant's rope header is copied over every copy the JIT can read
# (/work and the installed runtime trees), each variant with its own empty kernel cache; the guard prints the math
# (trisc1) ELF hashes each variant's JIT built, which must differ between variants with different headers.
# usage: run_gtest.sh <variant> ...
set +e +o pipefail
cd /work
H=tt_metal/tt-llk/tt_llk_blackhole/common/inc/sfpu/experimental/ckernel_sfpu_rope.h
B=$(ls -d /work/build*/test/tt_metal/unit_tests_llk 2>/dev/null | head -1)
echo "== gtest binary: ${B:-none}"
[[ -n $B ]] || { ls /work/build* 2>/dev/null | head; exit 0; }
mapfile -t COPIES < <( { echo /work/$H; find / -xdev -path "*site-packages*$H" 2>/dev/null; } | sort -u )
echo "== header copies: ${COPIES[*]}"
ORIG=/tmp/rope_orig.h; cp /work/$H $ORIG
for v in "$@"; do
  src=ci_ab/variants/$v/$H; [[ -f $src ]] || src=$ORIG
  for c in "${COPIES[@]}"; do cp "$src" "$c"; done
  export TT_METAL_CACHE=/tmp/ttcache_$v; rm -rf $TT_METAL_CACHE
  timeout 900 $B --gtest_filter='*FusedRope*' > /tmp/gtest_$v.log 2>&1
  echo "== gtest $v rc=$? $(grep -E '\[  (PASSED|FAILED)  \]' /tmp/gtest_$v.log | tr '\n' ' ')"
  grep -E "\[  FAILED  \]|Failure|error" /tmp/gtest_$v.log | head -5
  echo "   $v math ELFs: $(find $TT_METAL_CACHE -name '*.elf' -path '*trisc1*' 2>/dev/null | xargs -r sha256sum | awk '{print substr($1,1,12)}' | sort -u | paste -sd' ')"
done
for c in "${COPIES[@]}"; do cp $ORIG "$c"; done
