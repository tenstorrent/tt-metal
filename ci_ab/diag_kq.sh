#!/usr/bin/env bash
# Scratch, not for merge: does the LLK_PACK_ADDR_RMW define of the tilize and untilize programs reach the compiler?
set +e +o pipefail
cd /work; export TT_METAL_HOME=/work; S=/work/ci_ab; OUT=/work/generated/diag; rm -rf $OUT; mkdir -p $OUT
python3 -c "import tracy" 2>/dev/null || export PYTHONPATH=/work:/work/tools
export PYTHONPATH=$S:${PYTHONPATH:-}
SP=$(cd /tmp && env -u PYTHONPATH python3 -c "import ttnn, os; print(os.path.dirname(os.path.dirname(os.path.abspath(ttnn.__file__))))")
rel=tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_pack.h
P=$(find "$SP" -path "*/$rel" -type f)
echo "== package header: $P"
for a in M K F; do
  cp $S/v/$a/$rel /work/$rel; for p in $P; do cp $S/v/$a/$rel $p; done
  export TT_METAL_CACHE=$OUT/cache_$a; mkdir -p $TT_METAL_CACHE
  timeout -s INT -k 60 900 python3 -m pytest -q -p no:cacheprovider -o timeout_method=thread -o addopts= $S/test_cls_ab.py -k "tilize_bf16_s1 or tilize_bf16_l1 or untilize_bf16_s1 or untilize_bf16_l1 or tilize_vp_bf16 or untilize_unpad_bf16 or typecast_bf16_fp32_s1" > $OUT/run_$a.log 2>&1
  echo "== run $a rc=$? $(grep -aE '[0-9]+ (passed|failed)' $OUT/run_$a.log | tail -1)"
  ELFKEY_DROP=NOTHING python3 $S/elfhash.py $a $OUT/cache_$a tilize typecast > $OUT/elf_$a.txt 2>&1
  grep '^ELF ' $OUT/elf_$a.txt | cut -c1-200
  echo "== kernel dirs $a: $(ls $OUT/cache_$a/*/kernels 2>/dev/null | head -40 | tr '\n' ' ')"
  for d in $(find $OUT/cache_$a -type d -path "*kernels/*tilize*" -maxdepth 6 | head -3); do echo "-- $d"; ls $d | head; grep -rl LLK_PACK_ADDR_RMW $d 2>/dev/null | head -3; done
done
