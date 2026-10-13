#!/usr/bin/env bash
# Scratch job (MoE gate route), not for merge: the 512-expert global gate test on H (#60154's head device files) and N
# (this head's), each its own fresh JIT cache; first error lines and the per-kernel compute .text hashes.
set +e +o pipefail
cd /work; export TT_METAL_HOME=/work HWLOCK_HELD=ci
S=/work/ci_ab; OUT=/work/generated/g512; rm -rf $OUT; mkdir -p $OUT
SP=$(cd /tmp && python3 -c "import ttnn, os; print(os.path.dirname(os.path.dirname(os.path.abspath(ttnn.__file__))))")
OBJ=$(find /work/runtime /opt /usr/local "$SP" -name riscv-tt-elf-objcopy -type f 2>/dev/null | head -1)
suffix() { echo "$1" | awk -F/ '{n=NF; print $(n-3)"/"$(n-2)"/"$(n-1)"/"$n}'; }
use() {
  while read -r rel; do
    cp "$S/variants3/$1/$rel" "/work/$rel"
    while read -r p; do [[ -n $p && $p != "/work/$rel" ]] && cp "$S/variants3/$1/$rel" "$p"; done < <(find "$SP" -path "*/$(suffix $rel)" -type f 2>/dev/null)
  done < $S/files3.txt
  export TT_METAL_CACHE=$OUT/cache_$1; mkdir -p $TT_METAL_CACHE
}
for a in ${ARMS:-H N}; do
  use $a
  timeout -s INT -k 60 1800 python3 -m pytest -p no:cacheprovider -o timeout_method=thread models/common/tests/modules/moe/test_generalized_moe_gate.py -k 512_global -rf > $OUT/ut_$a.log 2>&1
  echo "== $a 512_global rc=$? $(grep -aE '[0-9]+ (passed|failed)' $OUT/ut_$a.log | tail -1)"
  grep -aE '^E ' $OUT/ut_$a.log | sort | uniq -c | sort -rn | head -6 | cut -c1-300
  grep -a '^FAILED' $OUT/ut_$a.log | head -3 | cut -c1-300
  (cd $TT_METAL_CACHE && find . -path '*/kernels/*' -name 'trisc*.elf' ! -name '*.xip.elf' | while read -r e; do k=$(echo "$e" | sed 's|.*/kernels/||; s|/.*||'); echo "$k $($OBJ -O binary -j .text "$e" /dev/stdout 2>/dev/null | md5sum | cut -c1-12)"; done | sort -u) > $OUT/elf_$a.txt
done
diff $OUT/elf_H.txt $OUT/elf_N.txt > /dev/null && echo "IDENT512 same $(wc -l < $OUT/elf_H.txt) compute ELFs" || { echo "IDENT512 DIFF"; diff $OUT/elf_H.txt $OUT/elf_N.txt | head -10; }
echo "== done"
