#!/usr/bin/env bash
# Scratch ELF identity job (MoE gate route), not for merge. The gate tests and the DeepSeek gate test on H (#60154's head
# device files) and N (this head's), each its own fresh JIT cache; per kernel name, the set of compute .text hashes.
set +e +o pipefail
cd /work; export TT_METAL_HOME=/work HWLOCK_HELD=ci
S=/work/ci_ab; OUT=/work/generated/ident; rm -rf $OUT; mkdir -p $OUT
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
for a in H N; do
  use $a
  timeout -s INT -k 60 2400 python3 -m pytest -q -p no:cacheprovider -o timeout_method=thread models/common/tests/modules/moe/test_generalized_moe_gate.py -k "not 512_global" > $OUT/ut1_$a.log 2>&1
  echo "== $a test_generalized_moe_gate rc=$? $(grep -aE '[0-9]+ (passed|failed)' $OUT/ut1_$a.log | tail -1)"
  timeout -s INT -k 60 2400 python3 -m pytest -q -p no:cacheprovider -o timeout_method=thread models/demos/deepseek_v3_b1/tests/unit_tests/test_deepseek_moe_gate.py > $OUT/ut2_$a.log 2>&1
  echo "== $a test_deepseek_moe_gate rc=$? $(grep -aE '[0-9]+ (passed|failed)' $OUT/ut2_$a.log | tail -1)"
  (cd $TT_METAL_CACHE && find . -path '*/kernels/*' -name 'trisc*.elf' ! -name '*.xip.elf' | while read -r e; do k=$(echo "$e" | sed 's|.*/kernels/||; s|/.*||'); echo "$k $($OBJ -O binary -j .text "$e" /dev/stdout 2>/dev/null | md5sum | cut -c1-12)"; done | sort -u) > $OUT/elf_$a.txt
done
join -a1 -a2 -e none -o 0,1.2,2.2 <(awk '{h[$1]=h[$1]" "$2} END {for (k in h) print k, h[k]}' $OUT/elf_H.txt | awk '{k=$1; $1=""; n=split($0,a," "); asort(a); s=""; for(i=1;i<=n;i++) s=s a[i] ","; print k, s}' | sort) \
  <(awk '{h[$1]=h[$1]" "$2} END {for (k in h) print k, h[k]}' $OUT/elf_N.txt | awk '{k=$1; $1=""; n=split($0,a," "); asort(a); s=""; for(i=1;i<=n;i++) s=s a[i] ","; print k, s}' | sort) | \
  awk '{print "IDENT", $1, ($2==$3 ? "same" : "DIFF"), split($2,x,",")-1, split($3,y,",")-1}'
echo "== done"
