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
  (cd $TT_METAL_CACHE && find . -path '*/kernels/*' -name 'trisc*.elf' ! -name '*.xip.elf' | while read -r e; do k=$(echo "$e" | sed 's|.*/kernels/||'); echo "$k $($OBJ -O binary -j .text "$e" /dev/stdout 2>/dev/null | md5sum | cut -c1-12)"; done | sort -u) > $OUT/elf_$a.txt
done
join -j1 -a1 -a2 -e none -o 0,1.2,2.2 <(sort $OUT/elf_H.txt) <(sort $OUT/elf_N.txt) > $OUT/j.txt
awk '{split($1,p,"/"); k=p[1]; s=($2==$3)?"same":($2=="none"||$3=="none")?"unpaired":"diff"; n[k" "s]++} END {for (x in n) print "IDENTP", x, n[x]}' $OUT/j.txt | sort
awk '$2!=$3 {split($1,p,"/"); if (p[1]=="generalized_moe_gate_kernel") d[p[2]]=d[p[2]]" "p[3]} END {for (h in d) print "IDENTD", h, d[h]}' $OUT/j.txt | sort
h=$(awk '$2!=$3 {split($1,p,"/"); if (p[1]=="generalized_moe_gate_kernel") print p[2]}' $OUT/j.txt | head -1)
d=$(find $OUT/cache_N -path "*/kernels/generalized_moe_gate_kernel/$h" -type d | head -1); echo "== layout of $d"; (cd $d && find . -maxdepth 2 | head -30)
for f in $(find $d -maxdepth 2 -type f \( -name '*.h' -o -name '*.csv' -o -name '*.txt' \) | head -8); do echo "-- $f"; grep -a -m8 -iE "ungrouped|topk|num_blocks|n_group|compile|arg" $f | cut -c1-200; done
echo "== done"
