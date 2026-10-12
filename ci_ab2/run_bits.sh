#!/usr/bin/env bash
# ttnn raw-word dumps (typecast pairs, quantize, requantize, dequantize; bits_tc4.py) on one CI Blackhole runner for
# main and two device-file variants (the quantization levers, the typecast fix), each with its own fresh JIT cache. The
# variant's files go into the checkout and into the installed ttnn package (the JIT reads the LLK headers there);
# main's copies are restored after each run. Words are compared against main's run in the job.
set +e +o pipefail
export HWLOCK_HELD=ci
R=$(git rev-parse --show-toplevel); S=$R/ci_ab2; cd $R
SP=$(python -c "import ttnn, os; print(os.path.dirname(os.path.dirname(os.path.abspath(ttnn.__file__))))")
echo "== runner $(hostname) $(date -u +%FT%TZ), package root $SP"
OBJ=$(find / -name riscv-tt-elf-objcopy -type f 2>/dev/null | head -1); echo "== objcopy $OBJ"
B=/tmp/ra2_main_files; rm -rf $B; mkdir -p $B
while read -r rel; do mkdir -p $B/$(dirname $rel); cp $R/$rel $B/$rel; done < $S/files.txt
place() { # place <variant|main>
  while read -r rel; do
    local src=$B/$rel; [[ $1 != main && -f $S/variants/$1/tree/$rel ]] && src=$S/variants/$1/tree/$rel
    cp "$src" "$R/$rel"
    local suf; suf=$(echo "$rel" | awk -F/ '{n=NF; print $(n-3)"/"$(n-2)"/"$(n-1)"/"$n}')
    local n=0; while read -r p; do [[ -n $p && $p != "$R/$rel" ]] && { cp "$src" "$p"; n=$((n+1)); }; done < <(find "$SP" -path "*/$suf" -type f 2>/dev/null)
    [[ $1 == q || $1 == tc ]] && [[ $src != $B/* ]] && echo "   placed $rel (+$n package copies)"
  done < $S/files.txt
}
elfsum() { # per kernel name, one hash over the sorted .text hashes of its compute ELFs (any cache layout)
  local n; n=$(find "$1" -name 'trisc[012].elf' 2>/dev/null | wc -l); echo "   ($n compute ELFs under $1: $(ls "$1" 2>/dev/null | head -3 | tr '\n' ' '))"
  [[ $n == 0 ]] && echo "   other JIT caches: $(find / -maxdepth 5 -type d -name 'tt-metal-cache*' 2>/dev/null | head -6 | tr '\n' ' ')"
  find "$1" -path '*/kernels/*' -name 'trisc[012].elf' 2>/dev/null | while read -r e; do
    k=$(echo "$e" | sed -E 's#.*/kernels/([^/]+)/.*#\1#'); $OBJ -O binary -j .text "$e" /tmp/elf.bin 2>/dev/null; echo "$k $(md5sum < /tmp/elf.bin | cut -c1-12)"
  done | sort | awk '{h[$1]=h[$1] $2} END {for (k in h) print k, h[k]}' | while read -r k hs; do echo "   $k $(echo $hs | md5sum | cut -c1-12)"; done | sort
}
OUT=/tmp/ra2_bits; rm -rf $OUT; mkdir -p $OUT
for v in main q tc main2; do
  vv=${v%2}; place $vv
  export TT_METAL_CACHE=/tmp/ra2cache_$v; rm -rf $TT_METAL_CACHE; mkdir -p $TT_METAL_CACHE
  echo "== $v $(date -u +%T)"
  R3_BITS_OUT=$OUT/$v timeout -s INT -k 60 1500 python -m pytest -p no:cacheprovider -o timeout_method=thread -q -rfE $S/tools/bits_tc4.py > $OUT/log_$v.txt 2>&1
  echo "== $v rc=$? $(grep -aE '[0-9]+ (passed|failed)' $OUT/log_$v.txt | tail -1) $(date -u +%T)"
  grep -a -m5 -E 'FAILED|error:|TIMED OUT' $OUT/log_$v.txt
  echo "== compute ELFs of $v:"; elfsum $TT_METAL_CACHE
done
place main
python3 $S/tools/cmp_bits_ci.py $OUT
exit 0
