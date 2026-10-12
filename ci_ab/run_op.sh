#!/usr/bin/env bash
# Scratch A/B job of the misc family (headroom round), not for merge. One job, one machine: ttnn.experimental.dropout
# device time with the device files of main (X), the #60122 head merged (Y), the head with the ADDR_MOD_6 store (Z)
# and an A/A copy of Y (Y2), each with its own fresh JIT cache, swapped into the checkout and the installed package,
# rotated X Y Z Y2 Y2 Z Y X in two blocks, every case in its own tracy process.
set +e +o pipefail
cd /work; export TT_METAL_HOME=/work TRACY_NO_WEB_SERVER=1
S=/work/ci_ab; OUT=/work/generated/op; rm -rf $OUT; mkdir -p $OUT
SP=$(cd /tmp && python3 -c "import ttnn, os; print(os.path.dirname(os.path.dirname(os.path.abspath(ttnn.__file__))))")
echo "== $(date -u +%FT%TZ) $(git log -1 --format='%h %s') package root $SP"
OBJ=$(find / -name riscv-tt-elf-objcopy -type f 2>/dev/null | head -1)
while read -r rel; do suf=$(echo "$rel" | awk -F/ '{n=NF; print $(n-3)"/"$(n-2)"/"$(n-1)"/"$n}'); echo "== $rel -> $(find "$SP" -path "*/$suf" -type f 2>/dev/null | tr '\n' ' ')"; done < $S/files.txt
use() {  # use <variant>: its files into the checkout and every copy under the package root, its own JIT cache
  while read -r rel; do
    cp "$S/v/$1/$rel" "/work/$rel"
    suf=$(echo "$rel" | awk -F/ '{n=NF; print $(n-3)"/"$(n-2)"/"$(n-1)"/"$n}')
    while read -r p; do [[ -n $p && $p != "/work/$rel" ]] && cp "$S/v/$1/$rel" "$p"; done < <(find "$SP" -path "*/$suf" -type f 2>/dev/null)
  done < $S/files.txt
  export TT_METAL_CACHE=$OUT/cache_$1; mkdir -p $TT_METAL_CACHE
}
elfsum() { for e in $(ls $1/*/kernels/*ropout*/*/trisc1/*.elf 2>/dev/null | sort); do $OBJ -O binary -j .text "$e" /tmp/e.bin 2>/dev/null && md5sum < /tmp/e.bin; done | md5sum | cut -c1-12; }
CASES="dropout_32x32 dropout_1024x1024 dropout_4096x2048"
r=0
for blk in 1 2; do for v in X Y Z Y2 Y2 Z Y X; do
  r=$((r+1)); use $v
  for c in $CASES; do
    o=$OUT/${c}__${v}__r$r; mkdir -p $o
    timeout -s INT -k 60 900 python3 -m tracy -r -p --no-web-server -o $o $S/prof1.py $c > $o.log 2>&1
    rc=$?; [[ $rc -ne 0 ]] && { echo "RAB FAIL $c $v $r rc=$rc"; grep -a -m3 -E 'Error|error|FAILED' $o.log | cut -c1-200; }
    python3 $S/sum1.py $o $c $v $r
  done
  [[ $r -le 4 ]] && echo "RAB ELF $v $(elfsum $TT_METAL_CACHE)"
done; done
use X
echo "== done $(date -u +%FT%TZ)"
