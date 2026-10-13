#!/usr/bin/env bash
# Scratch proof job for #60103 (typecast), not for merge. One machine: (1) ttnn raw words of the typecast pairs
# (bits_tc4.py -k typecast) with the device files of main, K (fp32_to_uint16 unrolled, no drain), A (K plus the unrolled
# 32-bit input bodies) and main again, each with a fresh JIT cache, and the per-kernel compute ELF hashes; (2) typecast
# device time (tracy) with the head's files (H), K, A and an A/A copy H2, rotated H K A H2 H2 A K H in two blocks.
set +e +o pipefail
export HWLOCK_HELD=ci TT_METAL_HOME=$(git rev-parse --show-toplevel) TRACY_NO_WEB_SERVER=1
R=$TT_METAL_HOME; S=$R/ci_ab2; cd $R
SP=$(cd /tmp && python3 -c "import ttnn, os; print(os.path.dirname(os.path.dirname(os.path.abspath(ttnn.__file__))))")
echo "== runner $(hostname) $(date -u +%FT%TZ) $(git log -1 --format='%h') package root $SP"
OBJ=$(find / -name riscv-tt-elf-objcopy -type f 2>/dev/null | head -1); echo "== objcopy $OBJ"
use() {  # use <side>: its files into the checkout and every copy under the package root, its own JIT cache
  while read -r rel; do
    cp "$S/v/$1/$rel" "$R/$rel"; local n=0
    suf=$(echo "$rel" | awk -F/ '{n=NF; print $(n-3)"/"$(n-2)"/"$(n-1)"/"$n}')
    while read -r p; do [[ -n $p && $p != "$R/$rel" ]] && { cp "$S/v/$1/$rel" "$p"; n=$((n+1)); }; done < <(find "$SP" -path "*/$suf" -type f 2>/dev/null)
    echo "   $1: $rel (+$n package copies) $(md5sum < "$R/$rel" | cut -c1-8)"
  done < $S/files.txt
  export TT_METAL_CACHE=/tmp/tccache_$1; mkdir -p $TT_METAL_CACHE
}
elfsum() {
  find "$1" -path '*/kernels/*' -name 'trisc[012].elf' 2>/dev/null | while read -r e; do
    k=$(echo "$e" | sed -E 's#.*/kernels/([^/]+)/.*#\1#'); $OBJ -O binary -j .text "$e" /tmp/elf.bin 2>/dev/null; echo "$k $(md5sum < /tmp/elf.bin | cut -c1-12)"
  done | sort | awk '{h[$1]=h[$1] $2} END {for (k in h) print k, h[k]}' | while read -r k hs; do echo "   ELF $2 $k $(echo $hs | md5sum | cut -c1-12)"; done | sort
}
OUT=/tmp/tc_bits; rm -rf $OUT /tmp/tccache_*; mkdir -p $OUT
for v in main K A main2; do
  use $v; echo "== bits $v $(date -u +%T)"
  R3_BITS_OUT=$OUT/$v timeout -s INT -k 60 1800 python3 -m pytest -p no:cacheprovider -o timeout_method=thread -q -rfE -k typecast $S/tools/bits_tc4.py > $OUT/log_$v.txt 2>&1
  echo "== bits $v rc=$? $(grep -aE '[0-9]+ (passed|failed)' $OUT/log_$v.txt | tail -1) $(date -u +%T)"; grep -a -m5 -E 'FAILED|error:' $OUT/log_$v.txt
  elfsum $TT_METAL_CACHE $v
done
python3 - $OUT <<'PY'
import glob, os, sys
import numpy as np
d = sys.argv[1]
load = lambda v: {os.path.basename(p)[:-4]: p for p in glob.glob(os.path.join(d, v, "*.npz"))}
m = load("main")
for v in ("main2", "K", "A"):
    o = load(v); same = diff = words = 0; bad = []
    for n in sorted(set(m) | set(o)):
        if n not in m or n not in o: bad.append(n + "(missing)"); continue
        a, b = np.load(m[n]), np.load(o[n]); nd = 0
        for k in a.files:
            if k not in b.files: nd += 1; continue
            x, y = a[k], b[k]; words += x.size
            if x.dtype != y.dtype or x.shape != y.shape: nd += x.size or 1; continue
            s = x.dtype.itemsize
            nd += int((np.ascontiguousarray(x).view(np.uint8).reshape(-1, s) != np.ascontiguousarray(y).view(np.uint8).reshape(-1, s)).any(axis=1).sum())
        if nd: diff += 1; bad.append(f"{n}({nd})")
        else: same += 1
    print(f"== bits {v} against main: {same} cases identical, {diff} differ, {words} words; {' '.join(bad[:20])}")
PY
rm -rf /tmp/tccache_*
CASES="tc_fp32_u16 tc_bf16_u16 tc_i32_u16 tc_u32_bf16 tc_u32_fp32 tc_fp32_bf16 tc_i32_fp32 tc_i32_bf16 tc_bf16_i32"
r=0
for blk in 1 2; do for v in H K A H2 H2 A K H; do
  r=$((r+1)); use $v > /dev/null; o=/tmp/tcop/r${r}_$v; mkdir -p $o
  timeout -s INT -k 60 1500 python3 -m tracy -r -p --no-web-server -o $o $S/tools/prof_tc.py $CASES > $o.log 2>&1
  rc=$?; [[ $rc -ne 0 ]] && { echo "RAB FAIL $v $r rc=$rc"; grep -a -m3 -E 'Error|error|FAILED' $o.log | cut -c1-200; }
  grep -a "CASE .* FAILED" $o.log | head -3
  python3 $S/tools/sum_tc.py $o $v $r
done; done
use H > /dev/null
echo "== done $(date -u +%FT%TZ)"
