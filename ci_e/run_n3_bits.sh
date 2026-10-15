#!/usr/bin/env bash
# Scratch job (split plan, N3 bits), not for merge. Same setup as run_split_elf.sh; only the device bits loop. The BITS hook is a registered plugin object (a conftest hook only sees items below its own directory), -s keeps the lines in the log.
# ci_e/v/<side>/, a .absent marker for a file the side lacks) go into the checkout and the installed package before its
# runs, and each run gets a fresh JIT cache. Sides: D0 old #58819 merged with main, D1 its restore merged with main;
# T0 #59143's restore with the old pool kernels, T1 the restore; P0 main, P1 N3 (main plus the pool and upsample init).
#  1. device (P150): P0 and P1 run the pool family cases and N3's new test with every ttnn.to_torch result hashed (bits),
#     then grid_diag.py (the #59368 readback: five C = 384 configurations, the op run twice, the input read back)
#  2. mock p100a: P0 P1 T0 T1 the pool family cases; mock N150: P0 P1; ELF hashes of the pool, upsample, grid kernels
#  3. mock p100a: D0 and D1 the 90 DATACOPY ELF cases (prof_ops12.py), every kernel's ELF hash
set +e +o pipefail
cd /work; export TT_METAL_HOME=/work
S=/work/ci_e; OUT=/work/generated/se; rm -rf $OUT; mkdir -p $OUT
python3 -c "import tracy" 2>/dev/null || export PYTHONPATH=/work:/work/tools
SP=$(cd /tmp && python3 -c "import ttnn, os; print(os.path.dirname(os.path.dirname(os.path.abspath(ttnn.__file__))))")
echo "== $(date -u +%FT%TZ) $(git log -1 --format='%h %s') package root $SP"
suffix() { echo "$1" | awk -F/ '{n=NF; print $(n-3)"/"$(n-2)"/"$(n-1)"/"$n}'; }
pkgdirs() { local s; s=$(dirname "$1" | awk -F/ '{n=NF; print $(n-2)"/"$(n-1)"/"$n}'); find "$SP" -type d -path "*/$s" 2>/dev/null; }
nopkg=0
while read -r rel; do [[ -z $(find "$SP" -path "*/$(suffix $rel)" -type f 2>/dev/null | head -1) ]] && nopkg=$((nopkg+1)); done < $S/files.txt
echo "== $(wc -l < $S/files.txt) side files, $nopkg without a package copy on main"
place() {  # place <side>
  while read -r rel; do
    if [[ -f $S/v/$1/$rel.absent ]]; then
      rm -f "/work/$rel"
      while read -r p; do [[ -n $p ]] && rm -f "$p"; done < <(find "$SP" -path "*/$(suffix $rel)" -type f 2>/dev/null)
      continue
    fi
    mkdir -p "$(dirname /work/$rel)"; cp "$S/v/$1/$rel" "/work/$rel"
    local n=0
    while read -r p; do [[ -n $p && $p != "/work/$rel" ]] && { cp "$S/v/$1/$rel" "$p"; n=$((n+1)); }; done < <(find "$SP" -path "*/$(suffix $rel)" -type f 2>/dev/null)
    if [[ $n -eq 0 ]]; then for d in $(pkgdirs "$rel"); do cp "$S/v/$1/$rel" "$d/"; done; fi
  done < $S/files.txt
  echo "SE FILES $1 $(while read -r rel; do cat "/work/$rel" 2>/dev/null; done < $S/files.txt | md5sum | cut -c1-8)"
}
fresh() { export TT_METAL_CACHE=$OUT/cache_$1; rm -rf "$TT_METAL_CACHE"; mkdir -p "$TT_METAL_CACHE"; }
P=tests/ttnn/unit_tests/operations/pool
PMODS="$P/test_maxpool2d.py $P/test_avgpool2d.py $P/test_upsample.py $P/test_grid_sample.py"
PT="python3 -m pytest -q -p no:cacheprovider -o timeout_method=thread -o addopts= -rfE $S/plugin"
fail=0

PT="python3 -m pytest -s -q -p no:cacheprovider -o timeout_method=thread -o addopts= -rfE $S/plugin"
fail=0
for s in P0 P1; do
  place $s; fresh dev_$s
  HWLOCK_HELD=ci BITS_TAG=$s SEL_FILE=$S/pool_ids.txt timeout -s INT -k 60 1500 $PT $PMODS > $OUT/bits_$s.log 2>&1
  echo "== bits $s rc=$? $(grep -aE '[0-9]+ (passed|failed)' $OUT/bits_$s.log | tail -1) $(grep -a '^SEL' $OUT/bits_$s.log | tail -1)"
  grep -a -o "BITS $s .*" $OUT/bits_$s.log | sed "s/^BITS $s //" | sort > $OUT/bits_$s.txt
  echo "== BITS $s lines $(wc -l < $OUT/bits_$s.txt)"
done
echo "== BITSCMP P0 only $(comm -23 $OUT/bits_P0.txt $OUT/bits_P1.txt | wc -l), P1 only $(comm -13 $OUT/bits_P0.txt $OUT/bits_P1.txt | wc -l), common $(comm -12 $OUT/bits_P0.txt $OUT/bits_P1.txt | wc -l)"
awk 'function key(l){sub(/ [^ ]+$/,"",l); return l} function hv(l){sub(/^.* /,"",l); return l}
  NR==FNR{h[key($0)]=hv($0); next}
  {k=key($0); seen[k]=1; if(k in h){c++; if(h[k]==hv($0)) e++; else {d++; print "DIFF " k " P0 " h[k] " P1 " hv($0)}} else p1++}
  END{for(k in h) if(!(k in seen)) p0++; print "== BITSKEY common " c+0 ", equal " e+0 ", differ " d+0 ", P0 only " p0+0 ", P1 only " p1+0}' $OUT/bits_P0.txt $OUT/bits_P1.txt
[[ -s $OUT/bits_P0.txt && -s $OUT/bits_P1.txt ]] || { echo "== BITSCMP VOID: a side printed no BITS lines"; fail=1; }
echo "-- P0 only"; comm -23 $OUT/bits_P0.txt $OUT/bits_P1.txt | cut -c1-300
echo "-- P1 only"; comm -13 $OUT/bits_P0.txt $OUT/bits_P1.txt | cut -c1-300
echo "-- P0 hashes"; cat $OUT/bits_P0.txt | cut -c1-300
place P0
echo "== done $(date -u +%FT%TZ) fail=$fail"
exit $fail
