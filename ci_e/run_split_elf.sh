#!/usr/bin/env bash
# Scratch job (split plan, carried pieces), not for merge. Built from main; each side's device files (ci_e/files.txt,
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

# 1. device: bits and the #59368 readback, main against N3
for s in P0 P1; do
  place $s; fresh dev_$s
  HWLOCK_HELD=ci BITS_TAG=$s SEL_FILE=$S/pool_ids.txt timeout -s INT -k 60 1500 $PT $PMODS > $OUT/bits_$s.log 2>&1
  echo "== bits $s rc=$? $(grep -aE '[0-9]+ (passed|failed)' $OUT/bits_$s.log | tail -1) $(grep -a '^SEL' $OUT/bits_$s.log | tail -1)"
  grep -a '^BITS ' $OUT/bits_$s.log
  grep -a -E '^(FAILED|ERROR)' $OUT/bits_$s.log | cut -c1-250
  HWLOCK_HELD=ci timeout -s INT -k 60 900 python3 $S/grid_diag.py $OUT/gd_$s $s > $OUT/gd_$s.log 2>&1
  echo "== grid_diag $s rc=$?"; grep -a -E "^$s c|DIAG DONE" $OUT/gd_$s.log | cut -c1-600
  python3 $S/elfhash.py dev_$s $TT_METAL_CACHE pool bilinear grid
done

# 2. mock: pool, upsample and grid kernels
for spec in P0:bh P1:bh T0:bh T1:bh P0:wh P1:wh; do
  s=${spec%%:*}; a=${spec#*:}
  desc=$([[ $a == wh ]] && echo /work/tt_metal/third_party/umd/tests/cluster_descriptor_examples/wormhole_N150.yaml || echo $S/p100a_cluster.yaml)
  [[ -f $desc ]] || { echo "== no mock descriptor $desc"; fail=1; continue; }
  place $s; fresh mock_${a}_$s
  TT_METAL_MOCK_CLUSTER_DESC_PATH=$desc SEL_FILE=$S/pool_ids.txt timeout -s INT -k 60 1200 $PT $PMODS > $OUT/mock_${a}_$s.log 2>&1
  echo "== mock $a $s rc=$? $(grep -aE '[0-9]+ (passed|failed|error)' $OUT/mock_${a}_$s.log | tail -1)"
  python3 $S/elfhash.py ${a}_$s $TT_METAL_CACHE pool bilinear grid
done

# 3. mock: the DATACOPY cases, old #58819 against its restore (both merged with main)
echo "== DATACOPY mock runs one after the other (one checkout, one package) $(date -u +%T)"
for s in D0 D1; do
  place $s; fresh mock_bh_$s
  HWLOCK_HELD= TT_METAL_MOCK_CLUSTER_DESC_PATH=$S/p100a_cluster.yaml R3_NMEAS=1 R3_L1_SMALL=16384 \
    timeout -s INT -k 60 2400 python3 $S/prof_ops12.py $(cat $S/elf_cases.txt) > $OUT/mock_bh_$s.log 2>&1
  echo "== mock bh $s rc=$? ok=$(grep -c ' ok$' $OUT/mock_bh_$s.log) failed=$(grep -c FAILED $OUT/mock_bh_$s.log) $(date -u +%T)"
  grep -a -m5 FAILED $OUT/mock_bh_$s.log | cut -c1-250
  python3 $S/elfhash.py bh_$s $TT_METAL_CACHE
done
place P0
echo "== done $(date -u +%FT%TZ) fail=$fail"
exit $fail
