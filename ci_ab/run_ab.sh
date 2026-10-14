#!/usr/bin/env bash
# Scratch, not for merge: op-level A/B of the MoE DRAM expert matmul on one P150 (17_CI_AB.md). Each arm's files
# (ci_ab/files.txt, from ci_ab/v/<arm>/) go into the checkout and the installed ttnn package before its run; every run has
# its own fresh JIT cache; arms named X2 are the A/A copy of X. Order from $ORDER, blocks from $BLOCKS.
set +e +o pipefail
R=$(git rev-parse --show-toplevel 2>/dev/null); R=${R:-/work}; S=$R/ci_ab; cd $R
export PYTHONPATH=$R:$R/tools:${PYTHONPATH:-}
RT=$(cd /tmp && env -u PYTHONPATH python3 -c "import ttnn, os; print(os.path.dirname(ttnn.__file__))" 2>/dev/null)
ORDER=${ORDER:-"M H E E2 E2 E H M"}; BLOCKS=${BLOCKS:-2}; GUARD=${GUARD:-"M H E"}; EXTRA=${EXTRA:-"P ZA:sum ZEA:sum"}
echo "== $(date -u +%FT%TZ) $(git log -1 --format='%h %s') runtime $RT order $ORDER blocks $BLOCKS"
T=models/demos/deepseek_v3_b1/tests/unit_tests/per_core_allocation
cp $S/test_mexp_ab.py $T/test_mexp_ab.py
: > $S/dest.txt
while read -r rel; do
  [[ -z $rel ]] && continue
  cmp -s $S/v/H/$rel $R/$rel || { echo "VOID: H differs from the tree: $rel"; exit 1; }
  suf=$(echo "$rel" | awk -F/ '{n=NF; print $(n-3)"/"$(n-2)"/"$(n-1)"/"$n}')
  for p in $(find "$RT" -path "*/$suf" -type f 2>/dev/null); do echo "$rel $p" >> $S/dest.txt; done
done < $S/files.txt
echo "== $(wc -l < $S/dest.txt) package copies of $(wc -l < $S/files.txt) files"
for a in $GUARD; do echo "== files $a: $(cd $S/v/$a && cat $S/files.txt | xargs md5sum | md5sum | cut -c1-10)"; done
python3 -m pytest --collect-only -q $T/test_mexp_ab.py 2>/dev/null | grep -oE 'test_mexp\[[^]]*\]' | sort -u | sed "s|^|$T/test_mexp_ab.py::|" > $S/ids.txt
cat $S/ids.txt; [[ -s $S/ids.txt ]] || exit 1
declare -A EH SEEN
run1() {
  local arm=$1 r=$2 extra=$3 out=$O/${1}_$2
  rm -rf $out /tmp/ttc_cur; mkdir -p $out
  local t0=$(date +%s)
  TT_METAL_CACHE=/tmp/ttc_cur IDS_FILE=$S/ids.txt timeout -s INT -k 60 900 python3 -m tracy -p -r --no-web-server $extra -o $out -m ci_ab.tools.run_nodes > $out.log 2>&1
  local rc=$?
  echo "== $arm $r rc=$rc $(( $(date +%s) - t0 )) s: $(grep -aE '[0-9]+ (passed|failed)' $out.log | tail -1)"
  [[ $rc == 124 ]] && { echo "== TIMEOUT in $arm, stopping"; exit 1; }
  grep -aqE "passed|failed" $out.log || { echo NO_PYTEST_OUTPUT; tail -c 3000 $out.log; }
  grep -aE "^(FAILED|ERROR)" $out.log | cut -c1-200 | head -4
  echo "BITSUM $arm $r $(grep -a '^BITS ' $out.log | md5sum | cut -c1-12) $(grep -ac '^BITS ' $out.log)"
  [[ $r == 1a ]] && grep -a '^BITS ' $out.log > $O/bits_${arm}.txt
  if [[ -z ${EH[$arm]} ]]; then EH[$arm]=$(python3 $S/tools/elfsum.py $arm /tmp/ttc_cur); echo "${EH[$arm]}"; fi
  python3 $S/tools/dlog.py $out $arm $r
  rm -rf /tmp/ttc_cur $out
}
O=/work/generated/ab; mkdir -p $O
use() { local src=${1%2}; src=${src%:*}; while read -r rel; do [[ -n $rel ]] && cp $S/v/$src/$rel $R/$rel; done < $S/files.txt
  while read -r rel p; do cp $S/v/$src/$rel $p; done < $S/dest.txt; }
for b in $(seq $BLOCKS); do
  for arm in $ORDER; do
    r=$b$([[ -n ${SEEN[$arm$b]} ]] && echo b || echo a); SEEN[$arm$b]=1
    use $arm
    run1 $arm $r ""
  done
  if [[ $b == 1 ]]; then
    h() { echo "${EH[$1]}" | sed 's/^ELF [A-Za-z0-9]* //'; }
    for x in $GUARD; do for y in $GUARD; do [[ $x < $y && $(h $x) == $(h $y) ]] && { echo "== GUARD FAIL: $x and $y ELFs identical: $(h $x)"; exit 1; }; done; done
    for x in $ORDER; do [[ $x == *2 && $(h $x) != $(h ${x%2}) ]] && echo "== NOTE: $x ELFs differ from ${x%2}: $(h $x) vs $(h ${x%2})"; done
    echo "== GUARD ok"
  fi
done
for x in $EXTRA; do a=${x%%:*}; f=""; [[ $x == *:sum ]] && f=--enable-sum-profiling; use $a; run1 $a x "$f"; done
for a in $GUARD; do [[ -f $O/bits_$a.txt ]] && echo "BITS-ARM $a $(md5sum < $O/bits_$a.txt | cut -c1-12) $(wc -l < $O/bits_$a.txt)"; done
for a in $GUARD; do [[ $a != H && -f $O/bits_$a.txt ]] && { diff <(cut -d' ' -f2- $O/bits_H.txt) <(cut -d' ' -f2- $O/bits_$a.txt) > /dev/null && echo "BITS-SAME H $a" || { echo "BITS-DIFF H $a"; diff $O/bits_H.txt $O/bits_$a.txt | head -6; }; }; done
use H
echo "== DONE $(date -u +%T)"
