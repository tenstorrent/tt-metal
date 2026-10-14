#!/usr/bin/env bash
# Scratch, not for merge: op-level A/B of the MoE DRAM expert matmul on one P150 (17_CI_AB.md). Each arm's files
# (ci_ab/files.txt, from ci_ab/v/<arm>/) go into the checkout and the installed ttnn package before its run; every run has
# its own fresh JIT cache; arms named X2 are the A/A copy of X. Order from $ORDER, blocks from $BLOCKS.
set +e +o pipefail
R=$(git rev-parse --show-toplevel 2>/dev/null); R=${R:-/work}; S=$R/ci_ab; cd $R
export PYTHONPATH=$R:$R/tools:${PYTHONPATH:-}
RT=$(cd /tmp && env -u PYTHONPATH python3 -c "import ttnn, os; print(os.path.dirname(ttnn.__file__))" 2>/dev/null)
ORDER=${ORDER:-"M H L L2 L2 L H M"}; BLOCKS=${BLOCKS:-2}; GUARD=${GUARD:-"M H L"}
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
O=/work/generated/ab; mkdir -p $O
use() { local src=${1%2}; while read -r rel; do [[ -n $rel ]] && cp $S/v/$src/$rel $R/$rel; done < $S/files.txt
  while read -r rel p; do cp $S/v/$src/$rel $p; done < $S/dest.txt; }
for b in $(seq $BLOCKS); do
  for arm in $ORDER; do
    r=$b$([[ -n ${SEEN[$arm$b]} ]] && echo b || echo a); SEEN[$arm$b]=1
    use $arm
    out=$O/${arm}_$r; rm -rf $out /tmp/ttc_${arm}_$r; mkdir -p $out
    t0=$(date +%s)
    TT_METAL_CACHE=/tmp/ttc_${arm}_$r IDS_FILE=$S/ids.txt timeout -s INT -k 60 1500 python3 -m tracy -p -r --no-web-server -o $out -m ci_ab.tools.run_nodes > $out.log 2>&1
    echo "== $arm $r rc=$? $(( $(date +%s) - t0 )) s: $(grep -aE '[0-9]+ (passed|failed)' $out.log | tail -1)"
    grep -aqE "passed|failed" $out.log || { echo NO_PYTEST_OUTPUT; tail -c 3000 $out.log; }
    grep -aE "^(FAILED|ERROR)" $out.log | cut -c1-200 | head -4
    if [[ -z ${EH[$arm]} ]]; then EH[$arm]=$(python3 $S/tools/elfsum.py $arm /tmp/ttc_${arm}_$r); echo "${EH[$arm]}"; fi
    python3 $S/tools/dlog.py $out $arm $r
    rm -rf /tmp/ttc_${arm}_$r $out
  done
  if [[ $b == 1 ]]; then
    h() { echo "${EH[$1]}" | sed 's/^ELF [A-Za-z0-9]* //'; }
    for x in $GUARD; do for y in $GUARD; do [[ $x < $y && $(h $x) == $(h $y) ]] && { echo "== GUARD FAIL: $x and $y ELFs identical: $(h $x)"; exit 1; }; done; done
    for x in $ORDER; do [[ $x == *2 && $(h $x) != $(h ${x%2}) ]] && echo "== NOTE: $x ELFs differ from ${x%2}: $(h $x) vs $(h ${x%2})"; done
    echo "== GUARD ok"
  fi
done
use H
echo "== DONE $(date -u +%T)"
