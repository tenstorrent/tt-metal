#!/usr/bin/env bash
# Scratch job (#58820 zero-regression fix, results), not for merge. Functional reduce tests on head, X and Y (run_reduce_fix.sh
# sides): test_reduce.py ROW SUM/AVG (the tile path's changed SETRWC) and all of test_reduce_block.py. Each side builds once in
# a fresh dir; runs head X Y head with bitdump.py, which hashes every result; head twice is the A/A of the stimuli.
set +e +o pipefail
R=$(git rev-parse --show-toplevel); S=$R/ci_ab; cd $R
echo "== checkout $(git log --oneline -1 | cut -c1-60), $(date -u +%T)"
place() { while read -r rel; do [[ -n $rel ]] && cp "$S/v/$1/$rel" "$R/$rel"; done < $S/v/files.txt; }
python -m venv $HOME/llkvenv && source $HOME/llkvenv/bin/activate
pip install -q -r tt_metal/tt-llk/tests/requirements.txt > /tmp/pip.txt 2>&1; echo "== pip rc=$?"
(cd tt_metal/tt-llk/tests && ./setup_testing_env.sh > /tmp/setup.txt 2>&1); echo "== setup rc=$?"
export CHIP_ARCH=blackhole LLK_HOME=$R/tt_metal/tt-llk TT_LLK_DISABLE_ASSERTS=1 PYTHONPATH=$S/sweep:${PYTHONPATH:-}
cd $R/tt_metal/tt-llk/tests/python_tests
{ python -m pytest --collect-only -q -p no:cacheprovider test_reduce.py -k "Row and (Sum or Average)" 2>/dev/null | grep "::"
  python -m pytest --collect-only -q -p no:cacheprovider test_reduce_block.py 2>/dev/null | grep "::"; } > /tmp/ids.txt
echo "== ids $(wc -l < /tmp/ids.txt)"
mapfile -t IDS < /tmp/ids.txt
B=$HOME/redbits_b; rm -rf $B; mkdir -p $B; NP=$(( $(nproc) < 16 ? $(nproc) : 16 ))
for side in head Z; do
  place $side; d=$B/$side; mkdir -p $d
  RUNNER_TEMP=$d TMPDIR=$d timeout -s INT -k 60 3000 python -m pytest --compile-producer -n $NP -p no:cacheprovider -q "${IDS[@]}" > /tmp/b_$side.txt 2>&1
  echo "== build $side rc=$? $(date -u +%T) $(grep -aE '[0-9]+ (passed|failed|error)' /tmp/b_$side.txt | tail -1)"
done
mkdir -p $B/out
for run in head1:head Z:Z head2:head; do
  tag=${run%%:*}; side=${run##*:}
  RUNNER_TEMP=$B/$side TMPDIR=$B/$side BITDUMP=$B/out/bits_$tag.tsv timeout -s INT -k 60 3000 python -m pytest --compile-consumer \
    -p bitdump -p no:cacheprovider -q -o timeout_method=thread --timeout 120 "${IDS[@]}" > /tmp/r_$tag.txt 2>&1
  echo "== run $tag rc=$? $(date -u +%T) $(grep -aE '[0-9]+ (passed|failed)' /tmp/r_$tag.txt | tail -1) hashes $(wc -l < $B/out/bits_$tag.tsv)"
  grep -a -E "^FAILED" /tmp/r_$tag.txt | cut -c1-200 | head -5
done
cd $B/out
for p in "head1 head2" "head1 Z"; do set -- $p
  echo "== bits $1 vs $2: $(join -t $'\t' <(sort bits_$1.tsv) <(sort bits_$2.tsv) | awk -F'\t' '{n++; if ($3==$5) s++} END {print s+0" identical of "n+0}')"
done
echo "== payload begin"; tar czf - bits_*.tsv | base64 -w 200; echo "== payload end"
echo "== DONE $(date -u +%T)"
