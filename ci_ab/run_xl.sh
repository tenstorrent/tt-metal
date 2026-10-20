#!/usr/bin/env bash
# Scratch: bits (A vs B output md5, then the ttnn test file on B) then the op-level A/B (run_tk_ab.sh, prof_ops.py = topk_large_indices cases).
set +e +o pipefail
cd /work; export TT_METAL_HOME=/work HWLOCK_HELD=ci; S=/work/ci_ab
RT=$(cd /tmp && env -u PYTHONPATH python3 -c "import ttnn, os; print(os.path.dirname(os.path.abspath(ttnn.__file__)))")
use() { while read -r rel; do cp "$S/variants/$1/$rel" "/work/$rel"; [[ -f $RT/$rel ]] && cp "$S/variants/$1/$rel" "$RT/$rel"; done < $S/files_ops.txt; }
mkdir -p /work/generated/bits
for a in A B A B; do use $a; export TT_METAL_CACHE=/work/generated/bits/cache_$a$RANDOM; timeout -s INT -k 60 1500 python3 $S/bits_xl.py > /work/generated/bits/$a.$RANDOM.log 2>&1; echo "== bits $a rc=$?"; done
for f in /work/generated/bits/A.*.log; do cat $f; done | sort > /tmp/bitsA.txt; for f in /work/generated/bits/B.*.log; do cat $f; done | sort > /tmp/bitsB.txt
echo "BITS cases $(sort -u /tmp/bitsA.txt | wc -l) A, $(sort -u /tmp/bitsB.txt | wc -l) B; lines differing between A and B: $(sort -u /tmp/bitsA.txt | comm -3 - <(sort -u /tmp/bitsB.txt) | wc -l); not ok: $(grep -c 'False\|FAILED' /tmp/bitsB.txt)"
sort -u /tmp/bitsA.txt | comm -3 - <(sort -u /tmp/bitsB.txt) | head -10
use B; export TT_METAL_CACHE=/work/generated/bits/cache_pt
timeout -s INT -k 60 3000 python3 -m pytest -q -x -p no:cacheprovider tests/ttnn/nightly/unit_tests/operations/experimental/test_topk_large_indices.py > /work/generated/bits/pytest_B.log 2>&1; echo "BITS pytest B rc=$? $(tail -n 1 /work/generated/bits/pytest_B.log)"
use A
bash $S/run_tk_ab.sh
