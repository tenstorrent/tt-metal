#!/usr/bin/env bash
# Scratch: bits (A vs B output md5, then the ttnn test file on B) then the op-level A/B (run_tk_ab.sh, prof_ops.py = topk_large_indices cases).
set +e +o pipefail
cd /work; export TT_METAL_HOME=/work HWLOCK_HELD=ci; S=/work/ci_ab
RT=$(cd /tmp && env -u PYTHONPATH python3 -c "import ttnn, os; print(os.path.dirname(os.path.abspath(ttnn.__file__)))")
use() { while read -r rel; do cp "$S/variants/$1/$rel" "/work/$rel"; [[ -f $RT/$rel ]] && cp "$S/variants/$1/$rel" "$RT/$rel"; done < $S/files_ops.txt; }
mkdir -p /work/generated/bits
for a in A B A B; do use $a; export TT_METAL_CACHE=/work/generated/bits/cache_$a$RANDOM; timeout -s INT -k 60 1500 python3 $S/bits_xl.py > /work/generated/bits/$a.$RANDOM.log 2>&1; echo "== bits $a rc=$?"; done
for a in A B; do for f in /work/generated/bits/$a.*.log; do grep -a '^BITS ' $f; done | sort > /tmp/bits$a.txt; done
echo "BITS cases $(sort -u /tmp/bitsA.txt | wc -l) A, $(sort -u /tmp/bitsB.txt | wc -l) B; A runs disagree: $(sort /tmp/bitsA.txt | uniq -c | awk '$1==1' | wc -l); A vs B differing lines: $(sort -u /tmp/bitsA.txt | comm -3 - <(sort -u /tmp/bitsB.txt) | wc -l); B not ok: $(grep -a -c 'False\|FAILED' /tmp/bitsB.txt); A not ok: $(grep -a -c 'False\|FAILED' /tmp/bitsA.txt)"
sort -u /tmp/bitsA.txt | comm -3 - <(sort -u /tmp/bitsB.txt) | head -40
for a in A B; do use $a; export TT_METAL_CACHE=/work/generated/bits/cache_pt$a
timeout -s INT -k 60 3000 python3 -m pytest -q -p no:cacheprovider -rf tests/ttnn/nightly/unit_tests/operations/experimental/test_topk_large_indices.py > /work/generated/bits/pytest_$a.log 2>&1; echo "BITS pytest $a rc=$? $(grep -a -E '[0-9]+ (passed|failed)' /work/generated/bits/pytest_$a.log | tail -n 1)"; grep -a '^FAILED' /work/generated/bits/pytest_$a.log | head -5 | cut -c1-200; done
