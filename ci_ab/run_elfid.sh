#!/usr/bin/env bash
# Scratch job (#58820, the MATH_FIDELITY define for the unpack thread), not for merge. Runs the 86-op sample of the #58817
# check (mock_ops.py) on the card with a fresh JIT cache and prints a hash of every JIT ELF's loaded sections, keyed by its
# cache path; two jobs (branch with and without the genfiles hunk) are compared offline.
set +e +o pipefail
cd /work; export TT_METAL_HOME=/work HWLOCK_HELD=1 TT_METAL_CACHE=/work/generated/elfid_cache; rm -rf $TT_METAL_CACHE; mkdir -p $TT_METAL_CACHE
echo "== $(date -u +%FT%TZ) $(git log -1 --format='%h %s' | cut -c1-70)"
timeout -s INT -k 60 3000 python3 /work/ci_ab/elfid/mock_ops.py > /tmp/ops.log 2>&1; echo "== ops rc=$? $(grep -o 'done ok=.*' /tmp/ops.log)"
grep -c "^FAIL" /tmp/ops.log
python3 /work/ci_ab/sweep/elfdump.py $TT_METAL_CACHE > /tmp/elf.txt; echo "== elfs $(wc -l < /tmp/elf.txt)"
echo "== payload begin"; gzip -c /tmp/elf.txt | base64 -w 200; echo "== payload end"
echo "== DONE $(date -u +%T)"
