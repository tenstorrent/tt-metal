#!/usr/bin/env bash
# Scratch A/B for #58786 on the custom_mm branch, not for merge: the fused MoE's plain SRAM path (enable_sram_bspm off, the decode
# default) with the PR's gated caller (banked from 3 SRAM experts per section) against main's one-bank caller, plus an A/A
# copy of the gated caller, alternating
# in one job on one build at one path; each arm has its own kernel cache. Device kernel time per device from tracy.
set -uo pipefail
cd /work
T=models/demos/deepseek_v3_b1/tests/unit_tests/test_moe_mlp.py
SRAM=models/demos/deepseek_v3_b1/unified_kernels/matmul_expert_compressed_sram.hpp
ROUNDS=${ROUNDS:-3}
python3 -c "import tracy" 2>/dev/null || export PYTHONPATH=/work:/work/tools
echo "== $(date -u +%FT%TZ) $(git log -1 --format='%h %s' 2>/dev/null)"
mapfile -t IDS < <(python3 -m pytest --collect-only -q "$T::test_moe_fused_with_reduce" 2>/dev/null |
  grep -F "rigged_groups1" | grep -E "t4_all_picked|t8_all_picked")
printf 'node %s\n' "${IDS[@]}"
[[ ${#IDS[@]} -gt 0 ]] || { echo "no node ids collected"; python3 -m pytest --collect-only -q "$T::test_moe_fused_with_reduce" 2>&1 | tail -20; exit 1; }
fail=0
for r in $(seq "$ROUNDS"); do
  for v in banked onebank banked2; do
    cp ci_ab/sram_$v.hpp $SRAM
    out=/work/generated/ab/${v}_$r; rm -rf "$out"; mkdir -p "$out"
    t0=$(date +%s)
    TT_METAL_CACHE=/tmp/ttcache_$v python3 -m tracy -p -r --no-web-server -o "$out" \
      -m pytest -q -p no:cacheprovider -o timeout_method=thread "${IDS[@]}" > "$out.log" 2>&1
    rc=$?
    echo "== round $r $v rc=$rc $(( $(date +%s) - t0 )) s: $(grep -E '[0-9]+ (passed|failed)' "$out.log" | tail -1)"
    [[ $rc -eq 0 ]] || { fail=1; tail -40 "$out.log"; }
    python3 ci_ab/devtime.py "$out" "$v" "$r"
  done
done
cp ci_ab/sram_banked.hpp $SRAM
python3 ci_ab/devtime.py --summary /work/generated/ab
exit $fail
