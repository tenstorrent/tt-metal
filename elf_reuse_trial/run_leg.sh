#!/usr/bin/env bash
# Trial sessions for one merge-gate leg. Part "a" runs at commit A, part "b" at commit B.
#   a: R1 cold, cache off | R2 cold, empty cache (fills it, both keys) | snapshot | R3 A warm (A->A floor)
#   b: R6 B cold, cache off (reference) | R4 B warm direct | R5 B warm pre | compare each with R6
# env: ARCH GROUP NGROUPS OUT RUNNER_TEMP GITHUB_WORKSPACE
set -euo pipefail
PART=$1
TRIAL=$(cd "$(dirname "$0")" && pwd)
C="$RUNNER_TEMP/elfcache"
mkdir -p "$OUT"
python3 "$TRIAL/inject.py" "$GITHUB_WORKSPACE" "$TRIAL/elf_cache.py"
( cd "$GITHUB_WORKSPACE/tt_metal/tt-llk/tests" && ./setup_testing_env.sh > "$OUT/setup-$PART.log" 2>&1 )
OBJCOPY="$GITHUB_WORKSPACE/tt_metal/tt-llk/tests/sfpi/compiler/bin/riscv-tt-elf-objcopy"
"$GITHUB_WORKSPACE/tt_metal/tt-llk/tests/sfpi/compiler/bin/riscv-tt-elf-g++" --version | head -1 > "$OUT/gxx-$PART.txt"

snap() { rm -rf "$2"; cp -al "$1" "$2"; }

if [[ "$PART" == a ]]; then
  "$TRIAL/session.sh" R1_A_cold_off "" direct
  rm -rf "$C/A"; mkdir -p "$C/A"
  "$TRIAL/session.sh" R2_A_cold_fill "$C/A" both
  # Cache size and a save/restore round trip (tar + zstd, as an upload/download would need).
  python3 - "$C/A" > "$OUT/cache_size.json" <<'EOF'
import os, sys, json
n = b = 0
for root, _, files in os.walk(sys.argv[1]):
    for f in files:
        n += 1; b += os.path.getsize(os.path.join(root, f))
print(json.dumps({"files": n, "bytes": b}))
EOF
  t0=$(date +%s.%N)
  tar -C "$C/A" -cf - . | zstd -q -T0 -3 > "$RUNNER_TEMP/cacheA.tar.zst"
  t1=$(date +%s.%N)
  mkdir -p "$RUNNER_TEMP/restore" && zstd -q -dc -T0 "$RUNNER_TEMP/cacheA.tar.zst" | tar -C "$RUNNER_TEMP/restore" -xf -
  t2=$(date +%s.%N)
  python3 -c 'import json,sys,os; t0,t1,t2=map(float,sys.argv[1:4]); print(json.dumps({"save_s":round(t1-t0,1),"restore_s":round(t2-t1,1),"archive_bytes":os.path.getsize(sys.argv[4])}))' "$t0" "$t1" "$t2" "$RUNNER_TEMP/cacheA.tar.zst" > "$OUT/cache_roundtrip.json"
  rm -rf "$RUNNER_TEMP/restore" "$RUNNER_TEMP/cacheA.tar.zst"
  snap "$C/A" "$C/R3"
  "$TRIAL/session.sh" R3_A_warm_direct "$C/R3" direct
  rm -rf "$C/R3"
  python3 "$TRIAL/compare.py" "$OUT/R1_A_cold_off" "$OUT/R2_A_cold_fill" > "$OUT/cmp_R1_R2.json"
  python3 "$TRIAL/compare.py" "$OUT/R1_A_cold_off" "$OUT/R3_A_warm_direct" > "$OUT/cmp_R1_R3.json"
else
  KEEP_TREE=1 "$TRIAL/session.sh" R6_B_cold_off "" direct
  REF="$RUNNER_TEMP/tree_R6_B_cold_off"
  B="$RUNNER_TEMP/tt-llk-build"
  snap "$C/A" "$C/R4"
  "$TRIAL/session.sh" R4_B_warm_direct "$C/R4" direct
  python3 "$TRIAL/compare.py" "$OUT/R6_B_cold_off" "$OUT/R4_B_warm_direct" "$REF" "$B" "$OBJCOPY" > "$OUT/cmp_R6_R4.json"
  rm -rf "$C/R4"
  snap "$C/A" "$C/R5"
  "$TRIAL/session.sh" R5_B_warm_pre "$C/R5" pre
  python3 "$TRIAL/compare.py" "$OUT/R6_B_cold_off" "$OUT/R5_B_warm_pre" "$REF" "$B" "$OBJCOPY" > "$OUT/cmp_R6_R5.json"
  rm -rf "$C/R5"
  cat "$OUT"/cmp_*.json
fi
