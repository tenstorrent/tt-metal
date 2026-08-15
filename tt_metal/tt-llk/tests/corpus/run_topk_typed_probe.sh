#!/usr/bin/env bash
# Compile the TopK typed-SFPI semantic-gap probe for WH and BH.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLK_TESTS="$(cd "$HERE/.." && pwd)"
SFPI_ROOT="${SFPI_ROOT:-$LLK_TESTS/sfpi}"
CXX="${CXX:-$SFPI_ROOT/compiler/bin/riscv-tt-elf-g++}"
OUT="${1:-/localdev/nkapre/topk-typed-probe}"
SRC="$HERE/topk_typed_index_tracking_probe.cpp"

mkdir -p "$OUT"
for arch in bh wh; do
    "$CXX" -mcpu="tt-$arch-tensix" -O3 -std=c++17 -fno-exceptions -fno-rtti \
        -I "$SFPI_ROOT/include" -S "$SRC" -o "$OUT/$arch.s"
    test "$(rg -c '^[[:space:]]*SFPSWAP' "$OUT/$arch.s")" = 1
    test "$(rg -c '^[[:space:]]*SFPTRANSP' "$OUT/$arch.s")" = 1
    rg -q '# READ L4' "$OUT/$arch.s"
    rg -q '# READ L7' "$OUT/$arch.s"
done

{
    printf 'source\t%s\n' "$SRC"
    printf 'sfpi_root\t%s\n' "$(readlink -f "$SFPI_ROOT")"
    printf 'compiler_sha256\t'; sha256sum "$CXX" | awk '{print $1}'
    printf 'tt_metal_head\t'; git -C "$(cd "$LLK_TESTS/../../.." && pwd)" rev-parse HEAD
} > "$OUT/provenance.tsv"
sha256sum "$OUT/bh.s" "$OUT/wh.s" > "$OUT/assembly.sha256"
printf 'probe\t%s\n' "$OUT"
