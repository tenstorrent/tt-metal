#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "usage: $0 TARGET_CXX SFPI_INCLUDE [OUTPUT_DIR] (optional ACCESS_ONLY_CXX environment variable)" >&2
    exit 2
fi

here=$(cd "$(dirname "$0")" && pwd)
source_file="$here/raw_lreg_metadata_abi.cpp"
target_cxx=$1
sfpi_include=$2
if [[ $# -eq 3 ]]; then
    mkdir -p "$3"
    output=$(cd "$3" && pwd)
    if [[ -n $(find "$output" -mindepth 1 -print -quit) ]]; then
        echo "validation output must be empty: $output" >&2
        exit 2
    fi
else
    output=$(mktemp -d)
    trap 'rm -f "$output"/*.s "$output"/*.log; rmdir "$output"' EXIT
fi

for arch in wh bh qsr32; do
    "$target_cxx" -std=c++17 -O2 -mcpu="tt-$arch-tensix" -I"$sfpi_include" \
        -DTT_LLK_REQUIRE_RAW_LREG_METADATA -S "$source_file" \
        -o "$output/$arch.s"
    if ! grep -q '# RAWLREG_EFFECT 1, 2' "$output/$arch.s"; then
        echo "raw-LREG effect metadata was not emitted for $arch" >&2
        exit 1
    fi
    if ! grep -Eq 'SFPLOAD[[:space:]]+L[1-7], 1, 0, 0' "$output/$arch.s"; then
        echo "effect compiler did not reserve raw L0 across the typed temporary on $arch" >&2
        exit 1
    fi
    echo "PASS: $arch emitted raw-LREG metadata"
done

# The older access builtin's first mask RELEASES a live raw value. It cannot
# implement the effect builtin's first READ mask. Source compatibility is a
# no-op, while required-effect mode must reject that compiler.
if [[ -n ${ACCESS_ONLY_CXX:-} ]]; then
    if "$ACCESS_ONLY_CXX" -std=c++17 -O2 -mcpu=tt-bh-tensix \
        -DTEST_REQUIRE_ACCESS_ONLY_COMPILER -DTT_LLK_REQUIRE_RAW_LREG_METADATA \
        -S "$source_file" -o "$output/access-only-required.s" \
        > "$output/access-only-required.log" 2>&1; then
        echo "access-only compiler unexpectedly accepted required effect metadata" >&2
        exit 1
    fi
    if ! grep -q 'raw-LREG effect metadata required' "$output/access-only-required.log"; then
        cat "$output/access-only-required.log" >&2
        echo "access-only rejection did not come from the required-effect guard" >&2
        exit 1
    fi
    "$ACCESS_ONLY_CXX" -std=c++17 -O2 -mcpu=tt-bh-tensix \
        -DTEST_REQUIRE_ACCESS_ONLY_COMPILER -S "$source_file" \
        -o "$output/access-only-compatible.s"
    if grep -q '# RAWLREG' "$output/access-only-compatible.s"; then
        echo "access-only compatibility path emitted misleading metadata" >&2
        exit 1
    fi
    if ! grep -Eq 'SFPLOAD[[:space:]]+L0, 1, 0, 0' "$output/access-only-compatible.s"; then
        echo "access-only compatibility fixture changed allocation; inspect assembly" >&2
        exit 1
    fi
    echo "PASS: access-only compiler keeps source compatibility but rejects required effects"
fi

# A compiler without either builtin must fail the required-metadata mode,
# rather than silently compiling the helper as a no-op.
if command -v c++ >/dev/null 2>&1; then
    # An uninstalled target backend is selected through these environment
    # variables. Keep it out of the independent host-compiler refusal check.
    if env -u COMPILER_PATH -u GCC_EXEC_PREFIX -u CPLUS_INCLUDE_PATH \
        -u C_INCLUDE_PATH -u CPATH -u LIBRARY_PATH \
        c++ -std=c++17 -DTT_LLK_REQUIRE_RAW_LREG_METADATA \
        -fsyntax-only "$source_file" > "$output/host-required.log" 2>&1; then
        echo "required raw-LREG metadata unexpectedly accepted by host C++" >&2
        exit 1
    fi
    # The fixture also calls target-only builtins, so any failure is not enough.
    if ! grep -q 'raw-LREG effect metadata required' "$output/host-required.log"; then
        cat "$output/host-required.log" >&2
        echo "host rejection did not come from the required-effect guard" >&2
        exit 1
    fi
    echo "PASS: unsupported host compiler rejects required metadata"
fi
