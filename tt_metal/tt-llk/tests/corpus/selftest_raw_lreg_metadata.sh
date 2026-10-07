#!/usr/bin/env bash

set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
source_file="$here/raw_lreg_macro_metadata_compile.cpp"

tested=0
for compiler in "${CXX:-c++}" clang++; do
    if ! command -v "$compiler" >/dev/null 2>&1; then
        continue
    fi
    for arch in WH BH QSR; do
        case $arch in
            WH) target_extension=wh ;;
            BH) target_extension=bh ;;
            QSR) target_extension=qsr ;;
        esac
        for compatibility in marker fallback legacy-target; do
            flags=(-std=c++17 -Wall -Wextra -Werror -fsyntax-only "-DTEST_$arch")
            if [[ $compatibility == fallback ]]; then
                flags+=(-DTEST_FALLBACK)
            elif [[ $compatibility == legacy-target ]]; then
                flags+=(-DTEST_DEFAULT_FALLBACK "-D__riscv_xtttensix${target_extension}")
            fi
            "$compiler" "${flags[@]}" "$source_file"
            tested=$((tested + 1))
        done
    done
done

if ((tested == 0)); then
    echo "ERROR: no C++ compiler found" >&2
    exit 1
fi

echo "PASS: $tested raw-LREG metadata compile checks"
