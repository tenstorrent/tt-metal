#!/usr/bin/env bash

set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
source_file="$here/raw_lreg_macro_metadata_compile.cpp"

# Optional real target check. Host syntax checks below deliberately mock the
# builtin and cannot establish that the compiler implements the interface.
if [[ ${1:-} == --target-cxx ]]; then
    [[ $# == 2 ]] || { echo "usage: $0 --target-cxx /path/to/riscv-tt-elf-g++" >&2; exit 2; }
    target_cxx=$2
    scratch=$(mktemp -d)
    trap 'rm -f "$scratch/WH.s" "$scratch/BH.s" "$scratch/QSR.s" "$scratch/gap.s"; rmdir "$scratch"' EXIT
    for arch in WH BH QSR; do
        case $arch in
            WH) cpu=tt-wh-tensix ;;
            BH) cpu=tt-bh-tensix ;;
            QSR) cpu=tt-qsr32-tensix ;;
        esac
        "$target_cxx" -std=c++17 -O2 -mcpu="$cpu" -DTEST_TARGET_EFFECT \
            -DTEST_DEFAULT_FALLBACK "-DTEST_$arch" -S "$source_file" -o "$scratch/$arch.s"
        count=$(grep -c '# RAWLREG_EFFECT' "$scratch/$arch.s" || true)
        [[ $count == 13 ]] || { echo "FAIL: $arch emitted $count effects, expected 13" >&2; exit 1; }
        echo "PASS: $cpu compiled with 13 real effect markers (not a hardware test)"
    done
    "$target_cxx" -O2 -mcpu=tt-bh-tensix -DUSE_EFFECT -S \
        "$here/raw_lreg_annotation_gap.cpp" -o "$scratch/gap.s"
    grep -Eq 'SFPLOAD[[:space:]]+L[1-7], 1, 0, 0' "$scratch/gap.s" || {
        echo "FAIL: temporary load must not overwrite raw L0" >&2; exit 1;
    }
    echo "PASS: raw input survives intervening typed load/store (assembly check)"
    exit 0
elif [[ $# != 0 ]]; then
    echo "usage: $0 [--target-cxx /path/to/riscv-tt-elf-g++]" >&2
    exit 2
fi

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
