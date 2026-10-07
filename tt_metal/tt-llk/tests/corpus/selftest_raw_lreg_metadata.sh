#!/usr/bin/env bash

set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
source_file="$here/raw_lreg_macro_metadata_compile.cpp"

# Optional real target check. Host syntax checks below deliberately mock the
# builtin and cannot establish that the compiler implements the interface.
if [[ ${1:-} == --target-cxx ]]; then
    [[ $# == 2 || ( $# == 4 && $3 == --sfpi-include ) ]] || {
        echo "usage: $0 --target-cxx /path/to/riscv-tt-elf-g++ [--sfpi-include /path/to/sfpi/include]" >&2; exit 2;
    }
    target_cxx=$2
    scratch=$(mktemp -d)
    trap 'rm -f "$scratch/WH.s" "$scratch/BH.s" "$scratch/QSR.s" "$scratch/gap.s" "$scratch/mul-int.s" "$scratch/baseline.s"; rmdir "$scratch"' EXIT
    for arch in WH BH QSR; do
        case $arch in
            WH) cpu=tt-wh-tensix ;;
            BH) cpu=tt-bh-tensix ;;
            QSR) cpu=tt-qsr32-tensix ;;
        esac
        for opt in -O0 -O2; do
            "$target_cxx" -std=c++17 "$opt" -mcpu="$cpu" -DTEST_TARGET_EFFECT \
                -DTEST_DEFAULT_FALLBACK "-DTEST_$arch" -S "$source_file" -o "$scratch/$arch.s"
            count=$(grep -c '# RAWLREG_EFFECT' "$scratch/$arch.s" || true)
            [[ $count == 13 ]] || { echo "FAIL: $arch $opt emitted $count effects, expected 13" >&2; exit 1; }
            if grep -Eq '[[:space:]]call[[:space:]]' "$scratch/$arch.s"; then
                echo "FAIL: metadata fixture emitted a runtime helper call at $opt" >&2; exit 1
            fi
            "$target_cxx" -std=c++17 "$opt" -mcpu="$cpu" -DTEST_DEFAULT_FALLBACK \
                '-DTT_LLK_SFPRAWLREG_EFFECT(r,w)=((void)0)' "-DTEST_$arch" \
                -S "$source_file" -o "$scratch/baseline.s"
            stores=$(grep -Ec '^[[:space:]]+(sw|sd)[[:space:]]' "$scratch/$arch.s" || true)
            baseline_stores=$(grep -Ec '^[[:space:]]+(sw|sd)[[:space:]]' "$scratch/baseline.s" || true)
            [[ $stores == "$baseline_stores" ]] || {
                echo "FAIL: metadata added scalar stores at $arch $opt" >&2; exit 1;
            }
            echo "PASS: $cpu $opt compiled with 13 real effect markers (not a hardware test)"
            "$target_cxx" -std=c++17 "$opt" -mcpu="$cpu" -DTEST_TARGET_EFFECT \
                -DTEST_DEFAULT_FALLBACK -DTEST_LUT_MODES "-DTEST_$arch" \
                -S "$source_file" -o "$scratch/$arch.s"
            count=$(grep -c '# RAWLREG_EFFECT' "$scratch/$arch.s" || true)
            [[ $count == 25 ]] || { echo "FAIL: LUT modifier fixture emitted $count effects, expected 25" >&2; exit 1; }
            echo "PASS: $cpu $opt LUT modifier combinations"
        done
    done
    "$target_cxx" -O2 -mcpu=tt-bh-tensix -DUSE_EFFECT -S \
        "$here/raw_lreg_annotation_gap.cpp" -o "$scratch/gap.s"
    grep -Eq 'SFPLOAD[[:space:]]+L[1-7], 1, 0, 0' "$scratch/gap.s" || {
        echo "FAIL: temporary load must not overwrite raw L0" >&2; exit 1;
    }
    echo "PASS: raw input survives intervening typed load/store (assembly check)"
    "$target_cxx" -O2 -mcpu=tt-bh-tensix -DUSE_MACRO -S \
        "$here/raw_lreg_annotation_gap.cpp" -o "$scratch/gap.s"
    grep -Eq 'SFPLOAD[[:space:]]+L[1-7], 1, 0, 0' "$scratch/gap.s" &&
        grep -q '# RAWLREG_EFFECT 1, 1' "$scratch/gap.s" || {
        echo "FAIL: raw write must preserve its old destination for inactive lanes" >&2; exit 1;
    }
    echo "PASS: raw destination preserved across typed load/store (assembly check)"
    # Run all sides of the comparison. Report what existing builtins actually
    # do; do not require them to fail or label an allocation scan silicon proof.
    for issue in TTI TT; do
        issue_flags=()
        [[ $issue == TT ]] && issue_flags+=(-DUSE_MMIO)
        for scheduling in default scheduled; do
            schedule_flags=()
            [[ $scheduling == scheduled ]] && schedule_flags+=(-fschedule-insns -fschedule-insns2)
            for scheme in 0 1 2; do
                "$target_cxx" -O2 -mcpu=tt-bh-tensix "${issue_flags[@]}" "${schedule_flags[@]}" \
                    "-DSCHEME=$scheme" -S "$here/raw_lreg_full_annotation.cpp" -o "$scratch/gap.s"
                if grep -Eq 'SFPLOAD[[:space:]]+L0, 1, 0, 0' "$scratch/gap.s"; then
                    echo "OBSERVED: $issue $scheduling scheme=$scheme temporary overwrites L0"
                    [[ $scheme != 2 ]] || { echo "FAIL: effect reservation lost raw L0" >&2; exit 1; }
                elif grep -Eq 'SFPLOAD[[:space:]]+L[1-7], 1, 0, 0' "$scratch/gap.s"; then
                    echo "OBSERVED: $issue $scheduling scheme=$scheme temporary avoids L0"
                else
                    echo "FAIL: unrecognized allocation; inspect comparator assembly" >&2; exit 1
                fi
            done
        done
    done
    if [[ $# == 4 ]]; then
        root=$(cd "$here/../../../.." && pwd)
        llk="$root/tt_metal/tt-llk/tt_llk_blackhole"
        "$target_cxx" -std=c++17 -O2 -mcpu=tt-bh-tensix \
            -DTENSIX_FIRMWARE -DCOMPILE_FOR_TRISC -DARCH_BLACKHOLE \
            -I"$root/tt_metal/hw/inc/internal/tt-1xx/blackhole" \
            -I"$root/tt_metal/hw/inc/internal" -I"$root/tt_metal/hw/inc" \
            -I"$llk/common/inc" -I"$llk/llk_lib" -I"$4" \
            -S "$here/raw_lreg_mul_int_compile.cpp" -o "$scratch/mul-int.s"
        grep -q '# RAWLREG_EFFECT' "$scratch/mul-int.s" || {
            echo "FAIL: production mul-int compiled without effects" >&2; exit 1;
        }
        echo "PASS: production Blackhole mul-int compiled with effects (not a hardware test)"
    fi
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
            flags=(-std=c++17 -Wall -Wextra -Werror -fsyntax-only -DTEST_LUT_MODES "-DTEST_$arch")
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
