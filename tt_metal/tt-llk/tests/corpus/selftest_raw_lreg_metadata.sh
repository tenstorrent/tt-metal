#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
source_file="$here/raw_lreg_macro_metadata_compile.cpp"
# Effect markers raw_lreg_macro_metadata_compile.cpp issues: six TT_ words in
# tt_mmio_constant_lreg, one each in tt_mmio_runtime_lreg,
# tt_mmio_runtime_address and tt_mmio_load_into, six TTI_ words in
# tti_asm_constant_lreg. TEST_LUT_MODES adds lut_modifier_issue's eight.
fixture_effects=17
lut_mode_effects=8

# Optional real target check. Host syntax checks below deliberately mock the
# marker and cannot establish that the compiler implements the interface.
# The target check builds against the SFPI headers that match the compiler,
# as firmware does, so markers go through lltt::lreg_effect where SFPI has it.
if [[ ${1:-} == --target-cxx ]]; then
    [[ $# == 4 && $3 == --sfpi-include ]] || {
        echo "usage: $0 --target-cxx /path/to/riscv-tt-elf-g++ --sfpi-include /path/to/sfpi/include" >&2; exit 2;
    }
    target_cxx=$2
    sfpi_include=$4
    # Complete the producer and check its exit before searching.  grep -q on
    # a live compiler pipe can close it early and turn a present capability
    # into a false negative under pipefail (SIGPIPE).
    capability_macros=$("$target_cxx" -mcpu=tt-bh-tensix -I"$sfpi_include" -dM -E -x c++ - <<< '#include "lltt.h"')
    if grep -q LLTT_HAS_LREG_EFFECT <<< "$capability_macros"; then
        echo "INFO: effect markers use lltt::lreg_effect"
    else
        echo "INFO: SFPI has no lltt::lreg_effect; effect markers use the compiler builtin"
    fi
    scratch=$(mktemp -d)
    trap 'rm -f "$scratch/WH.s" "$scratch/BH.s" "$scratch/QSR.s" "$scratch/gap.s" "$scratch/mul-int.s" "$scratch/baseline.s" "$scratch/pressure.s" "$scratch/pressure.log" "$scratch/lut-refuse.log"; rmdir "$scratch"' EXIT
    for arch in WH BH QSR; do
        case $arch in
            WH) cpu=tt-wh-tensix ;;
            BH) cpu=tt-bh-tensix ;;
            QSR) cpu=tt-qsr32-tensix ;;
        esac
        for opt in -O0 -O2; do
            "$target_cxx" -I"$sfpi_include" -std=c++17 "$opt" -mcpu="$cpu" -DTEST_TARGET_EFFECT \
                -DTEST_DEFAULT_FALLBACK "-DTEST_$arch" -S "$source_file" -o "$scratch/$arch.s"
            count=$(grep -c '# RAWLREG_EFFECT' "$scratch/$arch.s" || true)
            [[ $count == "$fixture_effects" ]] || {
                echo "FAIL: $arch $opt emitted $count effects, expected $fixture_effects" >&2; exit 1;
            }
            # A lane-merge compiler owns inactive destination preservation.
            capability_macros=$("$target_cxx" -mcpu="$cpu" -dM -E -x c++ /dev/null)
            if grep -q __riscv_xtt_sfprawlreg_merge <<< "$capability_macros"; then
                runtime_read=0
                inlined_read=0
                macro_read=128
            else
                runtime_read=136
                inlined_read=68
                macro_read=131
            fi
            # LREG3 load with a runtime address: L3 and its paired L7.
            grep -q "# RAWLREG_EFFECT $runtime_read, 136" "$scratch/$arch.s" || {
                echo "FAIL: $arch $opt runtime address hid the constant LREG fields" >&2; exit 1;
            }
            # LREG2 load issued from an inline helper: L2 and its paired L6
            # once inlined, otherwise the all-LREG fallback.
            inlined=0
            grep -q "# RAWLREG_EFFECT $inlined_read, 68" "$scratch/$arch.s" && inlined=1
            [[ $inlined == $([[ $opt == -O2 ]] && echo 1 || echo 0) ]] || {
                echo "FAIL: $arch $opt inlined LREG operand decoded unexpectedly" >&2; exit 1;
            }
            if grep -Eq '[[:space:]]call[[:space:]]' "$scratch/$arch.s"; then
                echo "FAIL: metadata fixture emitted a runtime helper call at $opt" >&2; exit 1
            fi
            "$target_cxx" -I"$sfpi_include" -std=c++17 "$opt" -mcpu="$cpu" -DTEST_DEFAULT_FALLBACK \
                '-DTT_LLK_SFPRAWLREG_EFFECT(r,w)=((void)0)' "-DTEST_$arch" \
                -S "$source_file" -o "$scratch/baseline.s"
            stores=$(grep -Ec '^[[:space:]]+(sw|sd)[[:space:]]' "$scratch/$arch.s" || true)
            baseline_stores=$(grep -Ec '^[[:space:]]+(sw|sd)[[:space:]]' "$scratch/baseline.s" || true)
            [[ $stores == "$baseline_stores" ]] || {
                echo "FAIL: metadata added scalar stores at $arch $opt" >&2; exit 1;
            }
            echo "PASS: $cpu $opt compiled with $fixture_effects real effect markers (not a hardware test)"
            grep -q '# RAWLREG_EFFECT 0, 0' "$scratch/$arch.s" &&
                grep -q "# RAWLREG_EFFECT $macro_read, 3" "$scratch/$arch.s" || {
                echo "FAIL: $cpu $opt lost explicit template or known-macro effects" >&2; exit 1;
            }
            "$target_cxx" -I"$sfpi_include" -std=c++17 "$opt" -mcpu="$cpu" -DTEST_TARGET_EFFECT \
                -DTEST_DEFAULT_FALLBACK -DTEST_LUT_MODES "-DTEST_$arch" \
                -S "$source_file" -o "$scratch/$arch.s"
            count=$(grep -c '# RAWLREG_EFFECT' "$scratch/$arch.s" || true)
            expected=$((fixture_effects + lut_mode_effects))
            [[ $count == "$expected" ]] || {
                echo "FAIL: LUT modifier fixture emitted $count effects, expected $expected" >&2; exit 1;
            }
            echo "PASS: $cpu $opt LUT modifier combinations"
        done
        if "$target_cxx" -I"$sfpi_include" -std=c++17 -O2 -mcpu="$cpu" -DTEST_TARGET_EFFECT \
            -DTEST_DEFAULT_FALLBACK -DTEST_RESERVED_LUT "-DTEST_$arch" \
            -S "$source_file" -o "$scratch/$arch.s" > "$scratch/lut-refuse.log" 2>&1; then
            echo "FAIL: $cpu accepted reserved SFPLUTFP32 modifier 8" >&2; exit 1
        fi
        grep -q 'unsupported raw SFPU effect' "$scratch/lut-refuse.log" || {
            cat "$scratch/lut-refuse.log" >&2
            echo "FAIL: $cpu reserved LUT rejection did not come from metadata gate" >&2; exit 1
        }
        echo "PASS: $cpu rejects reserved SFPLUTFP32 modifier 8"
    done
    "$target_cxx" -I"$sfpi_include" -O2 -mcpu=tt-bh-tensix -DUSE_EFFECT -S \
        "$here/raw_lreg_annotation_gap.cpp" -o "$scratch/gap.s"
    grep -Eq 'SFPLOAD[[:space:]]+L[1-7], 1, 0, 0' "$scratch/gap.s" || {
        echo "FAIL: temporary load must not overwrite raw L0" >&2; exit 1;
    }
    echo "PASS: raw input survives intervening typed load/store (assembly check)"
    "$target_cxx" -I"$sfpi_include" -O2 -mcpu=tt-bh-tensix -DUSE_MACRO -S \
        "$here/raw_lreg_annotation_gap.cpp" -o "$scratch/gap.s"
    grep -Eq 'SFPLOAD[[:space:]]+L[1-7], 1, 0, 0' "$scratch/gap.s" || {
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
            for scheme in 0 1 2 3; do
                pass_flags=()
                # The working alternative must not borrow protection from
                # the pass whose necessity this comparison is evaluating.
                [[ $scheme == 3 ]] && pass_flags+=(-fdisable-rtl-rvtt_lreg_livein)
                "$target_cxx" -I"$sfpi_include" -O2 -mcpu=tt-bh-tensix "${issue_flags[@]}" "${schedule_flags[@]}" \
                    "${pass_flags[@]}" \
                    "-DSCHEME=$scheme" -S "$here/raw_lreg_full_annotation.cpp" -o "$scratch/gap.s"
                if grep -Eq 'SFPLOAD[[:space:]]+L0, 1, 0, 0' "$scratch/gap.s"; then
                    echo "OBSERVED: $issue $scheduling scheme=$scheme temporary overwrites L0"
                    [[ $scheme != 2 && $scheme != 3 ]] || { echo "FAIL: inspect preservation of raw L0 in scheme=$scheme" >&2; exit 1; }
                elif grep -Eq 'SFPLOAD[[:space:]]+L[1-7], 1, 0, 0' "$scratch/gap.s"; then
                    echo "OBSERVED: $issue $scheduling scheme=$scheme temporary avoids L0"
                else
                    echo "FAIL: unrecognized allocation; inspect comparator assembly" >&2; exit 1
                fi
            done
            "$target_cxx" -I"$sfpi_include" -std=c++17 -O2 -mcpu=tt-bh-tensix \
                "${issue_flags[@]}" "${schedule_flags[@]}" \
                -fdisable-rtl-rvtt_lreg_livein -DUSE_SFPI_API -DSCHEME=3 \
                -S "$here/raw_lreg_full_annotation.cpp" -o "$scratch/gap.s"
            grep -Eq 'SFPLOAD[[:space:]]+L[1-7], 1, 0, 0' "$scratch/gap.s" || {
                echo "FAIL: inspect public LRegFile lifetime comparator assembly" >&2; exit 1;
            }
            echo "PASS: $issue $scheduling public LRegFile threaded lifetime, pass disabled (assembly check)"
            # Register-pressure gate: compile-only, never a correctness pass.
            for live in 2 4 7 8; do
                for scheme in 2 3; do
                    pass_flags=()
                    [[ $scheme == 3 ]] && pass_flags+=(-fdisable-rtl-rvtt_lreg_livein)
                    if "$target_cxx" -I"$sfpi_include" -std=c++17 -O2 -mcpu=tt-bh-tensix \
                        "${issue_flags[@]}" "${schedule_flags[@]}" "${pass_flags[@]}" \
                        "-DSCHEME=$scheme" "-DLIVE_REGS=$live" \
                        -S "$here/raw_lreg_threading_pressure.cpp" -o "$scratch/pressure.s" \
                        > "$scratch/pressure.log" 2>&1; then
                        if [[ $live == 8 ]]; then
                            echo "FAIL: $issue $scheduling scheme=$scheme accepted eight preserved raw values plus a temporary" >&2
                            exit 1
                        fi
                        echo "OBSERVED: $issue $scheduling scheme=$scheme $live raw values + temporary compiled (not a device result)"
                    elif [[ $live == 8 ]] && grep -Eq 'too few lregs|lreg-pressure-exceeded' "$scratch/pressure.log"; then
                        echo "OBSERVED: $issue $scheduling scheme=$scheme register-capacity rejection (not a correctness pass)"
                    else
                        cat "$scratch/pressure.log" >&2
                        exit 1
                    fi
                done
            done
        done
    done
    root=$(cd "$here/../../../.." && pwd)
    llk="$root/tt_metal/tt-llk/tt_llk_blackhole"
    "$target_cxx" -I"$sfpi_include" -std=c++17 -O2 -mcpu=tt-bh-tensix \
        -DTENSIX_FIRMWARE -DCOMPILE_FOR_TRISC -DARCH_BLACKHOLE \
        -I"$root/tt_metal/hw/inc/internal/tt-1xx/blackhole" \
        -I"$root/tt_metal/hw/inc/internal" -I"$root/tt_metal/hw/inc" \
        -I"$llk/common/inc" -I"$llk/llk_lib" \
        -S "$here/raw_lreg_mul_int_compile.cpp" -o "$scratch/mul-int.s"
    grep -q '# RAWLREG_EFFECT' "$scratch/mul-int.s" || {
        echo "FAIL: production mul-int compiled without effects" >&2; exit 1;
    }
    echo "PASS: production Blackhole mul-int compiled with effects (not a hardware test)"
    exit 0
elif [[ $# != 0 ]]; then
    echo "usage: $0 [--target-cxx /path/to/riscv-tt-elf-g++ --sfpi-include /path/to/sfpi/include]" >&2
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
        for compatibility in marker lane-merge fallback legacy-target; do
            flags=(-std=c++17 -Wall -Wextra -Werror -fsyntax-only -DTEST_LUT_MODES -DTEST_EXPRESSION_POSITION "-DTEST_$arch")
            if [[ $compatibility == fallback ]]; then
                flags+=(-DTEST_FALLBACK)
            elif [[ $compatibility == lane-merge ]]; then
                flags+=(-D__riscv_xtt_sfprawlreg_merge)
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
