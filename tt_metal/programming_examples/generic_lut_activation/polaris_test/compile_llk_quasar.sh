#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Standalone SFPI compile-only sanity check for the Quasar generic-LUT
# eltwise tests. Compiles a per-eval-method Quasar tt-llk test source
# (the same sources the pytest harness runs) for all three TRISC threads,
# WITHOUT a device or simulator.
#
# This is a *syntax / SFPU-intrinsic* gate: it proves the selected source
# compiles with the real qsr32 SFPI toolchain, mirroring the include/flag
# set the pytest harness builds with (tests/python_tests/helpers/test_config.py).
# For an end-to-end PCC run under the pinned sim, use run_quasar.sh instead.
#
# Usage:
#   ./compile_llk_quasar.sh [eval_method] [activation]
#
#   eval_method : polynomial | rational | parity | expalu | newton_root
#                 (default: polynomial)
#   activation  : informational tag only, used to name the output dir
#                 (default: gelu)
#
# Examples:
#   ./compile_llk_quasar.sh                      # polynomial, gelu
#   ./compile_llk_quasar.sh rational atanh
#   ./compile_llk_quasar.sh newton_root sqrt
#   ./compile_llk_quasar.sh parity tanh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TT_METAL_ROOT="${TT_METAL_HOME:-/localdev/nkapre/tt-metal}"
LLK_TESTS_DIR="$TT_METAL_ROOT/tt_metal/tt-llk/tests"
LLK_ROOT="$TT_METAL_ROOT/tt_metal/tt-llk"
ARCH_LLK_ROOT="tt_llk_quasar"
BUILD_DIR="/tmp/tt-llk-build-standalone"

# --- Arguments -------------------------------------------------------------
EVAL_METHOD="${1:-polynomial}"
ACTIVATION="${2:-gelu}"

# Map eval_method -> Quasar test source. These are the SAME sources the pytest
# harness compiles & runs; keep this table in sync if new methods are added.
case "$EVAL_METHOD" in
    poly|polynomial)
        KERNEL_BASENAME="generic_lut_activation_quasar_test.cpp" ;;
    rat|rational)
        KERNEL_BASENAME="generic_lut_rational_quasar_test.cpp" ;;
    parity)
        KERNEL_BASENAME="generic_lut_parity_quasar_test.cpp" ;;
    expalu|exponent_alu)
        KERNEL_BASENAME="generic_lut_expalu_quasar_test.cpp" ;;
    newton_root|newton|root)
        KERNEL_BASENAME="generic_lut_newton_root_quasar_test.cpp" ;;
    *)
        echo "ERROR: unknown eval_method '$EVAL_METHOD'"
        echo "       expected one of: polynomial | rational | parity | expalu | newton_root"
        exit 2 ;;
esac

KERNEL_SRC="$LLK_TESTS_DIR/sources/quasar/$KERNEL_BASENAME"
CONFIG_NAME="${EVAL_METHOD}_${ACTIVATION}"
OUTPUT_DIR="$SCRIPT_DIR/quasar_elfs/$CONFIG_NAME"

echo "=============================================="
echo "Quasar LLK compile-only sanity check"
echo "=============================================="
echo "  eval_method : $EVAL_METHOD"
echo "  activation  : $ACTIVATION (tag only)"
echo "  source      : $KERNEL_SRC"
echo ""

if [[ ! -f "$KERNEL_SRC" ]]; then
    echo "ERROR: kernel source not found: $KERNEL_SRC"
    exit 1
fi

# --- SFPI toolchain --------------------------------------------------------
# The pytest harness symlinks tests/sfpi -> runtime/sfpi; honour the same.
if [[ ! -e "$LLK_TESTS_DIR/sfpi" ]]; then
    echo "Creating SFPI symlink..."
    ln -sf "$TT_METAL_ROOT/runtime/sfpi" "$LLK_TESTS_DIR/sfpi"
fi
GXX="$LLK_TESTS_DIR/sfpi/compiler/bin/riscv-tt-elf-g++"
if [[ ! -f "$GXX" ]]; then
    echo "ERROR: SFPI compiler not found at $GXX"
    echo "       Make sure tt-metal is built with the SFPI toolchain."
    exit 1
fi
echo "Compiler: $GXX"
echo ""

# --- Build dir + minimal build.h ------------------------------------------
TEST_BUILD_DIR="$BUILD_DIR/$CONFIG_NAME"
mkdir -p "$TEST_BUILD_DIR/obj"
mkdir -p "$OUTPUT_DIR"

# Minimal, valid build.h. It supplies ONLY the harness-provided symbols the
# sources reference (RUNTIME_PARAMETERS / RuntimeParams, FormatConfig, the
# dest/format constexprs). Every LUT_* / EXPALU_* / NEWTON_ROOT_* define is
# guarded by #ifndef in the sources, so the per-method built-in defaults kick
# in here -- this is purely a compile gate, not a numerically-configured run.
# Layout mirrors a real harness-generated build.h.
cat > "$TEST_BUILD_DIR/build.h" << 'BUILDH_EOF'
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// AUTO-GENERATED minimal config header for the standalone Quasar compile gate.
#pragma once

#include <array>
#include <type_traits>

#include "operand.h"
#include "llk_defs.h"
#include "tensix_types.h"

#define RUNTIME_PARAMETERS [[maybe_unused]] const struct RuntimeParams&

constexpr bool l1_acc_en = 0;
constexpr bool unpack_to_dest = false;

struct FormatConfig
{
    std::uint32_t unpack_A_src = 0;
    std::uint32_t unpack_B_src = 0;
    std::uint32_t unpack_S_src = 0;
    std::uint32_t unpack_A_dst = 0;
    std::uint32_t unpack_B_dst = 0;
    std::uint32_t unpack_S_dst = 0;
    std::uint32_t math = 0;
    std::uint32_t sfpu_math = 0;
    std::uint32_t pack_src = 0;
    std::uint32_t pack_dst = 0;
    std::uint32_t pack_S_src = 0;
    std::uint32_t pack_S_dst = 0;
};

constexpr bool is_fp32_dest_acc_en = true;

constexpr auto SFPU_UNARY_OPERATION = SfpuType::sigmoid;
constexpr bool IMPLIED_MATH_FORMAT = true;
constexpr auto DATA_COPY_TYPE = ckernel::DataCopyType::A2D;
constexpr std::uint32_t UNPACKER_ENGINE_SEL = p_unpacr::UNP_A;
constexpr auto dest_sync = ckernel::DstSync::SyncHalf;

struct RuntimeParams {
    std::uint32_t TILE_SIZE_PACK;
    std::uint32_t TILE_SIZE_UNPACK_A;
    std::uint32_t TILE_SIZE_UNPACK_B;
    FormatConfig formats;
    Operand buffer_A;
    Operand buffer_B;
    Operand buffer_Res;
    std::uint32_t TILE_CNT;
    std::uint32_t num_faces;
    std::uint32_t num_faces_A;
    std::uint32_t num_faces_B;
    std::uint32_t TEST_FACE_R_DIM;
    std::uint32_t TEST_FACE_C_DIM;
    int DST_INDEX;
};
BUILDH_EOF
# The rational source is the one method with NO #ifndef built-in LUT defaults
# (the harness always injects RAT_* via GENERIC_LUT_DATA). Append a minimal
# default rational LUT (1 segment, n1/d1) so the compile gate has the symbols
# it needs. Numerically meaningless -- this is a syntax gate only.
if [[ "$KERNEL_BASENAME" == "generic_lut_rational_quasar_test.cpp" ]]; then
cat >> "$TEST_BUILD_DIR/build.h" << 'RATH_EOF'

// Minimal default rational LUT for the standalone compile gate (n1/d1, 1 seg).
constexpr std::uint32_t RAT_NUM_SEGMENTS = 1;
constexpr std::uint32_t RAT_NUM_DEGREE   = 1;
constexpr std::uint32_t RAT_DEN_DEGREE   = 1;
constexpr std::array<float, RAT_NUM_SEGMENTS + 1> RAT_BOUNDARIES = {-1.0f, 1.0f};
constexpr std::array<float, RAT_NUM_SEGMENTS*(RAT_NUM_DEGREE + 1)> RAT_NUM_COEFFS = {0.0f, 1.0f};
constexpr std::array<float, RAT_NUM_SEGMENTS*(RAT_DEN_DEGREE + 1)> RAT_DEN_COEFFS = {1.0f, 0.0f};
RATH_EOF
fi

echo "Generated: $TEST_BUILD_DIR/build.h"
echo ""

# --- Includes + flags (mirror tests/python_tests/helpers/test_config.py) ---
cd "$LLK_TESTS_DIR"

INCLUDES=(
    "-I$TEST_BUILD_DIR"
    "-Isfpi/include"
    "-I../$ARCH_LLK_ROOT/llk_lib"
    "-I../$ARCH_LLK_ROOT/common/inc"
    "-I../$ARCH_LLK_ROOT/common/inc/sfpu"
    "-I../common"
    "-I../../hw/inc"
    "-Ifirmware/riscv/common"
    "-Ihelpers/include"
    "-I../../hostdevcommon/api"
    "-I../../hw/inc/internal/tt-2xx/quasar"
    "-I../../hw/ckernels/quasar/metal/llk_api"
)

DEFINES=(
    "-DTENSIX_FIRMWARE"
    "-DENV_LLK_INFRA"
    "-DKERNEL_BUILD"
    "-DENABLE_LLK_ASSERT"
    "-DARCH_QUASAR"
    # Harness sets this whenever formats are runtime (the default); the sources
    # gate the `params.formats` local on it (#if defined(RUNTIME_FORMATS)).
    "-DRUNTIME_FORMATS"
)

# qsr32: non-compute threads use -mcpu=tt-qsr32, compute uses -mcpu=tt-qsr32-tensix.
CFLAGS_COMMON=(
    "-std=c++17"
    "-ftt-nttp" "-ftt-constinit" "-ftt-consteval" "-ftt-no-dyninit"
    "-g" "-O3"
    "-ffast-math"
    "-fno-exceptions" "-fno-rtti" "-fno-use-cxa-atexit"
    "-Wall"
)

echo "Compiling (compile-only) for all three TRISC threads..."
echo "----------------------------------------------"
COMPILE_OK=true
for TRISC in unpack math pack; do
    TRISC_UPPER=$(echo "$TRISC" | tr '[:lower:]' '[:upper:]')
    OBJ="$TEST_BUILD_DIR/obj/kernel_$TRISC.o"
    if [[ "$TRISC" == "math" ]]; then MCPU="-mcpu=tt-qsr32-tensix"; else MCPU="-mcpu=tt-qsr32"; fi

    echo "  [$TRISC] $MCPU"
    if $GXX $MCPU "${CFLAGS_COMMON[@]}" "${DEFINES[@]}" "${INCLUDES[@]}" \
        -DLLK_TRISC_$TRISC_UPPER \
        -c -o "$OBJ" "$KERNEL_SRC" 2>&1; then
        cp "$OBJ" "$OUTPUT_DIR/"
        echo "    -> $OBJ"
    else
        echo "    FAILED: $TRISC"
        COMPILE_OK=false
    fi
done

echo ""
echo "=============================================="
if [[ "$COMPILE_OK" == "true" ]]; then
    echo "COMPILE OK: $EVAL_METHOD ($KERNEL_BASENAME)"
    echo "Objects in: $OUTPUT_DIR"
    exit 0
else
    echo "COMPILE FAILED: $EVAL_METHOD ($KERNEL_BASENAME) -- see errors above"
    exit 1
fi
