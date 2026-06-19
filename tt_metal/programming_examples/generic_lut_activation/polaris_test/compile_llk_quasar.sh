#!/bin/bash
# Build piecewise LUT activation kernel for Quasar using LLK infrastructure
#
# This script compiles the piecewise_lut_activation kernel for quasar
# and extracts the ELFs for use with polaris/neosom simulator.
#
# Usage:
#   ./build_llk_quasar.sh [poly_degree] [num_segments] [activation]
#
# Examples:
#   ./build_llk_quasar.sh 1 4 gelu    # Linear, 4 segments (p1_s4)
#   ./build_llk_quasar.sh 2 8 sigmoid # Quadratic, 8 segments (p2_s8)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TT_METAL_ROOT="/localdev/nkapre/tt-metal"
LLK_TESTS_DIR="$TT_METAL_ROOT/tt_metal/third_party/tt_llk/tests"
LLK_ROOT="$TT_METAL_ROOT/tt_metal/third_party/tt_llk"
BUILD_DIR="/tmp/tt-llk-build"
OUTPUT_DIR="$SCRIPT_DIR/quasar_elfs"

# Default parameters
POLY_DEGREE="${1:-1}"
NUM_SEGMENTS="${2:-4}"
ACTIVATION="${3:-gelu}"

# Calculate LUT size
LUT_SIZE=$(( (NUM_SEGMENTS + 1) + NUM_SEGMENTS * (POLY_DEGREE + 1) ))

# Configuration name
CONFIG_NAME="${ACTIVATION}_p${POLY_DEGREE}_s${NUM_SEGMENTS}"

echo "=============================================="
echo "Building LLK Quasar Piecewise LUT Activation"
echo "=============================================="
echo ""
echo "Configuration: $CONFIG_NAME"
echo "  Polynomial degree: $POLY_DEGREE"
echo "  Segments: $NUM_SEGMENTS"
echo "  LUT size: $LUT_SIZE"
echo "  Activation: $ACTIVATION"
echo ""

# Ensure SFPI symlink exists
if [[ ! -L "$LLK_TESTS_DIR/sfpi" ]]; then
    echo "Creating SFPI symlink..."
    ln -sf "$TT_METAL_ROOT/runtime/sfpi" "$LLK_TESTS_DIR/sfpi"
fi

# Check if compiler exists
GXX="$LLK_TESTS_DIR/sfpi/compiler/bin/riscv-tt-elf-g++"
if [[ ! -f "$GXX" ]]; then
    echo "ERROR: SFPI compiler not found at $GXX"
    echo "Make sure tt-metal is built with SFPI toolchain"
    exit 1
fi

echo "Compiler: $GXX"
echo ""

# Create output directory and build directory
mkdir -p "$OUTPUT_DIR/$CONFIG_NAME"
TEST_BUILD_DIR="$BUILD_DIR/piecewise_lut_$CONFIG_NAME"
mkdir -p "$TEST_BUILD_DIR/obj"
mkdir -p "$TEST_BUILD_DIR/elf"

# Generate build.h file (required by params.h)
echo "Generating build.h..."
cat > "$TEST_BUILD_DIR/build.h" << 'BUILDH_EOF'
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
// AUTO-GENERATED CONFIGURATION HEADER FOR STANDALONE QUASAR BUILD

#pragma once

#include <array>
#include <type_traits>

#include "operand.h"
#include "llk_defs.h"
#include "llk_sfpu_types.h"
#include "tensix_types.h"

// Basic configuration
constexpr std::uint32_t TILE_SIZE_CNT = 0x1000;
constexpr std::uint32_t TILE_SIZE_PACK = 128;
constexpr std::uint32_t TILE_SIZE_UNPACK_A = 128;
constexpr std::uint32_t TILE_SIZE_UNPACK_B = 128;

// Data format configuration - FP32 for activation test
constexpr bool is_fp32_dest_acc_en = true;
constexpr bool unpack_to_dest = false;

// Single iteration format configuration
constexpr auto UNPACK_A_IN = static_cast<std::underlying_type_t<DataFormat>>(DataFormat::Float32);
constexpr auto UNPACK_A_OUT = static_cast<std::underlying_type_t<DataFormat>>(DataFormat::Float32);
constexpr auto MATH_FORMAT = static_cast<std::underlying_type_t<DataFormat>>(DataFormat::Float32);
constexpr auto PACK_IN = static_cast<std::underlying_type_t<DataFormat>>(DataFormat::Float32);
constexpr auto PACK_OUT = static_cast<std::underlying_type_t<DataFormat>>(DataFormat::Float32);

// Template parameters for datacopy (quasar uses ckernel::DataCopyType, not EltwiseUnaryDatacopyType)
constexpr ckernel::DataCopyType DATA_COPY_TYPE = ckernel::DataCopyType::A2D;
constexpr uint32_t UNPACKER_ENGINE_SEL = ckernel::p_unpacr::UNP_A;
constexpr ckernel::DstSync dest_sync = ckernel::DstSync::SyncFull;

// Buffer addresses (typical L1 layout for quasar)
constexpr std::array<uint32_t, 1> buffer_A = {0x10000};
constexpr std::array<uint32_t, 1> buffer_B = {0x20000};
constexpr std::array<uint32_t, 1> buffer_Res = {0x30000};

// Struct that has a runtime parameter layout
struct RuntimeParams {
    uint32_t TILE_CNT;
    uint32_t DST_INDEX;
    uint32_t TEST_FACE_R_DIM;
    uint32_t TEST_FACE_C_DIM;
    uint32_t num_faces;
};
BUILDH_EOF

echo "  Created: $TEST_BUILD_DIR/build.h"
echo ""

# Build the kernel using LLK Makefile
cd "$LLK_TESTS_DIR"

echo "Compiling for quasar..."
echo "----------------------------------------------"

# Set environment for the build
export CHIP_ARCH=quasar

# Include paths for quasar (order matters!)
INCLUDES=(
    "-I$TEST_BUILD_DIR"
    "-I$LLK_TESTS_DIR/helpers/include"
    "-I$LLK_TESTS_DIR/hw_specific/quasar/inc"
    "-I$LLK_ROOT/tt_llk_quasar/common/inc"
    "-I$LLK_ROOT/tt_llk_quasar/common/inc/sfpu"
    "-I$LLK_ROOT/tt_llk_quasar/llk_lib"
    "-I$LLK_TESTS_DIR/sfpi/include"
    "-I$LLK_TESTS_DIR/firmware/riscv/common"
)

# Defines
DEFINES=(
    "-DARCH_QUASAR"
    "-DTENSIX_FIRMWARE"
    "-DENV_LLK_INFRA"
    "-DENABLE_LLK_ASSERT"
    "-DLLK_BOOT_MODE_TRISC"
    "-DPOLY_DEGREE=$POLY_DEGREE"
    "-DNUM_SEGMENTS=$NUM_SEGMENTS"
    "-DLUT_SIZE=$LUT_SIZE"
)

# Compiler flags (quasar uses blackhole SFPI for now)
CFLAGS=(
    "-std=c++17"
    "-mcpu=tt-bh-tensix"
    "-g"
    "-O3"
    "-ffast-math"
    "-fno-use-cxa-atexit"
    "-Wall"
    "-fno-exceptions"
    "-fno-rtti"
    "-nostdlib"
    "-fno-builtin"
)

KERNEL_SRC="$LLK_TESTS_DIR/sources/quasar/piecewise_lut_activation_quasar_test.cpp"

echo "Source: $KERNEL_SRC"
echo ""

COMPILE_SUCCESS=true

# Compile for each TRISC core
for TRISC in unpack math pack; do
    TRISC_UPPER=$(echo "$TRISC" | tr '[:lower:]' '[:upper:]')
    OUTPUT_OBJ="$TEST_BUILD_DIR/obj/kernel_$TRISC.o"

    echo "Compiling $TRISC core..."

    if $GXX ${CFLAGS[*]} ${DEFINES[*]} ${INCLUDES[*]} \
        -DLLK_TRISC_$TRISC_UPPER \
        -c -o "$OUTPUT_OBJ" "$KERNEL_SRC" 2>&1; then
        echo "  Created: $OUTPUT_OBJ"
    else
        echo "  FAILED: $TRISC core compilation"
        COMPILE_SUCCESS=false
    fi
done

echo ""
echo "----------------------------------------------"
echo ""

# Check what was generated
if ls "$TEST_BUILD_DIR/obj"/*.o 1>/dev/null 2>&1; then
    echo "Generated object files:"
    ls -la "$TEST_BUILD_DIR/obj"/*.o
    echo ""

    # Copy object files to output directory
    cp "$TEST_BUILD_DIR/obj"/*.o "$OUTPUT_DIR/$CONFIG_NAME/"
    echo "Copied to: $OUTPUT_DIR/$CONFIG_NAME/"
else
    echo "No object files generated."
fi

echo ""
echo "=============================================="
echo "Build Summary"
echo "=============================================="
echo ""
echo "Kernel source: $KERNEL_SRC"
echo "Build directory: $TEST_BUILD_DIR"
echo "Output directory: $OUTPUT_DIR/$CONFIG_NAME"
echo ""

if [[ "$COMPILE_SUCCESS" == "true" ]]; then
    echo "Compilation SUCCESSFUL!"
    echo ""

    # Now link to create ELFs
    echo "Linking ELFs..."
    echo "----------------------------------------------"

    # Build shared objects if they don't exist
    SHARED_DIR="$BUILD_DIR/shared"
    mkdir -p "$SHARED_DIR/obj"
    mkdir -p "$SHARED_DIR/elf"

    LINKER_SCRIPTS="$LLK_TESTS_DIR/helpers/ld"
    HELPERS_SRC="$LLK_TESTS_DIR/helpers"
    MEMORY_LD="$LINKER_SCRIPTS/memory.quasar.ld"
    SECTIONS_LD="$LINKER_SCRIPTS/sections.ld"

    LINK_FLAGS=(
        "-fexceptions"
        "-Wl,-z,max-page-size=16"
        "-Wl,-z,common-page-size=16"
        "-nostartfiles"
    )

    # Build tmu-crt0.o if not present
    if [[ ! -f "$SHARED_DIR/obj/tmu-crt0.o" ]]; then
        echo "Building tmu-crt0.o..."
        $GXX -mcpu=tt-bh ${CFLAGS[*]} ${DEFINES[*]} ${INCLUDES[*]} \
            -c -o "$SHARED_DIR/obj/tmu-crt0.o" "$HELPERS_SRC/tmu-crt0.S"
    fi

    # Build main_*.o for each TRISC
    for TRISC in unpack math pack; do
        TRISC_UPPER=$(echo "$TRISC" | tr '[:lower:]' '[:upper:]')
        MAIN_OBJ="$SHARED_DIR/obj/main_$TRISC.o"

        if [[ ! -f "$MAIN_OBJ" ]]; then
            echo "Building main_$TRISC.o..."
            $GXX ${CFLAGS[*]} ${DEFINES[*]} ${INCLUDES[*]} \
                -DLLK_TRISC_$TRISC_UPPER \
                -c -o "$MAIN_OBJ" "$HELPERS_SRC/src/trisc.cpp"
        fi
    done

    LINK_SUCCESS=true

    # Link each TRISC core ELF
    for TRISC in unpack math pack; do
        KERNEL_OBJ="$TEST_BUILD_DIR/obj/kernel_$TRISC.o"
        MAIN_OBJ="$SHARED_DIR/obj/main_$TRISC.o"
        CRT_OBJ="$SHARED_DIR/obj/tmu-crt0.o"
        OUTPUT_ELF="$TEST_BUILD_DIR/elf/$TRISC.elf"
        TRISC_LD="$LINKER_SCRIPTS/$TRISC.ld"

        echo "Linking $TRISC.elf..."

        if $GXX -mcpu=tt-bh-tensix -g -O3 -std=c++17 -ffast-math \
            ${LINK_FLAGS[*]} \
            "$CRT_OBJ" "$MAIN_OBJ" "$KERNEL_OBJ" \
            -T"$MEMORY_LD" -T"$TRISC_LD" -T"$SECTIONS_LD" \
            -o "$OUTPUT_ELF" 2>&1; then
            echo "  Created: $OUTPUT_ELF"
            cp "$OUTPUT_ELF" "$OUTPUT_DIR/$CONFIG_NAME/"
        else
            echo "  FAILED: $TRISC.elf linking"
            LINK_SUCCESS=false
        fi
    done

    echo ""
    echo "----------------------------------------------"
    echo ""

    if [[ "$LINK_SUCCESS" == "true" ]]; then
        echo "BUILD FULLY SUCCESSFUL!"
        echo ""
        echo "Generated ELF files:"
        ls -la "$OUTPUT_DIR/$CONFIG_NAME/"*.elf 2>/dev/null || echo "  (no ELFs in output dir)"
        ls -la "$TEST_BUILD_DIR/elf/"*.elf 2>/dev/null || echo "  (no ELFs in build dir)"
    else
        echo "LINKING FAILED - see errors above"
        echo ""
        echo "Object files are available at: $OUTPUT_DIR/$CONFIG_NAME/"
    fi
else
    echo "BUILD FAILED - see errors above"
    echo ""
    echo "Common issues:"
    echo "  - Missing headers: check include paths"
    echo "  - SFPI compiler not found: rebuild tt-metal"
fi
echo ""
