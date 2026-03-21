# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# CMake target for running clang-tidy on kernel (device) code.
#
# Kernel code is normally compiled at runtime via JIT with a RISC-V cross-compiler.
# This target re-compiles the same source files with the *host* clang compiler so
# that clang-tidy can analyse them.  The target is EXCLUDE_FROM_ALL — it is only
# built on demand (e.g. via `cmake --build <dir> --target kernel_clang_tidy`).
#
# Key differences from actual JIT compilation:
#   • Host clang instead of riscv-tt-elf-g++
#   • No RISC-V flags (-mcpu=tt-wh etc.)
#   • -Wno-unknown-attributes to tolerate RISC-V __attribute__ extensions
#   • A prelude header provides stub values for JIT-generated content
#   • Fake KERNEL_COMPILE_TIME_ARGS so get_compile_time_arg_val() resolves
#
# Because the goal is static analysis (not code generation), perfect compilation
# is not required — we just need clang's frontend to parse the code.

# ── Gather all kernel source files ──────────────────────────────────────────

set(ALL_KERNEL_FILES "")

file(
    GLOB_RECURSE TT_METAL_KERNELS
    "${CMAKE_SOURCE_DIR}/tt_metal/kernels/*.cpp"
    "${CMAKE_SOURCE_DIR}/tt_metal/kernels/*.cc"
)
file(
    GLOB_RECURSE TT_METAL_DISPATCH_KERNELS
    "${CMAKE_SOURCE_DIR}/tt_metal/impl/dispatch/kernels/*.cpp"
    "${CMAKE_SOURCE_DIR}/tt_metal/impl/dispatch/kernels/*.cc"
)
file(
    GLOB_RECURSE TT_METAL_FABRIC_KERNELS
    "${CMAKE_SOURCE_DIR}/tt_metal/fabric/impl/kernels/*.cpp"
    "${CMAKE_SOURCE_DIR}/tt_metal/fabric/impl/kernels/*.cc"
)

# Only include files from kernel/ or kernels_ng/ directories under operations
file(
    GLOB_RECURSE ALL_TTNN_FILES
    "${CMAKE_SOURCE_DIR}/ttnn/cpp/ttnn/operations/*.cpp"
    "${CMAKE_SOURCE_DIR}/ttnn/cpp/ttnn/operations/*.cc"
)
set(TTNN_OPERATION_KERNELS "")
foreach(file ${ALL_TTNN_FILES})
    if(file MATCHES ".*/kernels/|.*/kernels_ng/")
        list(APPEND TTNN_OPERATION_KERNELS ${file})
    endif()
endforeach()

list(
    APPEND
    ALL_KERNEL_FILES
    ${TT_METAL_KERNELS}
    ${TT_METAL_DISPATCH_KERNELS}
    ${TT_METAL_FABRIC_KERNELS}
    ${TTNN_OPERATION_KERNELS}
)

list(LENGTH ALL_KERNEL_FILES ALL_KERNEL_FILES_LEN)
message(STATUS "[kernel_clang_tidy] Found ${ALL_KERNEL_FILES_LEN} kernel files")

# ── Architecture selection (default: wormhole_b0) ──────────────────────────

if(NOT DEFINED ARCH_NAME)
    set(ARCH_NAME "wormhole_b0")
endif()

string(TOLOWER "${ARCH_NAME}" ARCH_NAME_LOWER)

if(ARCH_NAME_LOWER MATCHES "wormhole")
    set(KCT_ARCH_PREFIX "wormhole")
    set(KCT_ARCH_ALIAS "wormhole_b0")
    set(KCT_ARCH_DEFINE "ARCH_WORMHOLE")
    set(KCT_NUM_DRAM_BANKS 12)
    set(KCT_NUM_L1_BANKS 64)
elseif(ARCH_NAME_LOWER MATCHES "blackhole")
    set(KCT_ARCH_PREFIX "blackhole")
    set(KCT_ARCH_ALIAS "blackhole")
    set(KCT_ARCH_DEFINE "ARCH_BLACKHOLE")
    set(KCT_NUM_DRAM_BANKS 8)
    set(KCT_NUM_L1_BANKS 140)
else()
    message(WARNING "[kernel_clang_tidy] Unsupported ARCH_NAME=${ARCH_NAME}, defaulting to wormhole_b0")
    set(KCT_ARCH_PREFIX "wormhole")
    set(KCT_ARCH_ALIAS "wormhole_b0")
    set(KCT_ARCH_DEFINE "ARCH_WORMHOLE")
    set(KCT_NUM_DRAM_BANKS 12)
    set(KCT_NUM_L1_BANKS 64)
endif()

# ── Strip inherited host compile options ────────────────────────────────────
# The parent project adds host-specific flags via add_compile_options()
# (e.g. -march=x86-64-v3, -Wall, -Werror) that conflict with our RISC-V
# cross-compilation target.  Clear them; we set our own complete flag set
# via target_compile_options() below.
set_directory_properties(
    PROPERTIES
        COMPILE_OPTIONS
            ""
)

# ── Create the OBJECT library ───────────────────────────────────────────────

add_library(kernel_clang_tidy OBJECT EXCLUDE_FROM_ALL ${ALL_KERNEL_FILES})

# ── Include paths ───────────────────────────────────────────────────────────
# Mirrors the paths from:
#   • JitBuildEnv constructor (tt_metal/jit_build/build.cpp)
#   • HalJitBuildQuery{Wormhole,BlackHole}::includes() in tt_metal/llrt/hal/tt-1xx/

target_include_directories(
    kernel_clang_tidy
    PRIVATE
        # -- JIT-generated file stubs (must come first to shadow real generated files) --
        "${CMAKE_CURRENT_SOURCE_DIR}/jit_stubs"
        # -- Common includes (from JitBuildEnv) --
        "${CMAKE_SOURCE_DIR}"
        "${CMAKE_SOURCE_DIR}/ttnn"
        "${CMAKE_SOURCE_DIR}/ttnn/cpp"
        "${CMAKE_SOURCE_DIR}/tt_metal"
        "${CMAKE_SOURCE_DIR}/tt_metal/include"
        "${CMAKE_SOURCE_DIR}/tt_metal/hw/inc"
        "${CMAKE_SOURCE_DIR}/tt_metal/hostdevcommon/api"
        "${CMAKE_SOURCE_DIR}/tt_metal/hw/inc/debug"
        "${CMAKE_SOURCE_DIR}/tt_metal/api/"
        "${CMAKE_SOURCE_DIR}/tt_metal/api/tt-metalium/"
        # -- Architecture-specific includes (from HAL) --
        "${CMAKE_SOURCE_DIR}/tt_metal/hw/ckernels/${KCT_ARCH_ALIAS}/metal/common"
        "${CMAKE_SOURCE_DIR}/tt_metal/hw/ckernels/${KCT_ARCH_ALIAS}/metal/llk_io"
        "${CMAKE_SOURCE_DIR}/tt_metal/hw/inc/internal/tt-1xx"
        "${CMAKE_SOURCE_DIR}/tt_metal/hw/inc/internal/tt-1xx/${KCT_ARCH_PREFIX}"
        "${CMAKE_SOURCE_DIR}/tt_metal/hw/inc/internal/tt-1xx/${KCT_ARCH_PREFIX}/${KCT_ARCH_ALIAS}_defines"
        "${CMAKE_SOURCE_DIR}/tt_metal/hw/inc/internal/tt-1xx/${KCT_ARCH_PREFIX}/noc"
        "${CMAKE_SOURCE_DIR}/tt_metal/third_party/tt_llk/tt_llk_${KCT_ARCH_ALIAS}/common/inc"
        "${CMAKE_SOURCE_DIR}/tt_metal/third_party/tt_llk/tt_llk_${KCT_ARCH_ALIAS}/llk_lib"
        # -- Compute kernel includes (LLK API + SFPU) --
        "${CMAKE_SOURCE_DIR}/tt_metal/hw/ckernels/${KCT_ARCH_ALIAS}/metal/llk_api"
        "${CMAKE_SOURCE_DIR}/tt_metal/hw/ckernels/${KCT_ARCH_ALIAS}/metal/llk_api/llk_sfpu"
        # -- Firmware source (for headers referenced via firmware paths) --
        "${CMAKE_SOURCE_DIR}/tt_metal/hw/firmware/src/tt-1xx"
        # -- SFPU includes --
        "${CMAKE_SOURCE_DIR}/tt_metal/third_party/tt_llk/tt_llk_${KCT_ARCH_ALIAS}/common/inc/sfpu"
        # -- SFPI headers --
        # sfpi.h requires TT RISC-V compiler builtins that clang doesn't have.
        # Compute kernels using SFPI will hit a compile error in sfpi.h, but
        # clang-tidy still analyses the rest of their code (with -ferror-limit=0).
        "${CMAKE_SOURCE_DIR}/runtime/sfpi/include"
)

# ── Compiler flags ──────────────────────────────────────────────────────────
# We intentionally omit RISC-V flags (-mcpu=tt-wh, -flto, etc.) and instead
# use host-compatible flags.  -Wno-unknown-attributes silences errors from
# RISC-V-specific __attribute__ extensions (rvtt_l1_ptr, rvtt_reg_ptr).

target_compile_options(
    kernel_clang_tidy
    PRIVATE
        -std=c++17
        --target=riscv32-unknown-elf # Parse as 32-bit RISC-V (matching real target)
        # RISC-V standard library headers (from the SFPI toolchain)
        -nostdinc++
        -isystem${CMAKE_SOURCE_DIR}/runtime/sfpi/compiler/riscv-tt-elf/include/c++/15.1.0
        -isystem${CMAKE_SOURCE_DIR}/runtime/sfpi/compiler/riscv-tt-elf/include/c++/15.1.0/riscv-tt-elf
        -isystem${CMAKE_SOURCE_DIR}/runtime/sfpi/compiler/riscv-tt-elf/include
        -isystem${CMAKE_SOURCE_DIR}/runtime/sfpi/compiler/lib/gcc/riscv-tt-elf/15.1.0/include
        -O0 # No optimisation — just parse
        -fno-exceptions
        -ferror-limit=0 # Don't stop after 20 errors per file
        -Wno-unknown-attributes # Tolerate __attribute__((rvtt_l1_ptr)) etc.
        -Wno-c++11-narrowing # Suppress narrowing (expected: 32-bit target vs host types)
        -Wno-unknown-pragmas
        -Wno-deprecated-declarations
        -Wno-unused-variable
        -Wno-unused-function
        -Wno-error
        # Force-include the prelude that provides JIT-generated stubs
        -include
        "${CMAKE_CURRENT_SOURCE_DIR}/kernel_clang_tidy_prelude.h"
)

# ── Compile definitions ────────────────────────────────────────────────────
# Mirrors defines from JitBuildEnv + HAL.  We set sensible defaults for
# device-specific values (PCIE_NOC_X, NUM_DRAM_BANKS, etc.) and provide
# dummy KERNEL_COMPILE_TIME_ARGS so that get_compile_time_arg_val() resolves.

target_compile_definitions(
    kernel_clang_tidy
    PRIVATE
        # -- Core build mode --
        TENSIX_FIRMWARE
        LOCAL_MEM_EN=0
        KERNEL_BUILD
        # -- Processor identification --
        # We do NOT define COMPILE_FOR_TRISC so that firmware_common.h includes
        # the real dataflow_api.h.  Compute kernel stubs are provided by the
        # prelude and jit_stubs/.
        UCK_CHLKC_MATH
        NAMESPACE=chlkc_math
        PROCESSOR_INDEX=1
        TRISC_MATH
        # -- NOC configuration --
        NOC_INDEX=0
        NOC_MODE=0
        PCIE_NOC_X=0
        PCIE_NOC_Y=3
        ROUTING_FW_ENABLED
        # -- Device dimensions (reasonable defaults) --
        NUM_DRAM_BANKS=${KCT_NUM_DRAM_BANKS}
        NUM_L1_BANKS=${KCT_NUM_L1_BANKS}
        # -- Bank mapping (used by InterleavedAddrGen in dataflow_api_addrgen.h) --
        LOG_BASE_2_OF_NUM_DRAM_BANKS=4
        LOG_BASE_2_OF_NUM_L1_BANKS=6
        # -- Dispatch constants (normally set by JIT) --
        DISPATCH_MESSAGE_ADDR=0
        # -- Data format stub --
        DATA_FORMATS_DEFINED=1
        # -- Architecture --
        ${KCT_ARCH_DEFINE}
        # -- Compile-time arguments (dummy values so kernels parse) --
        # Provide enough entries (128) to satisfy all get_compile_time_arg_val() indices
        "KERNEL_COMPILE_TIME_ARGS=\
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,\
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,\
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,\
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,\
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,\
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,\
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,\
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1"
    # Named compile-time args: get_named_compile_time_arg_val() is provided
    # as a macro in kernel_clang_tidy_prelude.h (force-included) — see that
    # file for details.  We intentionally do NOT define
    # KERNEL_COMPILE_TIME_ARG_MAP here.
)
