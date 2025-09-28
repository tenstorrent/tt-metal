set(ALL_KERNEL_FILES "")
# Direct recursive searches for tt_metal directories
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

# First find all cpp/cc files under the operations directory
file(
    GLOB_RECURSE ALL_TTNN_FILES
    "${CMAKE_SOURCE_DIR}/ttnn/cpp/ttnn/operations/*.cpp"
    "${CMAKE_SOURCE_DIR}/ttnn/cpp/ttnn/operations/*.cc"
)

# Filter to only include files from kernel directories
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
message(STATUS "Found ${ALL_KERNEL_FILES_LEN} kernel files")

if(NOT DEFINED ARCH_NAME)
    set(ARCH_NAME "wormhole_b0")
endif()
set(TOLOWER
    "${ARCH_NAME}"
    ARCH_NAME_LOWER
)

if(ARCH_NAME_LOWER MATCHES "wormhole")
    set(ARCH_NAME_PREFIX "wormhole")
elseif(ARCH_NAME_LOWER MATCHES "blackhole")
    set(ARCH_NAME_PREFIX "blackhole")
endif()

add_library(jit_kernels_index OBJECT EXCLUDE_FROM_ALL ${ALL_KERNEL_FILES})
target_include_directories(
    jit_kernels_index
    PRIVATE
        "${CMAKE_SOURCE_DIR}/"
        "${CMAKE_SOURCE_DIR}/ttnn"
        "${CMAKE_SOURCE_DIR}/ttnn/cpp"
        "${CMAKE_SOURCE_DIR}/tt_metal"
        "${CMAKE_SOURCE_DIR}/tt_metal/include"
        "${CMAKE_SOURCE_DIR}/tt_metal/hw/inc"
        "${CMAKE_SOURCE_DIR}/tt_metal/hostdevcommon/api"
        "${CMAKE_SOURCE_DIR}/tt_metal/hw/inc/debug"
        "${CMAKE_SOURCE_DIR}/tt_metal/api/"
        "${CMAKE_SOURCE_DIR}/tt_metal/api/tt-metalium/"
        "${CMAKE_SOURCE_DIR}/tt_metal/hw/firmware/src"
        "${CMAKE_SOURCE_DIR}/tt_metal/hw/inc/${ARCH_NAME_PREFIX}"
        "${CMAKE_SOURCE_DIR}/tt_metal/hw/inc/${ARCH_NAME_PREFIX}/noc"
        "${CMAKE_SOURCE_DIR}/tt_metal/hw/inc/${ARCH_NAME_PREFIX}/${ARCH_NAME_LOWER}_defines"
        "${CMAKE_SOURCE_DIR}/tt_metal/hw/ckernels/${ARCH_NAME_LOWER}/metal/common"
        "${CMAKE_SOURCE_DIR}/tt_metal/third_party/tt_llk/tt_llk_${ARCH_NAME_LOWER}/common/inc"
        "${CMAKE_SOURCE_DIR}/tt_metal/third_party/tt_llk/tt_llk_${ARCH_NAME_LOWER}/llk_lib"
        "${CMAKE_SOURCE_DIR}/tt_metal/hw/ckernels/${ARCH_NAME_LOWER}/metal/common"
        "${CMAKE_SOURCE_DIR}/tt_metal/hw/ckernels/${ARCH_NAME_LOWER}/metal/llk_api"
        "${CMAKE_SOURCE_DIR}/tt_metal/hw/ckernels/${ARCH_NAME_LOWER}/metal/llk_api/llk_sfpu"
        "${CMAKE_SOURCE_DIR}/tt_metal/hw/ckernels/${ARCH_NAME_LOWER}/metal/llk_io"
        "${CMAKE_SOURCE_DIR}/tt_metal/third_party/tt_llk/tt_llk_${ARCH_NAME_LOWER}/common/inc/sfpu"
)

# Add all the specific compiler flags needed for device code compilation
target_compile_options(
    jit_kernels_index
    PRIVATE
        -O2
        -mcpu=tt-wh
        -std=c++20
        -flto=auto
        -ffast-math
        -g
        -fno-use-cxa-atexit
        -fno-exceptions
        -Wall
        # -Werror
        -Wno-unknown-pragmas
        -Wno-deprecated-declarations
        -Wno-error=multistatement-macros
        -Wno-error=parentheses
        -Wno-error=unused-but-set-variable
        -Wno-unused-variable
        -Wno-unused-function
        -fno-tree-loop-distribute-patterns
        -include
        "${CMAKE_SOURCE_DIR}/tt_metal/jit_build/fake_kernels_target/fake_jit_prelude.h"
)

# Add all the specific compile definitions needed for device code
target_compile_definitions(
    jit_kernels_index
    PRIVATE
        PCIE_NOC_X=0
        PCIE_NOC_Y=3
        # By default pretend that kernel is compiled for MATH core
        TRISC_MATH=1
        NAMESPACE=chlkc_math
        # TRISC_PACK=1
        # NAMESPACE=chlkc_pack
        # TRISC_UNPACK=1
        # NAMESPACE=chlkc_unpack
        TENSIX_FIRMWARE
        LOCAL_MEM_EN=0
        ROUTING_FW_ENABLED
        KERNEL_BUILD
        NOC_INDEX=0
        NOC_MODE=0
        DATA_FORMATS_DEFINED=1
        "KERNEL_COMPILE_TIME_ARGS=1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1"
)

if(ARCH_NAME_LOWER MATCHES "wormhole")
    target_compile_definitions(
        jit_kernels_index
        PRIVATE
            ARCH_WORMHOLE
            NUM_DRAM_BANKS=12
            NUM_L1_BANKS=64
    )
elseif(ARCH_NAME_LOWER MATCHES "blackhole")
    target_compile_definitions(
        jit_kernels_index
        PRIVATE
            ARCH_BLACKHOLE
            NUM_DRAM_BANKS=8
            NUM_L1_BANKS=140
    )
endif()
