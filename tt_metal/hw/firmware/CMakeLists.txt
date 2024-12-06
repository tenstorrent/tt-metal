set(ARCHS
    grayskull
    wormhole
    blackhole
)
set(PROCS
    brisc
    ncrisc
    trisc0
    trisc1
    trisc2
    erisc
    idle_erisc
    slave_idle_erisc
)

include(FetchContent)
FetchContent_Declare(
    sfpi
    URL
        https://github.com/tenstorrent/sfpi/releases/download/v6.0.0/sfpi-release.tgz
    URL_HASH MD5=d837d26a2312d27815179995fdea83bd
    SOURCE_DIR
    ${PROJECT_SOURCE_DIR}/runtime/sfpi
)
FetchContent_MakeAvailable(sfpi)

# Function appends b0 if its wormhole
function(get_alias INPUT_STRING OUTPUT_VAR)
    if("${INPUT_STRING}" STREQUAL "wormhole")
        set(${OUTPUT_VAR} "wormhole_b0" PARENT_SCOPE)
    else()
        set(${OUTPUT_VAR} "${INPUT_STRING}" PARENT_SCOPE)
    endif()
endfunction()

function(get_source INPUT_STRING OUTPUT_VAR)
    string(FIND "${INPUT_STRING}" "trisc" substring_found)
    if(substring_found EQUAL "-1")
        set(${OUTPUT_VAR} "${INPUT_STRING}" PARENT_SCOPE)
    else()
        set(${OUTPUT_VAR} "trisc" PARENT_SCOPE)
    endif()
endfunction()

# Define the compiler command
set(GPP_CMD ${PROJECT_SOURCE_DIR}/runtime/sfpi/compiler/bin/riscv32-unknown-elf-g++)

set(GPP_DEFINES_common
    -DTENSIX_FIRMWARE
    -DPCIE_NOC1_X=12
    -DPCIE_NOC1_Y=8
    -DPCIE_NOC_X=0
    -DPCIE_NOC_Y=4
    -DLOCAL_MEM_EN=0
    -DFW_BUILD
    -DDISPATCH_MESSAGE_ADDR=99152
)
set(GPP_DEFINES_grayskull
    -DARCH_GRAYSKULL
    -DLOG_BASE_2_OF_NUM_DRAM_BANKS=3
    -DLOG_BASE_2_OF_NUM_L1_BANKS=7
    -DNUM_DRAM_BANKS=8
    -DNUM_L1_BANKS=128
)
set(GPP_DEFINES_wormhole
    -DARCH_WORMHOLE
    -DLOG_BASE_2_OF_NUM_DRAM_BANKS=3
    -DLOG_BASE_2_OF_NUM_L1_BANKS=7
    -DNUM_DRAM_BANKS=8
    -DNUM_L1_BANKS=128
)
set(GPP_DEFINES_blackhole
    -DARCH_BLACKHOLE
    -DLOG_BASE_2_OF_NUM_DRAM_BANKS=3
    -DLOG_BASE_2_OF_NUM_L1_BANKS=7
    -DNUM_DRAM_BANKS=8
    -DNUM_L1_BANKS=128
)
set(GPP_DEFINES_brisc -DCOMPILE_FOR_BRISC)
set(GPP_DEFINES_ncrisc -DCOMPILE_FOR_BRISC)
set(GPP_DEFINES_trisc0
    -DUCK_CHLKC_UNPACK
    -DNAMESPACE=chlkc_unpack
    -DCOMPILE_FOR_TRISC=0
)
set(GPP_DEFINES_trisc1
    -DUCK_CHLKC_MATH
    -DNAMESPACE=chlkc_math
    -DCOMPILE_FOR_TRISC=1
)
set(GPP_DEFINES_trisc2
    -DUCK_CHLKC_PACK
    -DNAMESPACE=chlkc_pack
    -DCOMPILE_FOR_TRISC=2
)
set(GPP_DEFINES_erisc
    -DCOMPILE_FOR_ERISC
    -DERISC
    -DRISC_B0_HW
)
set(GPP_DEFINES_idle_erisc
    -DCOMPILE_FOR_IDLE_ERISC=0
    -DERISC
    -DRISC_B0_HW
)
set(GPP_DEFINES_slave_idle_erisc
    -DCOMPILE_FOR_IDLE_ERISC=1
    -DERISC
    -DRISC_B0_HW
)

# Define flags for each architecture
set(GPP_FLAGS_grayskull -mcpu=tt-gs)
set(GPP_FLAGS_wormhole -mcpu=tt-wh)
set(GPP_FLAGS_blackhole -mcpu=tt-bh)

# Define common flags for all architectures
set(GPP_FLAGS_common
    -std=c++17
    -flto
    -ffast-math
    -fno-use-cxa-atexit
    -fno-exceptions
    -Wall
    -Werror
    -Wno-unknown-pragmas
    -Wno-deprecated-declarations
    -Wno-error=multistatement-macros
    -Wno-error=parentheses
    -Wno-error=unused-but-set-variable
    -Wno-unused-variable
    -Wno-unused-function
)
set(GPP_FLAGS_brisc
    -Os
    -fno-tree-loop-distribute-patterns
)

set(GPP_FLAGS_ncrisc
    -Os
    -fno-tree-loop-distribute-patterns
)

set(GPP_FLAGS_trisc0 -O3)
set(GPP_FLAGS_trisc1 -O3)
set(GPP_FLAGS_trisc2 -O3)
set(GPP_FLAGS_erisc
    -Os
    -fno-delete-null-pointer-checks
)
set(GPP_FLAGS_idle_erisc
    -Os
    -fno-delete-null-pointer-checks
)
set(GPP_FLAGS_slave_idle_erisc
    -Os
    -fno-delete-null-pointer-checks
)

# Includes independent from ARCH
set(GPP_INCLUDES_common
    -I.
    -I..
    -I${PROJECT_SOURCE_DIR}
    -I${PROJECT_SOURCE_DIR}/tt_metal
    -I${PROJECT_SOURCE_DIR}/tt_metal/include
    -I${PROJECT_SOURCE_DIR}/tt_metal/hw/inc
    -I${PROJECT_SOURCE_DIR}/tt_metal/hw/inc/debug
)
#Linker flags
set(GPP_LINK_FLAGS_common
    -std=c++17
    -flto
    -ffast-math
    -fno-exceptions
    -Wl,-z,max-page-size=16
    -Wl,-z,common-page-size=16
    -nostartfiles
)

function(get_link_objects INPUT_ARCH INPUT_PROC INPUT_DIR OUTPUT_LIST)
    set(OBJ_LIST ${INPUT_DIR}substitutes.o)

    if("${INPUT_PROC}" STREQUAL "brisc")
        list(APPEND OBJ_LIST ${INPUT_DIR}tdma_xmov.o)
    endif()
    if(NOT "${INPUT_PROC}" STREQUAL "erisc")
        list(APPEND OBJ_LIST ${INPUT_DIR}tmu-crt0.o)
    endif()
    if("${INPUT_PROC}" STREQUAL "ncrisc")
        if("${INPUT_ARCH}" STREQUAL "grayskull" OR "${INPUT_ARCH}" STREQUAL "wormhole")
            list(APPEND OBJ_LIST ${INPUT_DIR}ncrisc-halt.o)
        endif()
    endif()

    if("${INPUT_PROC}" STREQUAL "brisc" OR "${INPUT_PROC}" STREQUAL "idle_erisc")
        list(APPEND OBJ_LIST ${INPUT_DIR}noc.o)
    endif()

    set(${OUTPUT_LIST} ${OBJ_LIST} PARENT_SCOPE)
endfunction()

# Build object files all combination of ARCH and PROC
foreach(ARCH IN LISTS ARCHS)
    get_alias(${ARCH} ARCH_B0)
    set(FW_OUTPUT_DIR "${PROJECT_SOURCE_DIR}/runtime/hw/firmware/${ARCH}")

    set(GPP_LINK_FLAGS_${ARCH}_brisc
        -Os
        -T${PROJECT_SOURCE_DIR}/runtime/hw/toolchain/${ARCH}/firmware_brisc.ld
    )
    set(GPP_LINK_FLAGS_${ARCH}_ncrisc
        -Os
        -T${PROJECT_SOURCE_DIR}/runtime/hw/toolchain/${ARCH}/firmware_ncrisc.ld
    )
    set(GPP_LINK_FLAGS_${ARCH}_trisc0
        -O3
        -T${PROJECT_SOURCE_DIR}/runtime/hw/toolchain/${ARCH}/firmware_trisc0.ld
    )
    set(GPP_LINK_FLAGS_${ARCH}_trisc1
        -O3
        -T${PROJECT_SOURCE_DIR}/runtime/hw/toolchain/${ARCH}/firmware_trisc1.ld
    )
    set(GPP_LINK_FLAGS_${ARCH}_trisc2
        -O3
        -T${PROJECT_SOURCE_DIR}/runtime/hw/toolchain/${ARCH}/firmware_trisc2.ld
    )

    set(GPP_LINK_FLAGS_${ARCH}_erisc
        -Os
        -L${PROJECT_SOURCE_DIR}/tt_metal/hw/toolchain
        -T${PROJECT_SOURCE_DIR}/tt_metal/hw/toolchain/erisc-b0-app.ld
    )

    set(GPP_LINK_FLAGS_${ARCH}_idle_erisc
        -Os
        -T${PROJECT_SOURCE_DIR}/runtime/hw/toolchain/${ARCH}/firmware_ierisc.ld
    )
    set(GPP_LINK_FLAGS_${ARCH}_slave_idle_erisc
        -Os
        -T${PROJECT_SOURCE_DIR}/runtime/hw/toolchain/${ARCH}/firmware_slave_ierisc.ld
    )

    # Architecture specific include paths
    set(GPP_INCLUDES_${ARCH}
        -I${PROJECT_SOURCE_DIR}/tt_metal/hw/inc/${ARCH}
        -I${PROJECT_SOURCE_DIR}/tt_metal/hw/inc/${ARCH}/${ARCH_B0}_defines
        -I${PROJECT_SOURCE_DIR}/tt_metal/hw/inc/${ARCH}/noc
        -I${PROJECT_SOURCE_DIR}/tt_metal/third_party/umd/device/api
        -I${PROJECT_SOURCE_DIR}/tt_metal/third_party/umd/device/${ARCH}
        -I${PROJECT_SOURCE_DIR}/tt_metal/hw/ckernels/${ARCH_B0}/metal/common
        -I${PROJECT_SOURCE_DIR}/tt_metal/hw/ckernels/${ARCH_B0}/metal/llk_io
        -I${PROJECT_SOURCE_DIR}/tt_metal/third_party/tt_llk_${ARCH_B0}/common/inc
        -I${PROJECT_SOURCE_DIR}/tt_metal/third_party/tt_llk_${ARCH_B0}/llk_lib
    )

    foreach(PROC IN LISTS PROCS)
        if("${ARCH}" STREQUAL "grayskull" AND "${PROC}" STREQUAL "erisc")
            continue() # Skip the iteration, grayskull doesn;t have idle_erisc or slave_idle_erisc
        elseif("${ARCH}" STREQUAL "grayskull" AND "${PROC}" STREQUAL "idle_erisc")
            continue() # Skip the iteration, grayskull doesn;t have idle_erisc or slave_idle_erisc
        elseif(NOT "${ARCH}" STREQUAL "blackhole" AND "${PROC}" STREQUAL "slave_idle_erisc")
            continue() # Skip the iteration, grayskull doesn;t have idle_erisc or slave_idle_erisc
        endif()

        # Set GPP_FLAGS based on ARCH
        set(GPP_FLAGS
            ${GPP_FLAGS_${ARCH}}
            ${GPP_FLAGS_common}
            ${GPP_FLAGS_${PROC}}
        )

        set(GPP_DEFINES
            ${GPP_DEFINES_${ARCH}}
            ${GPP_DEFINES_${PROC}}
            ${GPP_DEFINES_common}
        )

        set(GPP_INCLUDES
            ${GPP_INCLUDES_common}
            ${GPP_INCLUDES_${ARCH}}
        )

        if("${PROC}" STREQUAL "erisc")
            list(APPEND GPP_INCLUDES -I${PROJECT_SOURCE_DIR}/tt_metal/hw/inc/ethernet)
        elseif("${PROC}" STREQUAL "idle_erisc" OR "${PROC}" STREQUAL "slave_idle_erisc")
            list(APPEND GPP_INCLUDES -I${PROJECT_SOURCE_DIR}/tt_metal/hw/firmware/src)
        elseif("${PROC}" STREQUAL "trisc0" OR "${PROC}" STREQUAL "trisc1" OR "${PROC}" STREQUAL "trisc2")
            list(APPEND GPP_INCLUDES -I${PROJECT_SOURCE_DIR}/tt_metal/hw/ckernels/${ARCH_B0}/metal/llk_api)
            list(APPEND GPP_INCLUDES -I${PROJECT_SOURCE_DIR}/tt_metal/hw/ckernels/${ARCH_B0}/metal/llk_api/llk_sfpu)
            list(APPEND GPP_INCLUDES -I${PROJECT_SOURCE_DIR}/runtime/sfpi/include)
            list(APPEND GPP_INCLUDES -I${PROJECT_SOURCE_DIR}/tt_metal/hw/firmware/src)
        else()
            list(APPEND GPP_INCLUDES -I${PROJECT_SOURCE_DIR}/tt_metal/hw/firmware/src)
        endif()

        # These are the set of object files we are to build foreach ARCH and each PROC
        set(FWLIB "${PROC}")

        # Map each .o to its source file
        get_source(${PROC} PROC_SOURCE)
        set(${FWLIB}_SOURCE "${PROJECT_SOURCE_DIR}/tt_metal/hw/firmware/src/${PROC_SOURCE}.cc")

        set(FW_OBJECT_FILE "${FW_OUTPUT_DIR}/${FWLIB}.o")
        add_custom_command(
            OUTPUT
                ${FW_OBJECT_FILE}
            COMMAND
                ${CMAKE_COMMAND} -E make_directory ${FW_OUTPUT_DIR}
            COMMAND
                ${GPP_CMD} ${GPP_FLAGS} ${GPP_DEFINES} ${GPP_INCLUDES} -c -o ${FW_OUTPUT_DIR}/${FWLIB}.o
                ${${FWLIB}_SOURCE}
            DEPENDS
                ${${FWLIB}_SOURCE}
            COMMENT "Building fw lib ${FW_OBJECT_FILE}"
            VERBATIM
        )
        list(APPEND FIRMWARE_OBJECT_FILES ${FW_OBJECT_FILE})

        set(FW_BINARY_FILE "${FW_OUTPUT_DIR}/${FWLIB}.elf")
        set(GPP_LINK_FLAGS
            ${GPP_FLAGS_${ARCH}}
            ${GPP_LINK_FLAGS_common}
            ${GPP_LINK_FLAGS_${ARCH}_${PROC}}
        )

        get_link_objects(${ARCH} ${PROC} "${PROJECT_SOURCE_DIR}/runtime/hw/lib/${ARCH}/" LINK_OBJS)
        add_custom_command(
            OUTPUT
                ${FW_BINARY_FILE}
            #COMMAND
            #${CMAKE_COMMAND} -E make_directory ${FW_OUTPUT_DIR}
            COMMAND
                ${GPP_CMD} ${GPP_LINK_FLAGS} -o ${FW_BINARY_FILE} ${FW_OBJECT_FILE} ${LINK_OBJS}
            DEPENDS
                ${${LINK_OBJS}}
                ${FW_OBJECT_FILE}
            COMMENT "Building fw binary ${FW_BINARY_FILE}"
            VERBATIM
        )
        list(APPEND FIRMWARE_BINARY_FILES ${FW_BINARY_FILE})
    endforeach()
endforeach()

# custom target that depends on all the output files
add_custom_target(
    firmware_output
    ALL
    DEPENDS
        ${FIRMWARE_OBJECT_FILES}
        ${FIRMWARE_BINARY_FILES}
)
add_dependencies(firmware_output hw_toolchain)

add_library(firmware_binary INTERFACE)
add_library(Metalium::Metal::Firmware ALIAS firmware_binary)

set(currentArch "$ENV{ARCH_NAME}")
string(REPLACE "wormhole_b0" "wormhole" currentArch "${currentArch}")
target_include_directories(
    firmware_binary
    INTERFACE
        inc
        inc/${currentArch}
        $<$<STREQUAL:${currentArch},wormhole>:${CMAKE_CURRENT_SOURCE_DIR}/inc/wormhole/wormhole_b0_defines>
)
