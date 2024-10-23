# Temporary workaround for Issue #8767
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
    ierisc
)
set(TYPES
    firmware
    kernel
)

foreach(ARCH IN LISTS ARCHS)
    set(DEV_MEM_MAP "${PROJECT_SOURCE_DIR}/tt_metal/hw/inc/${ARCH}/dev_mem_map.h")
    set(HW_INCLUDES "${PROJECT_SOURCE_DIR}/tt_metal/hw/inc/${ARCH}")
    set(HW_OUTPUT_DIR "${PROJECT_SOURCE_DIR}/runtime/hw/toolchain/${ARCH}")
    foreach(PROC IN LISTS PROCS)
        foreach(TYPE IN LISTS TYPES)
            set(HW_OUTPUT_FILE "${HW_OUTPUT_DIR}/${TYPE}_${PROC}.ld")
            string(TOUPPER ${PROC} PROC_DEFINE)
            string(TOUPPER ${TYPE} TYPE_DEFINE)

            # custom command to preprocess/generate the output file
            add_custom_command(
                OUTPUT
                    ${HW_OUTPUT_FILE}
                COMMAND
                    ${CMAKE_COMMAND} -E make_directory ${HW_OUTPUT_DIR}
                COMMAND
                    ${CMAKE_CXX_COMPILER} -DLD_TARGET=${PROC_DEFINE} -DLD_TYPE=${TYPE_DEFINE} -DTARGET_${PROC_DEFINE}
                    -DTYPE_${TYPE_DEFINE} -DCOMPILE_FOR_${PROC_DEFINE} -I${HW_INCLUDES} -E -P -x c -o ${HW_OUTPUT_FILE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/toolchain/main.ld
                DEPENDS
                    ${CMAKE_CURRENT_SOURCE_DIR}/toolchain/main.ld
                    ${CMAKE_CURRENT_SOURCE_DIR}/toolchain/memory.ld
                    ${CMAKE_CURRENT_SOURCE_DIR}/toolchain/sections.ld
                    ${DEV_MEM_MAP}
                COMMENT "Preprocessing toolchain/${PROC}.ld"
                VERBATIM
            )

            # add output file to the custom target
            list(APPEND PREPROCESSED_LD_FILES ${HW_OUTPUT_FILE})
        endforeach(
            TYPE
            IN
            LISTS
            TYPES
        )
    endforeach(
        PROC
        IN
        LISTS
        PORCS
    )
endforeach(
    ARCH
    IN
    LISTS
    ARCHS
)

# Function appends b0 if its wormhole
function(get_alias INPUT_STRING OUTPUT_VAR)
    if("${INPUT_STRING}" STREQUAL "wormhole")
        set(${OUTPUT_VAR} "wormhole_b0" PARENT_SCOPE)
    else()
        set(${OUTPUT_VAR} "${INPUT_STRING}" PARENT_SCOPE)
    endif()
endfunction()

# Define the compiler command
set(GPP_CMD ${PROJECT_SOURCE_DIR}/tt_metal/third_party/sfpi/compiler/bin/riscv32-unknown-elf-g++)

set(GPP_DEFINES -DTENSIX_FIRMWARE)

# Define flags for each architecture
set(GPP_FLAGS_grayskull
    -mgrayskull
    -march=rv32iy
    -mtune=rvtt-b1
    -mabi=ilp32
)
set(GPP_FLAGS_wormhole
    -mwormhole
    -march=rv32imw
    -mtune=rvtt-b1
    -mabi=ilp32
)
set(GPP_FLAGS_blackhole
    -mblackhole
    -march=rv32iml
    -mtune=rvtt-b1
    -mabi=ilp32
)

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
    -Wno-error=multistatement-macros
    -Wno-error=parentheses
    -Wno-error=unused-but-set-variable
    -Wno-unused-variable
    -Wno-unused-function
    -Os
    -fno-tree-loop-distribute-patterns
)

# We are going to build 5 or 6 object files foreach ARCH
foreach(ARCH IN LISTS ARCHS)
    get_alias(${ARCH} ARCH_B0)

    # These are the set of object files we are to build foreach ARCH
    set(HWLIBS
        tmu-crt0
        tmu-crt0k
        substitutes
        tdma_xmov
        noc
        ncrisc-halt
    )

    # Map each .o to its source file
    set(tmu-crt0_SOURCE "${PROJECT_SOURCE_DIR}/tt_metal/hw/toolchain/tmu-crt0.S")
    set(tmu-crt0k_SOURCE "${PROJECT_SOURCE_DIR}/tt_metal/hw/toolchain/tmu-crt0k.S")
    set(substitutes_SOURCE "${PROJECT_SOURCE_DIR}/tt_metal/hw/toolchain/substitutes.cpp")
    set(tdma_xmov_SOURCE "${PROJECT_SOURCE_DIR}/tt_metal/hw/firmware/src/tdma_xmov.c")
    set(noc_SOURCE "${PROJECT_SOURCE_DIR}/tt_metal/hw/firmware/src/${ARCH}/noc.c")
    set(ncrisc-halt_SOURCE "${PROJECT_SOURCE_DIR}/tt_metal/hw/toolchain/ncrisc-halt.S")

    # Set GPP_FLAGS based on ARCH
    set(GPP_FLAGS
        ${GPP_FLAGS_${ARCH}}
        ${GPP_FLAGS_common}
    )

    # Dump object files to this directory
    set(HW_LIB_DIR ${PROJECT_SOURCE_DIR}/runtime/hw/lib/${ARCH})

    # Includes independent from ARCH
    set(GPP_INCLUDES
        -I.
        -I..
        -I${PROJECT_SOURCE_DIR}
        -I${PROJECT_SOURCE_DIR}/tt_metal
        -I${PROJECT_SOURCE_DIR}/tt_metal/include
        -I${PROJECT_SOURCE_DIR}/tt_metal/hw/inc
        -I${PROJECT_SOURCE_DIR}/tt_metal/hw/inc/debug
        -I${PROJECT_SOURCE_DIR}/tt_metal/hw/firmware/src
    )

    # Architecture specific include paths
    list(APPEND GPP_INCLUDES -I${PROJECT_SOURCE_DIR}/tt_metal/hw/inc/${ARCH})
    list(APPEND GPP_INCLUDES -I${PROJECT_SOURCE_DIR}/tt_metal/hw/inc/${ARCH}/${ARCH_B0}_defines)
    list(APPEND GPP_INCLUDES -I${PROJECT_SOURCE_DIR}/tt_metal/hw/inc/${ARCH}/noc)
    list(APPEND GPP_INCLUDES -I${PROJECT_SOURCE_DIR}/tt_metal/third_party/umd/device/${ARCH})
    list(APPEND GPP_INCLUDES -I${PROJECT_SOURCE_DIR}/tt_metal/hw/ckernels/${ARCH_B0}/metal/common)
    list(APPEND GPP_INCLUDES -I${PROJECT_SOURCE_DIR}/tt_metal/hw/ckernels/${ARCH_B0}/metal/llk_io)
    list(APPEND GPP_INCLUDES -I${PROJECT_SOURCE_DIR}/tt_metal/third_party/tt_llk_${ARCH_B0}/common/inc)
    list(APPEND GPP_INCLUDES -I${PROJECT_SOURCE_DIR}/tt_metal/third_party/tt_llk_${ARCH_B0}/llk_lib)

    foreach(HWLIB IN LISTS HWLIBS)
        if("${ARCH}" STREQUAL "blackhole" AND "${HWLIB}" STREQUAL "ncrisc-halt")
            continue() # Skip the iteration, blackhole doesn't have IRAM
        endif()

        set(HW_OUTPUT_FILE "${HW_LIB_DIR}/${HWLIB}.o")
        add_custom_command(
            OUTPUT
                ${HW_OUTPUT_FILE}
            COMMAND
                ${CMAKE_COMMAND} -E make_directory ${HW_LIB_DIR}
            COMMAND
                ${GPP_CMD} ${GPP_FLAGS} ${GPP_DEFINES} ${GPP_INCLUDES} -c -o ${HW_LIB_DIR}/${HWLIB}.o ${${HWLIB}_SOURCE}
            COMMENT "Building hw lib ${HWLIB}.o"
            VERBATIM
        )
        list(APPEND PREPROCESSED_O_FILES ${HW_OUTPUT_FILE})
    endforeach(
        HWLIB
        IN
        LISTS
        HWLIBS
    )
endforeach(
    ARCH
    IN
    LISTS
    ARCHS
)

# custom target that depends on all the output files
add_custom_target(
    hw_toolchain
    ALL
    DEPENDS
        ${PREPROCESSED_LD_FILES}
        ${PREPROCESSED_O_FILES}
)
