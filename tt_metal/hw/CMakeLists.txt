# Temporary workaround for Issue #8767
set(HW_OUTPUT_DIR ${PROJECT_SOURCE_DIR}/runtime/hw/toolchain)
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

if("$ENV{ARCH_NAME}" STREQUAL "wormhole_b0")
    set(DEV_MEM_MAP "${PROJECT_SOURCE_DIR}/tt_metal/hw/inc/wormhole/dev_mem_map.h")
    set(HW_INCLUDES "${PROJECT_SOURCE_DIR}/tt_metal/hw/inc/wormhole")
else()
    set(DEV_MEM_MAP "${PROJECT_SOURCE_DIR}/tt_metal/hw/inc/$ENV{ARCH_NAME}/dev_mem_map.h")
    set(HW_INCLUDES "${PROJECT_SOURCE_DIR}/tt_metal/hw/inc/$ENV{ARCH_NAME}")
endif()

foreach(PROC IN LISTS PROCS)
    foreach(TYPE IN LISTS TYPES)
        set(HW_OUTPUT_FILE "${HW_OUTPUT_DIR}/${TYPE}_${PROC}.ld")
        string(TOUPPER ${PROC} PROC_DEFINE)
        string(TOUPPER ${TYPE} TYPE_DEFINE)

        # custom command to preprocess/generate the output file
        add_custom_command(
          OUTPUT ${HW_OUTPUT_FILE}
          COMMAND ${CMAKE_COMMAND} -E make_directory ${HW_OUTPUT_DIR}
          COMMAND ${CMAKE_CXX_COMPILER} -DLD_TARGET=${PROC_DEFINE} -DLD_TYPE=${TYPE_DEFINE} -DTARGET_${PROC_DEFINE} -DTYPE_${TYPE_DEFINE} -DCOMPILE_FOR_${PROC_DEFINE} -I${HW_INCLUDES} -E -P -x c -o ${HW_OUTPUT_FILE} ${CMAKE_CURRENT_SOURCE_DIR}/toolchain/main.ld
          DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/toolchain/main.ld ${CMAKE_CURRENT_SOURCE_DIR}/toolchain/memory.ld ${CMAKE_CURRENT_SOURCE_DIR}/toolchain/sections.ld ${DEV_MEM_MAP}
          COMMENT "Preprocessing toolchain/${PROC}.ld"
          VERBATIM
          )

        # add output file to the custom target
        list(APPEND PREPROCESSED_LD_FILES ${HW_OUTPUT_FILE})
    endforeach()
endforeach()

# Build hw lib objects
if("$ENV{ARCH_NAME}" STREQUAL "grayskull")
    set(GPP_FLAGS -mgrayskull -march=rv32iy -mtune=rvtt-b1 -mabi=ilp32)
    set(ALIAS_ARCH_NAME "grayskull")
elseif("$ENV{ARCH_NAME}" STREQUAL "wormhole_b0" OR "$ENV{ARCH_NAME}" STREQUAL "wormhole")
    set(GPP_FLAGS -mwormhole -march=rv32imw -mtune=rvtt-b1 -mabi=ilp32)
    set(ALIAS_ARCH_NAME "wormhole")
elseif("$ENV{ARCH_NAME}" STREQUAL "blackhole")
    set(GPP_FLAGS -mblackhole -march=rv32iml -mtune=rvtt-b1 -mabi=ilp32)
    set(ALIAS_ARCH_NAME "blackhole")
endif()

set(GPP_FLAGS ${GPP_FLAGS} -std=c++17 -flto -ffast-math -fno-use-cxa-atexit -fno-exceptions -Wall -Werror -Wno-unknown-pragmas -Wno-error=multistatement-macros -Wno-error=parentheses -Wno-error=unused-but-set-variable -Wno-unused-variable -Wno-unused-function)

set(HW_LIB_DIR ${PROJECT_SOURCE_DIR}/runtime/hw/lib)
set(GPP_CMD ${PROJECT_SOURCE_DIR}/tt_metal/third_party/sfpi/compiler/bin/riscv32-unknown-elf-g++)
set(GPP_DEFINES -DTENSIX_FIRMWARE)
# -Os -fno-tree-loop-distribute-patterns -DARCH_$ENV{ARCH_NAME} -DCOMPILE_FOR_NCRISC -DLOCAL_MEM_EN=0 -DFW_BUILD
set(GPP_INCLUDES -I. -I.. -I${PROJECT_SOURCE_DIR} -I${PROJECT_SOURCE_DIR}/tt_metal -I${PROJECT_SOURCE_DIR}/tt_metal/include -I${PROJECT_SOURCE_DIR}/tt_metal/hw/inc -I${PROJECT_SOURCE_DIR}/tt_metal/hw/inc/debug -I${PROJECT_SOURCE_DIR}/tt_metal/hw/inc/${ALIAS_ARCH_NAME} -I${PROJECT_SOURCE_DIR}/tt_metal/hw/inc/${ALIAS_ARCH_NAME}/$ENV{ARCH_NAME}_defines -I${PROJECT_SOURCE_DIR}/tt_metal/hw/inc/${ALIAS_ARCH_NAME}/noc -I${PROJECT_SOURCE_DIR}/tt_metal/third_party/umd/device/$ENV{ARCH_NAME} -I${PROJECT_SOURCE_DIR}/tt_metal/hw/ckernels/$ENV{ARCH_NAME}/metal/common -I${PROJECT_SOURCE_DIR}/tt_metal/hw/ckernels/$ENV{ARCH_NAME}/metal/llk_io -I${PROJECT_SOURCE_DIR}/tt_metal/third_party/tt_llk_$ENV{ARCH_NAME}/common/inc -I${PROJECT_SOURCE_DIR}/tt_metal/third_party/tt_llk_$ENV{ARCH_NAME}/llk_lib -I${PROJECT_SOURCE_DIR}/tt_metal/hw/firmware/src -I${PROJECT_SOURCE_DIR}/tt_metal/hw/ckernels/$ENV{ARCH_NAME}/metal/common -I${PROJECT_SOURCE_DIR}/tt_metal/hw/ckernels/$ENV{ARCH_NAME}/metal/llk_io)

add_custom_command(
    OUTPUT tmu-crt0
    COMMAND ${CMAKE_COMMAND} -E make_directory ${HW_LIB_DIR}
    COMMAND ${GPP_CMD} ${GPP_FLAGS} ${GPP_DEFINES} ${GPP_INCLUDES} -c -o ${HW_LIB_DIR}/tmu-crt0.o ${PROJECT_SOURCE_DIR}/tt_metal/hw/toolchain/tmu-crt0.S
    COMMENT "Building hw lib tmu-crt0.o"
    VERBATIM
)

add_custom_command(
    OUTPUT tmu-crt0k
    COMMAND ${CMAKE_COMMAND} -E make_directory ${HW_LIB_DIR}
    COMMAND ${GPP_CMD} ${GPP_FLAGS} ${GPP_DEFINES} ${GPP_INCLUDES} -c -o ${HW_LIB_DIR}/tmu-crt0k.o ${PROJECT_SOURCE_DIR}/tt_metal/hw/toolchain/tmu-crt0k.S
    COMMENT "Building hw lib tmu-crt0k.o"
    VERBATIM
)

add_custom_command(
    OUTPUT substitutes
    COMMAND ${CMAKE_COMMAND} -E make_directory ${HW_LIB_DIR}
    COMMAND ${GPP_CMD} ${GPP_FLAGS} ${GPP_DEFINES} ${GPP_INCLUDES} -c -o ${HW_LIB_DIR}/substitutes.o ${PROJECT_SOURCE_DIR}/tt_metal/hw/toolchain/substitutes.cpp
    COMMENT "Building hw lib substitutes.o"
    VERBATIM
)

add_custom_command(
    OUTPUT tdma_xmov
    COMMAND ${CMAKE_COMMAND} -E make_directory ${HW_LIB_DIR}
    COMMAND ${GPP_CMD} ${GPP_FLAGS} ${GPP_DEFINES} ${GPP_INCLUDES} -c -o ${HW_LIB_DIR}/tdma_xmov.o ${PROJECT_SOURCE_DIR}/tt_metal/hw/firmware/src/tdma_xmov.c
    COMMENT "Building hw lib tdma_xmov.o"
    VERBATIM
)

add_custom_command(
    OUTPUT noc
    COMMAND ${CMAKE_COMMAND} -E make_directory ${HW_LIB_DIR}
    COMMAND ${GPP_CMD} ${GPP_FLAGS} ${GPP_DEFINES} ${GPP_INCLUDES} -c -o ${HW_LIB_DIR}/noc.o ${PROJECT_SOURCE_DIR}/tt_metal/hw/firmware/src/${ALIAS_ARCH_NAME}/noc.c
    COMMENT "Building hw lib noc.o"
    VERBATIM
)

list(APPEND HW_LIB tmu-crt0 tmu-crt0k substitutes tdma_xmov noc)

# Only build ncrisc haltl for Grayskull and Wormhole since Blackhole ncrisc does not have IRAM
if (ALIAS_ARCH_NAME STREQUAL "grayskull" OR ALIAS_ARCH_NAME STREQUAL "wormhole")
    add_custom_command(
        OUTPUT ncrisc-halt
        COMMAND ${CMAKE_COMMAND} -E make_directory ${HW_LIB_DIR}
        COMMAND ${GPP_CMD} ${GPP_FLAGS} ${GPP_DEFINES} ${GPP_INCLUDES} -c -o ${HW_LIB_DIR}/ncrisc-halt.o ${PROJECT_SOURCE_DIR}/tt_metal/hw/toolchain/ncrisc-halt.S
        COMMENT "Building hw lib ncrisc-halt.o"
        VERBATIM
    )
    list(APPEND HW_LIB ncrisc-halt)
endif()

# custom target that depends on all the output files
add_custom_target(hw_toolchain ALL DEPENDS ${PREPROCESSED_LD_FILES} ${HW_LIB})
