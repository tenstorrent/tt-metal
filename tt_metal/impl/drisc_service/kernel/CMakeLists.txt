# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# The DRISC service kernel is built here, not by the JIT, and linked like tt-metal's DRISC firmware (firmware_drisc.ld,
# tmu-crt0.o, substitutes.o). One build per NOC; the host loads
# <root_dir>/runtime/hw/lib/blackhole/drisc_service_noc<N>.elf, next to the other prebuilt device objects.
#
# GPP_CMD in tt_metal/hw/CMakeLists.txt is scoped to that directory, so the SFPI root is re-derived by the same rule.
if(TT_USE_SYSTEM_SFPI)
    set(DRISC_SERVICE_SFPI "/opt/tenstorrent/sfpi")
else()
    set(DRISC_SERVICE_SFPI "${PROJECT_SOURCE_DIR}/runtime/sfpi")
endif()
set(DRISC_SERVICE_GXX ${DRISC_SERVICE_SFPI}/compiler/bin/riscv-tt-elf-g++)
set(DRISC_SERVICE_HW_LIB ${PROJECT_SOURCE_DIR}/runtime/hw/lib/blackhole)
set(DRISC_SERVICE_LD ${PROJECT_SOURCE_DIR}/runtime/hw/toolchain/blackhole/firmware_drisc.ld)
set(DRISC_SERVICE_GUARD_LD ${CMAKE_CURRENT_SOURCE_DIR}/drisc_service_guard.ld)

set(DRISC_SERVICE_INCLUDES
    -I${PROJECT_SOURCE_DIR}
    -I${PROJECT_SOURCE_DIR}/tt_metal
    -I${PROJECT_SOURCE_DIR}/tt_metal/hw/inc
    -I${PROJECT_SOURCE_DIR}/tt_metal/hw/inc/internal/tt-1xx
    -I${PROJECT_SOURCE_DIR}/tt_metal/hw/inc/internal/tt-1xx/blackhole
    -I${PROJECT_SOURCE_DIR}/tt_metal/hw/inc/internal/tt-1xx/blackhole/blackhole_defines
    -I${PROJECT_SOURCE_DIR}/tt_metal/hw/inc/internal/tt-1xx/blackhole/noc
    -I${PROJECT_SOURCE_DIR}/tt_metal/hw/inc/experimental
    -I${PROJECT_SOURCE_DIR}/tt_metal/hw/ckernels/blackhole/metal/common
    -I${PROJECT_SOURCE_DIR}/tt_metal/hw/ckernels/blackhole/metal/llk_io
    -I${PROJECT_SOURCE_DIR}/tt_metal/tt-llk/tt_llk_blackhole/common/inc
    -I${PROJECT_SOURCE_DIR}/tt_metal/tt-llk/tt_llk_blackhole/llk_lib
    -I${PROJECT_SOURCE_DIR}/tt_metal/tt-llk/tt_llk_blackhole/llk_lib/hal
    -I${PROJECT_SOURCE_DIR}/tt_metal/tt-llk/common
    -I${PROJECT_SOURCE_DIR}/tt_metal/tt-llk/tools/include
    -I${PROJECT_SOURCE_DIR}/tt_metal/api
    -I${PROJECT_SOURCE_DIR}/tt_metal/hostdevcommon/api
    -I${DRISC_SERVICE_SFPI}/include
)
set(DRISC_SERVICE_FLAGS
    -std=c++20
    -O2
    -g
    -mcpu=tt-bh
    -fno-exceptions
    -fno-rtti
    -fno-threadsafe-statics
    -fno-use-cxa-atexit
    -ffreestanding
    -fno-tree-loop-distribute-patterns
    -msmall-data-limit=0
    -DARCH_BLACKHOLE
    -DCOMPILE_FOR_DRISC
    -DPROCESSOR_INDEX=0
    -DTENSIX_FIRMWARE
    -DLOCAL_MEM_EN=0
    -DKERNEL_BUILD
    -DNOC_MODE=DM_DEDICATED_NOC
    -DNUM_DRAM_BANKS=8
    -DNUM_L1_BANKS=110
    -DLOG_BASE_2_OF_NUM_DRAM_BANKS=3
    -DIS_NOT_POW2_NUM_L1_BANKS=1
)

set(DRISC_SERVICE_RUNTIME_ELFS)
foreach(noc 0 1)
    set(obj ${CMAKE_CURRENT_BINARY_DIR}/drisc_service_noc${noc}.o)
    set(elf ${CMAKE_CURRENT_BINARY_DIR}/drisc_service_noc${noc}.elf)
    set(runtime_elf ${DRISC_SERVICE_HW_LIB}/drisc_service_noc${noc}.elf)
    add_custom_command(
        OUTPUT
            ${elf}
            ${runtime_elf}
        COMMAND
            ${DRISC_SERVICE_GXX} ${DRISC_SERVICE_FLAGS} -DNOC_INDEX=${noc} ${DRISC_SERVICE_INCLUDES} -MMD -MP -MF
            ${CMAKE_CURRENT_BINARY_DIR}/drisc_service_noc${noc}.d -MT ${elf} -c
            ${CMAKE_CURRENT_SOURCE_DIR}/drisc_service.cpp -o ${obj}
        COMMAND
            ${DRISC_SERVICE_GXX} ${DRISC_SERVICE_FLAGS} -nostdlib -nostartfiles -Wl,-z,max-page-size=16
            -Wl,-z,common-page-size=16 -Wl,-T,${DRISC_SERVICE_LD} ${DRISC_SERVICE_HW_LIB}/tmu-crt0.o ${obj}
            ${DRISC_SERVICE_HW_LIB}/substitutes.o ${DRISC_SERVICE_GUARD_LD} -lgcc -o ${elf}
        COMMAND
            ${CMAKE_COMMAND} -E copy ${elf} ${runtime_elf}
        DEPFILE ${CMAKE_CURRENT_BINARY_DIR}/drisc_service_noc${noc}.d
        DEPENDS
            ${CMAKE_CURRENT_SOURCE_DIR}/drisc_service.cpp
            ${DRISC_SERVICE_LD}
            ${DRISC_SERVICE_GUARD_LD}
            ${DRISC_SERVICE_HW_LIB}/tmu-crt0.o
            ${DRISC_SERVICE_HW_LIB}/substitutes.o
        COMMENT "Building DRISC service kernel drisc_service_noc${noc}.elf"
        VERBATIM
    )
    list(APPEND DRISC_SERVICE_RUNTIME_ELFS ${runtime_elf})
endforeach()

add_custom_target(drisc_service_elf ALL DEPENDS ${DRISC_SERVICE_RUNTIME_ELFS})
add_dependencies(drisc_service_elf hw_toolchain)

if(TT_INSTALL)
    install(
        FILES
            ${DRISC_SERVICE_RUNTIME_ELFS}
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/runtime/hw/lib/blackhole
        COMPONENT metalium-runtime
    )
endif()
