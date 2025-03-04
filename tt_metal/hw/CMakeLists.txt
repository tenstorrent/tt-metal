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
    slave_ierisc
    aerisc
)
set(TYPES
    firmware
    kernel
)

include(FetchContent)
set(SFPI_x86_64_Linux_RELEASE
    "v6.4.0/sfpi-release.tgz"
    "979309b710e825b459e52b52c470c658"
)
if(DEFINED SFPI_${CMAKE_HOST_SYSTEM_PROCESSOR}_${CMAKE_HOST_SYSTEM_NAME}_RELEASE)
    set(SFPI_RELEASE "${SFPI_${CMAKE_HOST_SYSTEM_PROCESSOR}_${CMAKE_HOST_SYSTEM_NAME}_RELEASE}")
    list(GET SFPI_RELEASE 0 SFPI_FILE)
    list(GET SFPI_RELEASE 1 SFPI_MD5)
    FetchContent_Declare(
        sfpi
        URL
            "https://github.com/tenstorrent/sfpi/releases/download/${SFPI_FILE}"
        URL_HASH "MD5=${SFPI_MD5}"
        SOURCE_DIR
        "${PROJECT_SOURCE_DIR}/runtime/sfpi"
    )
    FetchContent_MakeAvailable(sfpi)
else()
    message(FATAL_ERROR "SFPI binaries for ${CMAKE_HOST_SYSTEM_PROCESSOR}-${CMAKE_HOST_SYSTEM_NAME} not available")
endif()

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
        endforeach()
    endforeach()
endforeach()

# Function appends b0 if its wormhole
function(get_alias INPUT_STRING OUTPUT_VAR)
    if("${INPUT_STRING}" STREQUAL "wormhole")
        set(${OUTPUT_VAR} "wormhole_b0" PARENT_SCOPE)
    else()
        set(${OUTPUT_VAR} "${INPUT_STRING}" PARENT_SCOPE)
    endif()
endfunction()

# Define the compiler command
set(GPP_CMD ${PROJECT_SOURCE_DIR}/runtime/sfpi/compiler/bin/riscv32-unknown-elf-g++)

set(GPP_DEFINES -DTENSIX_FIRMWARE)

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
    -Wno-deprecated-declarations
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
        tmu-crt0k-ncrisc
        substitutes
        tdma_xmov
        noc
        ncrisc-halt
        ncrisc-halt-wormhole
    )

    # Map each .o to its source file
    set(tmu-crt0_SOURCE "${PROJECT_SOURCE_DIR}/tt_metal/hw/toolchain/tmu-crt0.S")
    set(tmu-crt0k_SOURCE "${PROJECT_SOURCE_DIR}/tt_metal/hw/toolchain/tmu-crt0k.S")
    set(tmu-crt0k-ncrisc_SOURCE "${PROJECT_SOURCE_DIR}/tt_metal/hw/toolchain/tmu-crt0k-ncrisc.S")
    set(substitutes_SOURCE "${PROJECT_SOURCE_DIR}/tt_metal/hw/toolchain/substitutes.cpp")
    set(tdma_xmov_SOURCE "${PROJECT_SOURCE_DIR}/tt_metal/hw/firmware/src/tdma_xmov.c")
    set(noc_SOURCE "${PROJECT_SOURCE_DIR}/tt_metal/hw/firmware/src/${ARCH}/noc.c")
    set(ncrisc-halt_SOURCE "${PROJECT_SOURCE_DIR}/tt_metal/hw/toolchain/ncrisc-halt.S")
    set(ncrisc-halt-wormhole_SOURCE "${PROJECT_SOURCE_DIR}/tt_metal/hw/toolchain/ncrisc-halt-wormhole.S")

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
        -I${PROJECT_SOURCE_DIR}/tt_metal/api
        -I${PROJECT_SOURCE_DIR}/tt_metal/api/tt-metalium
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
    list(APPEND GPP_INCLUDES -I${PROJECT_SOURCE_DIR}/tt_metal/third_party/tt_llk/tt_llk_${ARCH_B0}/common/inc)
    list(APPEND GPP_INCLUDES -I${PROJECT_SOURCE_DIR}/tt_metal/third_party/tt_llk/tt_llk_${ARCH_B0}/llk_lib)

    foreach(HWLIB IN LISTS HWLIBS)
        if(
            "${ARCH}"
                STREQUAL
                "blackhole"
            AND (
                (
                    "${HWLIB}"
                        STREQUAL
                        "ncrisc-halt"
                )
                OR (
                    "${HWLIB}"
                        STREQUAL
                        "tmu-crt0k-ncrisc"
                )
            )
        )
            continue() # Skip the iteration, blackhole doesn't have IRAM
        endif()
        if((NOT "${ARCH}" STREQUAL "wormhole") AND "${HWLIB}" STREQUAL "ncrisc-halt-wormhole")
            continue() # Skip the iteration, only wormhole has ncrisc-halt-wormhole
        endif()

        set(HW_OUTPUT_FILE "${HW_LIB_DIR}/${HWLIB}.o")
        add_custom_command(
            OUTPUT
                ${HW_OUTPUT_FILE}
            COMMAND
                ${CMAKE_COMMAND} -E make_directory ${HW_LIB_DIR}
            COMMAND
                ${GPP_CMD} ${GPP_FLAGS} ${GPP_DEFINES} ${GPP_INCLUDES} -c -o ${HW_LIB_DIR}/${HWLIB}.o ${${HWLIB}_SOURCE}
            DEPENDS
                ${${HWLIB}_SOURCE}
            COMMENT "Building hw lib ${HWLIB}.o"
            VERBATIM
        )
        list(APPEND PREPROCESSED_O_FILES ${HW_OUTPUT_FILE})
    endforeach()
endforeach()

# custom target that depends on all the output files
add_custom_target(
    hw_toolchain
    ALL
    DEPENDS
        ${PREPROCESSED_LD_FILES}
        ${PREPROCESSED_O_FILES}
)

add_library(hw INTERFACE)
add_library(Metalium::Metal::Hardware ALIAS hw)

target_include_directories(hw INTERFACE inc)

if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.23)
    target_sources(
        hw
        PUBLIC
            FILE_SET jit_api
            TYPE HEADERS
            BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
            FILES
                ckernels/blackhole/metal/common/chlkc_list.h
                ckernels/blackhole/metal/llk_api/llk_math_common_api.h
                ckernels/blackhole/metal/llk_api/llk_math_matmul_api.h
                ckernels/blackhole/metal/llk_api/llk_math_unary_datacopy_api.h
                ckernels/blackhole/metal/llk_api/llk_pack_api.h
                ckernels/blackhole/metal/llk_api/llk_param_structs.h
                ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_converter.h
                ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_exp.h
                ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_recip.h
                ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_exp.h
                ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_init.h
                ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_params.h
                ckernels/blackhole/metal/llk_api/llk_sfpu_types.h
                ckernels/blackhole/metal/llk_api/llk_unpack_AB_matmul_api.h
                ckernels/blackhole/metal/llk_api/llk_unpack_A_api.h
                ckernels/blackhole/metal/llk_api/llk_unpack_common_api.h
                ckernels/blackhole/metal/llk_io/llk_io.h
                ckernels/blackhole/metal/llk_io/llk_io_pack.h
                ckernels/blackhole/metal/llk_io/llk_io_unpack.h
                ckernels/blackhole/metal/llk_io/llk_operands.h
                ckernels/blackhole/metal/llk_io/llk_outputs.h
                ckernels/wormhole_b0/metal/common/chlkc_list.h
                ckernels/wormhole_b0/metal/llk_api/llk_math_common_api.h
                ckernels/wormhole_b0/metal/llk_api/llk_math_matmul_api.h
                ckernels/wormhole_b0/metal/llk_api/llk_math_unary_datacopy_api.h
                ckernels/wormhole_b0/metal/llk_api/llk_pack_api.h
                ckernels/wormhole_b0/metal/llk_api/llk_param_structs.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_converter.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_exp.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_recip.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_exp.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_init.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_params.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h
                ckernels/wormhole_b0/metal/llk_api/llk_unpack_AB_matmul_api.h
                ckernels/wormhole_b0/metal/llk_api/llk_unpack_A_api.h
                ckernels/wormhole_b0/metal/llk_api/llk_unpack_common_api.h
                ckernels/wormhole_b0/metal/llk_io/llk_io.h
                ckernels/wormhole_b0/metal/llk_io/llk_io_pack.h
                ckernels/wormhole_b0/metal/llk_io/llk_io_unpack.h
                ckernels/wormhole_b0/metal/llk_io/llk_operands.h
                ckernels/wormhole_b0/metal/llk_io/llk_outputs.h
                inc/atomic_rwptr.h
                inc/bit_utils.h
                inc/blackhole/c_tensix_core.h
                inc/blackhole/cfg_defines.h
                inc/blackhole/core_config.h
                inc/blackhole/dev_mem_map.h
                inc/blackhole/eth_l1_address_map.h
                inc/blackhole/noc/noc.h
                inc/blackhole/noc/noc_overlay_parameters.h
                inc/blackhole/noc/noc_parameters.h
                inc/blackhole/noc_nonblocking_api.h
                inc/blackhole/stream_interface.h
                inc/blackhole/stream_io_map.h
                inc/blackhole/tdma_xmov.h
                inc/blackhole/tensix.h
                inc/blackhole/tensix_types.h
                inc/circular_buffer.h
                inc/circular_buffer_init.h
                inc/cmd_defs.h
                inc/compile_time_args.h
                inc/dataflow_api.h
                inc/dataflow_cmd_bufs.h
                inc/dataflow_internal.h
                inc/debug/assert.h
                inc/debug/dprint.h
                inc/debug/dprint_buffer.h
                inc/debug/dprint_tile.h
                inc/debug/fw_debug.h
                inc/debug/noc_logging.h
                inc/debug/ring_buffer.h
                inc/debug/sanitize_noc.h
                inc/debug/stack_usage.h
                inc/debug/watcher_common.h
                inc/debug/waypoint.h
                inc/ethernet/dataflow_api.h
                inc/ethernet/erisc.h
                inc/ethernet/tt_eth_api.h
                inc/ethernet/tt_eth_ss_regs.h
                inc/ethernet/tunneling.h
                inc/firmware_common.h
                inc/grayskull/c_tensix_core.h
                inc/grayskull/cfg_defines.h
                inc/grayskull/core_config.h
                inc/grayskull/dev_mem_map.h
                inc/grayskull/eth_l1_address_map.h
                inc/grayskull/noc/noc.h
                inc/grayskull/noc/noc_overlay_parameters.h
                inc/grayskull/noc/noc_parameters.h
                inc/grayskull/noc_nonblocking_api.h
                inc/grayskull/stream_io_map.h
                inc/grayskull/tdma_xmov.h
                inc/grayskull/tensix.h
                inc/grayskull/tensix_types.h
                inc/mod_div_lib.h
                inc/remote_circular_buffer_api.h
                inc/risc_attribs.h
                inc/risc_common.h
                inc/tensix_functions.h
                inc/utils/utils.h
                inc/vptr_uint.h
                inc/wormhole/c_tensix_core.h
                inc/wormhole/core_config.h
                inc/wormhole/dev_mem_map.h
                inc/wormhole/eth_l1_address_map.h
                inc/wormhole/noc/noc.h
                inc/wormhole/noc/noc_overlay_parameters.h
                inc/wormhole/noc/noc_parameters.h
                inc/wormhole/noc_nonblocking_api.h
                inc/wormhole/stream_interface.h
                inc/wormhole/stream_io_map.h
                inc/wormhole/tdma_xmov.h
                inc/wormhole/tensix.h
                inc/wormhole/wormhole_b0_defines/cfg_defines.h
                inc/wormhole/wormhole_b0_defines/tensix_types.h
                inc/dataflow_api_addrgen.h
                inc/dataflow_api_common.h
                # Toolchain
                toolchain/erisc-b0-app.ld
                toolchain/erisc-b0-app-sections.ld
                toolchain/erisc-b0-memory.ld
                toolchain/erisc-b0-kernel.ld
                toolchain/tensix-address.ld
    )
endif()

target_link_libraries(hw INTERFACE TT::Metalium::HostDevCommon)

add_subdirectory(firmware)

if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.23)
    install(
        TARGETS
            hw
        FILE_SET
        jit_api
            DESTINATION
                ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/tt_metal/hw # FIXME: fix the include paths for jit_build
            COMPONENT metalium-runtime
    )
endif()
