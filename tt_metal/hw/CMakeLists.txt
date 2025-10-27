# Temporary workaround for Issue #8767
set(ARCHS
    wormhole
    blackhole
    quasar
)
set(PROCS
    brisc
    ncrisc
    ierisc
    subordinate_ierisc
    trisc=0
    trisc=1
    trisc=2
    aerisc
    main_aerisc
    subordinate_aerisc
    dm=0
)
set(TYPES
    firmware
    kernel
)

# for wormhole, we need to generate two different linker scripts
set(WH_LD_SCRIPTS
    kernel # base name
    app # base name
)
set(IRAM_OPTIONS
    "" # No IRAM
    ENABLE_IRAM # With IRAM
)

# determine sfpi version information
set(SFPI_BASE "${PROJECT_SOURCE_DIR}/runtime")
file(MAKE_DIRECTORY ${SFPI_BASE})

# sfpi-info.sh generates a cmake script, which we include just below.
execute_process(
    COMMAND
        ${CMAKE_CURRENT_SOURCE_DIR}/../sfpi-info.sh CMAKE txz
    OUTPUT_FILE ${SFPI_BASE}/sfpi-version.cmake
    COMMAND_ERROR_IS_FATAL ANY
)
# sfpi-info.sh sources sfpi-version, if either changes we should reconfigure
set_property(
    DIRECTORY
    APPEND
    PROPERTY
        CMAKE_CONFIGURE_DEPENDS
            "../sfpi-info.sh;../sfpi-version"
)
# this script sets a bunch of variables of the form SFPI_snake_case_name
include(${SFPI_BASE}/sfpi-version.cmake)

if(TT_USE_SYSTEM_SFPI)
    # we'll use the system's version
    set(SFPI_BASE "/opt/tenstorrent")
elseif(NOT "${SFPI_md5}" STREQUAL "")
    # download a toolchain
    include(FetchContent)
    FetchContent_Declare(
        sfpi
        URL
            "${SFPI_url}/${SFPI_filename}"
        URL_HASH "MD5=${SFPI_md5}"
        SOURCE_DIR
        "${SFPI_BASE}/sfpi"
    )
    FetchContent_MakeAvailable(sfpi)
else()
    message(STATUS "No downloadable SFPI tarball for ${SFPI_arch} ${SFPI_dist}")
    if(IS_READABLE "${SFPI_BASE}/${SFPI_filename}")
        message(STATUS "Locally-built ${SFPI_filename} present")
    else()
        file(REMOVE_RECURSE "${SFPI_BASE}/sfpi")
        message(
            STATUS
            "Building SFPI, this could take some time (~10 CPU-hours)\n"
            "Logging to ${SFPI_BASE}/sfpi-build.log ..."
        )
        execute_process(
            COMMAND
                ${CMAKE_CURRENT_SOURCE_DIR}/../sfpi-info.sh BUILD ${SFPI_BASE}/sfpi-src
            OUTPUT_FILE ${SFPI_BASE}/sfpi-build.log
            COMMAND_ERROR_IS_FATAL ANY
        )
    endif()
    if(EXISTS "${SFPI_BASE}/sfpi")
        message(STATUS "Locally-built toolchain present")
    else()
        message(STATUS "Unpacking locally-built ${SFPI_filename}")
        execute_process(
            COMMAND
                tar xJf ${SFPI_BASE}/${SFPI_filename}
            WORKING_DIRECTORY ${SFPI_BASE}
            COMMAND_ERROR_IS_FATAL ANY
        )
    endif()
endif()

set(GPP_CMD "${SFPI_BASE}/sfpi/compiler/bin/riscv-tt-elf-g++")
if(NOT EXISTS "${GPP_CMD}")
    message(FATAL_ERROR "GPP_CMD path '${GPP_CMD}' does not exist. Please check your configuration.")
endif()

execute_process(
    COMMAND
        ${GPP_CMD} --version
    COMMAND
        sed -e "1{s/^[^ ]* //;p}" -e d
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE GPP_VERSION
)
message(STATUS "Using SFPI compiler: ${GPP_CMD} ${GPP_VERSION}")
# check this is the version we want (the system version might be wrong)
string(REGEX REPLACE "^\\([^:]*:([^)]*)\\) .*$" "\\1" GPP_VERSION ${GPP_VERSION})
if(NOT ${GPP_VERSION} STREQUAL ${SFPI_version})
    if(TT_USE_SYSTEM_SFPI)
        message(
            FATAL_ERROR
            "Installed SFPI version ${GPP_VERSION} does not match required version ${SFPI_version}\nYou need to run `${PROJECT_SOURCE_DIR}/install_dependencies.sh --sfpi`."
        )
    else()
        message(FATAL_ERROR "SFPI packaging malfunction, ${SFPI_version} delivers ${GPP_VERSION}")
    endif()
endif()

if(NOT TT_USE_SYSTEM_SFPI)
    # Check each executable is loadable, as the user might not have installed the dependencies
    # \tlibcsfml-window.so.2.2 => not found
    # undefined symbol: blob\t(./a.out)
    file(
        GLOB SFPI_BINARIES
        "${SFPI_BASE}/sfpi/compiler/bin/*"
        "${SFPI_BASE}/sfpi/compiler/libexec/gcc/riscv-tt-elf/*/*"
    )
    execute_process(
        COMMAND
            echo ${SFPI_BINARIES}
        COMMAND
            sed "s/;/ /g"
        COMMAND
            xargs file
        COMMAND
            sed -e "/ELF ..-bit .SB executable,/{s/:.*//;p}" -e d
        COMMAND
            xargs ldd -d
        COMMAND
            sed -e "/=> not found/p" -e "/undefined symbol/{s/\t(.*)$//;p}" -e d
        COMMAND
            sort -u
        OUTPUT_STRIP_TRAILING_WHITESPACE
        OUTPUT_VARIABLE SFPI_UNMET
    )
    if(NOT "${SFPI_UNMET}" STREQUAL "")
        message(NOTICE "Unsatisfied dependencies:\n${SFPI_UNMET}")
        message(
            FATAL_ERROR
            "SFPI has unsatisfied dependencies. Either\n* set TT_USE_SYSTEM_SFPI and run install_dependencies.sh --install-sfpi, or\n* install the missing packages, or\n* (expert level) build your own sfpi and point sfpi-version at it"
        )
    endif()
endif()

foreach(ARCH IN LISTS ARCHS)
    if(ARCH STREQUAL "quasar")
        set(HW_VERSION_PATH "tt-2xx")
    else()
        set(HW_VERSION_PATH "tt-1xx")
    endif()

    set(DEV_MEM_MAPS
        "${PROJECT_SOURCE_DIR}/tt_metal/hw/inc/${HW_VERSION_PATH}/${ARCH}/dev_mem_map.h"
        "${PROJECT_SOURCE_DIR}/tt_metal/hw/inc/${HW_VERSION_PATH}/${ARCH}/tensix_dev_map.h"
    )
    set(HW_INCLUDES "${PROJECT_SOURCE_DIR}/tt_metal/hw/inc/${HW_VERSION_PATH}/${ARCH}")
    set(HW_OUTPUT_DIR "${PROJECT_SOURCE_DIR}/runtime/hw/toolchain/${ARCH}")
    string(TOUPPER ${ARCH} ARCH_DEFINE)

    if(ARCH STREQUAL "wormhole")
        foreach(SCRIPT_TYPE IN LISTS WH_LD_SCRIPTS)
            foreach(IRAM_OPT IN LISTS IRAM_OPTIONS)
                if(IRAM_OPT)
                    set(OUTPUT_FILE "${HW_OUTPUT_DIR}/erisc-b0-${SCRIPT_TYPE}_iram.ld")
                    set(IRAM_FLAG "-D${IRAM_OPT}")
                    set(IRAM_COMMENT " with IRAM")
                else()
                    set(OUTPUT_FILE "${HW_OUTPUT_DIR}/erisc-b0-${SCRIPT_TYPE}.ld")
                    set(IRAM_FLAG "")
                    set(IRAM_COMMENT "")
                endif()

                add_custom_command(
                    OUTPUT
                        ${OUTPUT_FILE}
                    COMMAND
                        ${CMAKE_CXX_COMPILER} ${IRAM_FLAG} -I${HW_INCLUDES} -E -P -x c -o ${OUTPUT_FILE}
                        ${CMAKE_CURRENT_SOURCE_DIR}/toolchain/erisc-b0-${SCRIPT_TYPE}.ld
                    DEPENDS
                        ${CMAKE_CURRENT_SOURCE_DIR}/toolchain/erisc-b0-${SCRIPT_TYPE}.ld
                    COMMENT "Preprocessing toolchain/erisc-b0-${SCRIPT_TYPE}.ld${IRAM_COMMENT}"
                    VERBATIM
                )

                list(APPEND LD_FILES ${OUTPUT_FILE})
            endforeach()
        endforeach()
    endif()

    foreach(PROC IN LISTS PROCS)
        string(REPLACE "=" "" PROC_FILE "${PROC}")
        foreach(TYPE IN LISTS TYPES)
            set(HW_OUTPUT_FILE "${TYPE}_${PROC_FILE}.ld")
            string(TOUPPER ${PROC} PROC_DEFINE)
            string(TOUPPER ${TYPE} TYPE_DEFINE)

            # custom command to preprocess/generate the output file
            add_custom_command(
                OUTPUT
                    ${HW_OUTPUT_DIR}/${HW_OUTPUT_FILE}
                COMMAND
                    ${CMAKE_COMMAND} -E make_directory ${HW_OUTPUT_DIR}
                COMMAND
                    ${CMAKE_CXX_COMPILER} -DTYPE_${TYPE_DEFINE} -DCOMPILE_FOR_${PROC_DEFINE} -DARCH_${ARCH_DEFINE}
                    -I${HW_INCLUDES} -E -P -x c -o ${HW_OUTPUT_DIR}/${HW_OUTPUT_FILE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/toolchain/main.ld
                DEPENDS
                    ${CMAKE_CURRENT_SOURCE_DIR}/toolchain/main.ld
                    ${DEV_MEM_MAPS}
                COMMENT "Preprocessing toolchain/${HW_OUTPUT_FILE}"
                VERBATIM
            )

            # add output file to the custom target
            list(APPEND LD_FILES ${HW_OUTPUT_DIR}/${HW_OUTPUT_FILE})
        endforeach()
    endforeach()
endforeach()

set(GPP_DEFINES -DTENSIX_FIRMWARE)

# Define flags for each architecture
set(GPP_FLAGS_wormhole -mcpu=tt-wh)
set(GPP_FLAGS_blackhole -mcpu=tt-bh)

# Define common flags for all architectures
set(GPP_FLAGS_common
    -std=c++17
    -flto=auto
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
    if(ARCH STREQUAL "quasar")
        set(HW_VERSION_PATH "tt-2xx")
    else()
        set(HW_VERSION_PATH "tt-1xx")
    endif()
    # These are the set of object files we are to build foreach ARCH
    set(HW_OBJ_SRC_PAIRS
        "substitutes.o:tt_metal/hw/toolchain/substitutes.cpp"
        "tdma_xmov.o:tt_metal/hw/firmware/src/tt-1xx/tdma_xmov.c"
        "noc.o:tt_metal/hw/firmware/src/${HW_VERSION_PATH}/${ARCH}/noc.c"
        "tmu-crt0.o:tt_metal/hw/toolchain/tmu-crt0.S"
    )
    set(ARCH_ALIAS ${ARCH})

    if(ARCH STREQUAL "wormhole")
        set(ARCH_ALIAS wormhole_b0)
        list(
            APPEND
            HW_OBJ_SRC_PAIRS
            "wh-iram-start.o:tt_metal/hw/toolchain/wh-iram-start.S"
            "wh-iram-trampoline.o:tt_metal/hw/toolchain/wh-iram-trampoline.S"
        )
    endif()

    # TODO quasar should use crt0.S, missing some header file
    # if(ARCH STREQUAL "quasar")
    #     list(APPEND HW_OBJ_SRC_PAIRS "tmu-crt0.o:tt_metal/hw/firmware/src/${HW_VERSION_PATH}/${ARCH}/crt0.S")
    # else()
    #     list(APPEND HW_OBJ_SRC_PAIRS "tmu-crt0.o:tt_metal/hw/toolchain/tmu-crt0.S")
    # endif()

    # Set GPP_FLAGS based on ARCH
    set(GPP_FLAGS
        ${GPP_FLAGS_${ARCH}}
        ${GPP_FLAGS_common}
    )

    # Dump object files to this directory
    set(HW_OBJ_DIR ${PROJECT_SOURCE_DIR}/runtime/hw/lib/${ARCH})

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
        -I${PROJECT_SOURCE_DIR}/tt_metal/hw/firmware/src/tt-1xx
        -I${PROJECT_SOURCE_DIR}/tt_metal/hw/firmware/src/tt-2xx
    )

    # Architecture specific include paths
    list(
        APPEND
        GPP_INCLUDES
        -I${PROJECT_SOURCE_DIR}/tt_metal/hw/inc/${HW_VERSION_PATH}/${ARCH}
        -I${PROJECT_SOURCE_DIR}/tt_metal/hw/inc/${HW_VERSION_PATH}/${ARCH}/${ARCH_ALIAS}_defines
        -I${PROJECT_SOURCE_DIR}/tt_metal/hw/inc/${HW_VERSION_PATH}/${ARCH}/noc
        -I${PROJECT_SOURCE_DIR}/tt_metal/hw/inc/${HW_VERSION_PATH}/${ARCH}/overlay
        -I${PROJECT_SOURCE_DIR}/tt_metal/third_party/umd/device/${ARCH}
        -I${PROJECT_SOURCE_DIR}/tt_metal/hw/ckernels/${ARCH_ALIAS}/metal/common
        -I${PROJECT_SOURCE_DIR}/tt_metal/hw/ckernels/${ARCH_ALIAS}/metal/llk_io
        -I${PROJECT_SOURCE_DIR}/tt_metal/third_party/tt_llk/tt_llk_${ARCH_ALIAS}/common/inc
        -I${PROJECT_SOURCE_DIR}/tt_metal/third_party/tt_llk/tt_llk_${ARCH_ALIAS}/llk_lib
    )

    foreach(HW_PAIR IN LISTS HW_OBJ_SRC_PAIRS)
        string(REPLACE ":" ";" HW_PAIR_LIST ${HW_PAIR})
        list(GET HW_PAIR_LIST 0 HWOBJ)
        list(GET HW_PAIR_LIST 1 HWSRC)
        set(HW_OUTPUT_FILE "${HW_OBJ_DIR}/${HWOBJ}")
        add_custom_command(
            OUTPUT
                ${HW_OUTPUT_FILE}
            COMMAND
                ${CMAKE_COMMAND} -E make_directory ${HW_OBJ_DIR}
            COMMAND
                ${GPP_CMD} ${GPP_FLAGS} ${GPP_DEFINES} ${GPP_INCLUDES} -c -o ${HW_OBJ_DIR}/${HWOBJ}
                ${PROJECT_SOURCE_DIR}/${HWSRC}
            DEPENDS
                ${PROJECT_SOURCE_DIR}/${HWSRC}
                # If the compiler changes, rebuild
                ../sfpi-version
            COMMENT "Building ${ARCH} hw lib ${HWOBJ}"
            VERBATIM
        )
        list(APPEND OBJ_FILES ${HW_OUTPUT_FILE})
    endforeach()
endforeach()

# custom target that depends on all the output files
add_custom_target(
    hw_toolchain
    ALL
    DEPENDS
        ${LD_FILES}
        ${OBJ_FILES}
)

add_library(hw INTERFACE)
add_library(Metalium::Metal::Hardware ALIAS hw)

target_include_directories(hw INTERFACE inc)

# These headers are for the device, not host; will require cross compiling to verify.
set_target_properties(
    hw
    PROPERTIES
        VERIFY_INTERFACE_HEADER_SETS
            FALSE
)

# It's acceptable to use GLOB here because these files are not part of the build and developers aren't generating packages (for now).
# We may need to revisit later.
file(GLOB_RECURSE blackhole_kernels ckernels/blackhole/*)
file(GLOB_RECURSE wormhole_kernels ckernels/wormhole_b0/*)
target_sources(
    hw
    PUBLIC
        FILE_SET jit_api
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES
            ${blackhole_kernels}
            ${wormhole_kernels}
            inc/accessor/array_wrapper.h
            inc/accessor/const.h
            inc/accessor/dspec.h
            inc/accessor/helpers.h
            inc/accessor/tensor_accessor.h
            inc/accessor/tensor_accessor_args.h
            inc/accessor/shard_pages_address_iterator.h
            inc/accessor/pages_address_iterator.h
            inc/accessor/page.h
            inc/atomic_rwptr.h
            inc/bit_utils.h
            inc/tt-1xx/blackhole/c_tensix_core.h
            inc/tt-1xx/blackhole/cfg_defines.h
            inc/tt-1xx/blackhole/core_config.h
            inc/tt-1xx/blackhole/dev_mem_map.h
            inc/tt-1xx/blackhole/eth_chan_noc_mapping.h
            inc/tt-1xx/blackhole/eth_fw_api.h
            inc/tt-1xx/blackhole/eth_l1_address_map.h
            inc/tt-1xx/blackhole/noc/noc.h
            inc/tt-1xx/blackhole/noc/noc_overlay_parameters.h
            inc/tt-1xx/blackhole/noc/noc_parameters.h
            inc/tt-1xx/blackhole/noc_nonblocking_api.h
            inc/tt-1xx/blackhole/stream_interface.h
            inc/tt-1xx/blackhole/stream_io_map.h
            inc/tt-1xx/blackhole/tdma_xmov.h
            inc/tt-1xx/blackhole/tensix.h
            inc/tt-1xx/blackhole/tensix_dev_map.h
            inc/tt-1xx/blackhole/tensix_types.h
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
            inc/debug/dprint_pages.h
            inc/debug/dprint_tensix.h
            inc/debug/dprint_tile.h
            inc/debug/eth_link_status.h
            inc/debug/fw_debug.h
            inc/debug/noc_logging.h
            inc/debug/ring_buffer.h
            inc/debug/sanitize_noc.h
            inc/debug/stack_usage.h
            inc/debug/watcher_common.h
            inc/debug/waypoint.h
            inc/dev_msgs.h
            inc/ethernet/dataflow_api.h
            inc/ethernet/erisc.h
            inc/ethernet/tt_eth_api.h
            inc/ethernet/tt_eth_ss_regs.h
            inc/ethernet/tunneling.h
            inc/firmware_common.h
            inc/mod_div_lib.h
            inc/tt-2xx/quasar/c_tensix_core.h
            inc/tt-2xx/quasar/cfg_defines.h
            inc/tt-2xx/quasar/core_config.h
            inc/tt-2xx/quasar/dev_mem_map.h
            inc/tt-2xx/quasar/eth_chan_noc_mapping.h
            inc/tt-2xx/quasar/eth_fw_api.h
            inc/tt-2xx/quasar/eth_l1_address_map.h
            inc/tt-2xx/quasar/noc/noc.h
            inc/tt-2xx/quasar/noc/noc_overlay_parameters.h
            inc/tt-2xx/quasar/noc/noc_parameters.h
            inc/tt-2xx/quasar/noc_nonblocking_api.h
            inc/tt-2xx/quasar/stream_interface.h
            inc/tt-2xx/quasar/stream_io_map.h
            inc/tt-2xx/quasar/tdma_xmov.h
            inc/tt-2xx/quasar/tensix.h
            inc/tt-2xx/quasar/tensix_dev_map.h
            inc/tt-2xx/quasar/tensix_types.h
            inc/remote_circular_buffer_api.h
            inc/risc_attribs.h
            inc/tt-1xx/risc_common.h
            inc/tensix_functions.h
            inc/utils/bfloat16.h
            inc/utils/float32.h
            inc/utils/int32.h
            inc/utils/utils.h
            inc/vptr_uint.h
            inc/tt-1xx/wormhole/c_tensix_core.h
            inc/tt-1xx/wormhole/core_config.h
            inc/tt-1xx/wormhole/dev_mem_map.h
            inc/tt-1xx/wormhole/eth_chan_noc_mapping.h
            inc/tt-1xx/wormhole/eth_fw_api.h
            inc/tt-1xx/wormhole/eth_l1_address_map.h
            inc/tt-1xx/wormhole/noc/noc.h
            inc/tt-1xx/wormhole/noc/noc_overlay_parameters.h
            inc/tt-1xx/wormhole/noc/noc_parameters.h
            inc/tt-1xx/wormhole/noc_nonblocking_api.h
            inc/tt-1xx/wormhole/stream_interface.h
            inc/tt-1xx/wormhole/stream_io_map.h
            inc/tt-1xx/wormhole/tdma_xmov.h
            inc/tt-1xx/wormhole/tensix.h
            inc/tt-1xx/wormhole/tensix_dev_map.h
            inc/tt-1xx/wormhole/wormhole_b0_defines/cfg_defines.h
            inc/tt-1xx/wormhole/wormhole_b0_defines/tensix_types.h
            inc/dataflow_api_addrgen.h
            inc/dataflow_api_common.h
            inc/socket_api.h
            inc/socket.h
            # Toolchain
            toolchain/erisc-b0-app.ld
            toolchain/erisc-b0-app-sections.ld
            toolchain/erisc-b0-memory.ld
            toolchain/erisc-b0-kernel.ld
        FILE_SET toolchain
        TYPE HEADERS
        BASE_DIRS ${PROJECT_SOURCE_DIR}
        FILES ${LD_FILES} ${OBJ_FILES}
)

target_link_libraries(hw INTERFACE TT::Metalium::HostDevCommon)

add_subdirectory(firmware)

install(
    TARGETS
        hw
    FILE_SET
    jit_api
        DESTINATION
            ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/tt_metal/hw # FIXME: fix the include paths for jit_build
        COMPONENT metalium-runtime
    FILE_SET
    toolchain
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium
        COMPONENT metalium-runtime
)
