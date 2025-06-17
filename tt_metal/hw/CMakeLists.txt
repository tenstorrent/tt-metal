# Temporary workaround for Issue #8767
set(ARCHS
    wormhole
    blackhole
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

# Get the appropriate SFPI tarball
file(STRINGS "../sfpi-version.sh" SFPI)
set(SFPI_arch_os "${CMAKE_HOST_SYSTEM_PROCESSOR}_${CMAKE_HOST_SYSTEM_NAME}")
foreach(tuple ${SFPI})
    string(STRIP ${tuple} pair)
    string(REPLACE "=" ";" pair ${pair})
    list(LENGTH pair count)
    if(count EQUAL 2)
        list(GET pair 0 key)
        list(GET pair 1 value)
        if(key STREQUAL "sfpi_version")
            set(SFPI_VERSION "${value}")
        elseif(key STREQUAL "sfpi_url")
            set(SFPI_URL "${value}")
        elseif(key STREQUAL "sfpi_${SFPI_arch_os}_txz_md5")
            set(SFPI_TXZ_MD5 "${value}")
        endif()
    endif()
endforeach()

if(NOT DEFINED SFPI_TXZ_MD5)
    message(FATAL_ERROR "SFPI tarball for ${SFPI_arch_os} is not available")
endif()

include(FetchContent)
FetchContent_Declare(
    sfpi
    URL
        "${SFPI_URL}/${SFPI_VERSION}/sfpi-${SFPI_arch_os}.txz"
    URL_HASH "MD5=${SFPI_TXZ_MD5}"
    SOURCE_DIR
    "${PROJECT_SOURCE_DIR}/runtime/sfpi"
)
FetchContent_MakeAvailable(sfpi)

foreach(ARCH IN LISTS ARCHS)
    set(DEV_MEM_MAP "${PROJECT_SOURCE_DIR}/tt_metal/hw/inc/${ARCH}/dev_mem_map.h")
    set(HW_INCLUDES "${PROJECT_SOURCE_DIR}/tt_metal/hw/inc/${ARCH}")
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
                    ${DEV_MEM_MAP}
                COMMENT "Preprocessing toolchain/${HW_OUTPUT_FILE}"
                VERBATIM
            )

            # add output file to the custom target
            list(APPEND LD_FILES ${HW_OUTPUT_DIR}/${HW_OUTPUT_FILE})
        endforeach()
    endforeach()
endforeach()

if(TT_USE_SYSTEM_SFPI)
    set(GPP_CMD "/opt/tenstorrent/sfpi/compiler/bin/riscv32-tt-elf-g++")
else()
    set(GPP_CMD "${PROJECT_SOURCE_DIR}/runtime/sfpi/compiler/bin/riscv32-tt-elf-g++")
endif()

if(NOT EXISTS "${GPP_CMD}")
    message(FATAL_ERROR "GPP_CMD path '${GPP_CMD}' does not exist. Please check your configuration.")
endif()

message(STATUS "Using SFPI compiler: ${GPP_CMD}")

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
    # These are the set of object files we are to build foreach ARCH
    set(HW_OBJ_SRC_PAIRS
        "tmu-crt0.o:tt_metal/hw/toolchain/tmu-crt0.S"
        "substitutes.o:tt_metal/hw/toolchain/substitutes.cpp"
        "tdma_xmov.o:tt_metal/hw/firmware/src/tdma_xmov.c"
        "noc.o:tt_metal/hw/firmware/src/${ARCH}/noc.c"
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
        -I${PROJECT_SOURCE_DIR}/tt_metal/hw/firmware/src
    )

    # Architecture specific include paths
    list(
        APPEND
        GPP_INCLUDES
        -I${PROJECT_SOURCE_DIR}/tt_metal/hw/inc/${ARCH}
        -I${PROJECT_SOURCE_DIR}/tt_metal/hw/inc/${ARCH}/${ARCH_ALIAS}_defines
        -I${PROJECT_SOURCE_DIR}/tt_metal/hw/inc/${ARCH}/noc
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
                ../sfpi-version.sh
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

if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.23)
    # These headers are for the device, not host; will require cross compiling to verify.
    set_target_properties(
        hw
        PROPERTIES
            VERIFY_INTERFACE_HEADER_SETS
                FALSE
    )
    target_sources(
        hw
        PUBLIC
            FILE_SET jit_api
            TYPE HEADERS
            BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
            FILES
                ckernels/blackhole/metal/llk_api/llk_math_binary_api.h
                ckernels/blackhole/metal/llk_api/llk_math_reduce_api.h
                ckernels/blackhole/metal/llk_api/llk_math_unary_sfpu_api.h
                ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_abs.h
                ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_comp.h
                ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_exp.h
                ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_exp2.h
                ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_expm1.h
                ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_heaviside.h
                ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_log.h
                ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_log1p.h
                ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_max.h
                ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_recip.h
                ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_relu.h
                ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_unary_max_min.h
                ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_relu.h
                ckernels/blackhole/metal/llk_api/llk_unpack_untilize_api.h
                ckernels/wormhole_b0/metal/llk_api/llk_math_binary_api.h
                ckernels/wormhole_b0/metal/llk_api/llk_math_reduce_api.h
                ckernels/wormhole_b0/metal/llk_api/llk_math_unary_sfpu_api.h
                ckernels/wormhole_b0/metal/llk_api/llk_unpack_untilize_api.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_abs.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_add1.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_alt_complex_rotate90.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_binary.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_binary_bitwise.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_binary_max_min.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_binop_with_unary.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_bitwise_and.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_bitwise_not.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_bitwise_or.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_bitwise_xor.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_cast_fp32_to_fp16a.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_cdf.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_clamp.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_comp.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_copy_dest_values.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_cumsum.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_dropout.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_elu.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_erf_erfc.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_erfinv.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_exp.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_exp2.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_expm1.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_fill.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_fmod.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_gelu.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_hardtanh.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_heaviside.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_i0.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_i1.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_identity.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_int_sum.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_isinf_isnan.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_left_shift.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_log.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_log1p.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_logical_not_noti.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_mask.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_max.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_min.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_negative.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_power_iterative.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_prelu.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_quant.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_rand.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_recip.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_relu.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_remainder.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_reshuffle_rows.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_right_shift.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_rsqrt.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_shift.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_sigmoid.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_sigmoid_appx.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_sign.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_signbit.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_silu.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_softplus.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_sqrt.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_square.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_sub_int32.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_tanh.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_tanh_derivative.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_tiled_prod.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_topk.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_trigonometry.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_typecast.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_unary_comp.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_unary_max_min.h
                ckernels/blackhole/metal/common/chlkc_list.h
                ckernels/blackhole/metal/llk_api/llk_math_common_api.h
                ckernels/blackhole/metal/llk_api/llk_math_matmul_api.h
                ckernels/blackhole/metal/llk_api/llk_math_unary_datacopy_api.h
                ckernels/blackhole/metal/llk_api/llk_pack_api.h
                ckernels/blackhole/metal/llk_api/llk_param_structs.h
                ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_exp.h
                ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_recip.h
                ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_exp.h
                ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_init.h
                ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_params.h
                ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_rounding_ops.h
                ckernels/blackhole/metal/llk_api/llk_sfpu_types.h
                ckernels/blackhole/metal/llk_api/llk_unpack_AB_api.h
                ckernels/blackhole/metal/llk_api/llk_unpack_AB_matmul_api.h
                ckernels/blackhole/metal/llk_api/llk_unpack_A_api.h
                ckernels/blackhole/metal/llk_api/llk_unpack_common_api.h
                ckernels/blackhole/metal/llk_api/llk_unpack_reduce_api.h
                ckernels/blackhole/metal/llk_api/llk_unpack_tilize_api.h
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
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_exp.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_recip.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_exp.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_init.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_params.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h
                ckernels/wormhole_b0/metal/llk_api/llk_unpack_AB_api.h
                ckernels/wormhole_b0/metal/llk_api/llk_unpack_AB_matmul_api.h
                ckernels/wormhole_b0/metal/llk_api/llk_unpack_A_api.h
                ckernels/wormhole_b0/metal/llk_api/llk_unpack_common_api.h
                ckernels/wormhole_b0/metal/llk_api/llk_unpack_reduce_api.h
                ckernels/wormhole_b0/metal/llk_api/llk_unpack_tilize_api.h
                ckernels/wormhole_b0/metal/llk_io/llk_io.h
                ckernels/wormhole_b0/metal/llk_io/llk_io_pack.h
                ckernels/wormhole_b0/metal/llk_io/llk_io_unpack.h
                ckernels/wormhole_b0/metal/llk_io/llk_operands.h
                ckernels/wormhole_b0/metal/llk_io/llk_outputs.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_abs.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_alt_complex_rotate90.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_bitwise_and.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_bitwise_not.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_bitwise_or.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_bitwise_xor.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_comp.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_exp2.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_expm1.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_fill.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_fmod.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_heaviside.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_i1.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_init.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_left_shift.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_log.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_log1p.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_mask.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_max.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_negative.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_power.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_prelu.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_recip.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_relu.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_remainder.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_right_shift.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_rounding_ops.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_rsqrt.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_sigmoid.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_sign.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_signbit.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_silu.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_sqrt.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_square.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_tanh.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_tiled_prod.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_topk.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_trigonometry.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_unary_comp.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_unary_comp.h
                ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_unary_max_min.h
                inc/atomic_rwptr.h
                inc/bit_utils.h
                inc/blackhole/c_tensix_core.h
                inc/blackhole/cfg_defines.h
                inc/blackhole/core_config.h
                inc/blackhole/dev_mem_map.h
                inc/blackhole/eth_chan_noc_mapping.h
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
                inc/debug/dprint_pages.h
                inc/debug/dprint_tile.h
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
                inc/remote_circular_buffer_api.h
                inc/risc_attribs.h
                inc/risc_common.h
                inc/tensix_functions.h
                inc/utils/bfloat16.h
                inc/utils/utils.h
                inc/vptr_uint.h
                inc/wormhole/c_tensix_core.h
                inc/wormhole/core_config.h
                inc/wormhole/dev_mem_map.h
                inc/wormhole/eth_chan_noc_mapping.h
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
