
set(HW_OUTPUT_DIR ${PROJECT_BINARY_DIR}/hw/toolchain)
set(CORES
    brisc
    ncrisc
    trisc0
    trisc1
    trisc2
    idle-erisc
)

if("$ENV{ARCH_NAME}" STREQUAL "wormhole_b0")
    set(DEV_MEM_MAP "${PROJECT_SOURCE_DIR}/tt_metal/hw/inc/wormhole/dev_mem_map.h")
    set(HW_INCLUDES "${PROJECT_SOURCE_DIR}/tt_metal/hw/inc/wormhole")
else()
    set(DEV_MEM_MAP "${PROJECT_SOURCE_DIR}/tt_metal/hw/inc/$ENV{ARCH_NAME}/dev_mem_map.h")
    set(HW_INCLUDES "${PROJECT_SOURCE_DIR}/tt_metal/hw/inc/$ENV{ARCH_NAME}")
endif()

foreach(CORE IN LISTS CORES)
    set(HW_OUTPUT_FILE "${HW_OUTPUT_DIR}/${CORE}.ld")

    # custom command to preprocess/generate the output file
    add_custom_command(
        OUTPUT ${HW_OUTPUT_FILE}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${HW_OUTPUT_DIR}
        COMMAND ${CMAKE_CXX_COMPILER} -I${HW_INCLUDES} -E -P -x c -o ${HW_OUTPUT_FILE} ${CMAKE_CURRENT_SOURCE_DIR}/toolchain/${CORE}.ld
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/toolchain/${CORE}.ld ${DEV_MEM_MAP}
        COMMENT "Preprocessing toolchain/${CORE}.ld"
        VERBATIM
    )

    # add output file to the custom target
    list(APPEND PREPROCESSED_LD_FILES ${HW_OUTPUT_FILE})
endforeach()

# custom target that depends on all the output files
add_custom_target(hw_toolchain ALL DEPENDS ${PREPROCESSED_LD_FILES})
