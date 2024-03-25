
set(OUTPUT_DIR ${CMAKE_BINARY_DIR}/hw/toolchain)
set(CORES
    brisc
    ncrisc
    trisc0
    trisc1
    trisc2
    idle-erisc
)

if("$ENV{ARCH_NAME}" STREQUAL "wormhole_b0")
    set(DEV_MEM_MAP "${CMAKE_SOURCE_DIR}/tt_metal/hw/inc/wormhole/dev_mem_map.h")
    set(INCLUDES "${CMAKE_SOURCE_DIR}/tt_metal/hw/inc/wormhole")
else()
    set(DEV_MEM_MAP "${CMAKE_SOURCE_DIR}/tt_metal/hw/inc/$ENV{ARCH_NAME}/dev_mem_map.h")
    set(INCLUDES "${CMAKE_SOURCE_DIR}/tt_metal/hw/inc/$ENV{ARCH_NAME}")
endif()

foreach(CORE IN LISTS CORES)
    # Define the output file
    set(OUTPUT_FILE "${OUTPUT_DIR}/${CORE}.ld")

    # custom command to preprocess/generate the output file
    add_custom_command(
        OUTPUT ${OUTPUT_FILE}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${OUTPUT_DIR}
        COMMAND ${CMAKE_CXX_COMPILER} -I${INCLUDES} -E -P -x c -o ${OUTPUT_FILE} ${CMAKE_CURRENT_SOURCE_DIR}/toolchain/${CORE}.ld
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/toolchain/${CORE}.ld ${DEV_MEM_MAP}
        COMMENT "Preprocessing toolchain/${CORE}.ld"
        VERBATIM
    )

    # Add the output file to the custom target
    list(APPEND PREPROCESSED_LD_FILES ${OUTPUT_FILE})
endforeach()

# Add a custom target that depends on all the output files
add_custom_target(hw_toolchain ALL DEPENDS ${PREPROCESSED_LD_FILES})
