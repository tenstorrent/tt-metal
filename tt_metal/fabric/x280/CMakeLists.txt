# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# The X280 (Blackhole L2CPU) runtime and its firmware images. Built only with
# -DTT_METAL_BUILD_X280=ON; not part of the fabric library or of Metalium.

if(NOT TT_METAL_BUILD_TESTS)
    message(FATAL_ERROR "TT_METAL_BUILD_X280 requires TT_METAL_BUILD_TESTS: x280 links test_metal_common_libs")
endif()

include(x280_image.cmake)

x280_add_image(NAME firmware-lim-idle LIM_IDLE SOURCES src/lim_idle.c)
# The sentinel active image.
x280_add_image(NAME firmware SOURCES src/main.c)
# Preparation and teardown are separate trusted artifacts compiled from one source, each with
# exactly one operation define.
x280_add_image(NAME firmware-pinned-input-prepare SOURCES src/x280_pinned_input.c DEFINES X280_PINNED_INPUT_BUILD_PREPARE=1)
x280_add_image(NAME firmware-pinned-input-teardown SOURCES src/x280_pinned_input.c DEFINES X280_PINNED_INPUT_BUILD_TEARDOWN=1)

add_test(
    NAME x280_image_layout
    COMMAND
        ${CMAKE_COMMAND} -DREADELF=${X280_READELF} -DIMAGE_DIR=${X280_IMAGE_DIR} -P
        ${CMAKE_CURRENT_SOURCE_DIR}/x280_image_layout.cmake
)

# Each ABI header compiles on its own, as C and C++ (and the two that boot/entry.S includes, as
# assembly), with the image flags and -Werror.
set(_x280_abi_headers
    x280_address_map.h
    x280_align.h
    x280_noc_coord.h
    x280_hart_protocol.h
    x280.h
    socket_layout.h
    rt/x280_hw.h
    rt/x280_idle.h
    rt/x280_measurement.h
    rt/x280_noc_map.h
    rt/x280_pll.h
    rt/x280_pinned_input.h
)
set(_x280_asm_headers
    x280_address_map.h
    x280_hart_protocol.h
)
set(_x280_header_stamps "")
function(_x280_check_header header lang)
    string(MAKE_C_IDENTIFIER "${header}.${lang}" id)
    set(stamp "${X280_IMAGE_DIR}/headers/${id}.stamp")
    if(lang STREQUAL "c")
        set(flags
            -std=c11
            -fsyntax-only
        )
    elseif(lang STREQUAL "c++")
        set(flags
            -std=c++17
            -fno-exceptions
            -fno-rtti
            -fsyntax-only
        )
    else()
        set(flags
            -c
            -o
            /dev/null
        )
    endif()
    add_custom_command(
        OUTPUT
            "${stamp}"
        COMMAND
            "${X280_GCC}" ${X280_CFLAGS} ${flags} -MD -MT "${stamp}" -MF "${stamp}.d" -x ${lang} -include "${header}"
            /dev/null
        COMMAND
            ${CMAKE_COMMAND} -E touch "${stamp}"
        DEPENDS
            "${CMAKE_CURRENT_SOURCE_DIR}/include/${header}"
        DEPFILE "${stamp}.d"
        COMMENT "x280: compile-checking ${header} as ${lang}"
        VERBATIM
        COMMAND_EXPAND_LISTS
    )
    set(_x280_header_stamps
        ${_x280_header_stamps}
        "${stamp}"
        PARENT_SCOPE
    )
endfunction()
foreach(_header IN LISTS _x280_abi_headers)
    _x280_check_header(${_header} c)
    _x280_check_header(${_header} c++)
endforeach()
foreach(_header IN LISTS _x280_asm_headers)
    _x280_check_header(${_header} assembler-with-cpp)
endforeach()
add_custom_target(x280_abi_headers ALL DEPENDS ${_x280_header_stamps})

# src/x280_pinned_input.c refuses to compile without exactly one valid operation define. Each case
# passes only on its own #error, so a broken compiler invocation does not read as a refusal.
foreach(
    _case
    IN
    ITEMS
        "none|define exactly one pinned-input image operation"
        "X280_PINNED_INPUT_BUILD_PREPARE=0|X280_PINNED_INPUT_BUILD_PREPARE must be 1 when defined"
        "X280_PINNED_INPUT_BUILD_TEARDOWN=0|X280_PINNED_INPUT_BUILD_TEARDOWN must be 1 when defined"
        "X280_PINNED_INPUT_BUILD_PREPARE=1,X280_PINNED_INPUT_BUILD_TEARDOWN=1|define exactly one pinned-input image operation"
)
    string(REPLACE "|" ";" _case "${_case}")
    list(GET _case 0 _defines)
    list(GET _case 1 _expected)
    string(MAKE_C_IDENTIFIER "${_defines}" _id)
    if(_defines STREQUAL "none")
        set(_defines "")
    else()
        string(REPLACE "," ";" _defines "${_defines}")
        list(TRANSFORM _defines PREPEND "-D")
    endif()
    add_test(
        NAME x280_pinned_input_rejects_${_id}
        COMMAND
            ${X280_GCC} ${X280_CFLAGS} -std=c11 -fsyntax-only ${_defines}
            ${CMAKE_CURRENT_SOURCE_DIR}/src/x280_pinned_input.c
    )
    set_tests_properties(
        x280_pinned_input_rejects_${_id}
        PROPERTIES
            PASS_REGULAR_EXPRESSION
                "#error \"${_expected}\""
    )
endforeach()
