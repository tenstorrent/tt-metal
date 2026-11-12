# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# The X280 (Blackhole L2CPU) runtime and its firmware images. Built only with
# -DTT_METAL_BUILD_X280=ON; not part of the fabric library or of Metalium.

option(TT_METAL_BUILD_X280 "Enables build of the X280 runtime (needs TT_METAL_BUILD_TESTS)" OFF)
if(NOT TT_METAL_BUILD_X280)
    return()
endif()
if(NOT TT_METAL_BUILD_TESTS)
    message(FATAL_ERROR "TT_METAL_BUILD_X280 requires TT_METAL_BUILD_TESTS: x280 links test_metal_common_libs")
endif()

include(x280_image.cmake)

x280_add_image(NAME firmware-lim-idle LIM_IDLE SOURCES src/lim_idle.c)
# The sentinel active image.
x280_add_image(NAME firmware SOURCES src/main.c)
# Pinned-input preparation, a trusted artifact compiled with exactly one operation define.
x280_add_image(NAME firmware-pinned-input-prepare SOURCES src/x280_pinned_input.c DEFINES X280_PINNED_INPUT_BUILD_PREPARE=1)
# Test-only images: the atomics micro-benchmark, which the active-image contract also runs, and
# the deliberate non-conformer that traps and CEASEs hart 2.
x280_add_image(NAME firmware-atomic-bench SOURCES src/atomic_bench.c)
x280_add_image(NAME firmware-active-session-fault SOURCES src/active_session_fault.c DEFINES X280_ACTIVE_FAULT_HART=2)

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
    x280_noc_ordering.h
    x280_hart_protocol.h
    x280.h
    noc.h
    rt/measurement.h
    rt/x280_atomic.h
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
        "X280_PINNED_INPUT_BUILD_PREPARE=1,X280_PINNED_INPUT_BUILD_LIBRARY=1|define exactly one pinned-input image operation"
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

# ---------------------------------------------------------------------------
# The runtime. x280_state holds the pure classification and validation rules and needs no Metal;
# x280_device is the Metal-backed transport and boot state machine. The runtime reaches
# MetalContext and tt::Cluster the way in-tree tests do.
# ---------------------------------------------------------------------------
add_library(
    x280_state
    STATIC
    src/device/x280_active_state.cpp
    src/device/x280_state.cpp
    src/device/x280_tile_state.cpp
    src/device/x280_noc_window.cpp
    src/device/x280_pinned_input_record.cpp
)
target_include_directories(x280_state PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_library(
    x280_device
    STATIC
    src/device/x280_runtime.cpp
    src/device/x280_chip.cpp
    src/device/x280_tile.cpp
    src/device/x280_active.cpp
    src/device/x280_metal_transport.cpp
    src/device/x280_pll.cpp
    src/device/x280_lim_ecc.cpp
    src/device/x280_boot.cpp
    src/device/x280_pinned_input_prepare.cpp
)
target_link_libraries(x280_device PUBLIC x280_state PRIVATE TT::Metalium)
target_include_directories(
    x280_device
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/device
        "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>"
)
# _x280_uses_images(<target>): <target> defaults to the images this build produces (X280_IMAGE_DIR).
# A test property, not the runtime's: the runtime takes image paths from its caller.
function(_x280_uses_images target)
    target_compile_definitions(${target} PRIVATE X280_IMAGE_DIR="${X280_IMAGE_DIR}")
    add_dependencies(${target} x280_images)
endfunction()

set(X280_TEST_OUTPUT_DIR "${PROJECT_BINARY_DIR}/test/tt_metal/x280")

# Host-only tests: x280_state and gtest, no Metal and no device.
add_executable(
    x280_runtime_host_tests
    tests/host/test_x280_format.cpp
    tests/host/test_x280_state.cpp
    tests/host/test_x280_active.cpp
    tests/host/test_x280_tile_state.cpp
    tests/host/test_x280_noc_window.cpp
    tests/host/test_x280_noc_coord.cpp
    tests/host/test_x280_pinned_input.cpp
)
target_include_directories(
    x280_runtime_host_tests
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/src/device
)
target_link_libraries(
    x280_runtime_host_tests
    PRIVATE
        x280_state
        gtest_main
        TT::Metalium::HostDevCommon
)
set_target_properties(
    x280_runtime_host_tests
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${X280_TEST_OUTPUT_DIR}
)
add_test(NAME x280_runtime_host_tests COMMAND x280_runtime_host_tests)

# _x280_device_test(<name> <source>...): a gtest executable over x280_device and Metal. Executables
# stay separate where a test must not share a process: prime is destructive, active_failure
# retains a fault, and shape, pll and pinned_input change chip state behind an opt-in.
function(_x280_device_test name)
    add_executable(${name} ${ARGN})
    target_include_directories(
        ${name}
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}/src/device
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/device
            "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>"
    )
    target_link_libraries(
        ${name}
        PRIVATE
            x280_device
            test_metal_common_libs
    )
    set_target_properties(
        ${name}
        PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY
                ${X280_TEST_OUTPUT_DIR}
    )
    _x280_uses_images(${name})
endfunction()

_x280_device_test(x280_runtime_device_tests
    tests/device/test_x280_active_hw.cpp
    tests/device/test_x280_active_workloads_hw.cpp
    tests/device/test_x280_device_hw.cpp
)
_x280_device_test(x280_runtime_coordinate_tests tests/device/test_x280_coordinates.cpp)
_x280_device_test(x280_active_failure_test tests/device/test_x280_active_failure_hw.cpp)
_x280_device_test(x280_prime_test tests/device/test_x280_prime_hw.cpp)
_x280_device_test(x280_shape_test tests/device/test_x280_shape_hw.cpp)
_x280_device_test(x280_pinned_input_test tests/device/test_x280_pinned_input_hw.cpp)
_x280_device_test(x280_arc_test tests/device/test_x280_arc_hw.cpp)
_x280_device_test(x280_pll_test tests/device/test_x280_pll_hw.cpp)

# Mock-cluster runtime root for the coordinate tests. Mock mode reads the SoC descriptor YAML, and
# blackhole_140_arch.yaml lists the L2CPU tiles under router_only, so the tests run on an overlay
# of this tree whose descriptor names them; the source tree is untouched.
set(_coord_root "${CMAKE_CURRENT_BINARY_DIR}/coordinate-test-runtime")
file(MAKE_DIRECTORY "${_coord_root}/tt_metal/soc_descriptors")
foreach(_item runtime tt_stl .cpmcache)
    file(
        CREATE_LINK
            "${PROJECT_SOURCE_DIR}/${_item}"
            "${_coord_root}/${_item}"
        SYMBOLIC
    )
endforeach()
file(GLOB _metal_entries "${PROJECT_SOURCE_DIR}/tt_metal/*")
foreach(_entry IN LISTS _metal_entries)
    get_filename_component(_name "${_entry}" NAME)
    if(NOT _name STREQUAL "soc_descriptors")
        file(
            CREATE_LINK
                "${_entry}"
                "${_coord_root}/tt_metal/${_name}"
            SYMBOLIC
        )
    endif()
endforeach()
file(GLOB _soc_entries "${PROJECT_SOURCE_DIR}/tt_metal/soc_descriptors/*")
foreach(_entry IN LISTS _soc_entries)
    get_filename_component(_name "${_entry}" NAME)
    configure_file("${_entry}" "${_coord_root}/tt_metal/soc_descriptors/${_name}" COPYONLY)
endforeach()
file(READ "${PROJECT_SOURCE_DIR}/tt_metal/soc_descriptors/blackhole_140_arch.yaml" _soc)
if(_soc MATCHES "\nl2cpu:")
    message(FATAL_ERROR "Coordinate tests require the unchanged Metal descriptor without an l2cpu section")
endif()
foreach(_y 3 5 7 9)
    set(_before "${_soc}")
    string(REPLACE "8-${_y}, " "" _soc "${_soc}")
    if(_soc STREQUAL _before)
        message(FATAL_ERROR "Coordinate test overlay: 8-${_y} not found in the descriptor's router_only list")
    endif()
endforeach()
string(APPEND _soc "\nl2cpu: [8-3, 8-5, 8-7, 8-9]\n")
file(WRITE "${_coord_root}/tt_metal/soc_descriptors/blackhole_140_arch.yaml" "${_soc}")

file(
    READ
    "${PROJECT_SOURCE_DIR}/tt_metal/third_party/umd/tests/cluster_descriptor_examples/blackhole_P300_both_mmio.yaml"
    _cluster
)
set(_disagree "${_cluster}")
string(FIND "${_disagree}" "l2cpu_harvesting_mask: 0" _first)
if(_first LESS 0)
    message(FATAL_ERROR "Coordinate test fixture has no unharvested L2CPU mask")
endif()
string(LENGTH "l2cpu_harvesting_mask: 0" _mask_length)
math(EXPR _after_first "${_first} + ${_mask_length}")
string(SUBSTRING "${_disagree}" 0 ${_after_first} _prefix)
string(SUBSTRING "${_disagree}" ${_after_first} -1 _suffix)
if(NOT _suffix MATCHES "l2cpu_harvesting_mask: 0")
    message(FATAL_ERROR "Coordinate test fixture has no second unharvested L2CPU mask to disagree with")
endif()
string(REPLACE "l2cpu_harvesting_mask: 0" "l2cpu_harvesting_mask: 2" _suffix "${_suffix}")
file(WRITE "${_coord_root}/disagreement.yaml" "${_prefix}${_suffix}")
file(WRITE "${_coord_root}/unharvested.yaml" "${_cluster}")
string(REPLACE "l2cpu_harvesting_mask: 0" "l2cpu_harvesting_mask: 2" _cluster "${_cluster}")
file(WRITE "${_coord_root}/harvested.yaml" "${_cluster}")

foreach(_case Unharvested Harvested Disagreement)
    string(TOLOWER "${_case}" _fixture)
    if(_case STREQUAL "Unharvested")
        set(_filter "X280Coordinates.*-X280Coordinates.Harvested*:X280Coordinates.Disagreement*")
    else()
        set(_filter "X280Coordinates.${_case}*")
    endif()
    add_test(
        NAME x280_coordinates_${_fixture}
        COMMAND
            ${CMAKE_COMMAND} -E env --unset=TT_VISIBLE_DEVICES "TT_METAL_RUNTIME_ROOT=${_coord_root}"
            "TT_METAL_MOCK_CLUSTER_DESC_PATH=${_coord_root}/${_fixture}.yaml"
            $<TARGET_FILE:x280_runtime_coordinate_tests> "--gtest_filter=${_filter}"
    )
    # A filter that selects nothing exits 0; require at least one passed test. The pass pattern
    # replaces the exit-code check, so a failed test must be matched explicitly.
    set_tests_properties(
        x280_coordinates_${_fixture}
        PROPERTIES
            PASS_REGULAR_EXPRESSION
                "\\[  PASSED  \\] [1-9]"
            FAIL_REGULAR_EXPRESSION
                "\\[  FAILED  \\]"
    )
endforeach()
