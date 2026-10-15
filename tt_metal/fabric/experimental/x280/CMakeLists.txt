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
# _x280_check_header(<header> <lang> [<include-dir>...]): <header> is found in include/ or in one of
# the extra include directories, which are also on the compile line.
function(_x280_check_header header lang)
    set(path "${CMAKE_CURRENT_SOURCE_DIR}/include/${header}")
    foreach(dir IN LISTS ARGN)
        if(EXISTS "${dir}/${header}")
            set(path "${dir}/${header}")
        endif()
    endforeach()
    if(NOT EXISTS "${path}")
        message(FATAL_ERROR "x280: header check: ${header} not found")
    endif()
    list(TRANSFORM ARGN PREPEND "-I" OUTPUT_VARIABLE dirs)
    string(MAKE_C_IDENTIFIER "${header}.${lang}" id)
    set(stamp "${X280_IMAGE_DIR}/headers/${id}.stamp")
    if(lang STREQUAL "c")
        set(flags
            -std=c11
            -fsyntax-only
        )
    elseif(lang STREQUAL "c++")
        set(flags
            ${_x280_cxxflags}
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
            "${X280_GCC}" ${X280_CFLAGS} ${dirs} ${flags} -MD -MT "${stamp}" -MF "${stamp}.d" -x ${lang} -include
            "${header}" /dev/null
        COMMAND
            ${CMAKE_COMMAND} -E touch "${stamp}"
        DEPENDS
            "${path}"
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
# _x280_check_headers([DIRS <dir>...] [C_AND_CXX <header>...] [CXX <header>...] [ASM <header>...]):
# _x280_check_header for each header in each listed language, with the DIRS (relative to this tree's
# root) as extra include directories.
function(_x280_check_headers)
    cmake_parse_arguments(PARSE_ARGV 0 ARG "" "" "DIRS;C_AND_CXX;CXX;ASM")
    list(TRANSFORM ARG_DIRS PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")
    foreach(header IN LISTS ARG_C_AND_CXX)
        _x280_check_header(${header} c ${ARG_DIRS})
        _x280_check_header(${header} c++ ${ARG_DIRS})
    endforeach()
    foreach(header IN LISTS ARG_CXX)
        _x280_check_header(${header} c++ ${ARG_DIRS})
    endforeach()
    foreach(header IN LISTS ARG_ASM)
        _x280_check_header(${header} assembler-with-cpp ${ARG_DIRS})
    endforeach()
    set(_x280_header_stamps ${_x280_header_stamps} PARENT_SCOPE)
endfunction()
_x280_check_headers(C_AND_CXX ${_x280_abi_headers} ASM ${_x280_asm_headers})

# _x280_refusal_test(<prefix> <source> COMPILE <command>... CASES <id>|<defines>|<message>...): one CTest
# <prefix><id> per case, compiling <source> with the case's comma-separated defines (none when empty).
# Each passes only on its own `#error "<message>"` (a regular expression), so a broken compiler
# invocation does not read as a refusal.
function(_x280_refusal_test prefix source)
    cmake_parse_arguments(PARSE_ARGV 2 ARG "" "" "COMPILE;CASES")
    foreach(case IN LISTS ARG_CASES)
        string(REPLACE "|" ";" case "${case}")
        list(GET case 0 id)
        list(GET case 1 defines)
        list(GET case 2 message)
        string(REPLACE "," ";" defines "${defines}")
        list(TRANSFORM defines PREPEND "-D")
        add_test(
            NAME ${prefix}${id}
            COMMAND
                ${ARG_COMPILE} ${defines} ${source}
        )
        set_tests_properties(
            ${prefix}${id}
            PROPERTIES
                PASS_REGULAR_EXPRESSION
                    "#error \"${message}\""
        )
    endforeach()
endfunction()

# src/x280_pinned_input.c refuses to compile without exactly one valid operation define.
_x280_refusal_test(x280_pinned_input_rejects_ ${CMAKE_CURRENT_SOURCE_DIR}/src/x280_pinned_input.c
    COMPILE ${X280_GCC} ${X280_CFLAGS} -std=c11 -fsyntax-only
    CASES
        "none||define exactly one pinned-input image operation"
        "X280_PINNED_INPUT_BUILD_PREPARE_0|X280_PINNED_INPUT_BUILD_PREPARE=0|X280_PINNED_INPUT_BUILD_PREPARE must be 1 when defined"
        "X280_PINNED_INPUT_BUILD_PREPARE_1_X280_PINNED_INPUT_BUILD_LIBRARY_1|X280_PINNED_INPUT_BUILD_PREPARE=1,X280_PINNED_INPUT_BUILD_LIBRARY=1|define exactly one pinned-input image operation"
)

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

# _x280_device_test(<name> <source>... [KERNELS <dir>...] [SLOW_DISPATCH]): a gtest executable over
# x280_device and Metal. Executables stay separate where a test must not share a process: prime is
# destructive, active_failure retains a fault, and shape, pll and pinned_input change chip state behind
# an opt-in. KERNELS adds kernels/<dir> include roots; SLOW_DISPATCH links x280_slow_dispatch, which a
# test that launches on a core-local RISC needs (harness/slow_dispatch.hpp).
function(_x280_device_test name)
    cmake_parse_arguments(PARSE_ARGV 1 ARG "SLOW_DISPATCH" "" "KERNELS")
    add_executable(${name} ${ARG_UNPARSED_ARGUMENTS})
    list(TRANSFORM ARG_KERNELS PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/kernels/")
    target_include_directories(
        ${name}
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}/src/device
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/device
            ${ARG_KERNELS}
            "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>"
    )
    target_link_libraries(
        ${name}
        PRIVATE
            x280_device
            test_metal_common_libs
    )
    if(ARG_SLOW_DISPATCH)
        target_link_libraries(${name} PRIVATE x280_slow_dispatch)
    endif()
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

# _x280_gtest_ctest(<name> <target> <filter> [ENV <cmake -E env argument>...]): a CTest running <target>
# with --gtest_filter=<filter>. A filter that selects nothing exits 0, so the test passes only when
# at least one test passed. The pass pattern replaces the exit-code check, so a failed test is
# matched explicitly; the fail pattern takes precedence.
function(_x280_gtest_ctest name target filter)
    cmake_parse_arguments(PARSE_ARGV 3 ARG "" "" "ENV")
    set(env "")
    if(ARG_ENV)
        set(env
            ${CMAKE_COMMAND}
            -E
            env
            ${ARG_ENV}
        )
    endif()
    add_test(
        NAME ${name}
        COMMAND
            ${env} $<TARGET_FILE:${target}> "--gtest_filter=${filter}"
    )
    set_tests_properties(
        ${name}
        PROPERTIES
            PASS_REGULAR_EXPRESSION
                "\\[  PASSED  \\] [1-9]"
            FAIL_REGULAR_EXPRESSION
                "\\[  FAILED  \\]"
    )
endfunction()

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
    _x280_gtest_ctest(x280_coordinates_${_fixture} x280_runtime_coordinate_tests "${_filter}"
        ENV --unset=TT_VISIBLE_DEVICES "TT_METAL_RUNTIME_ROOT=${_coord_root}"
            "TT_METAL_MOCK_CLUSTER_DESC_PATH=${_coord_root}/${_fixture}.yaml"
    )
endforeach()

# ---------------------------------------------------------------------------
# The benchmark harness (header-only, under harness/) and the kernel primitives every benchmark
# family shares (kernels/shared/).
# ---------------------------------------------------------------------------
# Linked by exactly the executables that launch a Metal kernel on a core-local RISC (idle or active
# ERISC, DRISC); its static initialiser sets TT_METAL_SLOW_DISPATCH_MODE=1 before Metal reads it.
# An object library keeps the linker from dropping the initialiser.
add_library(x280_slow_dispatch OBJECT harness/slow_dispatch.cpp)

# Suite headers compile on their own, as C and C++ or, for templates, C++ only.
set(_x280_shared_dir "${CMAKE_CURRENT_SOURCE_DIR}/kernels/shared")
_x280_check_headers(
    DIRS kernels/shared
    C_AND_CXX
        ubench_cookies.h
        ubench_backend_ids.h
        ubench_region.h
        ubench_units.h
        ubench_layout_cookie.h
        ubench_section_load.h
        ubench_outcome.h
        tensix_live_timer.h
        core_sync_layout.h
        core_sync_core_family.h
        core_sync_probe_args.h
        credit_counter.h
        p2p_latency_frontend.h
        p2p_latency_layout.h
        fabric_shadow.h
    CXX
        core_sync_arrival.hpp
        core_sync_barrier.hpp
        core_sync_transport_x280.hpp
        core_sync_transport_metal.hpp
        p2p_latency_frontend_cxx.hpp
        p2p_latency_entry_cxx.hpp
        x280_section_load.hpp
        fabric_shadow_header.hpp
)

# A DRISC core_sync director is refused at compile time, on kMayDirect alone.
add_test(
    NAME x280_core_sync_director_refusal
    COMMAND
        ${CMAKE_COMMAND} "-DCOMPILER=${X280_GCC}" "-DFLAGS=${X280_CFLAGS};-I${_x280_shared_dir};${_x280_cxxflags}"
        -DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/core-sync-director-check -P
        ${CMAKE_CURRENT_SOURCE_DIR}/x280_core_sync_director_check.cmake
)

# Host-only benchmark tests: like x280_runtime_host_tests, no Metal and no device. Each family adds
# its sources and its kernels/<family> include root.
add_executable(
    x280_bench_host_tests
    tests/host/test_harness_artifacts.cpp
    tests/host/test_harness_inventory.cpp
    tests/host/test_harness_output.cpp
    tests/host/test_harness_regions.cpp
    tests/host/test_harness_slow_dispatch.cpp
    tests/host/test_measurement.cpp
    tests/host/test_core_sync_barrier.cpp
)
target_include_directories(
    x280_bench_host_tests
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/src/device
        ${_x280_shared_dir}
)
target_link_libraries(
    x280_bench_host_tests
    PRIVATE
        x280_state
        gtest_main
        TT::Metalium::HostDevCommon
)
set_target_properties(
    x280_bench_host_tests
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${X280_TEST_OUTPUT_DIR}
)
# The image-lookup rule is tested on the path alone; no image is opened.
target_compile_definitions(x280_bench_host_tests PRIVATE X280_IMAGE_DIR="${X280_IMAGE_DIR}")
# RecordRoundTrip runs in its own CTest, x280_record_round_trip.
add_test(
    NAME x280_bench_host_tests
    COMMAND
        x280_bench_host_tests --gtest_filter=-RecordRoundTrip.*
)

# Harness device tests; most need hardware. test_fabric_shadow_layout.cpp is static
# assertions only.
_x280_device_test(x280_bench_device_tests
    tests/device/test_harness_open_config.cpp
    tests/device/test_harness_targets.cpp
    tests/device/test_fabric_shadow_layout.cpp
    KERNELS shared
    SLOW_DISPATCH
)
# Its two offline suites need no device: the PLL knob and the DRISC endpoint choice against the
# Blackhole SoC descriptor under Metal's runtime root.
_x280_gtest_ctest(x280_bench_device_offline x280_bench_device_tests "X280PllMhzFromEnv.*:MetalTargetOffline.*"
    ENV "TT_METAL_RUNTIME_ROOT=${PROJECT_SOURCE_DIR}"
)

# The release selector on the unharvested-L2CPU coordinate overlay, whose chip 0 keeps the
# descriptor's harvested Tensix columns.
_x280_device_test(x280_bench_release_selector_tests tests/device/test_release_selector.cpp KERNELS shared)
_x280_gtest_ctest(x280_release_selector x280_bench_release_selector_tests "ReleaseSelector.*"
    ENV --unset=TT_VISIBLE_DEVICES "TT_METAL_RUNTIME_ROOT=${_coord_root}"
        "TT_METAL_MOCK_CLUSTER_DESC_PATH=${_coord_root}/unharvested.yaml"
)

# ---------------------------------------------------------------------------
# transport_bandwidth: Tensix -> X280 (a Tensix kernel writes into passive LIM; no X280 image) and
# X280 -> Tensix (hart 0 stores through one NoC window into Tensix L1).
# ---------------------------------------------------------------------------
x280_add_image(
    NAME firmware-transport-bandwidth-x280-to-tensix
    SOURCES src/transport_bandwidth_x280_to_tensix.cpp
    INCLUDES
        kernels/shared
        kernels/transport_bandwidth
        kernels/transport_bandwidth/backends
)
_x280_check_headers(
    DIRS kernels/shared kernels/transport_bandwidth kernels/transport_bandwidth/backends
    C_AND_CXX transport_bandwidth_common.h tensix_to_x280.h x280_to_tensix.h rt/x280_vec_m1.h
    CXX transport_bandwidth_frontend_cxx.hpp
)
target_sources(x280_bench_host_tests PRIVATE tests/host/test_transport_bandwidth.cpp)
target_include_directories(x280_bench_host_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/kernels/transport_bandwidth)
_x280_device_test(x280_transport_bandwidth_tensix_to_x280
    tests/device/test_transport_bandwidth_tensix_to_x280_hw.cpp
    KERNELS shared transport_bandwidth
)
_x280_device_test(x280_transport_bandwidth_x280_to_tensix
    tests/device/test_transport_bandwidth_x280_to_tensix_hw.cpp
    KERNELS shared transport_bandwidth
)

# ---------------------------------------------------------------------------
# p2p_latency: X280 hart to hart on one tile, X280 tile to tile, and an X280 against a Tensix,
# idle-ERISC or DRISC peer launched by Metal. The three Metal peers share one X280 image.
# ---------------------------------------------------------------------------
foreach(_image x280-same-tile x280-peer-tile metal-peer-tile)
    string(REPLACE "-" "_" _source "${_image}")
    x280_add_image(
        NAME firmware-p2p-latency-${_image}
        SOURCES src/p2p_latency_${_source}.cpp
        INCLUDES
            kernels/shared
            kernels/p2p_latency
            kernels/p2p_latency/backends
    )
endforeach()
_x280_check_headers(
    DIRS kernels/shared kernels/p2p_latency kernels/p2p_latency/backends
    C_AND_CXX p2p_latency_x280_noc_args.h metal_peer_args.h
    CXX x280_noc.hpp
)
# The Metal peer kernel selects its core type from exactly one P2P_LATENCY_TARGET_* define.
set(_cases "none||p2p_latency Metal peer requires exactly one target define")
foreach(_targets TENSIX,IDLE_ERISC TENSIX,DRISC IDLE_ERISC,DRISC TENSIX,IDLE_ERISC,DRISC)
    string(MAKE_C_IDENTIFIER "${_targets}" _id)
    string(TOLOWER "${_id}" _id)
    string(REPLACE "," "=1,P2P_LATENCY_TARGET_" _defines "P2P_LATENCY_TARGET_${_targets}=1")
    list(APPEND _cases "${_id}|${_defines}|p2p_latency Metal peer requires exactly one target define")
endforeach()
_x280_refusal_test(x280_metal_peer_selection_rejects_ ${CMAKE_CURRENT_SOURCE_DIR}/kernels/p2p_latency/metal_peer_target.hpp
    COMPILE ${X280_GXX} ${_x280_target_flags} -E -x c++
    CASES ${_cases}
)
target_sources(x280_bench_host_tests PRIVATE tests/host/test_p2p_latency_args.cpp)
target_include_directories(x280_bench_host_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/kernels/p2p_latency)
foreach(_member x280_same_tile x280_peer_tile tensix_peer_tile)
    _x280_device_test(x280_p2p_latency_${_member}
        tests/device/test_p2p_latency_${_member}_hw.cpp
        KERNELS shared p2p_latency
    )
endforeach()
# The idle-ERISC and DRISC peers launch only under slow dispatch.
foreach(_member erisc_peer_tile drisc_peer_tile)
    _x280_device_test(x280_p2p_latency_${_member}
        tests/device/test_p2p_latency_${_member}_hw.cpp
        KERNELS shared p2p_latency
        SLOW_DISPATCH
    )
endforeach()

# ---------------------------------------------------------------------------
# pipeline_stage: X280-only router micro-operations over the pinned input: route_compute (the
# destination-to-next-hop lookup) and route_forward (the forward decision). Both executables need
# the 14-way LIM profile.
# ---------------------------------------------------------------------------
foreach(_member compute forward)
    x280_add_image(
        NAME firmware-pipeline-stage-route-${_member}
        SOURCES src/pipeline_stage_route_${_member}.cpp
        INCLUDES
            kernels/shared
            kernels/pipeline_stage
            kernels/pipeline_stage/backends
    )
endforeach()
_x280_check_headers(
    DIRS kernels/shared kernels/pipeline_stage kernels/pipeline_stage/backends
    C_AND_CXX pipeline_stage_common.h route_compute_args.h route_forward_args.h
    CXX pipeline_stage_frontend_cxx.hpp x280_member_backend.hpp x280_route_compute.hpp x280_route_forward.hpp
)
target_sources(x280_bench_host_tests PRIVATE tests/host/test_pipeline_stage.cpp)
target_include_directories(x280_bench_host_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/kernels/pipeline_stage)
foreach(_member compute forward)
    _x280_device_test(x280_pipeline_stage_route_${_member}
        tests/device/test_pipeline_stage_route_${_member}_hw.cpp
        KERNELS shared pipeline_stage
    )
endforeach()

# ---------------------------------------------------------------------------
# core_sync: an X280 hart joins or directs core_sync_barrier over Tensix, ERISC and DRISC workers;
# the release probe releases one or more X280 multicast groups.
# ---------------------------------------------------------------------------
foreach(_member director release-probe)
    string(REPLACE "-" "_" _source "${_member}")
    x280_add_image(
        NAME firmware-core-sync-${_member}
        SOURCES src/core_sync_${_source}.cpp
        INCLUDES
            kernels/shared
            kernels/core_sync
    )
endforeach()
_x280_check_headers(DIRS kernels/shared kernels/core_sync C_AND_CXX core_sync_director_args.h core_sync_release_probe_args.h)
target_sources(x280_bench_host_tests PRIVATE tests/host/test_core_sync_director_args.cpp)
target_include_directories(x280_bench_host_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/kernels/core_sync)
target_sources(x280_bench_device_tests PRIVATE tests/device/test_core_sync_hw.cpp)
# Its memory-binding and probe-argument suites need no device.
_x280_gtest_ctest(x280_core_sync_offline x280_bench_device_tests "CoreSyncMemoryBindings.*:CoreSyncProbeArgs.*")
# The director's DRISC worker launches only under slow dispatch.
_x280_device_test(x280_core_sync_director
    tests/device/test_core_sync_x280_director_hw.cpp
    KERNELS shared core_sync
    SLOW_DISPATCH
)
# Built on request (ninja x280_core_sync_characterization), not by default.
_x280_device_test(x280_core_sync_characterization EXCLUDE_FROM_ALL
    tests/device/test_core_sync_characterization_hw.cpp
    KERNELS shared core_sync
)

# ---------------------------------------------------------------------------
# packet_injection: Tensix (BRISC/NCRISC) or idle-ERISC producers inject fabric packet headers into
# an X280 consumer. One image per mode and creation-flag variant; the latency variants prepare
# and tear down the pinned input themselves.
# ---------------------------------------------------------------------------
foreach(
    _variant
    IN
    ITEMS
        "latency-include|1|1"
        "latency-exclude|1|0"
        "throughput|2|0"
        "backpressure-include|3|1"
        "backpressure-exclude|3|0"
)
    string(REPLACE "|" ";" _variant "${_variant}")
    list(GET _variant 0 _name)
    list(GET _variant 1 _mode)
    list(GET _variant 2 _creation)
    set(_pinned_input "")
    if(_mode EQUAL 1)
        set(_pinned_input
            OBJECTS
            ${_x280_pinned_input_library}
        )
    endif()
    x280_add_image(
        NAME firmware-packet-injection-${_name}
        SOURCES src/packet_injection_x280.cpp
        ${_pinned_input}
        INCLUDES
            kernels
            kernels/shared
            kernels/packet_injection
        DEFINES
            PACKET_INJECTION_MODE=${_mode}
            PACKET_INJECTION_INCLUDE_CREATION=${_creation}
        LINK_OPTIONS -Wl,--defsym=X280_PACKET_INJECTION_LAYOUT=1
    )
endforeach()
_x280_check_headers(
    DIRS kernels/shared kernels/packet_injection
    C_AND_CXX packet_injection_abi.h
    CXX packet_injection_slot.hpp
)
# The consumer refuses to compile without one valid mode and creation flag.
_x280_refusal_test(x280_packet_injection_rejects_ ${CMAKE_CURRENT_SOURCE_DIR}/src/packet_injection_x280.cpp
    COMPILE
        ${X280_GXX} ${X280_CFLAGS} ${_x280_cxxflags} -I${CMAKE_CURRENT_SOURCE_DIR}/kernels -I${_x280_shared_dir}
        -I${CMAKE_CURRENT_SOURCE_DIR}/kernels/packet_injection -fsyntax-only
    CASES
        "none||PACKET_INJECTION_MODE and PACKET_INJECTION_INCLUDE_CREATION must be explicitly defined"
        "mode_4_include_creation_0|PACKET_INJECTION_MODE=4,PACKET_INJECTION_INCLUDE_CREATION=0|PACKET_INJECTION_MODE must be 1 \\(LATENCY\\), 2 \\(THROUGHPUT\\), or 3 \\(BACKPRESSURE\\)"
        "mode_1_include_creation_2|PACKET_INJECTION_MODE=1,PACKET_INJECTION_INCLUDE_CREATION=2|PACKET_INJECTION_INCLUDE_CREATION must be 0 or 1"
        "mode_2_include_creation_1|PACKET_INJECTION_MODE=2,PACKET_INJECTION_INCLUDE_CREATION=1|Throughput mode requires PACKET_INJECTION_INCLUDE_CREATION=0"
)
target_sources(
    x280_bench_host_tests
    PRIVATE
        tests/host/test_packet_injection.cpp
        tests/host/test_packet_injection_erisc.cpp
)
target_include_directories(x280_bench_host_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/kernels/packet_injection)
_x280_device_test(x280_packet_injection_tensix_to_x280
    tests/device/test_packet_injection_tensix_to_x280_hw.cpp
    KERNELS shared packet_injection
)
# The idle-ERISC producers launch only under slow dispatch.
_x280_device_test(x280_packet_injection_erisc_to_x280_header
    tests/device/test_packet_injection_erisc_to_x280_header_hw.cpp
    KERNELS shared packet_injection
    SLOW_DISPATCH
)
find_package(Threads REQUIRED)
# The records name the revision from tt-metal's generated version header.
foreach(_member tensix_to_x280 erisc_to_x280_header)
    target_link_libraries(x280_packet_injection_${_member} PRIVATE Threads::Threads)
    target_include_directories(x280_packet_injection_${_member} PRIVATE ${PROJECT_BINARY_DIR}/generated)
    add_dependencies(x280_packet_injection_${_member} tt_metal_version_header)
endforeach()

# ---------------------------------------------------------------------------
# eth_credit: credit flow control across the direct Ethernet link between two chips. X280
# forwarders on both chips; E0/E1 kernels on the linked active Ethernet core of each.
# ---------------------------------------------------------------------------
x280_add_image(
    NAME firmware-eth-credit-x280-fwd
    SOURCES src/eth_credit_x280_fwd.cpp
    INCLUDES
        kernels
        kernels/shared
        kernels/eth_credit
)
_x280_check_headers(
    DIRS kernels/shared kernels/eth_credit
    C_AND_CXX eth_credit_abi.h
    CXX eth_credit_header.hpp x280_fwd.hpp
)
# The E0/E1 kernels are built by the Metal JIT at launch; this compiles them offline the same way,
# for every shape and bring-up mode the suite runs. It also links them when a Metal runtime cache
# holds firmware built from this tree, and otherwise reports "compile only".
add_test(
    NAME x280_eth_credit_jit
    COMMAND
        ${CMAKE_CURRENT_SOURCE_DIR}/tools/eth_credit_jit_check.sh ${X280_GXX} ${PROJECT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR}/eth-credit-jit
)
target_sources(x280_bench_host_tests PRIVATE tests/host/test_eth_credit.cpp)
target_include_directories(x280_bench_host_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/kernels/eth_credit)
# The E0/E1 kernels launch on active Ethernet cores only under slow dispatch.
_x280_device_test(x280_eth_credit
    tests/device/test_eth_credit_hw.cpp
    KERNELS shared eth_credit
    SLOW_DISPATCH
)

# ---------------------------------------------------------------------------
# Results tooling: the record round trip, the tabulator and campaign-runner tests, and the
# STOP-poll check of every image that links the active harness.
# ---------------------------------------------------------------------------
find_package(Python3 REQUIRED COMPONENTS Interpreter)
# The round trip renders every family's records and validates them with the live tabulator and the
# manifest in this source tree.
target_sources(x280_bench_host_tests PRIVATE tests/host/test_record_round_trip.cpp)
target_compile_definitions(x280_bench_host_tests PRIVATE X280_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
_x280_gtest_ctest(x280_record_round_trip x280_bench_host_tests "RecordRoundTrip.*")
add_test(
    NAME x280_characterization
    COMMAND
        ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/test_characterization.py
)
# The STOP-poll check scans every image source that links the active harness. It is registered when
# this directory finishes, so an image added anywhere in this file is in the list.
function(_x280_add_stop_poll_test)
    get_property(sources GLOBAL PROPERTY X280_ACTIVE_IMAGE_SOURCES)
    list(REMOVE_DUPLICATES sources)
    add_test(
        NAME x280_stop_poll
        COMMAND
            ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/check_stop_poll.py ${sources}
    )
endfunction()
cmake_language(DEFER CALL _x280_add_stop_poll_test)

add_custom_target(x280_abi_headers ALL DEPENDS ${_x280_header_stamps})
