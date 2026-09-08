# Include source file lists (module owners maintain sources.cmake)
include(sources.cmake)

add_executable(
    fabric_unit_tests
    ${UNIT_TESTS_FABRIC_SRC}
    ${PROJECT_SOURCE_DIR}/tt_metal/impl/internal/disaggregation/kv_chunk_address_table_protobuf.cpp
)

target_link_libraries(
    fabric_unit_tests
    PRIVATE
        tt_metal
        # FSD proto (tt::scaleout_tools::fsd::proto) constructed directly by test_physical_descriptor_builder.cpp;
        # those symbols are hidden in libtt_metal.so (see tools/scaleout/CMakeLists.txt), so link explicitly.
        TT::ScaleoutTools
        test_common_libs
        yaml-cpp::yaml-cpp
        protobuf::libprotobuf
        cadical::cadical
)

target_include_directories(
    fabric_unit_tests
    PRIVATE
        ${UMD_HOME}
        ${PROJECT_SOURCE_DIR}/tests
        ${PROJECT_SOURCE_DIR}/tt_metal
        ${CMAKE_CURRENT_SOURCE_DIR}/common
    SYSTEM
    PRIVATE
        # physical_descriptor_builder test includes the generated FSD + PSD proto headers
        ${Metalium_BINARY_DIR}/tools/scaleout # protobuf/factory_system_descriptor.pb.h
        "$<TARGET_PROPERTY:fabric,BINARY_DIR>" # protobuf/physical_system_descriptor.pb.h
)

# The physical_descriptor_builder test includes generated FSD/PSD proto headers.
add_dependencies(
    fabric_unit_tests
    scaleout_generated_protos
    fabric_generated_protos
)

set_target_properties(
    fabric_unit_tests
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/tt_metal/tt_fabric
)

add_executable(test_physical_discovery ${UNIT_TESTS_PHYSICAL_DISCOVERY_SRC})

target_link_libraries(
    test_physical_discovery
    PRIVATE
        tt_metal
        test_common_libs
        yaml-cpp::yaml-cpp
        spdlog::spdlog
)

target_include_directories(
    test_physical_discovery
    PRIVATE
        ${UMD_HOME}
        ${PROJECT_SOURCE_DIR}/tests
        ${PROJECT_SOURCE_DIR}/tt_metal
        ${CMAKE_CURRENT_SOURCE_DIR}/common
)

set_target_properties(
    test_physical_discovery
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/tt_metal/tt_fabric
)

add_library(test_system_health_smoke OBJECT)
add_library(TT::SystemHealth::Smoke ALIAS test_system_health_smoke)

target_sources(test_system_health_smoke PRIVATE ${TEST_SYSTEM_HEALTH_SMOKE_SOURCES})

target_link_libraries(
    test_system_health_smoke
    PRIVATE
        tt_metal
        test_common_libs
)
add_executable(test_system_health)

target_include_directories(
    test_system_health_smoke
    PRIVATE
        ${UMD_HOME}
        ${PROJECT_SOURCE_DIR}/tests
        ${PROJECT_SOURCE_DIR}/tt_metal
        ${CMAKE_CURRENT_SOURCE_DIR}/common
)
set_target_properties(
    test_system_health
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/tt_metal/tt_fabric
)
target_link_libraries(test_system_health PRIVATE TT::SystemHealth::Smoke)

add_library(test_fabric_smoke OBJECT)
add_library(TT::Metalium::Test::Fabric::Smoke ALIAS test_fabric_smoke)

target_sources(test_fabric_smoke PRIVATE ${TEST_FABRIC_SMOKE_SOURCES})

target_include_directories(
    test_fabric_smoke
    PRIVATE
        ${UMD_HOME}
        ${PROJECT_SOURCE_DIR}/tests
        ${PROJECT_SOURCE_DIR}/tt_metal
        ${CMAKE_CURRENT_SOURCE_DIR}/common
        ${PROJECT_SOURCE_DIR}/tests/tt_metal/tt_fabric/test_infra
)

target_link_libraries(
    test_fabric_smoke
    PRIVATE
        tt_metal
        test_common_libs
)

add_executable(fabric_smoke_tests)
target_link_libraries(fabric_smoke_tests PRIVATE TT::Metalium::Test::Fabric::Smoke)

set_target_properties(
    fabric_smoke_tests
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/tt_metal/tt_fabric
)

set(TEST_TT_FABRIC_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/test_infra/test_tt_fabric.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_infra/tt_fabric_test_common_types.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_infra/tt_fabric_test_config.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_infra/tt_fabric_test_results.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_infra/tt_fabric_test_latency_results.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_infra/tt_fabric_test_bandwidth_results.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_infra/tt_fabric_test_context.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_infra/tt_fabric_test_bandwidth_profiler.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_infra/tt_fabric_test_device_setup.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_infra/tt_fabric_test_progress_monitor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_infra/tt_fabric_test_eth_readback.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_infra/tt_fabric_test_code_profiler.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_infra/tt_fabric_telemetry_manager.cpp
)

add_executable(test_tt_fabric ${TEST_TT_FABRIC_SOURCES})

# Previously (#48783) this target deliberately took TT::ScaleoutTools headers WITHOUT linking it,
# because scaleout_tools' protobuf symbols were exported from libtt_metal.so: linking a second copy
# meant both copies shared one exported descriptor_table_* object, so whichever module initialised
# first consumed its call_once flag and registered the file into only its own descriptor pool,
# tripping a libprotobuf fatal ("CHECK failed: file != nullptr") on the Torus paths. Now that those
# symbols are hidden (see tools/scaleout/CMakeLists.txt) nothing is shared, each module resolves to
# its own copy and its own pool, and linking normally is both safe and required -- the symbols are
# no longer available from libtt_metal.so.
target_link_libraries(
    test_tt_fabric
    PRIVATE
        TT::ScaleoutTools
        tt_metal
        test_common_libs
        yaml-cpp::yaml-cpp
)

# Redundant with the link above for the include path itself, but kept so these headers stay marked
# SYSTEM (link-propagated interface directories are not) and do not raise warnings here.
target_include_directories(
    test_tt_fabric
    SYSTEM
    PRIVATE
        $<TARGET_PROPERTY:TT::ScaleoutTools,INTERFACE_INCLUDE_DIRECTORIES>
)

target_include_directories(
    test_tt_fabric
    PRIVATE
        ${UMD_HOME}
        ${PROJECT_SOURCE_DIR}/tests
        ${PROJECT_SOURCE_DIR}/tests/tt_metal/test_utils
        ${PROJECT_SOURCE_DIR}/tt_metal
        ${CMAKE_CURRENT_SOURCE_DIR}/common
        ${CMAKE_CURRENT_SOURCE_DIR}/test_infra
)

set_target_properties(
    test_tt_fabric
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/tt_metal/tt_fabric/test_infra
)

# ---- Bench: unicast ----
add_executable(
    bench_unicast
    ${CMAKE_CURRENT_SOURCE_DIR}/benchmark/collectives/unicast/bench_unicast_main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/benchmark/collectives/unicast/run_unicast_once.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/benchmark/collectives/common/perf_helpers.cpp
)

target_link_libraries(
    bench_unicast
    PRIVATE
        tt_metal
        gtest
        test_common_libs
)

add_dependencies(bench_unicast tt_metal)

target_include_directories(
    bench_unicast
    PRIVATE
        ${PROJECT_SOURCE_DIR}
        ${UMD_HOME}
        ${PROJECT_SOURCE_DIR}/tests
        ${PROJECT_SOURCE_DIR}/tt_metal
        ${CMAKE_CURRENT_SOURCE_DIR}/common
        ${PROJECT_BINARY_DIR}/tt_metal/third_party/umd/device/api
        ${PROJECT_BINARY_DIR}/tt_metal/third_party/umd/device/generated
)

set_target_properties(
    bench_unicast
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/tt_metal/tt_fabric
)

# ---- Test: addrgen_write (unified test for all addrgen variants) ----
add_executable(
    test_addrgen_write
    ${CMAKE_CURRENT_SOURCE_DIR}/fabric_data_movement/addrgen_write/test_main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fabric_data_movement/addrgen_write/unicast_runner.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fabric_data_movement/addrgen_write/multicast_runner.cpp
)

target_link_libraries(
    test_addrgen_write
    PRIVATE
        tt_metal
        gtest
        test_common_libs
)

add_dependencies(test_addrgen_write tt_metal)

target_include_directories(
    test_addrgen_write
    PRIVATE
        ${PROJECT_SOURCE_DIR}
        ${UMD_HOME}
        ${PROJECT_SOURCE_DIR}/tests
        ${PROJECT_SOURCE_DIR}/tt_metal
        ${CMAKE_CURRENT_SOURCE_DIR}/common
        ${PROJECT_BINARY_DIR}/tt_metal/third_party/umd/device/api
        ${PROJECT_BINARY_DIR}/tt_metal/third_party/umd/device/generated
)

set_target_properties(
    test_addrgen_write
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/tt_metal/tt_fabric
)

# ---- Test: bandwidth telemetry validation ----
add_executable(
    test_bandwidth_telemetry_validation
    ${CMAKE_CURRENT_SOURCE_DIR}/test_bandwidth_telemetry_validation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/benchmark/collectives/unicast/run_unicast_once.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/benchmark/collectives/common/perf_helpers.cpp
)

target_link_libraries(
    test_bandwidth_telemetry_validation
    PRIVATE
        tt_metal
        gtest
        test_common_libs
)

add_dependencies(test_bandwidth_telemetry_validation tt_metal)

target_include_directories(
    test_bandwidth_telemetry_validation
    PRIVATE
        ${PROJECT_SOURCE_DIR}
        ${UMD_HOME}
        ${PROJECT_SOURCE_DIR}/tests
        ${PROJECT_SOURCE_DIR}/tt_metal
        ${CMAKE_CURRENT_SOURCE_DIR}/common
        ${PROJECT_BINARY_DIR}/tt_metal/third_party/umd/device/api
        ${PROJECT_BINARY_DIR}/tt_metal/third_party/umd/device/generated
)

set_target_properties(
    test_bandwidth_telemetry_validation
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/tt_metal/tt_fabric
)

# Reuse the shared test PCH (gtest + heavy tt-metalium headers) for all targets in this dir.
foreach(
    _fabric_test_target
    IN
    ITEMS
        fabric_unit_tests
        test_physical_discovery
        test_system_health_smoke
        test_fabric_smoke
        test_tt_fabric
        bench_unicast
        test_addrgen_write
        test_bandwidth_telemetry_validation
)
    target_precompile_headers(${_fabric_test_target} REUSE_FROM metal_test_pch)
endforeach()
