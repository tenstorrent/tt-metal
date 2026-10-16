add_compile_options("$<$<CXX_COMPILER_ID:Clang>:-Wunused-parameter>")

include(sources.cmake)

add_library(tt_metal)
add_library(TT::Metalium ALIAS tt_metal)
add_library(Metalium::Metal ALIAS tt_metal) # For backwards compatibility

set_target_properties(
    tt_metal
    PROPERTIES
        EXPORT_NAME
            Metalium
)

include(flatbuffers)
GENERATE_FBS_HEADER(
    api/tt-metalium/serialized_descriptors/mesh_coordinate.fbs
    TARGET TT::Metalium
    OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/api/tt-metalium/serialized_descriptors/
)
set(API_GENERATED_HEADERS ${FBS_GENERATED_HEADER_FILE})

add_custom_target(
    metalium_GeneratedHeaders
    DEPENDS
        ${API_GENERATED_HEADERS}
    COMMENT "Generating all FlatBuffer headers"
)

target_sources(
    tt_metal
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS api ${CMAKE_CURRENT_BINARY_DIR}/api
        FILES ${TT_METAL_PUBLIC_API}
    PRIVATE
        ${TT_METAL_SOURCES}
        ${API_GENERATED_HEADERS}
)

# clang-tidy reads configuration relative to the generated translation unit,
# not the included header. Overlay the API checks only for this verification
# target; the parent build directory's .clang-tidy supplies the normal checks.
configure_file(
    "${PROJECT_SOURCE_DIR}/cmake/clang-tidy.d/metalium-api/.clang-tidy"
    "${CMAKE_CURRENT_BINARY_DIR}/tt_metal_verify_interface_header_sets/.clang-tidy"
    COPYONLY
)

set_source_files_properties(
    api/tt-metalium/serialized_descriptors/mesh_coordinate.fbs
    PROPERTIES
        SKIP_LINTING
            ON
)

# TODO(afuller): this should be self-describing modules.
#   For now just cherry-pick all the files I discovered empirically by trying to run a test.
add_library(jitapi INTERFACE)
# These headers are for the device, not host; will require cross compiling to verify.
set_target_properties(
    jitapi
    PROPERTIES
        VERIFY_INTERFACE_HEADER_SETS
            FALSE
)

file(GLOB_RECURSE COMPUTE_KERNEL_API include/*)
file(
    GLOB_RECURSE FABRIC_HW_INC
    fabric/hw/inc/*.hpp
    fabric/hw/inc/*.h
)
file(
    GLOB_RECURSE TT_LLK_HEADERS
    tt-llk/common/*.h
    tt-llk/tools/include/**/*.h
    tt-llk/tt_llk_wormhole_b0/**/*.h
    tt-llk/tt_llk_blackhole/**/*.h
    tt-llk/tt_llk_quasar/**/*.h
)

target_sources(
    jitapi
    INTERFACE
        FILE_SET jit_api
        TYPE HEADERS
        FILES ${JITAPI_FILES} ${FABRIC_HW_INC} ${COMPUTE_KERNEL_API} ${TT_LLK_HEADERS}
)

include(CheckFunctionExists)

check_function_exists(
    backtrace
    HAVE_BACKTRACE
)

if(NOT HAVE_BACKTRACE)
    find_library(EXECINFO_LIB execinfo)
else()
    set(EXECINFO_LIB "")
endif()

target_link_libraries(
    tt_metal
    PUBLIC
        umd::tt-umd
        enchantum::enchantum
        fmt::fmt-header-only
        TracyClient
        nlohmann_json::nlohmann_json
        TT::Metalium::HostDevCommon
        TT::STL
        tt-logger::tt-logger
        TT::Metalium::Types
        TT::FabricTopologyUtils
        ${EXECINFO_LIB}
    PRIVATE
        Metalium::Metal::Impl
        metal_common_libs
        common
        jit_build
        llrt
        detail
        distributed
        fabric
        tt_metal_noc_estimator
        FlatBuffers::FlatBuffers
        TT::Metalium::Logging
        TT::ScaleoutTools
        Reflect::Reflect
)

tt_reuse_precompile_headers(tt_metal TT::CommonPCH)

target_include_directories(
    tt_metal
    PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/api>"
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/api>"
    PRIVATE
        api/tt-metalium # FIXME: Re-home the tests and remove this
)
target_compile_options(tt_metal PUBLIC -Wno-int-to-pointer-cast)

if(TT_METAL_USE_EMULE)
    target_compile_definitions(tt_metal PRIVATE TT_METAL_USE_EMULE=1)
    # PRIVATE: only libtt_metal.so itself needs -rdynamic (to export TLS symbols
    # for dlopen'd JIT kernels). No need to propagate to all consumers.
    target_link_options(tt_metal PRIVATE -rdynamic)
    # tt-emule::runtime is the emulation engine's OBJECT library; link its objects into
    # libtt_metal.so so the -rdynamic export above promotes their dlsym surface.
    target_link_libraries(tt_metal PRIVATE tt-emule::runtime)
endif()

add_dependencies(tt_metal hw_toolchain)

set_target_properties(
    tt_metal
    PROPERTIES
        INSTALL_RPATH
            "/opt/openmpi-v5.0.7-ulfm/lib;$ORIGIN"
        ADDITIONAL_CLEAN_FILES
            "${PROJECT_BINARY_DIR}/lib;${PROJECT_BINARY_DIR}/obj"
)

if(BUILD_PROGRAMMING_EXAMPLES)
    add_subdirectory(programming_examples)
endif()

# Allow internal files to access the public API "by default" and without the
# scoping that external consumers must use.  Scoping may still be used if desired.
include_directories(
    api
    api/tt-metalium
    ${CMAKE_CURRENT_BINARY_DIR}/api
)

# ---------------------------------------------------------------------------
# TT::Metalium::Types: runtime-free value types shared by libtt_metal and TT::FabricTopologyUtils (MeshShape /
# MeshCoordinate / ShapeBase, FabricNodeId & friends, PhysicalNodeId). A shared library so there is exactly one copy:
# libtt_metal.so and libtt-fabric-topology-utils.so both link it. Sources are added by common/ and fabric/.
# ---------------------------------------------------------------------------
add_library(tt_metalium_types SHARED)
add_library(TT::Metalium::Types ALIAS tt_metalium_types)
set_target_properties(
    tt_metalium_types
    PROPERTIES
        EXPORT_NAME
            Metalium::Types
        OUTPUT_NAME
            tt-metalium-types
        INSTALL_RPATH
            "$ORIGIN"
)
target_include_directories(
    tt_metalium_types
    PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/api>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR} # distributed/mesh_coord_utils.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/api/tt-metalium # bare "shape_base.hpp" / <mesh_coord.hpp> in common/ (same FIXME as tt_metal)
)
target_link_libraries(
    tt_metalium_types
    PUBLIC
        TT::STL
        fmt::fmt-header-only
    PRIVATE
        enchantum::enchantum
        Boost::asio # boost/move header included by common/mesh_coord.cpp
        umd::tt-umd # resolves spdlog from libtt-umd.so instead of bundling a second copy (ASan ODR)
)
target_compile_features(tt_metalium_types PUBLIC cxx_std_20)
TT_ENABLE_UNITY_BUILD(tt_metalium_types)
tt_reuse_precompile_headers(tt_metalium_types TT::CommonPCH)
install(TARGETS tt_metalium_types EXPORT Metalium LIBRARY COMPONENT metalium-runtime)

add_subdirectory(logging)
add_subdirectory(hw)
add_subdirectory(hostdevcommon)
add_subdirectory(common)
add_subdirectory(jit_build)
add_subdirectory(llrt)
add_subdirectory(tools)
add_subdirectory(impl)
add_subdirectory(detail)
add_subdirectory(distributed)
add_subdirectory(fabric)
add_subdirectory(impl/experimental/udm)
add_subdirectory(impl/experimental/noc_estimator)
if(TT_METAL_BUILD_TESTS)
    add_subdirectory(test)
endif()

if(TT_INSTALL)
    install(
        TARGETS
            tt_metal
        EXPORT Metalium
        LIBRARY
            COMPONENT metalium-runtime
        FILE_SET
        api
            DESTINATION COMPONENT
            metalium-dev
    )
    install(EXPORT Metalium NAMESPACE TT:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tt-metalium COMPONENT metalium-dev)

    install(
        TARGETS
            jitapi
        FILE_SET
        jit_api
            DESTINATION
                ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/tt_metal # FIXME: fix the include paths for jit_build
            COMPONENT metalium-runtime
    )

    # 3rd party dependencies we export in our public API.
    # We must ship not only these libraries, but their public headers as well
    # ... or refactor our public API.
    install(
        TARGETS
            reflect
            enchantum
            TracyClient
            tt-logger
        EXPORT Metalium
        FILE_SET
        api
            COMPONENT metalium-dev
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/metalium-thirdparty
    )
endif()
