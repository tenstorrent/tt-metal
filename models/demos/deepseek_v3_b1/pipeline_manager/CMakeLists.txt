cmake_minimum_required(VERSION 3.22)
project(pipeline_manager VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

option(PM_ENABLE_TSAN "Enable thread sanitizer" OFF)
option(PM_BUILD_TESTS "Build test binaries" ON)
option(PM_BUILD_TOOLS "Build launcher/harness" ON)

if(PM_ENABLE_TSAN)
    add_compile_options(
        -fsanitize=thread
        -g
    )
    add_link_options(-fsanitize=thread)
endif()

find_package(Threads REQUIRED)

# ===========================================================================
#  Detect TT::Metalium — may already be in scope from a parent
#  add_subdirectory(external/tt-metal), or found via find_package.
# ===========================================================================
if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium QUIET CONFIG)
endif()

# d2h_socket.hpp / h2d_socket.hpp are not yet part of the installed
# TT-Metalium package.  When building against an installed package,
# point TT_METAL_SOURCE_DIR at the tt-metal source tree so we can
# pick up the not-yet-exported socket headers.
set(TT_METAL_SOURCE_DIR "" CACHE PATH "Path to tt-metal source tree (needed until socket headers are installed)")

# ===========================================================================
#  Core library — scheduling logic only, no hardware/socket dependencies.
#  Always built. Produces libpipeline_manager_core.so.
# ===========================================================================
add_library(pipeline_manager_core SHARED src/pipeline_manager.cpp)
target_include_directories(
    pipeline_manager_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(pipeline_manager_core PUBLIC Threads::Threads)
set_target_properties(
    pipeline_manager_core
    PROPERTIES
        VERSION
            ${PROJECT_VERSION}
        SOVERSION
            1
        EXPORT_NAME
            Core
)
add_library(PipelineManager::Core ALIAS pipeline_manager_core)

# ===========================================================================
#  Full library — adds SocketPipeline backed by H2D/D2H sockets.
#  Only built when TT::Metalium is available.
#  Produces libpipeline_manager.so.
# ===========================================================================
if(TARGET TT::Metalium)
    add_library(
        pipeline_manager
        SHARED
        src/pipeline_manager.cpp
        src/socket_connector.cpp
        src/socket_pipeline.cpp
    )
    target_include_directories(
        pipeline_manager
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    # Workaround: d2h/h2d socket headers are not yet part of the
    # installed TT-Metalium package. Pull them from the source tree.
    if(TT_METAL_SOURCE_DIR)
        target_include_directories(pipeline_manager SYSTEM PUBLIC ${TT_METAL_SOURCE_DIR}/tt_metal/api)
    endif()
    target_link_libraries(
        pipeline_manager
        PUBLIC
            TT::Metalium
            Threads::Threads
    )
    target_compile_definitions(pipeline_manager PUBLIC PM_HAS_METALIUM=1)
    set_target_properties(
        pipeline_manager
        PROPERTIES
            VERSION
                ${PROJECT_VERSION}
            SOVERSION
                1
            EXPORT_NAME
                Full
    )
    add_library(PipelineManager::Full ALIAS pipeline_manager)

    message(STATUS "TT::Metalium found — building pipeline_manager (full) with socket support")
else()
    message(STATUS "TT::Metalium not found — building pipeline_manager_core only (mock pipeline)")
endif()

# ===========================================================================
#  Tests
# ===========================================================================
if(PM_BUILD_TESTS)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.13.0
        FIND_PACKAGE_ARGS
            NAMES
            GTest
    )
    set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
    enable_testing()

    # Mock tests — always available, no hardware needed
    add_executable(test_pipeline_manager tests/test_pipeline_manager.cpp)
    target_link_libraries(
        test_pipeline_manager
        PRIVATE
            pipeline_manager_core
            gtest
            gtest_main
    )
    add_test(NAME PipelineManagerMock COMMAND test_pipeline_manager)

    # Device integration tests — only with Metalium
    if(TARGET pipeline_manager)
        add_executable(test_device_pipeline tests/test_device_pipeline.cpp)
        target_link_libraries(
            test_device_pipeline
            PRIVATE
                pipeline_manager
                gtest
                gtest_main
        )
        add_test(NAME PipelineManagerDevice COMMAND test_device_pipeline)
    endif()

    # Umbrella target for all tests
    add_custom_target(pipeline_manager_tests DEPENDS test_pipeline_manager)
    if(TARGET test_device_pipeline)
        add_dependencies(pipeline_manager_tests test_device_pipeline)
    endif()
endif()

# ===========================================================================
#  Tools
# ===========================================================================
if(PM_BUILD_TOOLS AND TARGET pipeline_manager)
    add_executable(pipeline_launcher tools/pipeline_launcher.cpp)
    target_link_libraries(pipeline_launcher PRIVATE pipeline_manager)
endif()

# ===========================================================================
#  Install rules — produces a find_package(pipeline-manager CONFIG) target.
#
#  Consumers link:
#    PipelineManager::Core  — standalone, no hardware
#    PipelineManager::Full  — with TT::Metalium socket support
# ===========================================================================
install(
    TARGETS
        pipeline_manager_core
    EXPORT pipeline-manager-targets
    LIBRARY
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

if(TARGET pipeline_manager)
    install(
        TARGETS
            pipeline_manager
        EXPORT pipeline-manager-targets
        LIBRARY
            DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE
            DESTINATION ${CMAKE_INSTALL_LIBDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )
endif()

install(DIRECTORY include/pipeline_manager DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(
    EXPORT pipeline-manager-targets
    FILE pipeline-manager-targets.cmake
    NAMESPACE PipelineManager::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/pipeline-manager
)

configure_package_config_file(
    cmake/pipeline-manager-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/pipeline-manager-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/pipeline-manager
)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/pipeline-manager-config-version.cmake
    COMPATIBILITY SameMajorVersion
)
install(
    FILES
        ${CMAKE_CURRENT_BINARY_DIR}/pipeline-manager-config.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/pipeline-manager-config-version.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/pipeline-manager
)
