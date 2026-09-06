# GUI-free .tracy inspector used by the streaming-profiler test and by hand.
#
# It links the same TracyServer library that tracy-capture / tracy-csvexport use (defined by
# tt_metal/third_party/tracy/cmake/server.cmake when tt_metal/third_party adds those tools), so the
# Tracy struct layouts match the writer's -- compiling the server sources separately with different
# flags is an ODR trap.  The binary lands next to tracy-capture in build/tools/profiler/bin so Python
# tooling can find them via tools.tracy.common.PROFILER_BIN_DIR.

if(NOT TARGET TracyServer)
    message(STATUS "TracyServer target not available; skipping tracy_zone_csv")
    return()
endif()

find_package(Threads REQUIRED)

set(_tt_tracy_root "${PROJECT_SOURCE_DIR}/tt_metal/third_party/tracy")
set(_tt_tracy_cli_bin_dir "${CMAKE_BINARY_DIR}/tools/profiler/bin")

# Tracy's cmake/config.cmake turns on CMAKE_INTERPROCEDURAL_OPTIMIZATION for its own directory, so
# libTracyServer.a is thin-LTO bitcode; a non-LTO link of it fails ("unable to handle this LTO object
# file").  Mirror TracyServer's IPO setting so the tools link the same way tracy-capture does.
get_target_property(_tt_tracy_ipo TracyServer INTERPROCEDURAL_OPTIMIZATION)
if(NOT _tt_tracy_ipo)
    set(_tt_tracy_ipo OFF)
endif()

add_executable(tracy_zone_csv ${CMAKE_CURRENT_SOURCE_DIR}/tracy_zone_csv/tracy_zone_csv.cpp)
# The source includes "server/TracyWorker.hpp" and "public/common/TracyTTDeviceData.hpp" relative to the Tracy
# root; TracyServer only exports its public/common and server dirs themselves.
target_include_directories(tracy_zone_csv SYSTEM PRIVATE ${_tt_tracy_root})
target_link_libraries(
    tracy_zone_csv
    PRIVATE
        TracyServer
        Threads::Threads
        ${CMAKE_DL_LIBS}
)
# Same suppressions/baseline tt_metal/third_party applies to the Tracy CLI targets.
target_compile_options(tracy_zone_csv PRIVATE -Wno-deprecated-declarations)
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    target_compile_options(tracy_zone_csv PRIVATE -march=x86-64-v2)
endif()
set_target_properties(
    tracy_zone_csv
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            "${_tt_tracy_cli_bin_dir}"
        INTERPROCEDURAL_OPTIMIZATION
            ${_tt_tracy_ipo}
)

unset(_tt_tracy_root)
unset(_tt_tracy_cli_bin_dir)
unset(_tt_tracy_ipo)
