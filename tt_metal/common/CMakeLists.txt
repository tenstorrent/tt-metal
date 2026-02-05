add_library(common OBJECT)
add_library(TT::Metalium::Common ALIAS common)

# Enable unity builds for faster compilation
TT_ENABLE_UNITY_BUILD(common)

# Include source file lists (module owners maintain sources.cmake)
include(sources.cmake)

# Use target_sources to gather the necessary sources
target_sources(common PRIVATE ${COMMON_SOURCES})

target_include_directories(
    common
    PUBLIC
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/tt_metal
        ${PROJECT_SOURCE_DIR}/tt_metal/llrt # Hack - circular dep, generated dev_msgs.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(
    common
    PUBLIC
        nlohmann_json::nlohmann_json
        enchantum::enchantum
        fmt::fmt-header-only
        TT::STL
        umd::Firmware
        umd::device
        simde::simde
        Taskflow::Taskflow
    PRIVATE
        Metalium::Metal::Hardware
        Tracy::TracyClient
        TT::Metalium::HostDevCommon
        yaml-cpp::yaml-cpp
        Boost::asio
        tt-logger::tt-logger
)

# Ensure generated headers (e.g., tt_metal/llrt/hal/generated/dev_msgs.hpp) exist before compiling this target
# thread_pool.cpp depends on hal/generated/dev_msgs.hpp
# Note that this targets dependency on llrt is not captured in cmake
# This would introduce a circular dependency if we were to add it
add_dependencies(common all_generated_files)
