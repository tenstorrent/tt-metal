# Provide unique architecture specific translation units and include paths for each ARCH

include(sources.cmake)

# Usage requirements for compiling against llrt's headers, without llrt's object
# files. See TT::Metalium::Impl::Headers in ../impl/CMakeLists.txt (#57055).
add_library(llrt_headers INTERFACE)
add_library(TT::Metalium::LLRT::Headers ALIAS llrt_headers)
target_include_directories(llrt_headers INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

add_library(llrt OBJECT ${LLRT_SRC})
add_library(Metalium::Metal::LLRT ALIAS llrt)

# Enable unity builds for faster compilation
TT_ENABLE_UNITY_BUILD(llrt)
tt_reuse_precompile_headers(llrt metal_common_pch)

add_subdirectory(hal)
add_dependencies(llrt hal_generate_dev_msgs_header)

target_link_libraries(
    llrt
    PUBLIC
        llrt_headers
        umd::tt-umd
        TT::Metalium::HostDevCommon
    PRIVATE
        HAL::1xx
        HAL::2xx
        Tracy::TracyClient
        nlohmann_json::nlohmann_json
        Reflect::Reflect
        yaml-cpp::yaml-cpp
        enchantum::enchantum
        TT::Metalium::Common
        tt-logger::tt-logger
        Metalium::Metal::Hardware
)
# Public include directory lives on llrt_headers (linked PUBLIC above).
target_compile_options(llrt PRIVATE -Wno-int-to-pointer-cast)

include(GNUInstallDirs)

set(TT_METAL_DEFAULT_INSTALL_ROOT "/usr/${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium")
target_compile_definitions(llrt PRIVATE TT_METAL_INSTALL_ROOT="${TT_METAL_DEFAULT_INSTALL_ROOT}")

if(TT_METAL_USE_EMULE)
    target_compile_definitions(llrt PRIVATE TT_METAL_USE_EMULE=1)
endif()

message(STATUS "TT_METAL_INSTALL_ROOT == ${TT_METAL_DEFAULT_INSTALL_ROOT}")
