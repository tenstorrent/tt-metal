set(JIT_BUILD_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/build.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/build_env_manager.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/data_format.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/depend.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/genfiles.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/jit_build_cache.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel_args.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/jit_build_options.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/jit_build_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/precompiled.cpp
)

add_library(jit_build OBJECT ${JIT_BUILD_SRCS})

# Enable unity builds for faster compilation
TT_ENABLE_UNITY_BUILD(jit_build)
target_precompile_headers(jit_build REUSE_FROM metal_common_pch)

target_link_libraries(
    jit_build
    PUBLIC
        common
    PRIVATE
        TT::Metalium::Common
        Metalium::Metal::Impl
        Metalium::Metal::LLRT
        Tracy::TracyClient
        TT::Metalium::HostDevCommon
)

if(DEFINED VERSION_HASH)
    target_compile_definitions(jit_build PRIVATE "-DGIT_COMMIT_HASH=\"${VERSION_HASH}\"")
endif()

if(ENABLE_FAKE_KERNELS_TARGET)
    add_subdirectory(fake_kernels_target)
endif()
