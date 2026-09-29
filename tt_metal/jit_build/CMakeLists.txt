include(sources.cmake)

add_library(jit_build OBJECT ${JIT_BUILD_SRCS})

# Enable unity builds for faster compilation
TT_ENABLE_UNITY_BUILD(jit_build)
tt_reuse_precompile_headers(jit_build metal_common_pch)

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
        yaml-cpp::yaml-cpp
)

if(ENABLE_FAKE_KERNELS_TARGET)
    add_subdirectory(fake_kernels_target)
endif()
