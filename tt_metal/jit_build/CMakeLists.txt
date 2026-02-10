# Include source file lists (module owners maintain sources.cmake)
include(sources.cmake)

add_library(jit_build OBJECT ${JIT_BUILD_SRCS})

# Enable unity builds for faster compilation
TT_ENABLE_UNITY_BUILD(jit_build)

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
