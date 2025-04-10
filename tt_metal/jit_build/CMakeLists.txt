set(JIT_BUILD_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/build.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/build_env_manager.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/data_format.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/genfiles.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel_args.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/jit_build_options.cpp
)

add_library(jit_build OBJECT ${JIT_BUILD_SRCS})
target_link_libraries(
    jit_build
    PUBLIC
        common
    PRIVATE
        TT::Metalium::Common
        Metalium::Metal::Impl
        Metalium::Metal::LLRT
        Tracy::TracyClient
        Taskflow::Taskflow
        TT::Metalium::HostDevCommon
)

if(DEFINED VERSION_HASH)
    target_compile_definitions(jit_build PRIVATE "-DGIT_COMMIT_HASH=\"${VERSION_HASH}\"")
endif()
