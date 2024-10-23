set(JIT_BUILD_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/build.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/data_format.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/genfiles.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel_args.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/settings.cpp
)

add_library(jit_build OBJECT ${JIT_BUILD_SRCS})
target_link_libraries(jit_build PUBLIC common)
