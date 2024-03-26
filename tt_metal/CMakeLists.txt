
if ("$ENV{ARCH_NAME}" STREQUAL "grayskull")
    include_directories(${UMD_HOME}/src/firmware/riscv/$ENV{ARCH_NAME} ${UMD_HOME}/device/$ENV{ARCH_NAME})
else()
    include_directories(${UMD_HOME}/src/firmware/riscv/$ENV{ARCH_NAME} ${UMD_HOME}/device/$ENV{ARCH_NAME})
    if ("$ENV{ARCH_NAME}" STREQUAL "wormhole")
        include_directories(${UMD_HOME}/src/firmware/riscv/wormhole/wormhole_a0_defines)
    else()
        include_directories(${UMD_HOME}/src/firmware/riscv/wormhole/wormhole_b0_defines)
    endif()
endif()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/common)    # <- Tracy compiled in / linked to common
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/jit_build)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/llrt)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tools)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/impl)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/detail)


add_library(tt_metal OBJECT ${CMAKE_CURRENT_SOURCE_DIR}/tt_metal.cpp)
target_include_directories(tt_metal PUBLIC ${UMD_HOME})
target_include_directories(tt_metal PUBLIC ${CMAKE_SOURCE_DIR})
target_include_directories(tt_metal PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(tt_metal PUBLIC ${CMAKE_SOURCE_DIR}/tt_metal/third_party/fmt)
target_compile_options(tt_metal PUBLIC ${METAL_COMPILE_OPTIONS} -DFMT_HEADER_ONLY -Werror -Wno-int-to-pointer-cast)

if(${TT_METAL_CREATE_STATIC_LIB} EQUAL 1)
    add_library(tt_metal_libs STATIC $<TARGET_OBJECTS:tt_metal> $<TARGET_OBJECTS:profiler> $<TARGET_OBJECTS:common> $<TARGET_OBJECTS:jit_build> $<TARGET_OBJECTS:llrt> $<TARGET_OBJECTS:impl> $<TARGET_OBJECTS:detail> $<TARGET_OBJECTS:tracy>)
else()
    add_library(tt_metal_libs SHARED $<TARGET_OBJECTS:tt_metal> $<TARGET_OBJECTS:profiler> $<TARGET_OBJECTS:common> $<TARGET_OBJECTS:jit_build> $<TARGET_OBJECTS:llrt> $<TARGET_OBJECTS:impl> $<TARGET_OBJECTS:detail> $<TARGET_OBJECTS:tracy>)
endif()

target_link_libraries(tt_metal_libs PUBLIC device yaml-cpp)
target_link_directories(tt_metal_libs PUBLIC ${CMAKE_BINARY_DIR}/lib)

target_include_directories(tt_metal_libs PUBLIC ${CMAKE_SOURCE_DIR}/tt_metal/third_party/fmt)
target_include_directories(tt_metal_libs PUBLIC ${CMAKE_SOURCE_DIR}/tt_metal/hw)

target_compile_options(tt_metal_libs PUBLIC -DFMT_HEADER_ONLY)
target_compile_options(tt_metal_libs PRIVATE ${METAL_COMPILE_OPTIONS} -Wno-int-to-pointer-cast)
