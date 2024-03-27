
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/common)    # <- Tracy compiled in / linked to common
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/jit_build)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/llrt)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tools)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/impl)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/detail)


add_library(tt_metal OBJECT ${CMAKE_CURRENT_SOURCE_DIR}/tt_metal.cpp)
target_link_libraries(tt_metal PUBLIC compiler_flags metal_header_directories metal_common_libs)
target_include_directories(tt_metal PUBLIC
    ${UMD_HOME}
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/tt_metal/third_party/fmt
)
target_compile_options(tt_metal PUBLIC -Wno-int-to-pointer-cast)

if(${TT_METAL_CREATE_STATIC_LIB} EQUAL 1)
    add_library(tt_metal_libs STATIC $<TARGET_OBJECTS:tt_metal> $<TARGET_OBJECTS:profiler> $<TARGET_OBJECTS:common> $<TARGET_OBJECTS:jit_build> $<TARGET_OBJECTS:llrt> $<TARGET_OBJECTS:impl> $<TARGET_OBJECTS:detail> $<TARGET_OBJECTS:tracy_o>)
else()
    add_library(tt_metal_libs SHARED $<TARGET_OBJECTS:tt_metal> $<TARGET_OBJECTS:profiler> $<TARGET_OBJECTS:common> $<TARGET_OBJECTS:jit_build> $<TARGET_OBJECTS:llrt> $<TARGET_OBJECTS:impl> $<TARGET_OBJECTS:detail> $<TARGET_OBJECTS:tracy_o>)
endif()

target_link_libraries(tt_metal_libs PUBLIC compiler_flags metal_header_directories metal_common_libs device yaml-cpp)
target_link_directories(tt_metal_libs PUBLIC ${CMAKE_BINARY_DIR}/lib)
target_include_directories(tt_metal_libs PUBLIC ${CMAKE_SOURCE_DIR}/tt_metal/third_party/fmt)
target_compile_options(tt_metal_libs PUBLIC -Wno-int-to-pointer-cast)
