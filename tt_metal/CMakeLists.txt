
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/common)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/jit_build)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/llrt)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tools)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/impl)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/detail)

set(TT_METAL_OBJECTS
    ${CMAKE_CURRENT_SOURCE_DIR}/tt_metal.cpp
    $<TARGET_OBJECTS:profiler>
    $<TARGET_OBJECTS:common>
    $<TARGET_OBJECTS:jit_build>
    $<TARGET_OBJECTS:llrt>
    $<TARGET_OBJECTS:impl>
    $<TARGET_OBJECTS:detail>
)

add_library(tt_metal ${TT_METAL_OBJECTS})
if(BUILD_SHARED_LIBS)
    target_link_libraries(tt_metal PUBLIC device)
    add_dependencies(tt_metal umd_device)
else()
    target_link_libraries(tt_metal PUBLIC ${UMD_STATIC_LIB} metal_common_libs)
    add_dependencies(tt_metal umd_static_lib_target)
endif()

target_link_libraries(tt_metal PUBLIC compiler_flags linker_flags metal_header_directories yaml-cpp $<$<BOOL:$ENV{ENABLE_TRACY}>:TracyClient>)  # linker_flags = -rdynamic if tracy enabled
target_link_directories(tt_metal PUBLIC ${CMAKE_BINARY_DIR}/lib)    # required so tt_metal can find device library
target_include_directories(tt_metal PUBLIC
    ${UMD_HOME}
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/tt_metal/third_party/fmt
)
target_compile_options(tt_metal PUBLIC -Wno-int-to-pointer-cast)
add_dependencies(tt_metal hw_toolchain)

set_target_properties(tt_metal PROPERTIES
    INSTALL_RPATH "${CMAKE_BINARY_DIR}/lib"
    ADDITIONAL_CLEAN_FILES "${CMAKE_BINARY_DIR}/lib;${CMAKE_BINARY_DIR}/obj"
)
