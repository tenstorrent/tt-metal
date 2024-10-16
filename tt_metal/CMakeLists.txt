
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/hw)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/common)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/jit_build)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/llrt)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tools)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/impl)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/detail)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/distributed)

set(TT_METAL_OBJECTS
    ${CMAKE_CURRENT_SOURCE_DIR}/tt_metal.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/graph/graph_tracking.cpp
    $<TARGET_OBJECTS:profiler>
    $<TARGET_OBJECTS:common>
    $<TARGET_OBJECTS:jit_build>
    $<TARGET_OBJECTS:llrt>
    $<TARGET_OBJECTS:impl>
    $<TARGET_OBJECTS:detail>
    $<TARGET_OBJECTS:distributed>
)

add_library(tt_metal ${TT_METAL_OBJECTS})

target_link_libraries(tt_metal PRIVATE metal_header_directories umd_device metal_common_libs compiler_flags magic_enum fmt)

target_precompile_headers(tt_metal PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/tracy/public/tracy/Tracy.hpp
    <functional>
    <map>
    <memory>
    <unordered_map>
    <variant>
    <vector>
)

target_link_libraries(tt_metal PRIVATE $<$<BOOL:${ENABLE_TRACY}>:TracyClient>)
target_include_directories(tt_metal
    PRIVATE
    ${UMD_HOME}
    ${PROJECT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
)
add_dependencies(tt_metal hw_toolchain)

set_target_properties(tt_metal PROPERTIES
    INSTALL_RPATH "${PROJECT_BINARY_DIR}/lib"
    ADDITIONAL_CLEAN_FILES "${PROJECT_BINARY_DIR}/lib;${PROJECT_BINARY_DIR}/obj"
)

if(BUILD_PROGRAMMING_EXAMPLES)
    add_subdirectory(programming_examples)
endif(BUILD_PROGRAMMING_EXAMPLES)
