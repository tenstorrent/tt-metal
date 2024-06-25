
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tensor)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/queue)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tt_dnn)

set(TT_EAGER_OBJECTS
    $<TARGET_OBJECTS:tt_dnn>
    $<TARGET_OBJECTS:tensor>
    $<TARGET_OBJECTS:queue>
)

# Defaults to a shared lib, static lib for production python wheel
add_library(tt_eager ${TT_EAGER_OBJECTS})
target_link_libraries(tt_eager PUBLIC compiler_flags linker_flags tt_metal ttnn_lib)     # linker_flags = -rdynamic if tracy enabled
target_include_directories(tt_eager PUBLIC
    ${UMD_HOME}
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/tt_metal
    ${PROJECT_SOURCE_DIR}/tt_eager
    ${PROJECT_SOURCE_DIR}/ttnn/cpp/
)

set_target_properties(tt_eager PROPERTIES
    INSTALL_RPATH "${PROJECT_BINARY_DIR}/lib"
)

if(WITH_PYTHON_BINDINGS)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tt_lib)
endif()
