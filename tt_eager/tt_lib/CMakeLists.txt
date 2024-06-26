
if(NOT WITH_PYTHON_BINDINGS)
    message(FATAL_ERROR "Remove include of tt_lib when WITH_PYTHON_BINDINGS flag is OFF")
endif()

set(TT_LIB_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/tt_lib_bindings.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/type_caster.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/tt_lib_bindings_tensor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/tt_lib_bindings_tensor_composite_ops.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/tt_lib_bindings_tensor_backward_ops.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/tt_lib_bindings_tensor_pytensor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/tt_lib_bindings_tensor_dm_ops.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/tt_lib_bindings_tensor_custom_bmm_ops.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/tt_lib_bindings_tensor_xary_ops.cpp
)
# TODO: should be using pybind11_add_module, but right now it introduces many build problems
# pybinds will always be built as a shared library
add_library(tt_lib SHARED ${TT_LIB_SRCS})
target_link_libraries(tt_lib PUBLIC compiler_flags linker_flags tt_eager tt_metal pch_pybinds)  # linker_flags = -rdynamic if tracy enabled
target_include_directories(tt_lib PUBLIC
    ${UMD_HOME}
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/tt_metal
    ${PROJECT_SOURCE_DIR}/ttnn/cpp
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/tt_metal/third_party/pybind11/include
    ${Python3_INCLUDE_DIRS}
)
target_compile_options(tt_lib PUBLIC -fno-var-tracking -Wno-int-to-pointer-cast)
target_link_directories(tt_lib PUBLIC
    ${PROJECT_BINARY_DIR}/tt_eager
    ${PROJECT_BINARY_DIR}/tt_metal
    ${Python3_LIBRARIES}
)

# Make sure library built is _C.so and that it can find all it's linked libraries
# tt_lib breaks if -fvisibility=hidden, so CXX_VISIBILITY_PRESET set to default
set_target_properties(tt_lib PROPERTIES
    OUTPUT_NAME "_C"
    PREFIX ""
    SUFFIX ".so"
    BUILD_RPATH "${PROJECT_BINARY_DIR}/tt_metal;${PROJECT_BINARY_DIR}/tt_eager"
    INSTALL_RPATH "${PROJECT_BINARY_DIR}/lib"
    CXX_VISIBILITY_PRESET "default"
    ADDITIONAL_CLEAN_FILES "${PROJECT_SOURCE_DIR}/tt_eager/tt_lib/_C.so;${PROJECT_SOURCE_DIR}/tt_eager/metal_libs.egg-info"
)
