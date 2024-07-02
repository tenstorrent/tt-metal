
set(TENSOR_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/tensor_impl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tensor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/types.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tensor_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/serialization.cpp

    CACHE INTERNAL "Tensor sources to reuse in ttnn build"
)

# add_library(tensor OBJECT ${TENSOR_SRCS})

# target_link_libraries(tensor PUBLIC metal_header_directories compiler_flags)
# target_include_directories(tensor PUBLIC
#     ${UMD_HOME}
#     ${PROJECT_SOURCE_DIR}
#     ${PROJECT_SOURCE_DIR}/tt_metal
#     ${PROJECT_SOURCE_DIR}/tt_eager
#     ${CMAKE_CURRENT_SOURCE_DIR}
#     ${PROJECT_SOURCE_DIR}/tt_metal/third_party/fmt
# )
