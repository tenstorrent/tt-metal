
set(TENSOR_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/tensor_impl_wrapper.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tensor_impl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tensor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/types.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tensor_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/serialization.cpp
)

add_library(tensor STATIC ${TENSOR_SRCS})

target_link_libraries(tensor PUBLIC tt_metal_libs)
target_include_directories(tensor PUBLIC
                            ${UMD_HOME}
                            ${CMAKE_SOURCE_DIR}
                            ${CMAKE_SOURCE_DIR}/tt_metal
                            ${CMAKE_SOURCE_DIR}/tt_eager
                            ${CMAKE_CURRENT_SOURCE_DIR}
)
