
set(COMMON_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/core_descriptor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/metal_soc_descriptor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tt_backend_api_types.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils.cpp
)

add_library(common OBJECT ${COMMON_SRCS})
target_link_libraries(common PRIVATE yaml-cpp::yaml-cpp)
target_link_libraries(common PUBLIC compiler_flags metal_header_directories magic_enum fmt)

target_include_directories(common PUBLIC
    ${UMD_HOME}
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/tt_metal
    ${PROJECT_SOURCE_DIR}/tt_metal/third_party/fmt
)
