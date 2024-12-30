set(COMMON_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/core_assignment.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/core_coord.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/core_descriptor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/metal_soc_descriptor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tt_backend_api_types.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/work_split.cpp
)

add_library(common STATIC ${COMMON_SRCS}) # FIXME(14541): Should be OBJECT, but can't handle circular deps between Object libs
add_library(Metalium::Metal::Common ALIAS common)

target_link_libraries(common PRIVATE yaml-cpp::yaml-cpp)
target_link_libraries(
    common
    PUBLIC
        nlohmann_json::nlohmann_json
        magic_enum
        fmt::fmt-header-only
        span
        Metalium::Metal::STL
        Metalium::Metal::LLRT
        umd::Firmware
    PRIVATE
        Tracy::TracyClient
)

target_include_directories(
    common
    PUBLIC
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/tt_metal
)
