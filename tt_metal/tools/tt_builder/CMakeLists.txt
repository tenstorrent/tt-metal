add_library(tt_builder_lib OBJECT ${CMAKE_CURRENT_SOURCE_DIR}/builder.cpp)
target_link_libraries(tt_builder_lib PRIVATE tt_metal)
target_include_directories(
    tt_builder_lib
    PRIVATE
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/tt_metal
        ${PROJECT_SOURCE_DIR}/tt_metal/common
        ${CMAKE_CURRENT_SOURCE_DIR}
)

add_executable(tt_builder ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
target_link_libraries(
    tt_builder
    PRIVATE
        tt_metal
        tt_builder_lib
)
target_include_directories(
    tt_builder
    PRIVATE
        ${PROJECT_SOURCE_DIR}/tt_metal/common
        ${CMAKE_CURRENT_SOURCE_DIR}
)
set_target_properties(
    tt_builder
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/tools
)
