project(bert_example)
set(SOURCES main.cpp)
add_executable(bert_example ${SOURCES})
target_link_libraries(bert_example PRIVATE ttml)

# Add compile definitions for data and config paths
target_compile_definitions(
    bert_example
    PRIVATE
        DATA_FOLDER="${CMAKE_SOURCE_DIR}/data"
        CONFIGS_FOLDER="${CMAKE_SOURCE_DIR}/configs"
)
