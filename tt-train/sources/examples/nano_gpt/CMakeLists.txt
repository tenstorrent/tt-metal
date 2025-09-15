project(nano_gpt)

set(MAIN_SOURCES
    main.cpp
    utils.cpp
    3tier/remote_optimizer.cpp
    3tier/common.cpp
)

set(OPTIMIZER_SOURCES
    3tier/optimizer_worker.cpp
    3tier/common.cpp
)

set(AGGREGATOR_SOURCES
    3tier/aggregator_worker.cpp
    3tier/common.cpp
)

add_executable(nano_gpt ${MAIN_SOURCES})
target_link_libraries(nano_gpt PRIVATE ttml)
add_executable(nano_gpt_aggregator ${AGGREGATOR_SOURCES})
target_link_libraries(nano_gpt_aggregator PRIVATE ttml)
add_executable(nano_gpt_optimizer ${OPTIMIZER_SOURCES})
target_link_libraries(nano_gpt_optimizer PRIVATE ttml)

target_compile_definitions(
    nano_gpt
    PRIVATE
        DATA_FOLDER="${CMAKE_SOURCE_DIR}/data"
        CONFIGS_FOLDER="${CMAKE_SOURCE_DIR}/configs"
)
target_compile_definitions(
    nano_gpt_aggregator
    PRIVATE
        DATA_FOLDER="${CMAKE_SOURCE_DIR}/data"
        CONFIGS_FOLDER="${CMAKE_SOURCE_DIR}/configs"
)
target_compile_definitions(
    nano_gpt_optimizer
    PRIVATE
        DATA_FOLDER="${CMAKE_SOURCE_DIR}/data"
        CONFIGS_FOLDER="${CMAKE_SOURCE_DIR}/configs"
)

# Define the target file location
set(SHAKESPEARE_URL "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt")
set(SHAKESPEARE_FILE "${CMAKE_SOURCE_DIR}/data/shakespeare.txt")

# Check if the file already exists before downloading
if(NOT EXISTS "${SHAKESPEARE_FILE}")
    message(STATUS "Downloading Shakespeare text file to ${SHAKESPEARE_FILE}")
    file(
        DOWNLOAD
            ${SHAKESPEARE_URL}
            ${SHAKESPEARE_FILE}
        SHOW_PROGRESS
    )
else()
    message(STATUS "Shakespeare text file already exists at ${SHAKESPEARE_FILE}, skipping download.")
endif()
