project(nano_gpt)

set(MAIN_SOURCES
    main.cpp
    utils.cpp
)

set(llama_inference_SOURCES
    llama_inference.cpp
    utils.cpp
)

add_executable(nano_gpt ${MAIN_SOURCES})
target_link_libraries(nano_gpt PRIVATE ttml)
add_executable(llama_inference ${llama_inference_SOURCES})
target_link_libraries(llama_inference PRIVATE ttml)

target_compile_definitions(
    nano_gpt
    PRIVATE
        DATA_FOLDER="${CMAKE_SOURCE_DIR}/data"
        CONFIGS_FOLDER="${TT_TRAIN_SOURCE_DIR}/configs"
)
target_compile_definitions(llama_inference PRIVATE CONFIGS_FOLDER="${TT_TRAIN_SOURCE_DIR}/configs")

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

tt_reuse_precompile_headers(nano_gpt ttml_pch)
tt_reuse_precompile_headers(llama_inference ttml_pch)
